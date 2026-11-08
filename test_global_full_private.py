import pytest
import torch

from test_shared_prefix import PrefixToy, setup
from continual_lora.cqsm_accounting import cqsm_parameter_accounting
from continual_lora.methods import CQSMDualLoRALinear, shared_only_adapter_forward
from continual_lora.trainer import adapter_state_dict, load_adapter_checkpoint


def test_global_off_exactly_recovers_original_private_scope_and_outputs():
    old, oc, ok, x, old_report = setup(layout='late_parallel', shared=False)
    new, nc, nk, _, new_report = setup(layout='visual_residual', shared=False)
    assert old_report.group_ranks == new_report.group_ranks
    assert old_report.modules == new_report.modules
    old_counts = cqsm_parameter_accounting(old, oc, 0)['counts']
    new_counts = cqsm_parameter_accounting(new, nc, 0)['counts']
    assert old_counts == new_counts
    for name, state in old.state_dict().items():
        torch.testing.assert_close(state, new.state_dict()[name], rtol=0, atol=0, equal_nan=True,
                                   msg=lambda msg: name + ': ' + msg)
    old.eval(); new.eval(); oc.begin_anchor_batch(); nc.begin_anchor_batch()
    assert torch.equal(old(x), new(x))
    assert nk.loss_and_gradients() == (None, None)
    ok.close(); nk.close()


def test_global_is_additive_and_preserves_all_private_maps_and_selector():
    model, ctl, keeper, x, report = setup(layout='visual_residual')
    reference, rc, rk, _, rr = setup(layout='late_parallel', shared=False)
    expected = cqsm_parameter_accounting(reference, rc, 0)['counts']['private_current_optimizable']
    assert cqsm_parameter_accounting(model, ctl, 0)['counts']['private_current_optimizable'] == expected
    assert report.group_ranks == rr.group_ranks
    assert all(m.private_enabled for m in model.modules() if isinstance(m, CQSMDualLoRALinear))
    assert keeper.block_ids == [4, 5]
    model.eval(); ctl.begin_anchor_batch(); model(x)
    descriptor = ctl.anchor_router()._visual_descriptor.clone()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, CQSMDualLoRALinear):
                m.lora_B[0].normal_(std=.02)
        for _, m in keeper.shared:
            m.global_B.add_(.03*m.global_output_mask)
    ctl.begin_anchor_batch(); model(x)
    assert torch.equal(descriptor, ctl.anchor_router()._visual_descriptor)
    keeper.close(); rk.close()


@pytest.mark.parametrize('checkpointed', [False, True])
def test_private_changes_do_not_change_shared_retention_target(checkpointed):
    model, ctl, keeper, x, _ = setup(checkpointed, layout='visual_residual')
    keeper.begin_task(1)
    torch.manual_seed(117); ctl.begin_anchor_batch(); before = model(x)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, CQSMDualLoRALinear):
                m.lora_B[0].normal_(std=.2)
    torch.manual_seed(117); keeper.begin_batch(); ctl.begin_anchor_batch(); after = model(x)
    assert not torch.allclose(before, after)
    rng = torch.get_rng_state().clone()
    value, gradients = keeper.loss_and_gradients()
    assert value == 0 and all(torch.count_nonzero(g) == 0 for g in gradients)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    # The native graph remains usable after auxiliary checkpoint replay.
    after.square().mean().backward()
    assert any(m.lora_B.grad is not None and m.lora_B.grad.abs().sum() > 0
               for m in model.modules() if isinstance(m, CQSMDualLoRALinear))
    keeper.close()


@pytest.mark.parametrize('checkpointed', [False, True])
def test_shared_retention_gradients_do_not_depend_on_private_weights(checkpointed):
    model, ctl, keeper, x, _ = setup(checkpointed, layout='visual_residual')
    keeper.begin_task(1)
    with torch.no_grad():
        for _, m in keeper.shared: m.global_B.add_(.03*m.global_output_mask)
    results = []
    for perturb_private in (False, True):
        if perturb_private:
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, CQSMDualLoRALinear): m.lora_B[0].normal_(std=.2)
        torch.manual_seed(121); keeper.begin_batch(); ctl.begin_anchor_batch(); y = model(x)
        descriptor = ctl.anchor_router()._visual_descriptor.clone()
        route = ctl.anchor_router()._visual_route.clone()
        rng = torch.get_rng_state().clone()
        value, gradients = keeper.loss_and_gradients()
        assert value > 0
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.equal(descriptor, ctl.anchor_router()._visual_descriptor)
        assert torch.equal(route, ctl.anchor_router()._visual_route)
        assert all(p.grad is None for p in model.parameters())
        results.append((value, gradients))
    assert torch.equal(results[0][0], results[1][0])
    assert all(torch.equal(a, b) for a, b in zip(results[0][1], results[1][1]))
    y.square().mean().backward()
    private_before = {id(p): p.grad.clone() for n, p in model.named_parameters()
                      if p.grad is not None and not n.endswith('global_B')}
    report = keeper.apply(gradients)
    assert report['shared_retention_gradient_ratio'] == pytest.approx(.2, rel=1e-5)
    assert all(torch.equal(private_before[id(p)], p.grad)
               for p in model.parameters() if id(p) in private_before)
    assert not keeper.captures and not keeper.active
    keeper.close()


def test_multiple_native_forwards_average_retention_without_side_effects():
    model, ctl, keeper, x, _ = setup(True, layout='visual_residual')
    keeper.begin_task(1)
    with torch.no_grad():
        for _, m in keeper.shared: m.global_B.add_(.03*m.global_output_mask)
    result = []
    for count in (1, 2):
        keeper.begin_batch()
        for _ in range(count):
            torch.manual_seed(171); ctl.begin_anchor_batch(); model(x)
        assert len(keeper.captures) == count
        result.append(keeper.loss_and_gradients())
    assert torch.equal(result[0][0], result[1][0])
    assert all(torch.equal(a, b) for a, b in zip(result[0][1], result[1][1]))
    keeper.close()


def test_zero_ratio_uses_the_ordinary_full_private_forward_only():
    model, ctl, keeper, x, _ = setup(True, layout='visual_residual', retention_ratio=0.)
    keeper.begin_task(1); keeper.begin_batch(); ctl.begin_anchor_batch(); model(x)
    assert not keeper.captures and keeper.loss_and_gradients() == (None, None)
    keeper.close()


def test_shared_only_context_restores_native_path_after_exception():
    model, ctl, keeper, x, _ = setup(layout='visual_residual')
    model.eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, CQSMDualLoRALinear): m.lora_B[0].normal_(std=.2)
    ctl.begin_anchor_batch(); expected = model(x)
    with pytest.raises(RuntimeError, match='test exception'):
        with shared_only_adapter_forward():
            pure = model(x)
            assert not torch.allclose(expected, pure)
            raise RuntimeError('test exception')
    ctl.begin_anchor_batch(); assert torch.equal(expected, model(x))
    keeper.close()


@pytest.mark.parametrize('shared', [False, True])
def test_schema4_checkpoint_preserves_overlap_and_task_free_inference(tmp_path, monkeypatch, shared):
    from continual_lora import trainer
    monkeypatch.setattr(trainer, 'enable_lora_training_compatibility', lambda: None)
    model, ctl, keeper, x, report = setup(layout='visual_residual', shared=shared)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, CQSMDualLoRALinear): m.lora_B[0].normal_(std=.02)
        for _, m in keeper.shared: m.global_B.add_(.02*m.global_output_mask)
    ctl.end_task(); model.eval(); ctl.begin_anchor_batch(); expected = model(x)
    cfg = dict(method='cqsm_dual_lora', rank=16, alpha=8, cqsm_schema_version=4,
               cqsm_restore_legacy=True, cqsm_shared_layout='visual_residual', global_rank=10 if shared else 0,
               cqsm_shared=shared, cqsm_private=True, cqsm_bank_mode='all', cqsm_basis='covariance',
               energy_calibration_lambda=3., focus_orthogonality_weight=.001,
               focus_group_ranks=report.group_ranks, stream=[{}, {}])
    path = tmp_path/'full_private.pt'
    torch.save(dict(config=cfg, adapter=adapter_state_dict(model), controller=ctl.state_dict(),
                    completed_task=0), path)
    torch.manual_seed(19); other = PrefixToy(); oc, _, _ = load_adapter_checkpoint(other, path)
    other.eval(); oc.begin_anchor_batch()
    assert torch.equal(expected, other(x))
    assert oc.anchor_router().inference_oracle_stage is None
    assert all(m.private_enabled for m in other.modules() if isinstance(m, CQSMDualLoRALinear))
    for m in other.modules():
        if isinstance(m, CQSMDualLoRALinear) and m.shared_enabled:
            assert not m.global_A.requires_grad and m.global_B.requires_grad
    cfg['cqsm_schema_version'] = 3
    torch.save(dict(config=cfg), path)
    with pytest.raises(ValueError, match='require schema 4'): load_adapter_checkpoint(PrefixToy(), path)
    cfg['cqsm_schema_version'] = 4; cfg['cqsm_shared_layout'] = 'visual_prefix'
    torch.save(dict(config=cfg), path)
    with pytest.raises(ValueError, match='schema 4 requires'): load_adapter_checkpoint(PrefixToy(), path)
    keeper.close()
