import pytest
import torch
from torch import nn
from test_shared_prefix import setup, PrefixToy
from continual_lora.methods import shared_only_adapter_forward
from continual_lora.trainer import _focus_backward, adapter_state_dict, load_adapter_checkpoint
from continual_lora.global_coordination import clip_adapter_gradients, global_parameters
from continual_lora.cqsm_accounting import cqsm_parameter_accounting


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_B = nn.Parameter(torch.tensor([.2]))
        self.private = nn.Parameter(torch.tensor([.7]))


class ScalarController:
    method = 'cqsm_dual_lora'
    def __init__(self, model): self.model = model
    def focus_parameters(self, *args, **kwargs): return list(self.model.parameters())
    def selector_parameters(self): return []


@pytest.mark.parametrize('coordinated,expected', [(False, 7.), (True, 5.)])
def test_global_projection_removes_conflict_and_preserves_other_objectives(coordinated, expected):
    model = ScalarModel(); ctl = ScalarController(model)
    s = model.global_B.sum()
    functional = dict(semantic=3*s, box=-s, mask=0*s)
    extra = 5*s + model.private.square().sum()
    report = _focus_backward(sum(functional.values())+extra, functional, ctl, True,
                             strict_finite=True, global_conflict=coordinated)
    torch.testing.assert_close(model.global_B.grad, torch.tensor([expected]))
    torch.testing.assert_close(model.private.grad, torch.tensor([1.4]))
    if coordinated:
        assert report['global_functional_projection_applied'] == 1
        assert report['global_gradient_cosine_semantic_box'] == pytest.approx(-1.)
        assert report['global_functional_correction_norm'] > 0


def test_non_projection_steps_keep_native_gradients():
    model = ScalarModel(); ctl = ScalarController(model)
    s = model.global_B.sum(); functional = dict(semantic=3*s, box=-s, mask=0*s)
    report = _focus_backward(sum(functional.values()), functional, ctl, False,
                             strict_finite=True, global_conflict=True)
    assert model.global_B.grad.item() == 2.
    assert report['global_functional_projection_applied'] == 0


def test_global_clipping_does_not_scale_private_gradients():
    model = ScalarModel(); ctl = ScalarController(model)
    model.global_B.grad = torch.tensor([10.]); model.private.grad = torch.tensor([.2])
    report = clip_adapter_gradients(model, ctl, independent_global=True)
    torch.testing.assert_close(model.global_B.grad, torch.tensor([1.]))
    assert model.private.grad.item() == pytest.approx(.2)
    assert report['global_gradient_norm_before_clip'] == 10.
    assert report['private_cqsm_gradient_norm_before_clip'] == pytest.approx(.2)


def test_gate_initialization_allows_zero_global_output_factors_to_learn():
    model, ctl, keeper, x, _ = setup(layout='visual_residual', global_gating=True)
    module = keeper.shared[0][1]
    assert module.global_gate_logit.sigmoid().item() == pytest.approx(.1)
    assert torch.count_nonzero(module.global_B) == 0
    with shared_only_adapter_forward(): module(x).square().mean().backward()
    assert module.global_B.grad.abs().sum() > 0
    assert module.global_gate_logit.grad is not None and module.global_gate_logit.grad == 0
    assert not module.global_A.requires_grad
    keeper.close()


@pytest.mark.parametrize('checkpointed', [False, True])
def test_gate_changes_are_seen_by_retention_and_do_not_change_selector(checkpointed):
    model, ctl, keeper, x, _ = setup(checkpointed, layout='visual_residual', global_gating=True)
    with torch.no_grad():
        for _, m in keeper.shared: m.global_B.add_(.03*m.global_output_mask)
    keeper.begin_task(1)
    model.train(); torch.manual_seed(119); ctl.begin_anchor_batch()
    with torch.no_grad(): model(x)
    descriptor = ctl.anchor_router()._visual_descriptor.clone()
    with torch.no_grad():
        for _, m in keeper.shared: m.global_gate_logit.add_(.7)
    torch.manual_seed(119); keeper.begin_batch(); ctl.begin_anchor_batch(); y = model(x)
    assert torch.equal(descriptor, ctl.anchor_router()._visual_descriptor)
    value, gradients = keeper.loss_and_gradients()
    assert value > 0
    gate_grads = [g for (_, kind, _), g in zip(keeper.param_specs, gradients)
                  if kind == 'global_gate_logit']
    assert all(g.abs().sum() > 0 for g in gate_grads)
    y.square().mean().backward()
    global_ids = {id(p) for p in global_parameters(model)}
    native = {id(p): p.grad.clone() for p in model.parameters()
              if p.grad is not None and id(p) not in global_ids}
    report = keeper.apply(gradients)
    assert report['shared_retention_gradient_ratio'] == pytest.approx(.2, rel=1e-5)
    assert all(torch.equal(native[id(p)], p.grad) for p in model.parameters() if id(p) in native)
    keeper.close()


def test_gate_accounting_preserves_private_and_cqsm_capacity():
    old, oc, ok, _, _ = setup(layout='visual_residual')
    new, nc, nk, _, _ = setup(layout='visual_residual', global_gating=True)
    a = cqsm_parameter_accounting(old, oc, 0)['counts']
    b = cqsm_parameter_accounting(new, nc, 0)['counts']
    for key in a:
        assert b[key] == a[key] + (len(nk.shared) if key.startswith('global_') else 0)
    ok.close(); nk.close()


def test_schema5_restores_gates_and_task_free_predictions(tmp_path, monkeypatch):
    from continual_lora import trainer
    monkeypatch.setattr(trainer, 'enable_lora_training_compatibility', lambda: None)
    model, ctl, keeper, x, report = setup(layout='visual_residual', global_gating=True)
    with torch.no_grad():
        for _, m in keeper.shared:
            m.global_B.add_(.02*m.global_output_mask); m.global_gate_logit.add_(.6)
    ctl.end_task(); model.eval(); ctl.begin_anchor_batch(); expected = model(x)
    cfg = dict(method='cqsm_dual_lora', rank=16, alpha=8, cqsm_schema_version=5,
               cqsm_restore_legacy=True, cqsm_shared_layout='visual_residual', global_rank=10,
               cqsm_shared=True, cqsm_private=True, cqsm_bank_mode='all', cqsm_basis='covariance',
               cqsm_global_gating=True, cqsm_global_gate_init=.1,
               energy_calibration_lambda=3., focus_orthogonality_weight=.001,
               focus_group_ranks=report.group_ranks, stream=[{}, {}])
    path = tmp_path/'gated.pt'
    payload = dict(config=cfg, adapter=adapter_state_dict(model), controller=ctl.state_dict(), completed_task=0)
    torch.save(payload, path)
    torch.manual_seed(19); other = PrefixToy(); oc, _, _ = load_adapter_checkpoint(other, path)
    other.eval(); oc.begin_anchor_batch()
    assert torch.equal(expected, other(x))
    assert oc.anchor_router().inference_oracle_stage is None
    assert all(torch.equal(a.global_gate_logit, b.global_gate_logit)
               for a, b in zip([m for _, m in keeper.shared],
                               [m for m in other.modules() if getattr(m, 'global_gating', False)]))
    cfg['cqsm_schema_version'] = 4; torch.save(payload, path)
    with pytest.raises(ValueError, match='require schema 5'):
        load_adapter_checkpoint(PrefixToy(), path)
    keeper.close()
