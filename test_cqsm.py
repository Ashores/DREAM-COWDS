"""Mechanism and serialization tests for the new, separately named CQSM method."""
import copy

import pytest
import torch
from torch import nn

from continual_lora.methods import (
    AdapterController, CQSMDualLoRALinear, CovarianceQuerySubspaceMemory,
    inject_continual_lora,
)
from continual_lora import trainer
from test_continual_lora import SemanticFocusToy
from continual_lora.eval_protocol import checkpoint_stage_map, sha256_file
from continual_lora.cqsm_accounting import cqsm_parameter_accounting


def setup_model(*, shared=True, private=True, bank_mode="all", basis="covariance", num_tasks=3,
                restore_legacy=False):
    torch.manual_seed(812)
    model = SemanticFocusToy()
    report = inject_continual_lora(
        model, "cqsm_dual_lora", rank=16, alpha=8, num_tasks=num_tasks,
        cqsm_shared=shared, cqsm_private=private, cqsm_bank_mode=bank_mode, cqsm_basis=basis,
        cqsm_restore_legacy=restore_legacy)
    controller = AdapterController(model, "cqsm_dual_lora")
    selector_ids = {id(p) for p in controller.selector_parameters()}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad and id(p) not in selector_ids],
        lr=0.02, weight_decay=0.1)
    return model, controller, optimizer, report


def test_slice_accounting_separates_future_storage_and_optimizer_policy():
    model, controller, optimizer, _ = setup_model()
    x = torch.randn(2, 3, model.width)
    initialize_task(model, controller, optimizer, 0, x)
    first = cqsm_parameter_accounting(model, controller, 0)
    controller.end_task()
    initialize_task(model, controller, optimizer, 1, x)
    second = cqsm_parameter_accounting(model, controller, 1)
    assert first["joint_loss_effective_trainable"] == second["joint_loss_effective_trainable"]
    assert first["adapter_preallocated_parameters"] == second["adapter_preallocated_parameters"]
    assert second["counts"]["private_retained"] == 2 * first["counts"]["private_retained"]
    assert second["fixed_query_basis_retained_elements"] == 2 * first["fixed_query_basis_retained_elements"]
    assert first["tensor_requires_grad_count_not_effective"] > first["combined_optimizer_effective_trainable"]
    actual_adapter = sum(p.numel() for n,p in model.named_parameters() if ".base." not in n)
    # Toy has frozen non-adapter layers: count exact adapter and selector objects instead.
    assert first["adapter_preallocated_parameters"] <= actual_adapter
    disabled, dc, do, _ = setup_model(private=False, bank_mode="off")
    initialize_task(disabled, dc, do, 0, x)
    off = cqsm_parameter_accounting(disabled, dc, 0)
    assert off["joint_loss_effective_trainable"] == off["counts"]["global_current_optimizable"]


def initialize_task(model, controller, optimizer, stage, x):
    controller.begin_task(stage)
    controller.reset_focus_active_optimizer_state(optimizer)
    model.eval()
    controller.anchor_router().inference_oracle_stage = stage
    controller.start_energy_covariance_collection()
    controller.begin_anchor_batch()
    with torch.no_grad():
        model(x)
    report = controller.finalize_energy_object_subspaces(stage)
    controller.anchor_router().inference_oracle_stage = None
    model.train()
    return report


def step(model, controller, optimizer, x):
    controller.begin_anchor_batch()
    optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    optimizer.step()
    controller.restore_focus_protected_parameters()


def test_cqsm_injection_rank_scope_and_zero_initialization():
    model, controller, optimizer, report = setup_model()
    assert report.method == "cqsm_dual_lora"
    core = controller.energy_object_core()
    assert isinstance(core, CovarianceQuerySubspaceMemory)
    assert core.rank == core.general_rank == 2 and core.isolated_rank == 0
    assert not core.lora_A.requires_grad
    assert report.module_groups["global_shared"]
    for name in report.modules:
        module = model.get_submodule(name)
        assert isinstance(module, CQSMDualLoRALinear)
        assert not module.base.weight.requires_grad
        expected = name.startswith(("transformer.decoder.", "dot_prod_scoring.",
                                    "segmentation_head."))
        assert module.shared_enabled == expected
        if expected:
            assert module.global_A.shape == (2, module.in_features)
            assert module.global_B.shape == (module.out_features, 2)
            assert torch.count_nonzero(module.global_B) == 0
    controller.begin_task(0)
    controller.begin_anchor_batch()
    x = torch.randn(2, 3, model.width)
    model(x)
    for module in controller.adapters():
        assert torch.allclose(module(x), module.base(x))
    assert controller.regularization_loss().item() == 0


def test_cqsm_train_gradients_global_updates_and_history_protection():
    model, controller, optimizer, _ = setup_model()
    x = torch.randn(2, 3, model.width)
    initialize_task(model, controller, optimizer, 0, x)
    core = controller.energy_object_core()
    fixed_a = core.lora_A.clone()
    adapters = controller.adapters()
    global_module = next(m for m in adapters if m.shared_enabled)
    global_initial = global_module.global_B.detach().clone()
    for _ in range(3):
        step(model, controller, optimizer, x)
    assert not torch.equal(global_initial, global_module.global_B)
    assert core.lora_A.grad is None
    assert torch.equal(fixed_a, core.lora_A)
    assert all(m.lora_B.grad[0].abs().sum() > 0 for m in adapters)
    assert all(m.core_B.grad[0].abs().sum() > 0 for m in adapters if m.uses_object_core)
    controller.end_task()
    previous = [(m.lora_A[0].clone(), m.lora_B[0].clone(),
                 m.core_B[0].clone() if m.core_B is not None else None) for m in adapters]
    global_before_task1 = global_module.global_B.detach().clone()
    initialize_task(model, controller, optimizer, 1, x + 0.5)
    assert torch.equal(fixed_a[0], core.lora_A[0])
    for _ in range(3):
        step(model, controller, optimizer, x + 0.5)
    assert not torch.equal(global_before_task1, global_module.global_B)
    for m, (a, b, cb) in zip(adapters, previous):
        assert torch.equal(a, m.lora_A[0])
        assert torch.equal(b, m.lora_B[0])
        assert torch.count_nonzero(m.lora_A.grad[0]) == 0
        assert torch.count_nonzero(m.lora_B.grad[0]) == 0
        if cb is not None:
            assert torch.equal(cb, m.core_B[0])
            assert torch.count_nonzero(m.core_B.grad[0]) == 0
    controller.end_task()
    assert torch.equal(core.general_scale, torch.ones_like(core.general_scale))


def test_cqsm_memory_all_on_is_independent_of_private_route():
    model, controller, optimizer, _ = setup_model()
    x = torch.randn(2, 3, model.width)
    for stage in range(2):
        initialize_task(model, controller, optimizer, stage, x + stage)
        step(model, controller, optimizer, x + stage)
        controller.end_task()
    module = next(m for m in controller.adapters() if m.uses_object_core)
    core = controller.energy_object_core()
    with torch.no_grad():
        module.core_B[0].fill_(0.5)
        module.core_B[1].fill_(1.5)
        # Legacy scale must not affect CQSM, even if an external buffer is changed.
        core.general_scale.fill_(0.01)
    contributions = [torch.nn.functional.linear(core.project(x, s), module.core_B[s])
                     for s in range(2)]
    router = controller.anchor_router()
    outputs = []
    for route in (0, 1):
        router._route = torch.nn.functional.one_hot(
            torch.full((2,), route), 3).float()
        outputs.append(module.bank_update(x))
    assert torch.allclose(outputs[0], sum(contributions))
    assert torch.equal(outputs[0], outputs[1])
    module.bank_mode = "current"
    assert torch.allclose(module.bank_update(x), contributions[1])
    module.bank_mode = "routed"
    assert torch.allclose(module.bank_update(x), contributions[1])
    router._route = torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), 3).float()
    assert torch.allclose(module.bank_update(x), contributions[0])
    module.bank_mode = "off"
    assert torch.count_nonzero(module.bank_update(x)) == 0


def test_cqsm_statistics_use_per_call_mean_and_append_without_calibration():
    core = CovarianceQuerySubspaceMemory(4, 2, 3)
    x1 = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    x2 = torch.diag(torch.tensor([1.0, 5.0, 3.0, 2.0])).repeat(3, 1)
    history = torch.zeros(4, 4)
    old_bases = []
    for stage in range(2):
        core.begin_task(stage)
        core.start_collection()
        core.observe(x1 + stage)
        core.observe(x2 + stage)
        report = core.finalize_subspaces(stage)
        moment = ((x1 + stage).T @ (x1 + stage) / len(x1) +
                  (x2 + stage).T @ (x2 + stage) / len(x2)) / 2
        normalized = 4 * moment / moment.trace()
        assert report["collection_consumer_calls"] == 2
        assert torch.allclose(core.current_covariance, normalized, atol=1e-6)
        values, vectors = torch.linalg.eigh(history + normalized)
        projector = vectors[:, -2:] @ vectors[:, -2:].T
        basis = core.lora_A[stage]
        assert torch.allclose(basis @ basis.T, torch.eye(2), atol=1e-5)
        assert torch.allclose(basis.T @ basis, projector, atol=1e-5)
        for previous, expected in enumerate(old_bases):
            assert torch.equal(core.lora_A[previous], expected)
        old_bases.append(basis.clone())
        core.end_task()
        history += normalized
        assert torch.allclose(core.historical_covariance, history, atol=1e-6)
        assert torch.equal(core.general_scale, torch.ones_like(core.general_scale))
    with pytest.raises(RuntimeError, match="twice"):
        core.end_task()


@pytest.mark.parametrize("shared,private,bank_mode,basis", [
    (True, True, "all", "covariance"), (False, True, "all", "covariance"),
    (True, True, "off", "covariance"), (True, True, "current", "covariance"),
    (True, True, "routed", "covariance"), (True, False, "all", "covariance"),
    (True, True, "all", "random")])
def test_cqsm_checkpoint_round_trip_and_next_task(
        tmp_path, monkeypatch, shared, private, bank_mode, basis):
    monkeypatch.setattr(trainer, "enable_lora_training_compatibility", lambda: None)
    model, controller, optimizer, report = setup_model(
        shared=shared, private=private, bank_mode=bank_mode, basis=basis)
    x = torch.randn(2, 3, model.width)
    initialize_task(model, controller, optimizer, 0, x)
    step(model, controller, optimizer, x)
    controller.end_task()
    config = dict(method="cqsm_dual_lora", rank=16, alpha=8, cqsm_schema_version=1,
                  cqsm_shared=shared, cqsm_bank_mode=bank_mode,
                  cqsm_private=private, cqsm_basis=basis,
                  focus_group_ranks=report.group_ranks,
                  stream=[{"name": str(i), "split": "train"} for i in range(3)])
    payload = dict(config=config, adapter=trainer.adapter_state_dict(model),
                   controller=controller.state_dict(), completed_task=0)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    torch.manual_seed(812)
    restored = SemanticFocusToy()
    restored_ctl, restored_report, _ = trainer.load_adapter_checkpoint(restored, checkpoint)
    assert restored_report.method == "cqsm_dual_lora"
    for name, value in trainer.adapter_state_dict(restored).items():
        if value.is_floating_point():
            assert torch.allclose(value, payload["adapter"][name], rtol=0, atol=0,
                                  equal_nan=True), name
        else:
            assert torch.equal(value, payload["adapter"][name]), name
    for m, ctl in ((model, controller), (restored, restored_ctl)):
        m.eval()
        ctl.anchor_router().inference_oracle_stage = 0
        ctl.begin_anchor_batch()
    assert torch.allclose(model(x), restored(x), atol=1e-6)
    restored_ctl.begin_task(1)
    assert not restored_ctl.energy_object_core().lora_A.requires_grad
    for m in restored_ctl.adapters():
        assert m.lora_A.requires_grad == private
        assert m.lora_B.requires_grad == private
        if m.shared_enabled:
            assert not m.global_A.requires_grad and m.global_B.requires_grad
        if m.core_B is not None:
            assert m.core_B.requires_grad == (bank_mode != "off")
    if shared:
        missing = next(k for k in payload["adapter"] if k.endswith("global_A"))
        del payload["adapter"][missing]
        torch.save(payload, checkpoint)
        with pytest.raises(RuntimeError, match="Incomplete CQSM checkpoint"):
            trainer.load_adapter_checkpoint(SemanticFocusToy(), checkpoint)


def test_cqsm_rank_is_not_silently_clipped():
    with pytest.raises(ValueError, match="rank 2"):
        CovarianceQuerySubspaceMemory(12, 1, 3)
    with pytest.raises(ValueError, match="rank 2"):
        inject_continual_lora(SemanticFocusToy(), "cqsm_dual_lora", rank=8, num_tasks=3)


def test_cqsm_private_ablation_removes_gradient_and_residual():
    model, controller, optimizer, _ = setup_model(private=False)
    x = torch.randn(2, 3, model.width)
    initialize_task(model, controller, optimizer, 0, x)
    previous = [(m.lora_A.clone(), m.lora_B.clone()) for m in controller.adapters()]
    step(model, controller, optimizer, x)
    for m, (a, b) in zip(controller.adapters(), previous):
        assert m.lora_A.grad is None and m.lora_B.grad is None
        assert torch.equal(m.lora_A, a) and torch.equal(m.lora_B, b)
        before = m(x).detach()
        with torch.no_grad():
            m.lora_A.fill_(10)
            m.lora_B.fill_(10)
        assert torch.equal(m(x), before)


def test_cqsm_random_basis_is_covariance_independent_but_records_moments():
    left = CovarianceQuerySubspaceMemory(12, 2, 2, basis="random")
    right = CovarianceQuerySubspaceMemory(12, 2, 2, basis="random")
    for stage in range(2):
        for core, scale in ((left, torch.arange(1., 13.)),
                            (right, torch.arange(12., 0., -1.))):
            core.begin_task(stage)
            core.start_collection()
            core.observe(torch.diag(scale))
            core.finalize_subspaces(stage)
            core.end_task()
        assert torch.equal(left.lora_A[stage], right.lora_A[stage])
        assert not torch.equal(left.historical_covariance, right.historical_covariance)
        assert torch.allclose(left.lora_A[stage] @ left.lora_A[stage].T,
                              torch.eye(2), atol=1e-5)


def test_evaluation_routing_labels_follow_checkpoint_order(tmp_path):
    order = ["TrashCan-Instance", "Cityscapes", "Kvasir-Instrument", "HRSID", "PASCAL_VOC_2012"]
    payload = {"config": {"stream": [{"name": name} for name in order]}}
    assert checkpoint_stage_map(payload)["PASCAL_VOC_2012"] == 4
    assert checkpoint_stage_map(payload)["TrashCan-Instance"] == 0
    assert checkpoint_stage_map(None) == {}
    file = tmp_path / "data.bin"
    file.write_bytes(b"first")
    before = sha256_file(file)
    file.write_bytes(b"second")
    assert before != sha256_file(file)
