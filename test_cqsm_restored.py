"""Behavioral checks for restored CQSM; the unmodified preset is tested separately."""
import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from continual_lora import trainer
from continual_lora.methods import CovarianceQuerySubspaceMemory, EnergyDecomposedObjectCore
from test_cqsm import setup_model, initialize_task, step
from test_continual_lora import SemanticFocusToy


def test_restored_calibration_matches_legacy_and_preserves_history():
    old = EnergyDecomposedObjectCore(4, 2, 3, general_rank=2, calibration_lambda=3.)
    new = CovarianceQuerySubspaceMemory(4, 2, 3, energy_calibration=True,
                                      calibration_lambda=3.)
    for stage in range(3):
        x = torch.diag(torch.tensor([4., 3., 2., 1.])) + stage * .25
        scales = new.general_scale[:stage].clone()
        for core in (old, new):
            core.begin_task(stage)
            core.start_collection()
            core.observe(x)
            core.finalize_subspaces(stage)
            core.end_task()
        assert torch.equal(old.lora_A, new.lora_A) if stage == 2 else torch.equal(
            old.lora_A[:stage+1], new.lora_A[:stage+1])
        assert torch.equal(old.general_scale, new.general_scale)
        assert torch.equal(scales, new.general_scale[:stage])
        assert torch.equal(old.historical_covariance, new.historical_covariance)
    assert bool((new.general_scale[1:] < 1).all())
    with pytest.raises(RuntimeError, match="twice"):
        new.end_task()


def test_restored_memory_uses_scales_under_every_private_route():
    model, ctl, opt, _ = setup_model(restore_legacy=True)
    x = torch.randn(2, 3, model.width)
    for stage in range(2):
        initialize_task(model, ctl, opt, stage, x + stage)
        step(model, ctl, opt, x + stage)
        ctl.end_task()
    core = ctl.energy_object_core()
    module = next(m for m in ctl.adapters() if m.uses_object_core)
    expected = sum(torch.nn.functional.linear(
        core.project(x, s) * core.general_scale[s], module.core_B[s]) for s in range(2))
    outputs = []
    for route in (0, 1):
        ctl.anchor_router()._route = torch.nn.functional.one_hot(
            torch.full((2,), route), 3).float()
        outputs.append(module.bank_update(x))
    assert torch.allclose(outputs[0], expected)
    assert torch.equal(outputs[0], outputs[1])
    with torch.no_grad():
        core.general_scale.zero_()
    assert torch.count_nonzero(module.bank_update(x)) == 0


def test_restored_orthogonality_has_gradients_without_changing_old_private_slices():
    model, ctl, opt, _ = setup_model(restore_legacy=True)
    x = torch.randn(2, 3, model.width)
    initialize_task(model, ctl, opt, 0, x)
    step(model, ctl, opt, x)
    ctl.end_task()
    previous = {n: v[0].clone() for n, v in trainer.adapter_state_dict(model).items()
                if n.endswith((".lora_A", ".lora_B", ".core_B")) and v.ndim == 3}
    initialize_task(model, ctl, opt, 1, x + .5)
    module = ctl.adapters()[0]
    with torch.no_grad():
        module.lora_A[1].add_(.1 * module.lora_A[0])
    opt.zero_grad(set_to_none=True)
    regularizer = ctl.regularization_loss()
    assert regularizer.item() > 0
    regularizer.backward()
    assert module.lora_A.grad[1].abs().sum() > 0
    assert torch.count_nonzero(module.lora_A.grad[0]) == 0
    opt.step()
    ctl.restore_focus_protected_parameters()
    state = trainer.adapter_state_dict(model)
    assert all(torch.equal(state[n][0], v) for n, v in previous.items())


def test_conflict_projection_preserves_auxiliary_gradients():
    p = torch.nn.Parameter(torch.tensor([1., 1.]))
    ctl = SimpleNamespace(focus_parameters=lambda *args, **kwargs: [p])
    functional = dict(semantic=p[0], box=-2*p[0]+p[1], mask=.5*p[1])
    raw = [torch.autograd.grad(v, [p], retain_graph=True) for v in functional.values()]
    projected, _, _ = trainer._project_functional_gradients(raw)
    auxiliary = 3*p.sum()
    trainer._focus_backward(sum(functional.values()) + auxiliary, functional, ctl,
                            True, strict_finite=True)
    assert torch.allclose(p.grad, projected[0] + 3)


def test_restored_projection_rejects_nonfinite_instead_of_skipping():
    p = torch.nn.Parameter(torch.tensor(0.))
    ctl = SimpleNamespace(focus_parameters=lambda *args, **kwargs: [p])
    functional = dict(semantic=p.sqrt(), box=p*2, mask=p*3)
    with pytest.raises(FloatingPointError, match="before projection"):
        trainer._focus_backward(sum(functional.values()), functional, ctl,
                                True, strict_finite=True)
    assert p.grad is None


def test_auxiliary_loss_assembly_keeps_all_contributions(monkeypatch):
    monkeypatch.setitem(sys.modules, "sam3.model.utils.misc",
                        SimpleNamespace(copy_data_to_device=lambda x, dev: x))
    monkeypatch.setitem(sys.modules, "sam3.train.loss.loss_fns",
                        SimpleNamespace(CORE_LOSS_KEY="native"))
    p = torch.nn.Parameter(torch.tensor(2.))
    class Model:
        def __call__(self, batch): return {"p": p}
        def back_convert(self, target): return {"num_boxes": torch.ones(1)}
    ctl = SimpleNamespace(method="cqsm_dual_lora", begin_anchor_batch=lambda: None,
                          regularization_loss=lambda: .001*p, selector_loss=lambda: p*0)
    bridge = SimpleNamespace(begin_batch=lambda: None,
                             loss=lambda output: (4*p, {"instances": 1.}))
    monkeypatch.setattr(trainer, "_focus_mask_box_consistency", lambda out: 2*p)
    monkeypatch.setattr(trainer, "_weighted_functional_losses", lambda *a, **kw:
                        dict(semantic=p, box=p, mask=p, reconstruction_gap=p*0))
    loss, terms, _ = trainer._forward_focus_loss(
        Model(), lambda *a: {"native": 3*p},
        {"batch": SimpleNamespace(find_targets=[None])}, "cpu", ctl,
        consistency_weight=.1, unified_bridge=bridge, unified_object_weight=.1)
    loss.backward()
    assert p.grad.item() == pytest.approx(3 + .2 + .001 + .4)
    assert terms["focus_consistency_loss"] > 0
    assert terms["focus_regularization_loss"] > 0
    assert terms["unified_object_weighted_loss"] > 0


def test_restored_checkpoint_round_trip_retains_calibrated_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "enable_lora_training_compatibility", lambda: None)
    model, ctl, opt, report = setup_model(restore_legacy=True)
    x = torch.randn(2, 3, model.width)
    for stage in range(2):
        initialize_task(model, ctl, opt, stage, x + stage)
        step(model, ctl, opt, x + stage)
        ctl.end_task()
    config = dict(method="cqsm_dual_lora", rank=16, alpha=8, cqsm_schema_version=2,
        cqsm_restore_legacy=True, cqsm_shared=True, cqsm_private=True,
        cqsm_bank_mode="all", cqsm_basis="covariance", energy_calibration_lambda=3.,
        focus_orthogonality_weight=.001, focus_group_ranks=report.group_ranks,
        stream=[{"name": str(i), "split": "train"} for i in range(3)])
    payload = dict(config=config, adapter=trainer.adapter_state_dict(model),
                   controller=ctl.state_dict(), completed_task=1)
    path = tmp_path / "restored.pt"
    torch.save(payload, path)
    torch.manual_seed(812)
    other = SemanticFocusToy()
    oc, _, _ = trainer.load_adapter_checkpoint(other, path)
    assert oc.energy_object_core().energy_calibration
    outputs = []
    for m, controller in ((model, ctl), (other, oc)):
        m.eval()
        controller.anchor_router().inference_oracle_stage = 1
        controller.begin_anchor_batch()
        outputs.append(m(x))
    assert torch.equal(outputs[0], outputs[1])
    broken = copy.deepcopy(payload)
    broken["config"]["cqsm_schema_version"] = 1
    torch.save(broken, path)
    with pytest.raises(ValueError, match="schema"):
        trainer.load_adapter_checkpoint(SemanticFocusToy(), path)
