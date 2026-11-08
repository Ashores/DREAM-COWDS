"""Explicit optimization and diagnostics for the task-independent Global path."""
import torch
import torch.distributed as dist
from .methods import CQSMDualLoRALinear


def global_parameters(model):
    return [p for name, p in model.named_parameters()
            if (name.endswith(("global_B", "global_gate_logit")) or ".sdo_bridge." in name)
            and p.requires_grad]


def functional_correction(model, functional, project, world_size=1):
    """Project Global's three functional gradients after global-batch averaging.

    The caller adds (projected - original) to the ordinary backward gradients.
    Adding the same correction on every rank before their subsequent average
    preserves all other native objectives and matches global-batch projection.
    """
    params = global_parameters(model)
    if not params:
        return [], [], {"global_functional_projection_applied": 0}
    gradients = [torch.autograd.grad(functional[name], params, retain_graph=True,
                                     allow_unused=True)
                 for name in ("semantic", "box", "mask")]
    if world_size > 1:
        if not dist.is_initialized() or dist.get_world_size() != world_size:
            raise RuntimeError("Global projection requires the matching process group")
        averaged = []
        for group in gradients:
            row = []
            for p, value in zip(params, group):
                value = torch.zeros_like(p) if value is None else value.detach().clone()
                dist.all_reduce(value)
                row.append(value / world_size)
            averaged.append(row)
        gradients = averaged
    if any(not bool(g.isfinite().all()) for row in gradients for g in row if g is not None):
        raise FloatingPointError("nonfinite Global functional gradient before projection")
    replacement, original, cosines = project(gradients)
    corrections = [None if a is None or b is None else a - b
                   for a, b in zip(replacement, original)]
    if any(not bool(g.isfinite().all()) for g in corrections if g is not None):
        raise FloatingPointError("nonfinite Global functional gradient after projection")
    norm = lambda row: sum((g.float().square().sum() for g in row if g is not None),
                           params[0].new_zeros(())).sqrt()
    report = {"global_gradient_cosine_" + k: v for k, v in cosines.items()}
    report.update(global_functional_projection_applied=1,
                  global_functional_conflicting_pairs=sum(v < 0 for v in cosines.values()),
                  global_functional_original_norm=float(norm(original)),
                  global_functional_projected_norm=float(norm(replacement)),
                  global_functional_correction_norm=float(norm(corrections)))
    return params, corrections, report


def apply_correction(params, corrections):
    for p, delta in zip(params, corrections):
        if delta is None:
            continue
        if p.grad is None:
            p.grad = delta.to(p)
        else:
            p.grad.add_(delta.to(p.grad))


def clip_adapter_gradients(model, controller, independent_global=False):
    selector = controller.selector_parameters()
    selector_ids = {id(p) for p in selector}
    adapters = [p for p in model.parameters() if p.requires_grad and id(p) not in selector_ids]
    report = {}
    if independent_global:
        shared = global_parameters(model)
        shared_ids = {id(p) for p in shared}
        private = [p for p in adapters if id(p) not in shared_ids]
        for name, params in (("global", shared), ("private_cqsm", private)):
            if params:
                norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                report[name + "_gradient_norm_before_clip"] = float(norm)
                report[name + "_gradient_clip_factor"] = min(1., 1. / (float(norm) + 1e-6))
    elif adapters:
        torch.nn.utils.clip_grad_norm_(adapters, 1.)
    if selector:
        torch.nn.utils.clip_grad_norm_(selector, 1.)
    return report


@torch.no_grad()
def global_diagnostics(model):
    modules = [m for m in model.modules() if isinstance(m, CQSMDualLoRALinear) and m.shared_enabled]
    if not modules:
        return {"global_modules_active": 0}
    gates = [float(m.global_gate_logit.sigmoid()) if m.global_gate_logit is not None else 1.
             for m in modules]
    squared = [(m.global_B.float() * m.global_output_mask).square().sum() for m in modules]
    return {"global_modules_active": len(modules), "global_gate_values": gates,
            "global_gate_mean": sum(gates)/len(gates), "global_gate_min": min(gates),
            "global_gate_max": max(gates), "global_output_factor_norm": float(sum(squared).sqrt()),
            "global_effective_mapping_norm": float(sum(s*g*g for s, g in zip(squared, gates)).sqrt())}
