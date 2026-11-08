"""Shared-prefix feature retention with a frozen previous-stage adapter snapshot.

Teacher inference reuses immutable SAM weights and substitutes only the small
shared factors. It uses detached frozen-prefix inputs and matched dropout RNG.
Feature graphs are captured without differentiable hook-side operations, keeping
non-reentrant activation checkpoint replay consistent. Teachers are training-only.
"""
import re
import torch
import torch.distributed as dist
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint
from .methods import CQSMDualLoRALinear, shared_only_adapter_forward


class SharedPrefixRetention:
    def __init__(self, model, gradient_ratio=0.2, max_multiplier=1e6):
        if not 0 <= gradient_ratio <= 1 or max_multiplier <= 0:
            raise ValueError("invalid feature-retention gradient settings")
        self.ratio = float(gradient_ratio)
        self.max_multiplier = float(max_multiplier)
        self.shared = [(n, m) for n, m in model.named_modules()
                       if isinstance(m, CQSMDualLoRALinear) and m.shared_enabled]
        ids = sorted({int(re.search(r"\.blocks\.(\d+)\.", n)[1])
                      for n, _ in self.shared})
        if ids and ids != list(range(ids[0], ids[-1]+1)):
            raise ValueError("the shared prefix must be contiguous")
        self.blocks = [model.get_submodule(f"backbone.vision_backbone.trunk.blocks.{i}")
                       for i in ids]
        self.block_ids = ids
        self.param_specs = [(m, "global_B", m.global_B) for _, m in self.shared]
        self.param_specs += [(m, "global_gate_logit", m.global_gate_logit)
                             for _, m in self.shared if m.global_gate_logit is not None]
        self.params = [p for _, _, p in self.param_specs]
        self.previous = []
        self.active = False
        self.teacher_running = False
        self.teachers, self.students = [], []
        self.handles = []
        if self.blocks:
            self.handles = [self.blocks[0].register_forward_pre_hook(self._teacher),
                            self.blocks[-1].register_forward_hook(self._student)]

    def begin_task(self, task):
        self.active = False
        self.teachers.clear(); self.students.clear()
        self.previous = [p.detach().clone() for p in self.params] if task > 0 else []

    def begin_batch(self):
        self.teachers.clear(); self.students.clear()
        self.active = bool(self.previous) and self.ratio > 0

    def _teacher(self, _module, inputs):
        if not self.active or self.teacher_running:
            return
        x = inputs[0].detach()
        if inputs[0].requires_grad:
            raise RuntimeError("retention must start at an immutable visual prefix")
        previous = {id(p): value for p, value in zip(self.params, self.previous)}
        devices = [x.device.index] if x.device.type == "cuda" else []
        self.teacher_running = True
        try:
            # Restoring RNG means teacher and student see the same dropout masks.
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
                for block in self.blocks:
                    replacement = {name: previous[id(p)] for name, p in block.named_parameters()
                                   if id(p) in previous}
                    x = functional_call(block, replacement, (x,), strict=False)
                self.teachers.append(x.detach())
        finally:
            self.teacher_running = False

    def _student(self, _module, _inputs, output):
        if self.active and not self.teacher_running:
            self.students.append(output)

    def loss_and_gradients(self):
        self.active = False  # Backward checkpoint replay must not recapture.
        if not self.teachers:
            return None, None
        if len(self.teachers) != len(self.students):
            raise RuntimeError("teacher and student boundary captures differ")
        losses = []
        for current, teacher in zip(self.students, self.teachers):
            scale = teacher.float().square().mean().clamp_min(1e-6)
            losses.append((current.float()-teacher.float()).square().mean()/scale)
        value = torch.stack(losses).mean()
        self.teachers.clear(); self.students.clear()
        gradients = torch.autograd.grad(value, self.params, retain_graph=True, allow_unused=True)
        return value.detach(), gradients

    def apply(self, gradients, world_size=1):
        """Add balanced retention gradients only to shared factors and gates.

        Historical per-channel norms reassign importance, as inspired by CL-LoRA.
        Balance is computed after rank averaging, before adapter gradient clipping.
        """
        if gradients is None:
            return {"shared_retention_applied": 0, "shared_retention_gradient_ratio": 0.0}
        reassigned = []
        for (module, kind, p), old, g in zip(self.param_specs, self.previous, gradients):
            g = torch.zeros_like(p) if g is None else g.detach().clone()
            if world_size > 1:
                dist.all_reduce(g); g.div_(world_size)
            if kind == "global_B":
                mask = module.global_output_mask.to(g)
                importance = old.float().norm(dim=1, keepdim=True)*mask
                mean = importance.sum()/mask.sum().clamp_min(1)
                weights = ((importance/mean).clamp(0.2, 5.) if float(mean) > 1e-12
                           else torch.ones_like(importance))*mask
            else:
                weights = torch.ones_like(g)
            reassigned.append(g*weights)
        native_norm = torch.sqrt(sum((p.grad.float().square().sum() if p.grad is not None
                                      else p.new_zeros(())) for p in self.params))
        retention_norm = torch.sqrt(sum(g.float().square().sum() for g in reassigned))
        if not bool(torch.isfinite(native_norm) & torch.isfinite(retention_norm)):
            raise FloatingPointError("nonfinite shared feature gradients")
        multiplier = (self.ratio*native_norm/retention_norm.clamp_min(1e-12)).clamp_max(self.max_multiplier)
        if float(retention_norm) < 1e-12:
            multiplier = multiplier*0
        for p, g in zip(self.params, reassigned):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.add_(g.to(p.grad)*multiplier)
        return {"shared_native_gradient_norm": float(native_norm),
                "shared_retention_gradient_norm": float(retention_norm),
                "shared_retention_multiplier": float(multiplier),
                "shared_retention_gradient_ratio": float(multiplier*retention_norm/native_norm.clamp_min(1e-12)),
                "shared_retention_applied": int(float(multiplier) > 0)}

    def close(self):
        self.active = False
        self.teachers.clear(); self.students.clear()
        for handle in self.handles:
            handle.remove()


class SharedOnlyPrefixRetention(SharedPrefixRetention):
    """Retain Global's own feature mapping when private and Global overlap.

    Both reference and current paths contain the frozen backbone and Global
    only. Current private adaptation therefore cannot change the target or
    cause Global to compensate for private updates. This objective uses current
    images; it does not promise retention on unseen historical distributions.
    """

    def __init__(self, model, gradient_ratio=0.2, max_multiplier=1e6):
        super().__init__(model, gradient_ratio, max_multiplier)
        for handle in self.handles:
            handle.remove()
        self.captures = []
        self.handles = ([self.blocks[0].register_forward_pre_hook(self._capture)]
                        if self.blocks else [])

    def begin_task(self, task):
        super().begin_task(task)
        self.captures.clear()

    def begin_batch(self):
        super().begin_batch()
        self.captures.clear()

    def _capture(self, _module, inputs):
        if not self.active:
            return
        x = inputs[0]
        if x.requires_grad:
            raise RuntimeError("shared-only retention requires an immutable prefix input")
        device = x.device
        self.captures.append((
            x.detach(), torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type)))

    @staticmethod
    def _shared_block(block, x):
        # This function is also the activation-checkpoint replay function.
        with shared_only_adapter_forward():
            return block(x)

    def loss_and_gradients(self):
        self.active = False  # Auxiliary and native backward must not recapture.
        captures, self.captures = self.captures, []
        if not captures:
            return None, None
        previous = {id(p): value for p, value in zip(self.params, self.previous)}
        gradients = [torch.zeros_like(p) for p in self.params]
        values = []
        for x, cpu_rng, cuda_rng, autocast_enabled, autocast_dtype in captures:
            devices = [x.device.index] if x.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.autocast(
                    device_type=x.device.type, enabled=autocast_enabled, dtype=autocast_dtype):
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, x.device)
                with torch.no_grad(), shared_only_adapter_forward():
                    teacher = x
                    for block in self.blocks:
                        replacement = {name: previous[id(p)] for name, p in block.named_parameters()
                                       if id(p) in previous}
                        teacher = functional_call(block, replacement, (teacher,), strict=False)
                # Match teacher/student dropout, and leave the native RNG stream unchanged.
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, x.device)
                current = x
                for block in self.blocks:
                    current = checkpoint(self._shared_block, block, current, use_reentrant=False)
                scale = teacher.float().square().mean().clamp_min(1e-6)
                loss = (current.float()-teacher.float()).square().mean()/scale
                grads = torch.autograd.grad(loss, self.params, allow_unused=True)
                for accumulated, grad in zip(gradients, grads):
                    if grad is not None:
                        accumulated.add_(grad.detach()/len(captures))
                values.append(loss.detach())
        return torch.stack(values).mean(), gradients

    def close(self):
        super().close()
        self.captures.clear()
