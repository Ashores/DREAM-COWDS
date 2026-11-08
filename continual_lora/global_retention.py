"""Local shared-output retention; inspired by, not a reproduction of CL-LoRA KD.

Uses current inputs detached from upstream adapters. No raw-image replay, no
classifier teacher, and no gradients into private/CQSM branches. Captures only
the preceding stage's shared output factors. Checkpoints retain ordinary model
weights; teacher snapshots are training-only and rebuilt at task boundaries.
"""
import torch
from torch.nn import functional as F
from .methods import CQSMDualLoRALinear


class GlobalRetention:
    def __init__(self, model, weight=1.0):
        if weight < 0:
            raise ValueError('weight must be nonnegative')
        self.weight = weight
        self.modules = [m for m in model.modules()
                        if isinstance(m, CQSMDualLoRALinear) and m.shared_enabled]
        self.previous = {}
        self.terms = []
        self.active = False
        self.handles = [m.register_forward_pre_hook(self._hook) for m in self.modules]

    def begin_task(self, task):
        self.active = False
        self.terms.clear()
        self.previous = ({id(m): m.global_B.detach().clone() for m in self.modules}
                         if task > 0 else {})

    def begin_batch(self):
        self.terms.clear()
        self.active = bool(self.previous) and self.weight > 0

    def _hook(self, module, inputs):
        if not self.active or not module.training:
            return
        # Only cache detached rank-sized inputs inside checkpointed forwards.
        # Differentiable retention operations must live outside checkpoint replay.
        with torch.no_grad(), torch.autocast(device_type=module.global_B.device.type, enabled=False):
            z = F.linear(inputs[0].detach().float(), module.global_A.detach().float())
            self.terms.append((module, z))

    def loss(self):
        self.active = False  # Ignore backward checkpoint recomputations.
        if self.terms:
            losses = []
            for module, z in self.terms:
                with torch.autocast(device_type=module.global_B.device.type, enabled=False):
                    current = F.linear(z, module.global_B.float())
                    previous = F.linear(z, self.previous[id(module)].float())
                    losses.append(F.mse_loss(current, previous))
            value = torch.stack(losses).mean() * self.weight
            self.terms.clear()
            return value
        if self.modules:
            return self.modules[0].global_B.new_zeros(())
        return torch.tensor(0.)

    def close(self):
        self.active = False
        self.terms.clear()
        for handle in self.handles:
            handle.remove()
