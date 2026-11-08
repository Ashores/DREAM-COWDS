"""Continual LoRA methods adapted to task-agnostic SAM 3.

All variants use a fixed total rank budget and expose one merged inference
path; no dataset/task identifier is consumed during inference.
"""
from __future__ import annotations

import math
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn


_SHARED_ONLY_FORWARD = ContextVar("cqsm_shared_only_forward", default=False)


@contextmanager
def shared_only_adapter_forward():
    """Evaluate frozen maps and Global without private/bank/router side effects.

    Each retention checkpoint calls this context again during backward replay.
    It never changes adapter enable flags, parameters, or the inference route.
    """
    token = _SHARED_ONLY_FORWARD.set(True)
    try:
        yield
    finally:
        _SHARED_ONLY_FORWARD.reset(token)


DEFAULT_TARGETS = (
    r"^backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
    r"^transformer\.(encoder|decoder)\.layers\.\d+\.linear[12]$",
)

# FOCUS-LoRA separates model functions, never datasets or stream stages.  Every
# group is active for every input and no group consumes a task/domain identity.
FOCUS_TARGET_GROUPS = {
    "visual": (
        r"^backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.(qkv|proj)$",
    ),
    "semantic": (
        r"^transformer\.(encoder|decoder)\.layers\.\d+\.linear[12]$",
        r"^transformer\.decoder\.presence_token_head\.layers\.[01]$",
    ),
    "box": (
        r"^transformer\.decoder\.bbox_embed\.layers\.[01]$",
        r"^transformer\.decoder\.ref_point_head\.layers\.[01]$",
    ),
    "mask": (
        r"^segmentation_head\.mask_predictor\.mask_embed\.layers\.[012]$",
    ),
}


def _orthogonal_rows(rows: int, cols: int, *, device=None, dtype=None) -> torch.Tensor:
    x = torch.randn(cols, rows, device=device, dtype=dtype or torch.float32)
    return torch.linalg.qr(x, mode="reduced").Q.T.contiguous()


class ContinualAdapterLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.num_tasks = rank, num_tasks
        self.scaling = float(alpha) / rank
        self.task_id = 0

    def begin_task(self, task_id: int) -> None:
        self.task_id = int(task_id)

    def end_task(self) -> None:
        pass

    def regularization_loss(self) -> torch.Tensor:
        return self.base.weight.new_zeros(())

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        for name, p in self.named_parameters():
            if not name.startswith("base."):
                yield p


class SequentialLoRALinear(ContinualAdapterLinear):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5):
        super().__init__(base, rank, alpha, num_tasks)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class FocusLoRALinear(ContinualAdapterLinear):
    """Function-specific, fixed-capacity stable-plastic LoRA.

    ``function_group`` describes what a layer does (visual, semantic, box, or
    mask); it never identifies the dataset or stream stage.  A single stable
    state is repeatedly consolidated with a reusable plastic workspace, so the
    module count and checkpoint size remain constant throughout the stream.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5,
                 *, function_group: str, stable_ratio: float = 0.75,
                 merge_gamma: float = 0.75, orthogonality_weight: float = 1e-3):
        super().__init__(base, rank, alpha, num_tasks)
        if function_group not in FOCUS_TARGET_GROUPS:
            raise ValueError(f"Unknown FOCUS function group: {function_group}")
        if not 0.0 < stable_ratio < 1.0:
            raise ValueError("stable_ratio must be strictly between zero and one")
        if not 0.0 < merge_gamma <= 1.0:
            raise ValueError("merge_gamma must be in (0, 1]")
        self.function_group = function_group
        self.stable_rank = max(1, min(rank - 1, int(round(rank * stable_ratio))))
        self.plastic_rank = rank - self.stable_rank
        if self.plastic_rank < 1:
            raise ValueError("FOCUS-LoRA requires at least one stable and one plastic rank")
        self.merge_gamma = float(merge_gamma)
        self.orthogonality_weight = float(orthogonality_weight)

        self.stable_A = nn.Parameter(torch.empty(self.stable_rank, self.in_features))
        self.stable_B = nn.Parameter(torch.zeros(self.out_features, self.stable_rank))
        self.plastic_A = nn.Parameter(torch.empty(self.plastic_rank, self.in_features))
        self.plastic_B = nn.Parameter(torch.zeros(self.out_features, self.plastic_rank))
        nn.init.kaiming_uniform_(self.stable_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.plastic_A, a=math.sqrt(5))
        # A scalar is sufficient to restore trainability on resume without
        # retaining a task-indexed parameter bank.
        self.register_buffer("consolidations", torch.zeros((), dtype=torch.long))

    def begin_task(self, task_id: int) -> None:
        # The stage number is intentionally ignored.  Only the existence of a
        # previously consolidated global state changes trainability.
        del task_id
        first_stage = int(self.consolidations.item()) == 0
        self.stable_A.requires_grad_(first_stage)
        self.stable_B.requires_grad_(first_stage)
        self.plastic_A.requires_grad_(True)
        self.plastic_B.requires_grad_(True)

    def restore_trainability(self) -> None:
        """Restore lifecycle flags after loading a checkpoint."""
        first_stage = int(self.consolidations.item()) == 0
        self.stable_A.requires_grad_(first_stage)
        self.stable_B.requires_grad_(first_stage)
        self.plastic_A.requires_grad_(True)
        self.plastic_B.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep FP32 master factors for long continual runs.  Casting inside the
        # differentiable path preserves FP32 gradients/optimizer states while
        # matching SAM 3's BF16 activation dtype during matrix multiplications.
        stable_a = self.stable_A.to(dtype=x.dtype)
        stable_b = self.stable_B.to(dtype=x.dtype)
        plastic_a = self.plastic_A.to(dtype=x.dtype)
        plastic_b = self.plastic_B.to(dtype=x.dtype)
        stable = F.linear(F.linear(x, stable_a), stable_b)
        plastic = F.linear(F.linear(x, plastic_a), plastic_b)
        return self.base(x) + (stable + plastic) * self.scaling

    def regularization_loss(self) -> torch.Tensor:
        # Preserve room for future adaptation without associating a direction
        # with a task.  The stable factors are detached after stage one.
        stable = F.normalize(self.stable_A.float(), dim=1)
        plastic = F.normalize(self.plastic_A.float(), dim=1)
        overlap = plastic @ stable.T
        return overlap.square().mean().to(self.base.weight.dtype) * self.orthogonality_weight

    @staticmethod
    def _factor_svd(b: torch.Tensor, a: torch.Tensor, keep: int):
        """Truncated SVD of ``b @ a`` using only a small rank-by-rank SVD."""
        work_b, work_a = b.float(), a.float()
        q_b, r_b = torch.linalg.qr(work_b, mode="reduced")
        q_a, r_a = torch.linalg.qr(work_a.T, mode="reduced")
        small = r_b @ r_a.T
        u, singular, vh = torch.linalg.svd(small, full_matrices=False)
        keep = min(keep, singular.numel())
        root = singular[:keep].clamp_min(0).sqrt()
        new_b = (q_b @ u[:, :keep]) * root.unsqueeze(0)
        new_a = root.unsqueeze(1) * (vh[:keep] @ q_a.T)
        discarded = singular[keep:].square().sum()
        total = singular.square().sum().clamp_min(torch.finfo(singular.dtype).eps)
        return new_b, new_a, float((discarded / total).item())

    @torch.no_grad()
    def _reset_plastic(self) -> None:
        candidate = torch.randn_like(self.plastic_A.float())
        stable = F.normalize(self.stable_A.float(), dim=1)
        candidate = candidate - (candidate @ stable.T) @ stable
        # In degenerate low-dimensional toy layers, a second random draw avoids
        # retaining zero rows after projection.
        norms = candidate.norm(dim=1, keepdim=True)
        if bool((norms < 1e-6).any()):
            fallback = torch.randn_like(candidate)
            candidate = torch.where(norms < 1e-6, fallback, candidate)
        self.plastic_A.copy_(F.normalize(candidate, dim=1).to(self.plastic_A.dtype))
        self.plastic_B.zero_()

    @torch.no_grad()
    def end_task(self) -> dict[str, float | int | str]:
        # Weight the plastic residual globally; gamma is fixed for the complete
        # stream and is never selected from a dataset identity.
        b = torch.cat((self.stable_B, self.plastic_B * self.merge_gamma), dim=1)
        a = torch.cat((self.stable_A, self.plastic_A), dim=0)
        new_b, new_a, relative_error = self._factor_svd(b, a, self.stable_rank)
        self.stable_B.copy_(new_b.to(self.stable_B))
        self.stable_A.copy_(new_a.to(self.stable_A))
        self._reset_plastic()
        self.consolidations.add_(1)
        self.stable_A.requires_grad_(False)
        self.stable_B.requires_grad_(False)
        return {"group": self.function_group, "rank": self.rank,
                "stable_rank": self.stable_rank, "plastic_rank": self.plastic_rank,
                "relative_compression_error": relative_error}

    def plastic_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.plastic_A, self.plastic_B


class ImportancePoolFocusLoRALinear(ContinualAdapterLinear):
    """Function-specific fixed-rank pool with data-free importance protection.

    Rank-one atoms are shared by the complete stream and are never indexed by a
    task or dataset identity.  A small subset is admitted for learning at each
    stream transition.  Previously allocated atoms remain active at inference,
    while their parameters are protected exactly from both gradients and
    decoupled weight decay.  When the pool is full, only the globally least
    important atoms are recycled.

    Importance is a fixed-size parameter statistic accumulated while the
    current samples are available.  No image, annotation, feature, or logit is
    retained after a stream stage.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5,
                 *, function_group: str, admission_ratio: float = 0.25,
                 importance_decay: float = 0.99,
                 orthogonality_weight: float = 1e-3):
        super().__init__(base, rank, alpha, num_tasks)
        if function_group not in FOCUS_TARGET_GROUPS:
            raise ValueError(f"Unknown FOCUS function group: {function_group}")
        if not 0.0 < admission_ratio <= 1.0:
            raise ValueError("admission_ratio must be in (0, 1]")
        if not 0.0 < importance_decay <= 1.0:
            raise ValueError("importance_decay must be in (0, 1]")
        self.function_group = function_group
        self.admission_ratio = float(admission_ratio)
        self.importance_decay = float(importance_decay)
        self.orthogonality_weight = float(orthogonality_weight)

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Only allocated atoms contribute to inference.  Importance and the
        # allocation mask are the complete persistent continual state.
        self.register_buffer("allocated_mask", torch.zeros(rank, dtype=torch.bool))
        self.register_buffer("importance", torch.zeros(rank))
        self.register_buffer("consolidations", torch.zeros((), dtype=torch.long))
        # Transient training state is intentionally absent from checkpoints.
        self.register_buffer("active_mask", torch.zeros(rank, dtype=torch.bool),
                             persistent=False)
        self.register_buffer("score_accumulator", torch.zeros(rank), persistent=False)
        self.register_buffer("score_steps", torch.zeros((), dtype=torch.long),
                             persistent=False)
        self._recycled_this_stage = 0
        self._protected_A: torch.Tensor | None = None
        self._protected_B: torch.Tensor | None = None
        self._protected_basis: torch.Tensor | None = None
        self.lora_A.register_hook(
            lambda gradient: gradient * self.active_mask[:, None].to(gradient))
        self.lora_B.register_hook(
            lambda gradient: gradient * self.active_mask[None, :].to(gradient))

    def _protected_mask(self) -> torch.Tensor:
        return self.allocated_mask & ~self.active_mask

    @torch.no_grad()
    def _initialize_atoms(self, slots: torch.Tensor) -> None:
        if not len(slots):
            return
        candidate = torch.randn(len(slots), self.in_features, device=self.lora_A.device,
                                dtype=torch.float32)
        if self._protected_basis is not None:
            candidate = candidate - ((candidate @ self._protected_basis.T) @
                                     self._protected_basis)
        norms = candidate.norm(dim=1, keepdim=True)
        if bool((norms < 1e-6).any()):
            fallback = torch.randn_like(candidate)
            candidate = torch.where(norms < 1e-6, fallback, candidate)
        self.lora_A[slots] = F.normalize(candidate, dim=1).to(self.lora_A)
        self.lora_B[:, slots].zero_()

    @torch.no_grad()
    def begin_task(self, task_id: int) -> None:
        # The caller may expose a stream boundary, but the stage identity is not
        # used to select parameters or alter the forward path.
        del task_id
        self.importance.mul_(self.importance_decay)
        requested = max(1, min(self.rank, int(math.ceil(self.rank * self.admission_ratio))))
        free = torch.where(~self.allocated_mask)[0]
        selected = free[:requested]
        missing = requested - len(selected)
        recycled = self.importance.new_empty(0, dtype=torch.long)
        if missing > 0:
            allocated = torch.where(self.allocated_mask)[0]
            order = torch.argsort(self.importance[allocated], stable=True)
            recycled = allocated[order[:missing]]
            selected = torch.cat((selected, recycled))

        self.active_mask.zero_()
        self.active_mask[selected] = True
        self.importance[selected] = 0
        protected = self.lora_A[self._protected_mask()].float()
        # Protected atoms do not change during this stage, so their orthogonal
        # basis can be cached instead of recomputed in every loss evaluation.
        self._protected_basis = (torch.linalg.qr(protected.T, mode="reduced").Q.T
                                 if len(protected) else None)
        self._initialize_atoms(selected)
        self.allocated_mask[selected] = True
        self.score_accumulator.zero_()
        self.score_steps.zero_()
        self._recycled_this_stage = int(len(recycled))
        # AdamW applies decoupled weight decay even to zero-gradient rows.  Keep
        # an ephemeral copy so protected atoms remain bitwise stable.
        self._protected_A = self.lora_A.detach().clone()
        self._protected_B = self.lora_B.detach().clone()

    def restore_trainability(self) -> None:
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
        self._protected_A = None
        self._protected_B = None
        self._protected_basis = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.allocated_mask.to(device=x.device, dtype=x.dtype)
        a = self.lora_A.to(dtype=x.dtype) * mask[:, None]
        b = self.lora_B.to(dtype=x.dtype) * mask[None, :]
        return self.base(x) + F.linear(F.linear(x, a), b) * self.scaling

    def regularization_loss(self) -> torch.Tensor:
        active = self.lora_A[self.active_mask].float()
        if not len(active) or self._protected_basis is None:
            return self.base.weight.new_zeros(())
        active = F.normalize(active, dim=1)
        overlap = active @ self._protected_basis.T
        return (overlap.square().mean() * self.orthogonality_weight).to(
            self.base.weight.dtype)

    @torch.no_grad()
    def accumulate_importance(self) -> None:
        if self.lora_A.grad is None or self.lora_B.grad is None:
            return
        # First-order functional saliency for atom b_i a_i^T.  The product with
        # the parameter value is invariant to the most common A/B rescaling.
        score_a = (self.lora_A.grad.float() * self.lora_A.float()).square().mean(1).sqrt()
        score_b = (self.lora_B.grad.float() * self.lora_B.float()).square().mean(0).sqrt()
        score = (score_a + score_b) * self.active_mask
        self.score_accumulator.add_(score.to(self.score_accumulator))
        self.score_steps.add_(1)

    @torch.no_grad()
    def restore_protected_parameters(self) -> None:
        if self._protected_A is None or self._protected_B is None:
            return
        protected = self._protected_mask()
        self.lora_A[protected] = self._protected_A[protected]
        self.lora_B[:, protected] = self._protected_B[:, protected]

    @torch.no_grad()
    def reset_active_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        for parameter, mask in ((self.lora_A, self.active_mask[:, None]),
                                (self.lora_B, self.active_mask[None, :])):
            state = optimizer.state.get(parameter, {})
            for value in state.values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.masked_fill_(mask.expand_as(value), 0)

    @staticmethod
    def _normalize_active(values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(values)
        if bool(active.any()):
            chosen = values[active]
            scale = chosen.max().clamp_min(torch.finfo(chosen.dtype).eps)
            result[active] = chosen / scale
        return result

    @torch.no_grad()
    def end_task(self) -> dict[str, float | int | str]:
        active = self.active_mask
        steps = max(int(self.score_steps.item()), 1)
        gradient_score = self.score_accumulator / steps
        magnitude = self.lora_A.float().norm(dim=1) * self.lora_B.float().norm(dim=0)
        saliency = 0.5 * self._normalize_active(gradient_score, active)
        saliency.add_(0.5 * self._normalize_active(magnitude, active))
        self.importance[active] = torch.maximum(self.importance[active], saliency[active])
        self.consolidations.add_(1)
        report = {
            "group": self.function_group,
            "rank": self.rank,
            "allocated_rank": int(self.allocated_mask.sum().item()),
            "active_rank": int(active.sum().item()),
            "recycled_rank": self._recycled_this_stage,
            "mean_importance": float(self.importance[self.allocated_mask].mean().item()),
            # No rank truncation is performed in FOCUS-v2.
            "relative_compression_error": 0.0,
        }
        self._protected_A = None
        self._protected_B = None
        self._protected_basis = None
        return report


class CapacityPartitionFocusLoRALinear(ContinualAdapterLinear):
    """Function-specific capacity partition with input-conditioned inference.

    Training-stage boundaries select a disjoint, nearly equal slice of a single
    fixed rank pool.  Once a slice has been learned it is never updated or
    recycled.  At inference, all allocated atoms are available and a
    parameter-free activation gate selects the top-k atoms for each input token;
    neither the stage index nor a dataset identity enters the forward method.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5,
                 *, function_group: str, top_k: int = 4,
                 orthogonality_weight: float = 1e-3):
        super().__init__(base, rank, alpha, num_tasks)
        if function_group not in FOCUS_TARGET_GROUPS:
            raise ValueError(f"Unknown FOCUS function group: {function_group}")
        if rank < num_tasks:
            raise ValueError("capacity-partitioned FOCUS needs at least one rank per stage")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.function_group = function_group
        self.top_k = min(int(top_k), rank)
        self.orthogonality_weight = float(orthogonality_weight)
        self.routing_mode = "atom_topk"
        self.routing_blocks = 1
        self.oracle_stage: int | None = None

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.register_buffer("allocated_mask", torch.zeros(rank, dtype=torch.bool))
        self.register_buffer("consolidations", torch.zeros((), dtype=torch.long))
        self.register_buffer("active_mask", torch.zeros(rank, dtype=torch.bool),
                             persistent=False)
        self._protected_A: torch.Tensor | None = None
        self._protected_B: torch.Tensor | None = None
        self._protected_basis: torch.Tensor | None = None
        self.lora_A.register_hook(
            lambda gradient: gradient * self.active_mask[:, None].to(gradient))
        self.lora_B.register_hook(
            lambda gradient: gradient * self.active_mask[None, :].to(gradient))

    def _stage_slots(self, stage: int) -> torch.Tensor:
        if not 0 <= stage < self.num_tasks:
            raise ValueError(f"stage {stage} is outside [0, {self.num_tasks})")
        boundaries = torch.linspace(0, self.rank, self.num_tasks + 1,
                                    device=self.allocated_mask.device).round().long()
        lo, hi = int(boundaries[stage]), int(boundaries[stage + 1])
        if hi <= lo:
            raise RuntimeError("rank partition unexpectedly produced an empty stage")
        return torch.arange(lo, hi, device=self.allocated_mask.device)

    def _protected_mask(self) -> torch.Tensor:
        return self.allocated_mask & ~self.active_mask

    @torch.no_grad()
    def begin_task(self, task_id: int) -> None:
        # The stage index controls training ownership only.  It is neither saved
        # as a model key nor consumed by the inference path.
        slots = self._stage_slots(int(task_id))
        if bool(self.allocated_mask[slots].any()):
            raise RuntimeError("a capacity slice cannot be allocated more than once")
        self.active_mask.zero_()
        self.active_mask[slots] = True
        protected = self.lora_A[self._protected_mask()].float()
        self._protected_basis = (torch.linalg.qr(protected.T, mode="reduced").Q.T
                                 if len(protected) else None)
        candidate = torch.randn(len(slots), self.in_features, device=self.lora_A.device,
                                dtype=torch.float32)
        if self._protected_basis is not None:
            candidate = candidate - ((candidate @ self._protected_basis.T) @
                                     self._protected_basis)
        norms = candidate.norm(dim=1, keepdim=True)
        if bool((norms < 1e-6).any()):
            candidate = torch.where(norms < 1e-6, torch.randn_like(candidate), candidate)
        self.lora_A[slots] = F.normalize(candidate, dim=1).to(self.lora_A)
        self.lora_B[:, slots].zero_()
        self.allocated_mask[slots] = True
        self._protected_A = self.lora_A.detach().clone()
        self._protected_B = self.lora_B.detach().clone()

    def restore_trainability(self) -> None:
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
        self._protected_A = None
        self._protected_B = None
        self._protected_basis = None

    def set_routing(self, mode: str = "atom_topk", *, blocks: int = 1,
                    oracle_stage: int | None = None) -> None:
        choices = {"atom_topk", "block_energy", "oracle_block"}
        if mode not in choices:
            raise ValueError(f"Unknown capacity routing mode {mode}; choose from {sorted(choices)}")
        if blocks < 1:
            raise ValueError("routing blocks must be positive")
        if mode == "oracle_block":
            if oracle_stage is None or not 0 <= oracle_stage < self.num_tasks:
                raise ValueError("oracle_block needs a valid oracle_stage")
        self.routing_mode = mode
        self.routing_blocks = min(int(blocks), self.num_tasks)
        self.oracle_stage = oracle_stage

    def _block_allowed_mask(self, activations: torch.Tensor) -> torch.Tensor:
        """Return a per-token atom mask selected without learned routing weights."""
        valid = self.allocated_mask.to(activations.device)
        if self.routing_mode == "oracle_block":
            allowed = torch.zeros_like(valid)
            slots = self._stage_slots(int(self.oracle_stage)).to(allowed.device)
            allowed[slots] = valid[slots]
            return allowed.expand(*activations.shape[:-1], self.rank)

        a_norm = self.lora_A.float().norm(dim=1).clamp_min(1e-6)
        normalized = activations.float().abs() / a_norm.to(activations.device)
        block_scores = []
        block_masks = []
        for stage in range(self.num_tasks):
            slots = self._stage_slots(stage).to(activations.device)
            stage_valid = valid[slots]
            mask = torch.zeros(self.rank, device=activations.device, dtype=torch.bool)
            mask[slots] = stage_valid
            block_masks.append(mask)
            if bool(stage_valid.any()):
                score = normalized[..., slots][..., stage_valid].square().mean(dim=-1)
            else:
                score = normalized.new_full(normalized.shape[:-1], -torch.inf)
            block_scores.append(score)
        scores = torch.stack(block_scores, dim=-1)
        available = int(torch.isfinite(scores.reshape(-1, self.num_tasks)[0]).sum().item())
        k = min(self.routing_blocks, max(available, 1))
        chosen = scores.topk(k, dim=-1).indices
        selected_blocks = torch.zeros_like(scores, dtype=torch.bool)
        selected_blocks.scatter_(-1, chosen, True)
        masks = torch.stack(block_masks, dim=0)
        return (selected_blocks.unsqueeze(-1) & masks).any(dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valid = self.allocated_mask
        if not bool(valid.any()):
            return self.base(x)
        mask = valid.to(device=x.device, dtype=x.dtype)
        a = self.lora_A.to(dtype=x.dtype) * mask[:, None]
        b = self.lora_B.to(dtype=x.dtype) * mask[None, :]
        activations = F.linear(x, a)
        logits = activations.float().to(x.dtype)
        allowed = (valid.to(x.device).expand(*logits.shape[:-1], self.rank)
                   if self.routing_mode == "atom_topk"
                   else self._block_allowed_mask(activations))
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        k = min(self.top_k, int(valid.sum().item()))
        if self.routing_mode == "oracle_block":
            k = min(k, int(allowed.reshape(-1, self.rank)[0].sum().item()))
        elif self.routing_mode == "block_energy":
            k = min(k, int(allowed.reshape(-1, self.rank)[0].sum().item()))
        values, indices = logits.topk(k, dim=-1)
        sparse = torch.full_like(logits, torch.finfo(logits.dtype).min)
        sparse.scatter_(-1, indices, values)
        gates = sparse.softmax(dim=-1)
        return self.base(x) + F.linear(activations * gates, b) * self.scaling

    def regularization_loss(self) -> torch.Tensor:
        active = self.lora_A[self.active_mask].float()
        if not len(active) or self._protected_basis is None:
            return self.base.weight.new_zeros(())
        overlap = F.normalize(active, dim=1) @ self._protected_basis.T
        return (overlap.square().mean() * self.orthogonality_weight).to(
            self.base.weight.dtype)

    @torch.no_grad()
    def restore_protected_parameters(self) -> None:
        if self._protected_A is None or self._protected_B is None:
            return
        protected = self._protected_mask()
        self.lora_A[protected] = self._protected_A[protected]
        self.lora_B[:, protected] = self._protected_B[:, protected]

    @torch.no_grad()
    def reset_active_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        for parameter, mask in ((self.lora_A, self.active_mask[:, None]),
                                (self.lora_B, self.active_mask[None, :])):
            state = optimizer.state.get(parameter, {})
            for value in state.values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.masked_fill_(mask.expand_as(value), 0)

    @torch.no_grad()
    def end_task(self) -> dict[str, float | int | str]:
        report = {
            "group": self.function_group,
            "rank": self.rank,
            "allocated_rank": int(self.allocated_mask.sum().item()),
            "active_rank": int(self.active_mask.sum().item()),
            "remaining_rank": int((~self.allocated_mask).sum().item()),
            "recycled_rank": 0,
            "relative_compression_error": 0.0,
        }
        self.consolidations.add_(1)
        self._protected_A = None
        self._protected_B = None
        self._protected_basis = None
        return report


class MultiAnchorTaskRouter(nn.Module):
    """Image-level, task-agnostic routing from frozen visual activations.

    Each stream stage contributes only additive first/second-order statistics of
    a normalized image descriptor.  At inference the closest distributions are
    mixed softly; no task identifier, learned router, replay sample, top-k, or
    temperature is required.
    """

    def __init__(self, feature_dim: int, num_tasks: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_tasks = int(num_tasks)
        self.register_buffer("anchor_sum", torch.zeros(num_tasks, feature_dim))
        self.register_buffer("anchor_square_sum", torch.zeros(num_tasks, feature_dim))
        self.register_buffer("anchor_count", torch.zeros(num_tasks))
        self.active_stage: int | None = None
        self._route: torch.Tensor | None = None
        self._accumulated = False

    def begin_task(self, stage: int) -> None:
        if not 0 <= int(stage) < self.num_tasks:
            raise ValueError(f"stage {stage} is outside [0, {self.num_tasks})")
        self.active_stage = int(stage)
        self._route = None
        self._accumulated = False

    def begin_batch(self) -> None:
        self._route = None
        self._accumulated = False

    @staticmethod
    def _descriptor(x: torch.Tensor) -> torch.Tensor:
        value = x.detach().float()
        if value.ndim < 1:
            raise ValueError("anchor source activation must include a feature dimension")
        # SAM 3 partitions one image into local-window batches before its first
        # qkv layer (for example [9, 28, 28, C]).  The continual trainer and
        # evaluator intentionally use one image per GPU, so all leading axes
        # belong to that image rather than representing independent samples.
        value = value.reshape(-1, value.shape[-1]).mean(dim=0, keepdim=True)
        return F.normalize(value, dim=-1)

    @torch.no_grad()
    def prepare(self, x: torch.Tensor, *, training: bool) -> None:
        descriptor = self._descriptor(x)
        if descriptor.shape[-1] != self.feature_dim:
            raise RuntimeError(
                f"anchor descriptor width {descriptor.shape[-1]} != {self.feature_dim}")
        if training:
            if self.active_stage is None:
                raise RuntimeError("begin_task must be called before anchor training")
            stage = self.active_stage
            if not self._accumulated:
                self.anchor_sum[stage].add_(descriptor.sum(dim=0).to(self.anchor_sum))
                self.anchor_square_sum[stage].add_(
                    descriptor.square().sum(dim=0).to(self.anchor_square_sum))
                self.anchor_count[stage].add_(descriptor.shape[0])
                self._accumulated = True
            route = descriptor.new_zeros(descriptor.shape[0], self.num_tasks)
            route[:, stage] = 1.0
            self._route = route
            return

        available = self.anchor_count > 0
        if not bool(available.any()):
            # This path is used only by pre-training smoke tests.
            route = descriptor.new_zeros(descriptor.shape[0], self.num_tasks)
            route[:, 0] = 1.0
            self._route = route
            return
        count = self.anchor_count[available].float().unsqueeze(1)
        mean = self.anchor_sum[available].float() / count
        variance = (self.anchor_square_sum[available].float() / count - mean.square()).clamp_min(0)
        # A relative numerical floor prevents a near-constant frozen feature
        # coordinate from dominating the distribution distance.
        floor = variance.mean(dim=1, keepdim=True).clamp_min(1e-6) * 1e-3
        distance = ((descriptor[:, None, :] - mean[None, :, :]).square() /
                    (variance[None, :, :] + floor[None, :, :])).mean(dim=-1)
        # Student-t style scores retain a useful soft mixture even when a
        # high-dimensional Mahalanobis distance is numerically large.
        posterior = (-torch.log1p(distance)).softmax(dim=-1)
        route = descriptor.new_zeros(descriptor.shape[0], self.num_tasks)
        route[:, available] = posterior
        self._route = route

    def route_for(self, x: torch.Tensor) -> torch.Tensor:
        if self._route is None:
            raise RuntimeError("the visual anchor source must run before task adapters")
        route = self._route.to(device=x.device, dtype=x.dtype)
        batch = route.shape[0]
        if batch == 1:
            return route.reshape(*([1] * (x.ndim - 1)), self.num_tasks)
        candidates = [axis for axis, size in enumerate(x.shape[:-1]) if size == batch]
        if not candidates:
            raise RuntimeError(
                f"cannot broadcast anchor batch {batch} into activation shape {tuple(x.shape)}")
        shape = [1] * x.ndim
        shape[candidates[0]] = batch
        shape[-1] = self.num_tasks
        return route.reshape(shape)

    @torch.no_grad()
    def synchronize_active(self) -> None:
        if self.active_stage is None:
            return
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return
        stage = self.active_stage
        for value in (self.anchor_sum[stage], self.anchor_square_sum[stage],
                      self.anchor_count[stage]):
            dist.all_reduce(value)

    def diagnostics(self) -> dict[str, object]:
        available = self.anchor_count > 0
        return {
            "anchors": int(available.sum().item()),
            "samples_per_anchor": [int(x) for x in self.anchor_count.tolist()],
            "feature_dim": self.feature_dim,
        }


class MultiAnchorTaskDecoupledLoRALinear(CapacityPartitionFocusLoRALinear):
    """Function-specific LoRA blocks controlled by one shared image route."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5,
                 *, function_group: str, orthogonality_weight: float = 1e-3):
        super().__init__(base, rank, alpha, num_tasks,
                         function_group=function_group, top_k=rank,
                         orthogonality_weight=orthogonality_weight)
        self._anchor_router: MultiAnchorTaskRouter | None = None
        self._is_anchor_source = False

    def attach_anchor_router(self, router: MultiAnchorTaskRouter, *, source: bool) -> None:
        if source:
            # Register the shared state exactly once so ordinary adapter
            # checkpoint extraction includes the anchor statistics.
            self.anchor_router = router
        object.__setattr__(self, "_anchor_router", router)
        self._is_anchor_source = bool(source)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router = self._anchor_router
        if router is None:
            raise RuntimeError("multi-anchor adapter has no shared router")
        if self._is_anchor_source:
            router.prepare(x, training=self.training)
        valid = self.allocated_mask
        if not bool(valid.any()):
            return self.base(x)
        mask = valid.to(device=x.device, dtype=x.dtype)
        a = self.lora_A.to(dtype=x.dtype) * mask[:, None]
        b = self.lora_B.to(dtype=x.dtype) * mask[None, :]
        activations = F.linear(x, a)
        stage_route = router.route_for(activations)
        atom_route = activations.new_zeros(*stage_route.shape[:-1], self.rank)
        for stage in range(self.num_tasks):
            slots = self._stage_slots(stage).to(activations.device)
            atom_route[..., slots] = stage_route[..., stage].unsqueeze(-1)
        return self.base(x) + F.linear(activations * atom_route, b) * self.scaling


class PromptMLPTaskRouter(nn.Module):
    """SAMCL-style shared visual selector with lightweight feature replay.

    The class name is retained for checkpoint/config compatibility, but prompts
    are deliberately not selector inputs. One image-level decision controls all
    visual, semantic, box, and mask adapters, preserving task consistency.
    """

    def __init__(self, visual_dim: int, num_tasks: int, embeddings_per_task: int = 300,
                 routing_block: int | None = None):
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.num_tasks = int(num_tasks)
        self.embeddings_per_task = int(embeddings_per_task)
        self.routing_block = routing_block
        hidden = max(4, self.visual_dim // 4)
        # Match SAMCL: four linear layers, with a D -> D -> D/4 -> D/4 -> T
        # layout and BatchNorm on the first three layers.
        self.selector = nn.Sequential(
            nn.Linear(self.visual_dim, self.visual_dim),
            nn.BatchNorm1d(self.visual_dim),
            nn.Linear(self.visual_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.num_tasks),
        )
        self.register_buffer(
            "embedding_buffer",
            torch.zeros(self.num_tasks, self.embeddings_per_task, self.visual_dim,
                        dtype=torch.float16),
        )
        self.register_buffer("embedding_count", torch.zeros(self.num_tasks, dtype=torch.long))
        self.register_buffer("selector_fit_loss", torch.full((), float("nan")))
        self.register_buffer("selector_fit_accuracy", torch.full((), float("nan")))
        self.active_stage: int | None = None
        self._visual_route: torch.Tensor | None = None
        self._route: torch.Tensor | None = None
        self._visual_posterior: torch.Tensor | None = None
        self._posterior: torch.Tensor | None = None
        self._visual_accumulated = False
        self._visual_descriptor: torch.Tensor | None = None
        # Evaluation-only diagnostic. Training and normal task-free inference
        # always leave this unset and use the learned selector.
        self.inference_oracle_stage: int | None = None

    def begin_task(self, stage: int) -> None:
        if not 0 <= int(stage) < self.num_tasks:
            raise ValueError(f"stage {stage} is outside [0, {self.num_tasks})")
        self.active_stage = int(stage)
        self.begin_batch()

    def begin_batch(self) -> None:
        self._visual_route = self._route = None
        self._visual_posterior = self._posterior = None
        self._visual_accumulated = False
        self._visual_descriptor = None

    @staticmethod
    def _visual_descriptors_from(x: torch.Tensor) -> torch.Tensor:
        value = x.detach().float()
        if value.ndim == 1:
            value = value.unsqueeze(0)
        elif value.ndim > 2:
            # The routing hook observes a trunk block input [B, ..., D], before
            # task-specific layers. Pool only spatial/token axes, never batch.
            value = value.reshape(value.shape[0], -1, value.shape[-1]).mean(dim=1)
        # SAMCL keeps the raw pooled embedding. BatchNorm in the selector is
        # responsible for feature scaling; L2 normalization would discard
        # domain-dependent activation magnitude.
        return value

    @staticmethod
    def _one_hot(indices: torch.Tensor, tasks: int) -> torch.Tensor:
        return F.one_hot(indices, num_classes=tasks).to(torch.float32)

    @staticmethod
    def _posterior_from(logits: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        masked = logits.masked_fill(
            ~available.to(logits.device).unsqueeze(0), -torch.inf)
        return masked.softmax(dim=-1)

    @torch.no_grad()
    def _store_current(self, descriptors: torch.Tensor) -> None:
        if self.active_stage is None:
            raise RuntimeError("begin_task must be called before selector feature collection")
        stage = self.active_stage
        start = int(self.embedding_count[stage].item())
        amount = min(len(descriptors), self.embeddings_per_task - start)
        if amount <= 0:
            return
        self.embedding_buffer[stage, start:start + amount].copy_(
            descriptors[:amount].to(self.embedding_buffer))
        self.embedding_count[stage].add_(amount)

    def prepare_visual(self, x: torch.Tensor, *, training: bool) -> None:
        # A block pre-hook normally calls this. The first adapted qkv also has a
        # fallback call so lightweight model wrappers remain supported.
        if self._visual_route is not None:
            return
        descriptor = self._visual_descriptors_from(x)
        if descriptor.shape[-1] != self.visual_dim:
            raise RuntimeError(
                f"visual selector width {descriptor.shape[-1]} != {self.visual_dim}")
        self._visual_descriptor = descriptor
        if training:
            if self.active_stage is None:
                raise RuntimeError("begin_task must be called before selector training")
            if not self._visual_accumulated:
                self._store_current(descriptor)
                self._visual_accumulated = True
            indices = torch.full((len(descriptor),), self.active_stage,
                                 device=descriptor.device, dtype=torch.long)
            posterior = self._one_hot(indices, self.num_tasks).to(descriptor)
        elif self.inference_oracle_stage is not None:
            stage = int(self.inference_oracle_stage)
            if not 0 <= stage < self.num_tasks:
                raise ValueError(
                    f"oracle anchor {stage} is outside [0, {self.num_tasks})")
            indices = torch.full((len(descriptor),), stage,
                                 device=descriptor.device, dtype=torch.long)
            posterior = self._one_hot(indices, self.num_tasks).to(descriptor)
        else:
            available = self.embedding_count > 0
            if not bool(available.any()):
                available = torch.zeros_like(available)
                available[0] = True
            logits = self.selector(descriptor)
            posterior = self._posterior_from(logits, available)
            indices = posterior.argmax(dim=-1)
        route = self._one_hot(indices, self.num_tasks).to(descriptor)
        self._visual_route = self._route = route
        self._visual_posterior = self._posterior = posterior

    def prepare_text(self, _x: torch.Tensor, *, training: bool) -> None:
        # Kept as a no-op compatibility surface: text/prompt information must
        # not alter the single image-level route.
        if self._route is None:
            raise RuntimeError("visual selector must run before model heads")

    def route_for(self, x: torch.Tensor, *, group: str | None = None) -> torch.Tensor:
        del group
        if self._route is None:
            raise RuntimeError("selector source did not run before task adapters")
        route = self._route.to(device=x.device, dtype=x.dtype)
        batch = len(route)
        if batch == 1:
            return route.reshape(*([1] * (x.ndim - 1)), self.num_tasks)
        candidates = [axis for axis, size in enumerate(x.shape[:-1]) if size == batch]
        if not candidates:
            raise RuntimeError(
                f"cannot broadcast selector batch {batch} into {tuple(x.shape)}")
        shape = [1] * x.ndim
        shape[candidates[0]], shape[-1] = batch, self.num_tasks
        return route.reshape(shape)

    def selector_loss(self) -> torch.Tensor:
        # SAMCL trains the selector separately on the embedding buffer after a
        # task. Adapter optimization therefore cannot distort the gate.
        return self.embedding_buffer.sum().float() * 0.0

    @torch.no_grad()
    def synchronize_active(self) -> None:
        if self.active_stage is None:
            return
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return
        stage = self.active_stage
        world = dist.get_world_size()
        local_count = self.embedding_count[stage].clone()
        counts = [torch.zeros_like(local_count) for _ in range(world)]
        dist.all_gather(counts, local_count)
        local_buffer = self.embedding_buffer[stage].contiguous()
        buffers = [torch.zeros_like(local_buffer) for _ in range(world)]
        dist.all_gather(buffers, local_buffer)
        chunks = [value[:int(count.item())] for value, count in zip(buffers, counts)
                  if int(count.item()) > 0]
        merged = torch.cat(chunks, dim=0) if chunks else local_buffer[:0]
        if len(merged) > self.embeddings_per_task:
            # Evenly retain samples from all workers instead of letting rank 0
            # monopolize a full buffer.
            indices = torch.linspace(0, len(merged) - 1, self.embeddings_per_task,
                                     device=merged.device).round().long()
            merged = merged[indices]
        self.embedding_buffer[stage].zero_()
        self.embedding_buffer[stage, :len(merged)].copy_(merged)
        self.embedding_count[stage].fill_(len(merged))

    def _buffer_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        active = ((self.active_stage + 1) if self.active_stage is not None
                  else int((self.embedding_count > 0).sum().item()))
        features, labels = [], []
        for stage in range(active):
            count = int(self.embedding_count[stage].item())
            if count:
                features.append(self.embedding_buffer[stage, :count].float())
                labels.append(torch.full((count,), stage, device=self.embedding_buffer.device,
                                         dtype=torch.long))
        if not features:
            return (self.embedding_buffer.new_zeros((0, self.visual_dim)).float(),
                    self.embedding_count.new_zeros((0,)))
        return torch.cat(features), torch.cat(labels)

    def fit_selector(self, *, epochs: int = 25, lr: float = 0.01,
                     batch_size: int = 24, seed: int = 2412) -> dict[str, float | int]:
        features, labels = self._buffer_dataset()
        active = int((self.embedding_count > 0).sum().item())
        if active <= 1 or len(features) < 2:
            self.selector_fit_loss.fill_(0.0)
            self.selector_fit_accuracy.fill_(1.0 if len(features) else float("nan"))
            return {"loss": float(self.selector_fit_loss),
                    "accuracy": float(self.selector_fit_accuracy),
                    "samples": len(features), "epochs": 0}
        self.selector.train()
        optimizer = torch.optim.SGD(self.selector.parameters(), lr=lr)
        generator = torch.Generator(device="cpu").manual_seed(seed + active)
        last_loss = 0.0
        for _ in range(epochs):
            order = torch.randperm(len(features), generator=generator).to(features.device)
            epoch_loss, batches = 0.0, 0
            for start in range(0, len(order), batch_size):
                index = order[start:start + batch_size]
                if len(index) < 2:  # BatchNorm needs at least two examples.
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = self.selector(features[index])[:, :active]
                loss = F.cross_entropy(logits, labels[index])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
                batches += 1
            last_loss = epoch_loss / max(batches, 1)
        self.selector.eval()
        with torch.no_grad():
            predicted = self.selector(features)[:, :active].argmax(dim=-1)
            accuracy = float((predicted == labels).float().mean())
            self.selector_fit_loss.fill_(last_loss)
            self.selector_fit_accuracy.fill_(accuracy)
        return {"loss": last_loss, "accuracy": accuracy,
                "samples": len(features), "epochs": epochs}

    def diagnostics(self) -> dict[str, object]:
        used = int(sum(int(x) for x in self.embedding_count.tolist()))
        return {
            "anchors": int((self.embedding_count > 0).sum().item()),
            "samples_per_anchor": [int(x) for x in self.embedding_count.tolist()],
            "visual_dim": self.visual_dim,
            "routing_block": self.routing_block,
            "selector": "samcl_four_layer_visual_mlp_hard_top1",
            "shared_route": ["visual", "semantic", "box", "mask"],
            "selector_buffer_accuracy": float(self.selector_fit_accuracy),
            "selector_fit_loss": float(self.selector_fit_loss),
            "embedding_storage_bytes_used": used * self.visual_dim * 2,
            "embedding_storage_bytes_capacity": self.embedding_buffer.numel() * 2,
        }


class SAMCLSharedA(nn.Module):
    """Task modules' cross-layer shared down projection from original SAMCL.

    The official SAMCL AugModule creates one shared ``A`` projection for every
    task and reuses it in all adapted image-encoder blocks. SAM 3's visual
    attention layers after the routing cut have the same input width, so the
    same construction transfers directly. Prompt heatmaps are deliberately not
    represented here: the current benchmark exposes image/text inputs only and
    must not derive an interactive point from a test mask.
    """

    def __init__(self, in_features: int, rank: int, num_tasks: int):
        super().__init__()
        self.in_features = int(in_features)
        self.rank = int(rank)
        self.num_tasks = int(num_tasks)
        self.lora_A = nn.Parameter(torch.empty(num_tasks, rank, in_features))
        self.lora_A_bias = nn.Parameter(torch.zeros(num_tasks, rank))
        self.register_buffer("active_mask", torch.zeros(num_tasks, dtype=torch.bool),
                             persistent=False)
        self.register_buffer("allocated_mask", torch.zeros(num_tasks, dtype=torch.bool))
        self._protected_A: torch.Tensor | None = None
        self._protected_bias: torch.Tensor | None = None
        self.lora_A.register_hook(
            lambda gradient: gradient * self.active_mask[:, None, None].to(gradient))
        self.lora_A_bias.register_hook(
            lambda gradient: gradient * self.active_mask[:, None].to(gradient))
        with torch.no_grad():
            for stage in range(num_tasks):
                nn.init.kaiming_uniform_(self.lora_A[stage], a=math.sqrt(5))

    @torch.no_grad()
    def begin_task(self, stage: int) -> None:
        if not 0 <= int(stage) < self.num_tasks:
            raise ValueError(f"stage {stage} is outside [0, {self.num_tasks})")
        if bool(self.allocated_mask[stage]):
            raise RuntimeError("a SAMCL SLoRA task module cannot be allocated twice")
        self.active_mask.zero_()
        self.active_mask[stage] = True
        nn.init.kaiming_uniform_(self.lora_A[stage], a=math.sqrt(5))
        self.lora_A_bias[stage].zero_()
        self.allocated_mask[stage] = True
        self._protected_A = self.lora_A.detach().clone()
        self._protected_bias = self.lora_A_bias.detach().clone()

    def project(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        return F.linear(
            x,
            self.lora_A[stage].to(dtype=x.dtype),
            self.lora_A_bias[stage].to(dtype=x.dtype),
        )

    @torch.no_grad()
    def restore_protected_parameters(self) -> None:
        if self._protected_A is None or self._protected_bias is None:
            return
        protected = ~self.active_mask
        self.lora_A[protected] = self._protected_A[protected]
        self.lora_A_bias[protected] = self._protected_bias[protected]

    @torch.no_grad()
    def reset_active_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        for parameter, mask in (
                (self.lora_A, self.active_mask[:, None, None]),
                (self.lora_A_bias, self.active_mask[:, None])):
            for value in optimizer.state.get(parameter, {}).values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.masked_fill_(mask.expand_as(value), 0)

    def restore_trainability(self) -> None:
        self.lora_A.requires_grad_(True)
        self.lora_A_bias.requires_grad_(True)
        self._protected_A = None
        self._protected_bias = None

    def end_task(self) -> None:
        self._protected_A = None
        self._protected_bias = None


class SAMCLSLoRALinear(ContinualAdapterLinear):
    """SAM 3 port of SAMCL's SLoRA task modules and MLP module selector.

    A task owns one full-rank slice rather than a fraction of a fixed pool. The
    down projection is shared across all image-encoder layers through
    :class:`SAMCLSharedA`, while every layer retains the task-specific up
    projection and LayerNorm used by the official AugModule. At inference the
    visual MLP selects a hard top-1 module; no task or dataset identifier is an
    input.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5,
                 *, shared_a: SAMCLSharedA, function_group: str = "visual"):
        # Official SLoRA does not apply the usual alpha/r scale. Keep alpha in
        # the public signature for checkpoint/config compatibility, but force a
        # unit residual scale here.
        super().__init__(base, rank, alpha, num_tasks)
        if base.in_features != shared_a.in_features:
            raise ValueError(
                f"shared SLoRA A width {shared_a.in_features} != layer width {base.in_features}")
        self.scaling = 1.0
        self.function_group = function_group
        self.lora_B = nn.Parameter(torch.zeros(num_tasks, self.out_features, rank))
        self.lora_B_bias = nn.Parameter(torch.zeros(num_tasks, self.out_features))
        self.norm_weight = nn.Parameter(torch.ones(num_tasks, rank))
        self.norm_bias = nn.Parameter(torch.zeros(num_tasks, rank))
        self.register_buffer("active_mask", torch.zeros(num_tasks, dtype=torch.bool),
                             persistent=False)
        self.register_buffer("allocated_mask", torch.zeros(num_tasks, dtype=torch.bool))
        self._shared_a: SAMCLSharedA | None = None
        self._anchor_router: PromptMLPTaskRouter | None = None
        self._is_anchor_source = False
        self._protected_B: torch.Tensor | None = None
        self._protected_B_bias: torch.Tensor | None = None
        self._protected_norm_weight: torch.Tensor | None = None
        self._protected_norm_bias: torch.Tensor | None = None
        self.lora_B.register_hook(
            lambda gradient: gradient * self.active_mask[:, None, None].to(gradient))
        self.lora_B_bias.register_hook(
            lambda gradient: gradient * self.active_mask[:, None].to(gradient))
        self.norm_weight.register_hook(
            lambda gradient: gradient * self.active_mask[:, None].to(gradient))
        self.norm_bias.register_hook(
            lambda gradient: gradient * self.active_mask[:, None].to(gradient))
        object.__setattr__(self, "_shared_a", shared_a)

    def attach_samcl_state(self, router: PromptMLPTaskRouter, shared_a: SAMCLSharedA,
                           *, source: bool) -> None:
        if source:
            # Register shared objects exactly once so the ordinary adapter-only
            # checkpoint captures both SLoRA A and the selector state.
            self.shared_a = shared_a
            self.anchor_router = router
        object.__setattr__(self, "_shared_a", shared_a)
        object.__setattr__(self, "_anchor_router", router)
        self._is_anchor_source = bool(source)

    @torch.no_grad()
    def begin_task(self, task_id: int) -> None:
        super().begin_task(task_id)
        stage = int(task_id)
        if bool(self.allocated_mask[stage]):
            raise RuntimeError("a SAMCL SLoRA layer module cannot be allocated twice")
        self.active_mask.zero_()
        self.active_mask[stage] = True
        self.lora_B[stage].zero_()
        self.lora_B_bias[stage].zero_()
        self.norm_weight[stage].fill_(1.0)
        self.norm_bias[stage].zero_()
        self.allocated_mask[stage] = True
        self._protected_B = self.lora_B.detach().clone()
        self._protected_B_bias = self.lora_B_bias.detach().clone()
        self._protected_norm_weight = self.norm_weight.detach().clone()
        self._protected_norm_bias = self.norm_bias.detach().clone()
        if self._is_anchor_source:
            if self._shared_a is None:
                raise RuntimeError("SAMCL source adapter has no shared A")
            self._shared_a.begin_task(stage)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router, shared_a = self._anchor_router, self._shared_a
        if router is None or shared_a is None:
            raise RuntimeError("SAMCL SLoRA adapter has no shared selector/A state")
        if self._is_anchor_source and router._visual_route is None:
            router.prepare_visual(x, training=self.training)
        route = router.route_for(x)
        base = self.base(x)
        update = torch.zeros_like(base)
        for stage in range(self.num_tasks):
            if not bool(self.allocated_mask[stage]):
                continue
            hidden = shared_a.project(x, stage)
            hidden = F.layer_norm(
                hidden,
                (self.rank,),
                self.norm_weight[stage].to(dtype=x.dtype),
                self.norm_bias[stage].to(dtype=x.dtype),
            )
            delta = F.linear(
                hidden,
                self.lora_B[stage].to(dtype=x.dtype),
                self.lora_B_bias[stage].to(dtype=x.dtype),
            )
            delta = delta * route[..., stage].unsqueeze(-1)
            update = update + delta
        return base + update

    @torch.no_grad()
    def restore_protected_parameters(self) -> None:
        snapshots = (self._protected_B, self._protected_B_bias,
                     self._protected_norm_weight, self._protected_norm_bias)
        if any(value is None for value in snapshots):
            return
        protected = ~self.active_mask
        self.lora_B[protected] = self._protected_B[protected]
        self.lora_B_bias[protected] = self._protected_B_bias[protected]
        self.norm_weight[protected] = self._protected_norm_weight[protected]
        self.norm_bias[protected] = self._protected_norm_bias[protected]
        if self._is_anchor_source and self._shared_a is not None:
            self._shared_a.restore_protected_parameters()

    @torch.no_grad()
    def reset_active_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        entries = (
            (self.lora_B, self.active_mask[:, None, None]),
            (self.lora_B_bias, self.active_mask[:, None]),
            (self.norm_weight, self.active_mask[:, None]),
            (self.norm_bias, self.active_mask[:, None]),
        )
        for parameter, mask in entries:
            for value in optimizer.state.get(parameter, {}).values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.masked_fill_(mask.expand_as(value), 0)
        if self._is_anchor_source and self._shared_a is not None:
            self._shared_a.reset_active_optimizer_state(optimizer)

    def restore_trainability(self) -> None:
        for parameter in (self.lora_B, self.lora_B_bias,
                          self.norm_weight, self.norm_bias):
            parameter.requires_grad_(True)
        self._protected_B = None
        self._protected_B_bias = None
        self._protected_norm_weight = None
        self._protected_norm_bias = None
        if self._is_anchor_source and self._shared_a is not None:
            self._shared_a.restore_trainability()

    @torch.no_grad()
    def end_task(self) -> dict[str, float | int | str]:
        stage = int(self.task_id)
        report = {
            "group": self.function_group,
            "rank": self.rank * self.num_tasks,
            "allocated_rank": self.rank * (stage + 1),
            "active_rank": self.rank,
            "remaining_rank": self.rank * (self.num_tasks - stage - 1),
            "recycled_rank": 0,
            "relative_compression_error": 0.0,
        }
        self._protected_B = None
        self._protected_B_bias = None
        self._protected_norm_weight = None
        self._protected_norm_bias = None
        if self._is_anchor_source and self._shared_a is not None:
            self._shared_a.end_task()
        return report


class CompositionalPromptMLPTaskRouter(PromptMLPTaskRouter):
    """Reuse frozen earlier anchors through one learned task-consistent route.

    The SAMCL MLP still predicts exactly one leaf anchor.  A global strictly
    lower-triangular matrix then expands that leaf into its own residual plus a
    bounded signed mixture of earlier anchors.  The same expansion is consumed
    by every visual, semantic, box, and mask adapter, so detection and
    segmentation cannot silently select different histories.

    Each row starts at zero, making a new anchor exactly equivalent to the
    independent multi-anchor baseline before optimization.  Only the current
    row can change; all earlier rows and all LoRA atoms remain protected.
    """

    def __init__(self, visual_dim: int, num_tasks: int, embeddings_per_task: int = 300,
                 routing_block: int | None = None):
        super().__init__(visual_dim, num_tasks, embeddings_per_task, routing_block)
        self.composition_logits = nn.Parameter(torch.zeros(num_tasks, num_tasks))
        self.register_buffer(
            "composition_active_mask",
            torch.zeros(num_tasks, num_tasks, dtype=torch.bool),
            persistent=False,
        )
        self._protected_composition: torch.Tensor | None = None
        self.composition_logits.register_hook(
            lambda gradient: gradient * self.composition_active_mask.to(gradient))

    def begin_task(self, stage: int) -> None:
        super().begin_task(stage)
        self.composition_active_mask.zero_()
        if stage > 0:
            self.composition_active_mask[stage, :stage] = True
        self._protected_composition = self.composition_logits.detach().clone()

    def composition_matrix(self) -> torch.Tensor:
        inherited = torch.tril(torch.tanh(self.composition_logits), diagonal=-1)
        return inherited + torch.eye(
            self.num_tasks,
            device=self.composition_logits.device,
            dtype=self.composition_logits.dtype,
        )

    def route_for(self, x: torch.Tensor, *, group: str | None = None) -> torch.Tensor:
        # PromptMLPTaskRouter returns the hard leaf route.  Right multiplication
        # by the shared composition matrix activates the leaf residual with
        # coefficient one and any useful frozen predecessors with bounded
        # signed coefficients.
        leaf_route = super().route_for(x, group=group)
        matrix = self.composition_matrix().to(
            device=leaf_route.device, dtype=leaf_route.dtype)
        return torch.matmul(leaf_route, matrix)

    @torch.no_grad()
    def restore_protected_composition(self) -> None:
        if self._protected_composition is None:
            return
        protected = ~self.composition_active_mask
        self.composition_logits[protected] = self._protected_composition[protected]

    def diagnostics(self) -> dict[str, object]:
        report = super().diagnostics()
        matrix = self.composition_matrix().detach().float().cpu()
        report.update(
            selector="samcl_mlp_leaf_plus_learned_compositional_anchor",
            anchor_composition="shared_signed_lower_triangular_residual",
            composition_matrix=matrix.tolist(),
            composition_nonzero=int((matrix.tril(-1).abs() > 1e-6).sum().item()),
            composition_shared_across=["visual", "semantic", "box", "mask"],
            composition_uses_task_id_at_inference=False,
        )
        return report


class ObjectConsistentSkillRouter(PromptMLPTaskRouter):
    """Compose a newly learned residual with an automatically selected old skill.

    Parent skills are selected during training from joint semantic, box, mask,
    and box--mask consistency evidence.  At inference the ordinary visual MLP
    predicts only the leaf skill; this router deterministically expands it to
    the learned residual path, without receiving a task or dataset identifier.
    """

    def __init__(self, visual_dim: int, num_tasks: int, embeddings_per_task: int = 300,
                 routing_block: int | None = None):
        super().__init__(visual_dim, num_tasks, embeddings_per_task, routing_block)
        self.register_buffer("skill_parent", torch.full((num_tasks,), -1, dtype=torch.long))
        self.register_buffer("skill_parent_score", torch.full((num_tasks,), float("nan")))
        self.register_buffer("skill_base_score", torch.full((num_tasks,), float("nan")))
        self.register_buffer(
            "utility_buffer",
            torch.full((self.num_tasks, self.embeddings_per_task, self.num_tasks),
                       float("nan"), dtype=torch.float16),
        )
        self._forced_stage: int | None = None

    def _local_owner_limit(self) -> int:
        """Reserve the final buffer slots for counterfactual utility examples."""
        import torch.distributed as dist
        world = (dist.get_world_size()
                 if dist.is_available() and dist.is_initialized() else 1)
        utility_reserve = min(64, self.embeddings_per_task // 4)
        return max(1, (self.embeddings_per_task - utility_reserve) // world)

    @torch.no_grad()
    def _store_owner(self, descriptors: torch.Tensor) -> int:
        """Store dense anchor-ownership examples collected throughout training."""
        if self.active_stage is None:
            raise RuntimeError("begin_task must be called before owner collection")
        stage = self.active_stage
        descriptors = descriptors.detach().reshape(-1, self.visual_dim)
        start = int(self.embedding_count[stage].item())
        amount = min(len(descriptors), self._local_owner_limit() - start)
        if amount <= 0:
            return 0
        stop = start + amount
        self.embedding_buffer[stage, start:stop].copy_(
            descriptors[:amount].to(self.embedding_buffer))
        self.utility_buffer[stage, start:stop].fill_(float("nan"))
        # The active anchor is the ownership target. This is internal training
        # state, not an inference-time task or dataset identifier.
        self.utility_buffer[stage, start:stop, stage].zero_()
        self.embedding_count[stage].add_(amount)
        return amount

    @torch.no_grad()
    def set_parent(self, stage: int, parent: int, *, score: float, base_score: float) -> None:
        stage, parent = int(stage), int(parent)
        if not 0 <= stage < self.num_tasks:
            raise ValueError(f"stage {stage} is outside [0, {self.num_tasks})")
        if parent >= stage or parent < -1:
            raise ValueError(f"parent {parent} must be -1 or an earlier skill than {stage}")
        self.skill_parent[stage] = parent
        self.skill_parent_score[stage] = float(score)
        self.skill_base_score[stage] = float(base_score)

    def force_stage(self, stage: int | None) -> None:
        if stage is not None and not -1 <= int(stage) < self.num_tasks:
            raise ValueError(f"forced stage {stage} is invalid")
        self._forced_stage = None if stage is None else int(stage)
        self.begin_batch()

    def _compose(self, indices: torch.Tensor) -> torch.Tensor:
        route = indices.new_zeros((len(indices), self.num_tasks), dtype=torch.float32)
        for row, leaf_value in enumerate(indices.tolist()):
            leaf = int(leaf_value)
            visited: set[int] = set()
            while leaf >= 0:
                if leaf in visited:
                    raise RuntimeError("object-skill parent graph contains a cycle")
                visited.add(leaf)
                route[row, leaf] = 1.0
                leaf = int(self.skill_parent[leaf].item())
        return route

    def prepare_visual(self, x: torch.Tensor, *, training: bool) -> None:
        if self._visual_route is not None:
            return
        descriptor = self._visual_descriptors_from(x)
        if descriptor.shape[-1] != self.visual_dim:
            raise RuntimeError(
                f"visual selector width {descriptor.shape[-1]} != {self.visual_dim}")
        self._visual_descriptor = descriptor
        if training and self._forced_stage is None and not self._visual_accumulated:
            self._store_owner(descriptor)
            self._visual_accumulated = True
        if self._forced_stage is not None:
            indices = torch.full((len(descriptor),), self._forced_stage,
                                 device=descriptor.device, dtype=torch.long)
            posterior = (self._one_hot(indices, self.num_tasks).to(descriptor)
                         if self._forced_stage >= 0
                         else descriptor.new_zeros((len(descriptor), self.num_tasks)))
        elif training:
            if self.active_stage is None:
                raise RuntimeError("begin_task must be called before selector training")
            indices = torch.full((len(descriptor),), self.active_stage,
                                 device=descriptor.device, dtype=torch.long)
            posterior = self._one_hot(indices, self.num_tasks).to(descriptor)
        else:
            available = self.embedding_count > 0
            if not bool(available.any()):
                available = torch.zeros_like(available)
                available[0] = True
            logits = self.selector(descriptor)
            posterior = self._posterior_from(logits, available)
            indices = posterior.argmax(dim=-1)
        route = self._compose(indices).to(descriptor)
        self._visual_route = self._route = route
        self._visual_posterior = self._posterior = posterior

    @torch.no_grad()
    def store_utility(self, descriptors: torch.Tensor, utilities: torch.Tensor) -> int:
        """Store frozen descriptors with counterfactual joint-utility targets."""
        if self.active_stage is None:
            raise RuntimeError("begin_task must be called before utility collection")
        descriptors = descriptors.detach().reshape(-1, self.visual_dim)
        utilities = utilities.detach().reshape(len(descriptors), -1)
        if utilities.shape[1] > self.num_tasks:
            raise ValueError("utility target has more entries than available skill slots")
        stage = self.active_stage
        start = int(self.embedding_count[stage].item())
        amount = min(len(descriptors), self.embeddings_per_task - start)
        if amount <= 0:
            return 0
        stop = start + amount
        self.embedding_buffer[stage, start:stop].copy_(
            descriptors[:amount].to(self.embedding_buffer))
        self.utility_buffer[stage, start:stop].fill_(float("nan"))
        self.utility_buffer[stage, start:stop, :utilities.shape[1]].copy_(
            utilities[:amount].to(self.utility_buffer))
        self.embedding_count[stage].add_(amount)
        return amount

    @torch.no_grad()
    def synchronize_active(self) -> None:
        if self.active_stage is None:
            return
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return
        stage = self.active_stage
        world = dist.get_world_size()
        local_count = self.embedding_count[stage].clone()
        counts = [torch.zeros_like(local_count) for _ in range(world)]
        dist.all_gather(counts, local_count)
        gathered_features = [torch.zeros_like(self.embedding_buffer[stage])
                             for _ in range(world)]
        gathered_utilities = [torch.zeros_like(self.utility_buffer[stage])
                              for _ in range(world)]
        dist.all_gather(gathered_features, self.embedding_buffer[stage].contiguous())
        dist.all_gather(gathered_utilities, self.utility_buffer[stage].contiguous())
        feature_chunks, utility_chunks = [], []
        for feature, utility, count in zip(gathered_features, gathered_utilities, counts):
            amount = int(count.item())
            if amount:
                feature_chunks.append(feature[:amount])
                utility_chunks.append(utility[:amount])
        features = (torch.cat(feature_chunks) if feature_chunks
                    else self.embedding_buffer[stage, :0])
        utilities = (torch.cat(utility_chunks) if utility_chunks
                     else self.utility_buffer[stage, :0])
        if len(features) > self.embeddings_per_task:
            indices = torch.linspace(0, len(features) - 1, self.embeddings_per_task,
                                     device=features.device).round().long()
            features, utilities = features[indices], utilities[indices]
        self.embedding_buffer[stage].zero_()
        self.utility_buffer[stage].fill_(float("nan"))
        self.embedding_buffer[stage, :len(features)].copy_(features)
        self.utility_buffer[stage, :len(utilities)].copy_(utilities)
        self.embedding_count[stage].fill_(len(features))

    def _utility_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        active = ((self.active_stage + 1) if self.active_stage is not None
                  else int((self.embedding_count > 0).sum().item()))
        features, utilities = [], []
        for stage in range(active):
            count = int(self.embedding_count[stage].item())
            if count:
                features.append(self.embedding_buffer[stage, :count].float())
                utilities.append(self.utility_buffer[stage, :count, :active].float())
        if not features:
            return (self.embedding_buffer.new_zeros((0, self.visual_dim)).float(),
                    self.utility_buffer.new_zeros((0, active)).float())
        return torch.cat(features), torch.cat(utilities)

    def fit_selector(self, *, epochs: int = 25, lr: float = 0.01,
                     batch_size: int = 24, seed: int = 2412) -> dict[str, float | int]:
        """Fit by joint-utility preference rather than stream-stage labels."""
        features, utilities = self._utility_dataset()
        active = int((self.embedding_count > 0).sum().item())
        valid_rows = torch.isfinite(utilities).any(dim=1) if len(utilities) else None
        if valid_rows is not None:
            features, utilities = features[valid_rows], utilities[valid_rows]
        if active <= 1 or len(features) < 2:
            self.selector_fit_loss.fill_(0.0)
            self.selector_fit_accuracy.fill_(1.0 if len(features) else float("nan"))
            return {"loss": float(self.selector_fit_loss),
                    "accuracy": float(self.selector_fit_accuracy),
                    "samples": len(features), "epochs": 0}
        labels = utilities.masked_fill(~torch.isfinite(utilities), torch.inf).argmin(dim=1)
        utility_rows = torch.isfinite(utilities).sum(dim=1) > 1
        self.selector.train()
        optimizer = torch.optim.SGD(self.selector.parameters(), lr=lr)
        generator = torch.Generator(device="cpu").manual_seed(seed + active)
        last_loss = 0.0
        for _ in range(epochs):
            order = torch.randperm(len(features), generator=generator).to(features.device)
            epoch_loss, batches = 0.0, 0
            for start in range(0, len(order), batch_size):
                index = order[start:start + batch_size]
                if len(index) < 2:
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = self.selector(features[index])[:, :active]
                target = utilities[index]
                loss = F.cross_entropy(logits, labels[index])
                # Pairwise preference preserves more counterfactual information
                # than the best-skill label alone. No loss-scale temperature is
                # needed because utilities are normalized ranks in [0, 1].
                left = logits.unsqueeze(2) - logits.unsqueeze(1)
                target_difference = target.unsqueeze(2) - target.unsqueeze(1)
                pair_valid = (torch.isfinite(target_difference) &
                              (target_difference.abs() > 1e-6))
                preferred = -target_difference.sign()
                if bool(pair_valid.any()):
                    pairwise = F.softplus(-preferred * left)[pair_valid].mean()
                    loss = loss + 0.5 * pairwise
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
                batches += 1
            last_loss = epoch_loss / max(batches, 1)
        self.selector.eval()
        with torch.no_grad():
            predicted = self.selector(features)[:, :active].argmax(dim=-1)
            accuracy = float((predicted == labels).float().mean())
            self.selector_fit_loss.fill_(last_loss)
            self.selector_fit_accuracy.fill_(accuracy)
        return {"loss": last_loss, "accuracy": accuracy,
                "samples": len(features), "epochs": epochs,
                "ownership_samples": int((~utility_rows).sum().item()),
                "utility_samples": int(utility_rows.sum().item())}

    def diagnostics(self) -> dict[str, object]:
        report = super().diagnostics()
        report.update(
            selector="visual_mlp_leaf_plus_object_consistent_skill_path",
            skill_parent=[int(x) for x in self.skill_parent.tolist()],
            skill_parent_score=[float(x) for x in self.skill_parent_score.tolist()],
            skill_base_score=[float(x) for x in self.skill_base_score.tolist()],
            skill_composition="unweighted_residual_parent_path",
            selector_supervision="anchor_ownership_ce_plus_counterfactual_utility_ranking",
            selector_uses_stage_labels="anchor_ownership_only",
            selector_uses_task_id_at_inference=False,
            selector_training_examples="dense_anchor_ownership_plus_utility_hard_negatives",
        )
        return report


class AnchorTextObserver(nn.Module):
    """Observe SAM 3's native pooled prompt without changing its output."""

    def __init__(self, base: nn.Linear, router: PromptMLPTaskRouter):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        object.__setattr__(self, "_anchor_router", router)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._anchor_router.prepare_text(x, training=self.training)
        return self.base(x)


class MLPTaskDecoupledLoRALinear(CapacityPartitionFocusLoRALinear):
    """Function-specific LoRA blocks selected by one shared visual MLP."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5,
                 *, function_group: str, orthogonality_weight: float = 1e-3):
        super().__init__(base, rank, alpha, num_tasks,
                         function_group=function_group, top_k=rank,
                         orthogonality_weight=orthogonality_weight)
        self._anchor_router: PromptMLPTaskRouter | None = None
        self._is_anchor_source = False

    def attach_anchor_router(self, router: PromptMLPTaskRouter, *, source: bool) -> None:
        if source:
            self.anchor_router = router
        object.__setattr__(self, "_anchor_router", router)
        self._is_anchor_source = bool(source)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router = self._anchor_router
        if router is None:
            raise RuntimeError("MLP task-decoupled adapter has no selector")
        if self._is_anchor_source and router._visual_route is None:
            router.prepare_visual(x, training=self.training)
        valid = self.allocated_mask
        if not bool(valid.any()):
            return self.base(x)
        mask = valid.to(device=x.device, dtype=x.dtype)
        a = self.lora_A.to(dtype=x.dtype) * mask[:, None]
        b = self.lora_B.to(dtype=x.dtype) * mask[None, :]
        activations = F.linear(x, a)
        stage_route = router.route_for(activations)
        atom_route = activations.new_zeros(*stage_route.shape[:-1], self.rank)
        for stage in range(self.num_tasks):
            slots = self._stage_slots(stage).to(activations.device)
            atom_route[..., slots] = stage_route[..., stage].unsqueeze(-1)
        return self.base(x) + F.linear(activations * atom_route, b) * self.scaling


class IndependentCapacityTaskLoRALinear(ContinualAdapterLinear):
    """A full, disjoint LoRA module for every stream stage and function.

    Unlike the fixed rank pool, ``rank`` is the capacity of *each* task module.
    A single image-level MLP route is nevertheless shared by visual, semantic,
    box, and mask functions, so the extra capacity cannot make detection and
    segmentation silently choose different tasks. Historical modules are
    immutable; a new module is initialized outside their input subspace and may
    compose frozen predecessors through the shared compositional router.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5,
                 *, function_group: str, orthogonality_weight: float = 1e-3):
        super().__init__(base, rank, alpha, num_tasks)
        if function_group not in FOCUS_TARGET_GROUPS:
            raise ValueError(f"Unknown independent-capacity group: {function_group}")
        self.function_group = function_group
        self.orthogonality_weight = float(orthogonality_weight)
        # Unit residual scaling matches the SAMCL SLoRA capacity control. Rank
        # changes therefore affect representation width, not output magnitude.
        self.scaling = 1.0
        self.lora_A = nn.Parameter(torch.empty(num_tasks, rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_tasks, self.out_features, rank))
        self.register_buffer("allocated_mask", torch.zeros(num_tasks, dtype=torch.bool))
        self.register_buffer("active_mask", torch.zeros(num_tasks, dtype=torch.bool),
                             persistent=False)
        self._anchor_router: PromptMLPTaskRouter | None = None
        self._is_anchor_source = False
        self._protected_A: torch.Tensor | None = None
        self._protected_B: torch.Tensor | None = None
        self._protected_basis: torch.Tensor | None = None
        self.lora_A.register_hook(
            lambda gradient: gradient * self.active_mask[:, None, None].to(gradient))
        self.lora_B.register_hook(
            lambda gradient: gradient * self.active_mask[:, None, None].to(gradient))
        with torch.no_grad():
            for stage in range(num_tasks):
                nn.init.kaiming_uniform_(self.lora_A[stage], a=math.sqrt(5))

    def attach_anchor_router(self, router: PromptMLPTaskRouter, *, source: bool) -> None:
        if source:
            self.anchor_router = router
        object.__setattr__(self, "_anchor_router", router)
        self._is_anchor_source = bool(source)

    @torch.no_grad()
    def begin_task(self, task_id: int) -> None:
        super().begin_task(task_id)
        stage = int(task_id)
        if not 0 <= stage < self.num_tasks:
            raise ValueError(f"stage {stage} is outside [0, {self.num_tasks})")
        if bool(self.allocated_mask[stage]):
            raise RuntimeError("an independent task module cannot be allocated twice")
        self.active_mask.zero_()
        self.active_mask[stage] = True
        protected = self.lora_A[self.allocated_mask].float().reshape(-1, self.in_features)
        self._protected_basis = (torch.linalg.qr(protected.T, mode="reduced").Q.T
                                 if len(protected) else None)
        candidate = torch.randn(
            self.rank, self.in_features, device=self.lora_A.device, dtype=torch.float32)
        if self._protected_basis is not None:
            candidate = candidate - ((candidate @ self._protected_basis.T) @
                                     self._protected_basis)
        norms = candidate.norm(dim=1, keepdim=True)
        if bool((norms < 1e-6).any()):
            candidate = torch.where(norms < 1e-6, torch.randn_like(candidate), candidate)
        self.lora_A[stage] = F.normalize(candidate, dim=1).to(self.lora_A)
        self.lora_B[stage].zero_()
        self.allocated_mask[stage] = True
        self._protected_A = self.lora_A.detach().clone()
        self._protected_B = self.lora_B.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router = self._anchor_router
        if router is None:
            raise RuntimeError("independent task adapter has no shared selector")
        if self._is_anchor_source and router._visual_route is None:
            router.prepare_visual(x, training=self.training)
        route = router.route_for(x, group=self.function_group)
        base = self.base(x)
        update = torch.zeros_like(base)
        for stage in range(self.num_tasks):
            if not bool(self.allocated_mask[stage]):
                continue
            delta = F.linear(
                F.linear(x, self.lora_A[stage].to(dtype=x.dtype)),
                self.lora_B[stage].to(dtype=x.dtype),
            )
            update = update + delta * route[..., stage].unsqueeze(-1)
        return base + update * self.scaling

    def regularization_loss(self) -> torch.Tensor:
        if self._protected_basis is None or not bool(self.active_mask.any()):
            return self.base.weight.new_zeros(())
        active = self.lora_A[self.active_mask].float().reshape(-1, self.in_features)
        overlap = F.normalize(active, dim=1) @ self._protected_basis.T
        return (overlap.square().mean() * self.orthogonality_weight).to(
            self.base.weight.dtype)

    @torch.no_grad()
    def restore_protected_parameters(self) -> None:
        if self._protected_A is None or self._protected_B is None:
            return
        protected = ~self.active_mask
        self.lora_A[protected] = self._protected_A[protected]
        self.lora_B[protected] = self._protected_B[protected]

    @torch.no_grad()
    def reset_active_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        mask = self.active_mask[:, None, None]
        for parameter in (self.lora_A, self.lora_B):
            for value in optimizer.state.get(parameter, {}).values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.masked_fill_(mask.expand_as(value), 0)

    def restore_trainability(self) -> None:
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
        self._protected_A = None
        self._protected_B = None
        self._protected_basis = None

    @torch.no_grad()
    def end_task(self) -> dict[str, float | int | str]:
        allocated_tasks = int(self.allocated_mask.sum().item())
        report = {
            "group": self.function_group,
            "rank": self.rank * self.num_tasks,
            "rank_per_task": self.rank,
            "allocated_rank": self.rank * allocated_tasks,
            "active_rank": self.rank,
            "remaining_rank": self.rank * (self.num_tasks - allocated_tasks),
            "recycled_rank": 0,
            "relative_compression_error": 0.0,
        }
        self._protected_A = None
        self._protected_B = None
        self._protected_basis = None
        return report


class TaskObjectCore(nn.Module):
    """Task-private object subspaces shared by semantic, box, and mask heads.

    A stage owns one small down projection, but every prediction function reads
    that same projection.  The bank is registered exactly once on the visual
    routing source; prediction adapters only keep a non-registering reference.
    This preserves independent historical capacity while making the three
    prediction functions agree on what an object is.
    """

    def __init__(self, in_features: int, rank: int, num_tasks: int,
                 orthogonality_weight: float = 1e-3):
        super().__init__()
        if rank <= 0:
            raise ValueError("object-core rank must be positive")
        self.in_features = int(in_features)
        self.rank = int(rank)
        self.num_tasks = int(num_tasks)
        self.orthogonality_weight = float(orthogonality_weight)
        self.lora_A = nn.Parameter(torch.empty(num_tasks, rank, in_features))
        self.register_buffer("allocated_mask", torch.zeros(num_tasks, dtype=torch.bool))
        self.register_buffer("active_mask", torch.zeros(num_tasks, dtype=torch.bool),
                             persistent=False)
        self._protected_A: torch.Tensor | None = None
        self._protected_basis: torch.Tensor | None = None
        self.lora_A.register_hook(
            lambda gradient: gradient * self.active_mask[:, None, None].to(gradient))
        with torch.no_grad():
            for stage in range(num_tasks):
                nn.init.kaiming_uniform_(self.lora_A[stage], a=math.sqrt(5))

    @torch.no_grad()
    def begin_task(self, task_id: int) -> None:
        stage = int(task_id)
        if not 0 <= stage < self.num_tasks:
            raise ValueError(f"stage {stage} is outside [0, {self.num_tasks})")
        if bool(self.allocated_mask[stage]):
            raise RuntimeError("an object-core task subspace cannot be allocated twice")
        self.active_mask.zero_()
        self.active_mask[stage] = True
        protected = self.lora_A[self.allocated_mask].float().reshape(
            -1, self.in_features)
        self._protected_basis = (torch.linalg.qr(protected.T, mode="reduced").Q.T
                                 if len(protected) else None)
        candidate = torch.randn(
            self.rank, self.in_features, device=self.lora_A.device,
            dtype=torch.float32)
        if self._protected_basis is not None:
            candidate = candidate - ((candidate @ self._protected_basis.T) @
                                     self._protected_basis)
        norms = candidate.norm(dim=1, keepdim=True)
        if bool((norms < 1e-6).any()):
            candidate = torch.where(
                norms < 1e-6, torch.randn_like(candidate), candidate)
        self.lora_A[stage] = F.normalize(candidate, dim=1).to(self.lora_A)
        self.allocated_mask[stage] = True
        self._protected_A = self.lora_A.detach().clone()

    def project(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        return F.linear(x, self.lora_A[stage].to(dtype=x.dtype))

    def regularization_loss(self) -> torch.Tensor:
        if self._protected_basis is None or not bool(self.active_mask.any()):
            return self.lora_A.new_zeros(())
        active = self.lora_A[self.active_mask].float().reshape(-1, self.in_features)
        overlap = F.normalize(active, dim=1) @ self._protected_basis.T
        return (overlap.square().mean() * self.orthogonality_weight).to(self.lora_A)

    @torch.no_grad()
    def restore_protected_parameters(self) -> None:
        if self._protected_A is not None:
            self.lora_A[~self.active_mask] = self._protected_A[~self.active_mask]

    @torch.no_grad()
    def reset_active_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        mask = self.active_mask[:, None, None]
        for value in optimizer.state.get(self.lora_A, {}).values():
            if torch.is_tensor(value) and value.shape == self.lora_A.shape:
                value.masked_fill_(mask.expand_as(value), 0)

    def restore_trainability(self) -> None:
        self.lora_A.requires_grad_(True)
        self._protected_A = None
        self._protected_basis = None

    def end_task(self) -> None:
        self._protected_A = None
        self._protected_basis = None


class ObjectCoupledTaskLoRALinear(IndependentCapacityTaskLoRALinear):
    """Independent task module with an inference-path shared object core.

    Private factors retain function-specific capacity.  Selected Query-facing
    layers additionally decode the same task-specific ``TaskObjectCore`` through
    their own up projections.  Thus semantic scoring, box regression, and mask
    prediction share an object representation without sharing a dataset ID.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float,
                 num_tasks: int = 5, *, function_group: str,
                 effective_rank: int | None = None,
                 object_core: TaskObjectCore | None = None,
                 uses_object_core: bool = False,
                 orthogonality_weight: float = 1e-3):
        super().__init__(base, rank, alpha, num_tasks,
                         function_group=function_group,
                         orthogonality_weight=orthogonality_weight)
        self.effective_rank = int(effective_rank or rank)
        self.uses_object_core = bool(uses_object_core)
        self._object_core: TaskObjectCore | None = None
        self._is_object_core_source = False
        if self.uses_object_core:
            if object_core is None or self.in_features != object_core.in_features:
                raise ValueError("object-core adapter has an incompatible input width")
            self.core_B = nn.Parameter(torch.zeros(
                num_tasks, self.out_features, object_core.rank))
            self.core_B.register_hook(
                lambda gradient: gradient * self.active_mask[:, None, None].to(gradient))
            self._protected_core_B: torch.Tensor | None = None
        else:
            self.register_parameter("core_B", None)
            self._protected_core_B = None
        object.__setattr__(self, "_object_core", object_core)

    def attach_object_core(self, core: TaskObjectCore, *, source: bool) -> None:
        if source:
            self.object_core = core
        object.__setattr__(self, "_object_core", core)
        self._is_object_core_source = bool(source)

    @torch.no_grad()
    def begin_task(self, task_id: int) -> None:
        super().begin_task(task_id)
        if self._is_object_core_source:
            if self._object_core is None:
                raise RuntimeError("object-core source has no core bank")
            self._object_core.begin_task(task_id)
        if self.uses_object_core:
            self.core_B[int(task_id)].zero_()
            self._protected_core_B = self.core_B.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        if not self.uses_object_core:
            return output
        core = self._object_core
        router = self._anchor_router
        if core is None or router is None:
            raise RuntimeError("object-coupled adapter has incomplete shared state")
        route = router.route_for(x, group=self.function_group)
        update = torch.zeros_like(output)
        for stage in range(self.num_tasks):
            if not bool(core.allocated_mask[stage]):
                continue
            delta = F.linear(
                core.project(x, stage), self.core_B[stage].to(dtype=x.dtype))
            update = update + delta * route[..., stage].unsqueeze(-1)
        return output + update

    def regularization_loss(self) -> torch.Tensor:
        loss = super().regularization_loss()
        if self._is_object_core_source and self._object_core is not None:
            loss = loss + self._object_core.regularization_loss().to(loss)
        return loss

    @torch.no_grad()
    def restore_protected_parameters(self) -> None:
        super().restore_protected_parameters()
        if self._is_object_core_source and self._object_core is not None:
            self._object_core.restore_protected_parameters()
        if self.uses_object_core and self._protected_core_B is not None:
            self.core_B[~self.active_mask] = self._protected_core_B[~self.active_mask]

    @torch.no_grad()
    def reset_active_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        super().reset_active_optimizer_state(optimizer)
        if self._is_object_core_source and self._object_core is not None:
            self._object_core.reset_active_optimizer_state(optimizer)
        if self.uses_object_core:
            mask = self.active_mask[:, None, None]
            for value in optimizer.state.get(self.core_B, {}).values():
                if torch.is_tensor(value) and value.shape == self.core_B.shape:
                    value.masked_fill_(mask.expand_as(value), 0)

    def restore_trainability(self) -> None:
        super().restore_trainability()
        if self._is_object_core_source and self._object_core is not None:
            self._object_core.restore_trainability()
        if self.uses_object_core:
            self.core_B.requires_grad_(True)
        self._protected_core_B = None

    @torch.no_grad()
    def end_task(self) -> dict[str, float | int | str]:
        report = super().end_task()
        report["rank"] = self.effective_rank * self.num_tasks
        report["rank_per_task"] = self.effective_rank
        report["object_core_rank"] = (
            self._object_core.rank if self.uses_object_core and self._object_core else 0)
        if self._is_object_core_source and self._object_core is not None:
            self._object_core.end_task()
        self._protected_core_B = None
        return report


class EnergyDecomposedObjectCore(TaskObjectCore):
    """Fixed object projections decomposed into general and isolated energy.

    The first rank units are task-general directions extracted from accumulated
    Query covariance.  They are always active.  Remaining units maximize the
    current-to-history energy ratio and are consumed only by the image-selected
    task module.  Only covariance matrices are retained; no image or Query
    samples are stored.
    """

    def __init__(self, in_features: int, rank: int, num_tasks: int, *,
                 general_rank: int = 1, calibration_lambda: float = 3.0,
                 random_directions: bool = False):
        if not 0 <= general_rank <= rank:
            raise ValueError("general_rank must be in [0, object-core rank]")
        super().__init__(in_features, rank, num_tasks, orthogonality_weight=0.0)
        self.general_rank = int(general_rank)
        self.isolated_rank = int(rank - general_rank)
        self.calibration_lambda = float(calibration_lambda)
        self.random_directions = bool(random_directions)
        self.lora_A.requires_grad_(False)
        self.register_buffer(
            "historical_covariance",
            torch.zeros(in_features, in_features, dtype=torch.float32),
        )
        self.register_buffer(
            "current_covariance",
            torch.zeros(in_features, in_features, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("general_scale", torch.ones(num_tasks, general_rank))
        self.register_buffer("subspace_ready", torch.zeros(num_tasks, dtype=torch.bool))
        self.register_buffer(
            "collection_covariance",
            torch.zeros(in_features, in_features, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "collection_count", torch.zeros((), dtype=torch.float64), persistent=False)
        self.collecting = False

    @torch.no_grad()
    def begin_task(self, task_id: int) -> None:
        stage = int(task_id)
        if not 0 <= stage < self.num_tasks:
            raise ValueError(f"stage {stage} is outside [0, {self.num_tasks})")
        if bool(self.allocated_mask[stage]):
            raise RuntimeError("an energy object subspace cannot be allocated twice")
        self.active_mask.zero_()
        self.active_mask[stage] = True
        self.allocated_mask[stage] = True
        self.subspace_ready[stage] = False
        self.general_scale[stage].fill_(1.0)
        self._protected_A = self.lora_A.detach().clone()
        self._protected_basis = None
        self.stop_collection()

    @torch.no_grad()
    def start_collection(self) -> None:
        self.collection_covariance.zero_()
        self.collection_count.zero_()
        self.collecting = True

    @torch.no_grad()
    def stop_collection(self) -> None:
        self.collecting = False

    @torch.no_grad()
    def observe(self, x: torch.Tensor, *, group: str | None = None) -> None:
        del group
        if not self.collecting or x.shape[-1] != self.in_features:
            return
        values = x.detach().reshape(-1, self.in_features).float()
        if not len(values):
            return
        # Average inside each consumer call before accumulating. This keeps the
        # six decoder layers from dominating the semantic/box/mask consumers
        # merely because their token counts differ.
        covariance = values.T @ values
        covariance.div_(float(len(values)))
        self.collection_covariance.add_(covariance)
        self.collection_count.add_(1.0)

    @staticmethod
    def _normalized_covariance(covariance: torch.Tensor) -> torch.Tensor:
        covariance = (covariance + covariance.T) * 0.5
        trace = covariance.trace().clamp_min(1e-8)
        return covariance * (covariance.shape[0] / trace)

    @staticmethod
    def _orthogonalize(candidate: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        if basis.numel():
            candidate = candidate - basis @ (basis.T @ candidate)
        return torch.linalg.qr(candidate, mode="reduced").Q

    @torch.no_grad()
    def finalize_subspaces(self, task_id: int) -> dict[str, float | int]:
        stage = int(task_id)
        if not bool(self.active_mask[stage]):
            raise RuntimeError("cannot finalize a non-active object subspace")
        self.stop_collection()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.collection_covariance)
            torch.distributed.all_reduce(self.collection_count)
        if float(self.collection_count.item()) <= 0:
            raise RuntimeError("no object Query activations were collected")
        current = self._normalized_covariance(
            self.collection_covariance / self.collection_count.float())
        history = self.historical_covariance

        if self.random_directions:
            generator = torch.Generator(device=current.device)
            generator.manual_seed(42 + 1009 * stage)
            if self.general_rank > 0:
                noise = torch.randn(
                    self.in_features, self.general_rank, generator=generator,
                    device=current.device, dtype=current.dtype)
                general = torch.linalg.qr(noise, mode="reduced").Q
                general_energy = float(torch.diagonal(
                    general.T @ current @ general).mean().item())
            else:
                general = current.new_zeros(self.in_features, 0)
                general_energy = 0.0
            isolated = current.new_zeros(self.in_features, 0)
            isolated_energy_ratio = 0.0
            directions = torch.cat((general, isolated), dim=1).T
            self.lora_A[stage].copy_(F.normalize(directions, dim=1).to(self.lora_A))
            self.current_covariance.copy_(current)
            self.subspace_ready[stage] = True
            return {
                "collection_consumer_calls": int(self.collection_count.item()),
                "current_covariance_trace": float(current.trace().item()),
                "general_energy": general_energy,
                "isolated_energy_ratio": isolated_energy_ratio,
                "random_directions": 1,
            }

        if self.general_rank > 0:
            shared_eigenvalues, shared_eigenvectors = torch.linalg.eigh(
                history + current)
            general = shared_eigenvectors[:, -self.general_rank:]
            general_energy = float(shared_eigenvalues[-self.general_rank:].mean().item())
        else:
            general = current.new_zeros(self.in_features, 0)
            general_energy = 0.0

        if self.isolated_rank <= 0:
            isolated = current.new_zeros(self.in_features, 0)
            isolated_energy_ratio = 0.0
        elif stage == 0 or float(history.trace().item()) <= 1e-8:
            # The first task has no historical contrast. Use the strongest
            # complementary current directions as its isolated residual.
            current_eigenvalues, current_eigenvectors = torch.linalg.eigh(current)
            take = self.general_rank + self.isolated_rank
            candidates = current_eigenvectors[:, -take:][:, :self.isolated_rank]
            isolated = self._orthogonalize(candidates, general)
            ratio_slice = (current_eigenvalues[-take:-self.general_rank]
                           if self.general_rank else current_eigenvalues[-self.isolated_rank:])
            isolated_energy_ratio = float(ratio_slice.mean().item())
        else:
            ridge = max(float(history.trace().item()) / self.in_features * 1e-4, 1e-5)
            history_regularized = history + torch.eye(
                self.in_features, device=history.device, dtype=history.dtype) * ridge
            hist_values, hist_vectors = torch.linalg.eigh(history_regularized)
            inverse_sqrt = ((hist_vectors * hist_values.clamp_min(ridge).rsqrt()) @
                            hist_vectors.T)
            contrast = inverse_sqrt @ current @ inverse_sqrt
            ratio_values, ratio_vectors = torch.linalg.eigh(
                (contrast + contrast.T) * 0.5)
            candidates = inverse_sqrt @ ratio_vectors[:, -self.isolated_rank:]
            isolated = self._orthogonalize(candidates, general)
            isolated_energy_ratio = float(ratio_values[-self.isolated_rank:].mean().item())
        directions = torch.cat((general, isolated), dim=1).T
        self.lora_A[stage].copy_(F.normalize(directions, dim=1).to(self.lora_A))
        self.current_covariance.copy_(current)
        self.subspace_ready[stage] = True
        return {
            "collection_consumer_calls": int(self.collection_count.item()),
            "current_covariance_trace": float(current.trace().item()),
            "general_energy": general_energy,
            "isolated_energy_ratio": isolated_energy_ratio,
        }

    def regularization_loss(self) -> torch.Tensor:
        return self.lora_A.sum() * 0.0

    def restore_trainability(self) -> None:
        # Data-derived down projections stay fixed. Only their function-specific
        # up projections are optimized by the joint prediction objectives.
        self.lora_A.requires_grad_(False)
        self._protected_A = None
        self._protected_basis = None

    @torch.no_grad()
    def reset_active_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.state.pop(self.lora_A, None)

    @torch.no_grad()
    def end_task(self) -> None:
        active = torch.where(self.active_mask)[0]
        if len(active) == 1:
            stage = int(active.item())
            if bool(self.subspace_ready[stage]):
                current = self.current_covariance
                if self.general_rank > 0:
                    directions = self.lora_A[stage, :self.general_rank].float().T
                    current_energy = torch.diagonal(directions.T @ current @ directions)
                    old_energy = torch.diagonal(
                        directions.T @ self.historical_covariance @ directions)
                    numerator = self.calibration_lambda * current_energy
                    self.general_scale[stage].copy_(
                        numerator / (numerator + old_energy).clamp_min(1e-8))
                self.historical_covariance.add_(current)
                self.current_covariance.zero_()
        self.stop_collection()
        self._protected_A = None
        self._protected_basis = None


class SemanticProtectedObjectCore(EnergyDecomposedObjectCore):
    """Multi-head object subspace whose shared directions preserve semantics.

    Semantic Query, box, and mask covariances are normalized independently.
    Their matrix harmonic mean defines directions supported by every head,
    while the frozen prompt projection is pulled back through SAM 3's frozen
    Query projection to prevent the shared object basis from discarding open-
    vocabulary structure. No examples, tokens, or per-instance features remain
    after the task prepass.
    """

    GROUP_INDEX = {"semantic": 0, "box": 1, "mask": 2}

    def __init__(self, in_features: int, rank: int, num_tasks: int, *,
                 general_rank: int = 2, calibration_lambda: float = 3.0):
        super().__init__(in_features, rank, num_tasks,
                         general_rank=general_rank,
                         calibration_lambda=calibration_lambda)
        self.register_buffer(
            "group_collection_covariance",
            torch.zeros(3, in_features, in_features, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "group_collection_count", torch.zeros(3, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "text_collection_covariance",
            torch.zeros(in_features, in_features, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "text_collection_count", torch.zeros((), dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "historical_specific_covariance",
            torch.zeros(in_features, in_features, dtype=torch.float32),
        )
        self.register_buffer(
            "current_specific_covariance",
            torch.zeros(in_features, in_features, dtype=torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def start_collection(self) -> None:
        super().start_collection()
        self.group_collection_covariance.zero_()
        self.group_collection_count.zero_()
        self.text_collection_covariance.zero_()
        self.text_collection_count.zero_()

    @torch.no_grad()
    def observe(self, x: torch.Tensor, *, group: str | None = None) -> None:
        super().observe(x, group=group)
        index = self.GROUP_INDEX.get(str(group))
        if not self.collecting or index is None or x.shape[-1] != self.in_features:
            return
        values = x.detach().reshape(-1, self.in_features).float()
        if not len(values):
            return
        self.group_collection_covariance[index].add_((values.T @ values) / len(values))
        self.group_collection_count[index].add_(1.0)

    @torch.no_grad()
    def observe_semantic_text(self, projected_text: torch.Tensor,
                              query_projection_weight: torch.Tensor) -> None:
        if not self.collecting or projected_text.shape[-1] != self.in_features:
            return
        text = projected_text.detach().reshape(-1, self.in_features).float()
        if not len(text):
            return
        text = F.normalize(text, dim=-1)
        text_covariance = (text.T @ text) / len(text)
        weight = query_projection_weight.detach().float()
        pulled_back = weight.T @ text_covariance @ weight
        self.text_collection_covariance.add_(pulled_back)
        self.text_collection_count.add_(1.0)

    @staticmethod
    def _harmonic_consensus(covariances: list[torch.Tensor]) -> torch.Tensor:
        width = covariances[0].shape[0]
        identity = torch.eye(width, device=covariances[0].device,
                             dtype=covariances[0].dtype)
        # Trace normalization makes 1e-3 a numerical ridge, not a tunable loss
        # coefficient. The harmonic mean rejects directions absent in any head.
        inverse_sum = sum(torch.linalg.inv(value + identity * 1e-3)
                          for value in covariances)
        consensus = len(covariances) * torch.linalg.inv(inverse_sum)
        return SemanticProtectedObjectCore._normalized_covariance(consensus)

    @torch.no_grad()
    def finalize_subspaces(self, task_id: int) -> dict[str, float | int]:
        stage = int(task_id)
        if not bool(self.active_mask[stage]):
            raise RuntimeError("cannot finalize a non-active object subspace")
        self.stop_collection()
        tensors = [self.collection_covariance, self.collection_count,
                   self.group_collection_covariance, self.group_collection_count,
                   self.text_collection_covariance, self.text_collection_count]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for value in tensors:
                torch.distributed.all_reduce(value)
        if bool((self.group_collection_count <= 0).any()):
            raise RuntimeError(
                f"missing multi-head covariance: {self.group_collection_count.tolist()}")

        heads = [self._normalized_covariance(
            self.group_collection_covariance[index] /
            self.group_collection_count[index].float()) for index in range(3)]
        text_calls = float(self.text_collection_count.item())
        if text_calls > 0:
            text_anchor = self._normalized_covariance(
                self.text_collection_covariance / self.text_collection_count.float())
            heads[0] = self._normalized_covariance((heads[0] + text_anchor) * 0.5)
        consensus = self._harmonic_consensus(heads)
        specific = self._normalized_covariance(sum(heads) / len(heads))

        shared_values, shared_vectors = torch.linalg.eigh(
            self.historical_covariance + consensus)
        general = shared_vectors[:, -self.general_rank:]
        history = self.historical_specific_covariance
        if stage == 0 or float(history.trace().item()) <= 1e-8:
            specific_values, specific_vectors = torch.linalg.eigh(specific)
            candidates = specific_vectors[:, -(
                self.general_rank + self.isolated_rank):,]
            candidates = candidates[:, :self.isolated_rank]
            ratio_values = specific_values[-(
                self.general_rank + self.isolated_rank):-self.general_rank]
        else:
            ridge = max(float(history.trace().item()) / self.in_features * 1e-4, 1e-5)
            regularized = history + torch.eye(
                self.in_features, device=history.device, dtype=history.dtype) * ridge
            history_values, history_vectors = torch.linalg.eigh(regularized)
            inverse_sqrt = ((history_vectors *
                             history_values.clamp_min(ridge).rsqrt()) @
                            history_vectors.T)
            contrast = inverse_sqrt @ specific @ inverse_sqrt
            ratio_values, ratio_vectors = torch.linalg.eigh(
                (contrast + contrast.T) * 0.5)
            candidates = inverse_sqrt @ ratio_vectors[:, -self.isolated_rank:]
            ratio_values = ratio_values[-self.isolated_rank:]
        isolated = self._orthogonalize(candidates, general)
        directions = torch.cat((general, isolated), dim=1).T
        self.lora_A[stage].copy_(F.normalize(directions, dim=1).to(self.lora_A))
        self.current_covariance.copy_(consensus)
        self.current_specific_covariance.copy_(specific)
        self.subspace_ready[stage] = True
        head_names = ("semantic", "box", "mask")
        report: dict[str, float | int] = {
            "collection_consumer_calls": int(self.collection_count.item()),
            "semantic_text_calls": int(self.text_collection_count.item()),
            "consensus_covariance_trace": float(consensus.trace().item()),
            "general_energy": float(shared_values[-self.general_rank:].mean().item()),
            "isolated_energy_ratio": float(ratio_values.mean().item()),
        }
        for name, value in zip(head_names, heads):
            report[f"{name}_covariance_trace"] = float(value.trace().item())
        return report

    @torch.no_grad()
    def end_task(self) -> None:
        active = torch.where(self.active_mask)[0]
        if len(active) == 1:
            stage = int(active.item())
            if bool(self.subspace_ready[stage]):
                directions = self.lora_A[stage, :self.general_rank].float().T
                current_energy = torch.diagonal(
                    directions.T @ self.current_covariance @ directions)
                old_energy = torch.diagonal(
                    directions.T @ self.historical_covariance @ directions)
                numerator = self.calibration_lambda * current_energy
                self.general_scale[stage].copy_(
                    numerator / (numerator + old_energy).clamp_min(1e-8))
                self.historical_covariance.add_(self.current_covariance)
                self.historical_specific_covariance.add_(
                    self.current_specific_covariance)
                self.current_covariance.zero_()
                self.current_specific_covariance.zero_()
        self.stop_collection()
        self._protected_A = None
        self._protected_basis = None


class EnergyObjectSubspaceLoRALinear(ObjectCoupledTaskLoRALinear):
    """Object-coupled task module with route-free general and routed isolation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve the function-private capacity and its single hard route.
        output = IndependentCapacityTaskLoRALinear.forward(self, x)
        if not self.uses_object_core:
            return output
        core = self._object_core
        router = self._anchor_router
        if not isinstance(core, EnergyDecomposedObjectCore) or router is None:
            raise RuntimeError("energy object adapter has incomplete shared state")
        core.observe(x, group=self.function_group)
        route = router.route_for(x, group=self.function_group)
        update = torch.zeros_like(output)
        for stage in range(self.num_tasks):
            if not bool(core.subspace_ready[stage]):
                continue
            hidden = core.project(x, stage)
            if core.general_rank > 0:
                general_hidden = hidden[..., :core.general_rank]
                scale_shape = [1] * (general_hidden.ndim - 1) + [core.general_rank]
                scaled_hidden = general_hidden * core.general_scale[stage].to(
                    device=x.device, dtype=x.dtype).reshape(scale_shape)
                update = update + F.linear(
                    scaled_hidden,
                    self.core_B[stage, :, :core.general_rank].to(dtype=x.dtype),
                )
            if core.isolated_rank > 0:
                isolated_hidden = hidden[..., core.general_rank:]
                isolated = F.linear(
                    isolated_hidden,
                    self.core_B[stage, :, core.general_rank:].to(dtype=x.dtype),
                )
                update = update + isolated * route[..., stage].unsqueeze(-1)
        return output + update

    @torch.no_grad()
    def end_task(self) -> dict[str, float | int | str]:
        report = super().end_task()
        core = self._object_core
        if isinstance(core, EnergyDecomposedObjectCore):
            report["general_object_rank"] = core.general_rank
            report["isolated_object_rank"] = core.isolated_rank
            report["object_subspace_ready"] = int(core.subspace_ready.sum().item())
        return report


class CovarianceQuerySubspaceMemory(EnergyDecomposedObjectCore):
    """Append-only CQSM, optionally restoring legacy energy calibration.

    Statistics are averaged per consumer invocation, then trace-normalized.
    History is a sum of normalized task moments, not a moving average.  The
    The default release keeps scales at one. The explicitly restored preset
    calibrates each new entry once at task completion using training moments.
    """

    def __init__(self, in_features: int, rank: int, num_tasks: int,
                 basis: str = "covariance", *, energy_calibration: bool = False,
                 calibration_lambda: float = 3.0, allow_experimental_rank: bool = False):
        if rank < 1 or in_features < rank or (rank != 2 and not allow_experimental_rank):
            raise ValueError("the preregistered CQSM basis has rank 2")
        if basis not in {"covariance", "random"}:
            raise ValueError("CQSM basis must be covariance or random")
        self.basis = basis
        self.energy_calibration = bool(energy_calibration)
        if not math.isfinite(calibration_lambda) or calibration_lambda <= 0:
            raise ValueError("calibration_lambda must be finite and positive")
        super().__init__(in_features, rank, num_tasks, general_rank=rank,
                         random_directions=basis == "random",
                         calibration_lambda=calibration_lambda)
        self.register_buffer("history_committed", torch.zeros(num_tasks, dtype=torch.bool))

    @torch.no_grad()
    def end_task(self) -> None:
        active = torch.where(self.active_mask)[0]
        if len(active) != 1:
            raise RuntimeError("CQSM consolidation requires one current task")
        stage = int(active.item())
        if not bool(self.subspace_ready[stage]):
            raise RuntimeError("CQSM basis must be estimated before consolidation")
        if bool(self.history_committed[stage]):
            raise RuntimeError("CQSM statistics cannot be committed twice")
        if self.energy_calibration:
            directions = self.lora_A[stage].float().T
            current_energy = torch.diagonal(directions.T @ self.current_covariance @ directions)
            old_energy = torch.diagonal(directions.T @ self.historical_covariance @ directions)
            numerator = self.calibration_lambda * current_energy
            scale = numerator / (numerator + old_energy).clamp_min(1e-8)
            if not bool(torch.isfinite(scale).all()):
                raise FloatingPointError("nonfinite CQSM energy calibration")
            self.general_scale[stage].copy_(scale)
        self.historical_covariance.add_(self.current_covariance)
        self.current_covariance.zero_()
        self.history_committed[stage] = True
        if not self.energy_calibration:
            self.general_scale.fill_(1.0)
        self.stop_collection()
        self._protected_A = None
        self._protected_basis = None


class CQSMDualLoRALinear(ObjectCoupledTaskLoRALinear):
    """Persistent global LoRA + one routed private LoRA + an ungated CQSM.

    Global factors have no task axis. The orthonormal input factor stays frozen;
    the output factor continues to train after task changes.
    The inherited protection hooks/snapshots apply only to private factors and
    bank readouts, including AdamW's decoupled weight decay. The base map and
    all estimated bank bases stay frozen. The restored preset re-enables the
    legacy private-factor orthogonality regularizer.
    """

    def __init__(self, *args, shared_enabled: bool = True, private_enabled: bool = True,
                 bank_mode: str = "all", restore_legacy: bool = False,
                 shared_rank: int = 2, shared_qv_only: bool = False,
                 shared_gating: bool = False, shared_gate_init: float = 0.1, **kwargs):
        if bank_mode not in {"all", "off", "current", "routed"}:
            raise ValueError("CQSM bank_mode must be all, off, current, or routed")
        self.restore_legacy = bool(restore_legacy)
        if not self.restore_legacy:
            kwargs["orthogonality_weight"] = 0.0
        super().__init__(*args, **kwargs)
        self.shared_enabled = bool(shared_enabled)
        self.private_enabled = bool(private_enabled)
        self.lora_A.requires_grad_(self.private_enabled)
        self.lora_B.requires_grad_(self.private_enabled)
        self.bank_mode = bank_mode
        self.global_rank = int(shared_rank) if self.shared_enabled else 0
        self.global_gating = bool(shared_gating and self.shared_enabled)
        if not 0 < shared_gate_init < 1:
            raise ValueError("shared gate initialization must lie strictly between 0 and 1")
        if self.global_gating:
            self.global_gate_logit = nn.Parameter(torch.tensor(
                math.log(shared_gate_init / (1.0 - shared_gate_init))))
        else:
            self.register_parameter("global_gate_logit", None)
        if self.shared_enabled:
            if not 0 < self.global_rank <= self.in_features:
                raise ValueError("shared rank must fit the input width")
            self.global_A = nn.Parameter(torch.empty(self.global_rank, self.in_features))
            self.global_B = nn.Parameter(torch.zeros(self.out_features, self.global_rank))
            mask = torch.ones(self.out_features, 1)
            if shared_qv_only:
                if self.out_features != 3 * self.in_features:
                    raise ValueError("Q/V-only shared LoRA requires a fused QKV projection")
                mask[self.in_features:2*self.in_features] = 0
            self.register_buffer("global_output_mask", mask, persistent=False)
            self.global_B.register_hook(lambda grad: grad * self.global_output_mask.to(grad))
            with torch.no_grad():
                q, _ = torch.linalg.qr(torch.randn(self.in_features, self.global_rank), mode="reduced")
                self.global_A.copy_(q.T)
            self.global_A.requires_grad_(False)
        else:
            self.register_parameter("global_A", None)
            self.register_parameter("global_B", None)
        if self.core_B is not None and self.bank_mode == "off":
            self.core_B.requires_grad_(False)

    def bank_update(self, x: torch.Tensor) -> torch.Tensor:
        update = x.new_zeros(*x.shape[:-1], self.out_features)
        if not self.uses_object_core:
            return update
        core = self._object_core
        if not isinstance(core, CovarianceQuerySubspaceMemory):
            raise RuntimeError("CQSM adapter requires its uncalibrated fixed basis bank")
        core.observe(x, group=self.function_group)
        if self.bank_mode == "off":
            return update
        ready = torch.where(core.subspace_ready)[0].tolist()
        if self.bank_mode == "current" and ready:
            # Latest learned entry, NOT an oracle task label for this image.
            ready = ready[-1:]
        route = (self._anchor_router.route_for(x, group=self.function_group)
                 if self.bank_mode == "routed" else None)
        for stage in ready:
            hidden = core.project(x, stage)
            if core.energy_calibration:
                hidden = hidden * core.general_scale[stage].to(hidden)
            delta = F.linear(hidden, self.core_B[stage].to(x.dtype))
            if route is not None:
                delta = delta * route[..., stage].unsqueeze(-1)
            update = update + delta
        return update

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_only = _SHARED_ONLY_FORWARD.get()
        if self.private_enabled and not shared_only:
            output = IndependentCapacityTaskLoRALinear.forward(self, x)
        else:
            if not shared_only and self._is_anchor_source and self._anchor_router._visual_route is None:
                self._anchor_router.prepare_visual(x, training=self.training)
            output = self.base(x)
        if self.shared_enabled:
            delta = F.linear(
                F.linear(x, self.global_A.to(x.dtype)),
                (self.global_B * self.global_output_mask).to(x.dtype))
            if self.global_gate_logit is not None:
                delta = delta * self.global_gate_logit.sigmoid().to(delta.dtype)
            if not shared_only and getattr(self, "sdo_bridge", None) is not None:
                delta = delta + self.sdo_bridge(x, self)
            output = output + delta
        return output if shared_only else output + self.bank_update(x)

    def regularization_loss(self) -> torch.Tensor:
        if self.restore_legacy and self.private_enabled:
            return super().regularization_loss()
        return self.base.weight.new_zeros(())

    def restore_trainability(self) -> None:
        super().restore_trainability()
        self.lora_A.requires_grad_(self.private_enabled)
        self.lora_B.requires_grad_(self.private_enabled)
        if self.shared_enabled:
            self.global_A.requires_grad_(False)
            self.global_B.requires_grad_(True)
            if self.global_gate_logit is not None:
                self.global_gate_logit.requires_grad_(True)
        if self.core_B is not None:
            self.core_B.requires_grad_(self.bank_mode != "off")


class ConsensusComplementLoRALinear(MLPTaskDecoupledLoRALinear):
    """Fixed-budget anchor whose active visual atoms have object/box/mask roles.

    The forward path is identical to the stable multi-anchor adapter. Roles only
    control how functional gradients are written during training, so inference
    still consumes one image-selected anchor and never a task identifier.
    """

    def role_parameter_masks(self) -> list[tuple[nn.Parameter, torch.Tensor,
                                                  torch.Tensor, torch.Tensor]]:
        slots = torch.where(self.active_mask)[0]
        roles = torch.full((self.rank,), -1, device=self.active_mask.device,
                           dtype=torch.long)
        if self.function_group == "visual":
            for position, slot in enumerate(slots.tolist()):
                roles[slot] = position % 3  # object, box complement, mask complement
        elif self.function_group == "semantic":
            roles[slots] = 0  # preserve the complete open-vocabulary semantic slice
        else:
            return []
        result = []
        for parameter, axis in ((self.lora_A, 0), (self.lora_B, 1)):
            shape = [1] * parameter.ndim
            shape[axis] = self.rank
            result.append((parameter,
                           (roles == 0).reshape(shape).expand_as(parameter),
                           (roles == 1).reshape(shape).expand_as(parameter),
                           (roles == 2).reshape(shape).expand_as(parameter)))
        return result

    def regularization_loss(self) -> torch.Tensor:
        loss = super().regularization_loss()
        if self.function_group != "visual":
            return loss
        slots = torch.where(self.active_mask)[0]
        if len(slots) < 2:
            return loss
        a = F.normalize(self.lora_A[slots].float(), dim=1)
        role = torch.arange(len(slots), device=a.device) % 3
        pairs = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            x, y = a[role == left], a[role == right]
            if len(x) and len(y):
                pairs.append((x @ y.T).square().mean())
        if pairs:
            complement_orthogonality = torch.stack(pairs).mean()
            loss = loss + (complement_orthogonality * self.orthogonality_weight).to(loss)
        return loss


class _AllocatedSlots(ContinualAdapterLinear):
    """Fixed rank pool allocated across tasks without parameter growth."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int):
        super().__init__(base, rank, alpha, num_tasks)
        self.register_buffer("active_mask", torch.zeros(rank))
        self.register_buffer("allocated_mask", torch.zeros(rank))

    def _task_slots(self, task_id: int) -> torch.Tensor:
        # Nearly equal deterministic partition; every slot is allocated at most once.
        boundaries = torch.linspace(0, self.rank, self.num_tasks + 1).round().long()
        lo, hi = int(boundaries[task_id]), int(boundaries[task_id + 1])
        if hi <= lo:
            hi = min(lo + 1, self.rank)
        return torch.arange(lo, hi)

    def begin_task(self, task_id: int) -> None:
        super().begin_task(task_id)
        slots = self._task_slots(task_id).to(self.active_mask.device)
        self.active_mask.zero_(); self.active_mask[slots] = 1
        self.allocated_mask[slots] = 1


class InfLoRALinear(_AllocatedSlots):
    """Interference-reduced LoRA with input-subspace-orthogonal A factors.

    Historical input directions are estimated online. New task A rows are
    projected onto their orthogonal complement; old rank slots remain frozen.
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5,
                 history_rank: int = 32, samples_per_forward: int = 32):
        super().__init__(base, rank, alpha, num_tasks)
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.history_rank = min(history_rank, self.in_features)
        self.samples_per_forward = samples_per_forward
        self.register_buffer("history_basis", torch.empty(0, self.in_features))
        self._samples: list[torch.Tensor] = []
        self.lora_A.register_hook(lambda g: g * self.active_mask[:, None])
        self.lora_B.register_hook(lambda g: g * self.active_mask[None, :])

    @torch.no_grad()
    def begin_task(self, task_id: int) -> None:
        super().begin_task(task_id)
        slots = torch.where(self.active_mask.bool())[0]
        a = torch.randn(len(slots), self.in_features, device=self.lora_A.device,
                        dtype=self.lora_A.dtype)
        if self.history_basis.numel():
            h = self.history_basis.to(a)
            a = a - (a @ h.T) @ h
        # Stable row normalization after null-space projection.
        a = F.normalize(a, dim=1)
        self.lora_A[slots] = a
        self.lora_B[:, slots].zero_()
        self._samples.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and len(self._samples) < 32:
            flat = x.detach().reshape(-1, self.in_features)
            if len(flat):
                step = max(len(flat) // self.samples_per_forward, 1)
                self._samples.append(flat[::step][: self.samples_per_forward].float().cpu())
        a = self.lora_A * self.allocated_mask[:, None]
        b = self.lora_B * self.allocated_mask[None, :]
        return self.base(x) + F.linear(F.linear(x, a), b) * self.scaling

    @torch.no_grad()
    def end_task(self) -> None:
        if not self._samples:
            return
        samples = torch.cat(self._samples, 0).to(self.lora_A.device)
        samples = samples - samples.mean(0, keepdim=True)
        if self.history_basis.numel():
            samples = torch.cat((self.history_basis.to(samples), samples), 0)
        _, _, vh = torch.linalg.svd(samples, full_matrices=False)
        self.history_basis = vh[: self.history_rank].detach().to(self.history_basis.device)
        self._samples.clear()


class BiLoRALinear(ContinualAdapterLinear):
    """Fixed-basis bilinear LoRA with task-disjoint blocks in one core matrix."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5):
        super().__init__(base, rank, alpha, num_tasks)
        self.register_buffer("left_basis", _orthogonal_rows(rank, self.out_features).T)
        self.register_buffer("right_basis", _orthogonal_rows(rank, self.in_features))
        self.core = nn.Parameter(torch.zeros(rank, rank))
        self.register_buffer("active_core", torch.zeros(rank, rank))
        self.register_buffer("allocated_core", torch.zeros(rank, rank))
        self.core.register_hook(lambda g: g * self.active_core)

    @torch.no_grad()
    def begin_task(self, task_id: int) -> None:
        super().begin_task(task_id)
        boundaries = torch.linspace(0, self.rank, self.num_tasks + 1).round().long()
        lo, hi = int(boundaries[task_id]), int(boundaries[task_id + 1])
        hi = max(hi, min(lo + 1, self.rank))
        self.active_core.zero_(); self.active_core[lo:hi, lo:hi] = 1
        self.allocated_core[lo:hi, lo:hi] = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        core = self.core * self.allocated_core
        # Evaluate the low-rank chain without materializing an out x in matrix.
        hidden = F.linear(x, self.right_basis)
        hidden = F.linear(hidden, core)
        return self.base(x) + F.linear(hidden, self.left_basis) * self.scaling


class Rank1ExpertLoRALinear(_AllocatedSlots):
    """Single LoRA decomposed into an input-routed pool of rank-1 experts."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5,
                 top_k: int = 4, router_stability: float = 1e-3, ago_weight: float = 1e-3):
        super().__init__(base, rank, alpha, num_tasks)
        self.expert_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.expert_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.router = nn.Linear(self.in_features, rank, bias=False)
        nn.init.kaiming_uniform_(self.expert_A, a=math.sqrt(5))
        nn.init.zeros_(self.router.weight)
        self.top_k = min(top_k, rank)
        self.router_stability, self.ago_weight = router_stability, ago_weight
        self.register_buffer("router_anchor", torch.zeros_like(self.router.weight))
        self.expert_A.register_hook(lambda g: g * self.active_mask[:, None])
        self.expert_B.register_hook(lambda g: g * self.active_mask[None, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x.float()).to(x.dtype)
        valid = self.allocated_mask.bool()
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        k = min(self.top_k, max(int(valid.sum()), 1))
        values, indices = logits.topk(k, dim=-1)
        sparse = torch.full_like(logits, torch.finfo(logits.dtype).min)
        sparse.scatter_(-1, indices, values)
        gates = sparse.softmax(-1)
        activations = F.linear(x, self.expert_A) * gates
        return self.base(x) + F.linear(activations, self.expert_B) * self.scaling

    def regularization_loss(self) -> torch.Tensor:
        allocated = self.allocated_mask.bool()
        a = F.normalize(self.expert_A[allocated].float(), dim=1)
        ago = a.new_zeros(())
        if len(a) > 1:
            gram = a @ a.T
            ago = (gram - torch.eye(len(a), device=gram.device)).square().mean()
        stability = (self.router.weight - self.router_anchor).square().mean()
        return ago * self.ago_weight + stability * self.router_stability

    @torch.no_grad()
    def end_task(self) -> None:
        self.router_anchor.copy_(self.router.weight)


class ParameterMatchedRank1ExpertLoRALinear(_AllocatedSlots):
    """Rank-1 experts with parameter-free activation routing.

    Reusing each expert's A projection as its routing score removes the extra
    router matrix, making the trainable parameter count exactly match ordinary
    LoRA at the same rank.
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_tasks: int = 5,
                 top_k: int = 4, ago_weight: float = 1e-3):
        super().__init__(base, rank, alpha, num_tasks)
        self.expert_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.expert_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.expert_A, a=math.sqrt(5))
        self.top_k, self.ago_weight = min(top_k, rank), ago_weight
        self.expert_A.register_hook(lambda g: g * self.active_mask[:, None])
        self.expert_B.register_hook(lambda g: g * self.active_mask[None, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = F.linear(x, self.expert_A)
        logits = activations.float().to(x.dtype)
        valid = self.allocated_mask.bool()
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        k = min(self.top_k, max(int(valid.sum()), 1))
        values, indices = logits.topk(k, dim=-1)
        sparse = torch.full_like(logits, torch.finfo(logits.dtype).min)
        sparse.scatter_(-1, indices, values)
        gates = sparse.softmax(-1)
        return self.base(x) + F.linear(activations * gates, self.expert_B) * self.scaling

    def regularization_loss(self) -> torch.Tensor:
        a = F.normalize(self.expert_A[self.allocated_mask.bool()].float(), dim=1)
        if len(a) <= 1: return self.base.weight.new_zeros(())
        gram = a @ a.T
        return (gram - torch.eye(len(a), device=gram.device)).square().mean() * self.ago_weight


@dataclass
class InjectionReport:
    method: str
    modules: list[str]
    trainable_parameters: int
    total_parameters: int
    persistent_adapter_parameters: int = 0
    module_groups: dict[str, list[str]] | None = None
    group_ranks: dict[str, int] | None = None


METHODS = {
    "sequential_lora": SequentialLoRALinear,
    "lora_ewc": SequentialLoRALinear,
    "inflora": InfLoRALinear,
    "bilora": BiLoRALinear,
    "rank1_expert": Rank1ExpertLoRALinear,
    "rank1_expert_pm": ParameterMatchedRank1ExpertLoRALinear,
    "lwf_lora": SequentialLoRALinear,
    "er_lora": SequentialLoRALinear,
    "derpp_lora": SequentialLoRALinear,
    "cumulative_lora": SequentialLoRALinear,
    "focus_lora": FocusLoRALinear,
    "focus_lora_v2": ImportancePoolFocusLoRALinear,
    "focus_lora_v3": CapacityPartitionFocusLoRALinear,
    "task_decoupled_lora": MultiAnchorTaskDecoupledLoRALinear,
    "task_decoupled_mlp": MLPTaskDecoupledLoRALinear,
    "compositional_anchor_lora": MLPTaskDecoupledLoRALinear,
    "object_consistent_lora": MLPTaskDecoupledLoRALinear,
    "shared_core_residual_lora": MLPTaskDecoupledLoRALinear,
    "unified_object_lora": MLPTaskDecoupledLoRALinear,
    "consensus_complement_lora": ConsensusComplementLoRALinear,
    "samcl_slora": SAMCLSLoRALinear,
    "independent_capacity_lora": IndependentCapacityTaskLoRALinear,
    "independent_visual_mlp_lora": IndependentCapacityTaskLoRALinear,
    "object_coupled_capacity_lora": ObjectCoupledTaskLoRALinear,
    "energy_object_subspace_lora": EnergyObjectSubspaceLoRALinear,
    "semantic_protected_subspace_lora": EnergyObjectSubspaceLoRALinear,
    "energy_object_compositional_lora": EnergyObjectSubspaceLoRALinear,
    "energy_visual_mlp_lora": EnergyObjectSubspaceLoRALinear,
    "energy_visual_mlp_general_only": EnergyObjectSubspaceLoRALinear,
    "energy_visual_mlp_independent_core": EnergyObjectSubspaceLoRALinear,
    "energy_visual_mlp_isolated_only": EnergyObjectSubspaceLoRALinear,
    "energy_visual_mlp_encoder": EnergyObjectSubspaceLoRALinear,
    "energy_visual_mlp_decoder": EnergyObjectSubspaceLoRALinear,
    "energy_visual_mlp_mlp_only": EnergyObjectSubspaceLoRALinear,
    "energy_visual_mlp_early_mlp": EnergyObjectSubspaceLoRALinear,
    "energy_visual_mlp_random_core": EnergyObjectSubspaceLoRALinear,
    "cqsm_dual_lora": CQSMDualLoRALinear,
}

CQSM_METHODS = {"cqsm_dual_lora"}

INDEPENDENT_CAPACITY_METHODS = {
    "independent_capacity_lora",
    "independent_visual_mlp_lora",
}

ENERGY_SUBSPACE_METHODS = {
    "energy_object_subspace_lora",
    "semantic_protected_subspace_lora",
    "energy_object_compositional_lora",
    "energy_visual_mlp_lora",
    "energy_visual_mlp_general_only",
    "energy_visual_mlp_independent_core",
    "energy_visual_mlp_isolated_only",
    "energy_visual_mlp_decoder",
    "energy_visual_mlp_mlp_only",
    "energy_visual_mlp_early_mlp",
    "energy_visual_mlp_random_core",
}

LORA_LOCATION_METHODS = {
    "energy_visual_mlp_encoder",
    "energy_visual_mlp_decoder",
}

VISUAL_MLP_PATTERNS = (
    r"^backbone\.vision_backbone\.trunk\.blocks\.\d+\.mlp\.fc[12]$",
)


def _linear_matches(model: nn.Module, patterns) -> list[tuple[str, nn.Linear]]:
    compiled = [re.compile(x) for x in patterns]
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and any(pattern.search(name) for pattern in compiled)]


def _focus_matches(model: nn.Module) -> dict[str, list[tuple[str, nn.Linear]]]:
    result = {group: _linear_matches(model, patterns)
              for group, patterns in FOCUS_TARGET_GROUPS.items()}
    owners: dict[str, str] = {}
    for group, matches in result.items():
        for name, _ in matches:
            if name in owners:
                raise RuntimeError(f"FOCUS target {name} belongs to both {owners[name]} and {group}")
            owners[name] = group
    missing = [group for group, matches in result.items() if not matches]
    if missing:
        raise RuntimeError(f"No SAM 3 linear layers matched FOCUS groups: {missing}")
    return result


def _focus_rank_plan(model: nn.Module, reference_rank: int,
                     explicit: Mapping[str, int] | None = None) -> tuple[dict[str, int], int]:
    """Choose group ranks under the ordinary rank-R LoRA parameter budget."""
    if reference_rank < 2:
        raise ValueError("FOCUS-LoRA requires --rank >= 2")
    reference = _linear_matches(model, DEFAULT_TARGETS)
    budget = sum(reference_rank * (m.in_features + m.out_features) for _, m in reference)
    grouped = _focus_matches(model)
    costs = {g: sum(m.in_features + m.out_features for _, m in matches)
             for g, matches in grouped.items()}
    if explicit is not None:
        ranks = {g: int(explicit[g]) for g in FOCUS_TARGET_GROUPS}
        if any(r < 2 for r in ranks.values()):
            raise ValueError("Every explicit FOCUS group rank must be >= 2")
        used = sum(costs[g] * ranks[g] for g in ranks)
        if used > budget:
            raise ValueError(f"Explicit FOCUS ranks use {used} parameters, over budget {budget}")
        return ranks, budget

    # Start from familiar ranks: full reference rank in shared/semantic layers
    # and half rank in the small prediction heads.  Search nearby integer ranks
    # for the closest feasible use of the exact reference parameter budget.
    preferred = {
        "visual": reference_rank,
        "semantic": reference_rank,
        "box": max(2, reference_rank // 2),
        "mask": max(2, reference_rank // 2),
    }
    ranges = {
        g: range(2, preferred[g] + 1) for g in preferred
    }
    best = None
    for rv in ranges["visual"]:
        for rs in ranges["semantic"]:
            for rb in ranges["box"]:
                for rm in ranges["mask"]:
                    candidate = {"visual": rv, "semantic": rs, "box": rb, "mask": rm}
                    used = sum(costs[g] * candidate[g] for g in candidate)
                    if used > budget:
                        continue
                    # Primary objective: consume the fair budget.  Secondary:
                    # stay close to the planned functional allocation.
                    distance = sum((preferred[g] - candidate[g]) ** 2 for g in candidate)
                    score = (used, -distance)
                    if best is None or score > best[0]:
                        best = (score, candidate)
    if best is None:
        raise RuntimeError("Could not fit minimum FOCUS ranks into the LoRA reference budget")
    return best[1], budget


def _replace_linear(model: nn.Module, name: str, adapter: nn.Module) -> None:
    if "." in name:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
    else:
        parent, child = model, name
    setattr(parent, child, adapter)


def _inject_focus_lora(model: nn.Module, rank: int, alpha: float, num_tasks: int,
                       *, focus_group_ranks: Mapping[str, int] | None = None,
                       focus_stable_ratio: float = 0.75,
                       focus_merge_gamma: float = 0.75,
                       focus_orthogonality_weight: float = 1e-3) -> InjectionReport:
    grouped = _focus_matches(model)
    group_ranks, reference_budget = _focus_rank_plan(model, rank, focus_group_ranks)
    model.requires_grad_(False)
    replaced: list[str] = []
    module_groups: dict[str, list[str]] = {}
    for group, matches in grouped.items():
        module_groups[group] = []
        for name, module in matches:
            adapter = FocusLoRALinear(
                module, rank=group_ranks[group], alpha=alpha, num_tasks=num_tasks,
                function_group=group, stable_ratio=focus_stable_ratio,
                merge_gamma=focus_merge_gamma,
                orthogonality_weight=focus_orthogonality_weight,
            )
            # Unlike ordinary baselines, FOCUS performs multi-objective
            # gradient projections and therefore retains FP32 master factors.
            adapter.to(device=module.weight.device)
            _replace_linear(model, name, adapter)
            replaced.append(name)
            module_groups[group].append(name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    persistent = sum(p.numel() for module in model.modules()
                     if isinstance(module, FocusLoRALinear)
                     for p in module.adapter_parameters())
    if persistent > reference_budget:
        raise RuntimeError(f"FOCUS adapter budget {persistent} exceeds reference {reference_budget}")
    total = sum(p.numel() for p in model.parameters())
    return InjectionReport("focus_lora", replaced, trainable, total,
                           persistent, module_groups, group_ranks)


def _inject_focus_lora_v2(model: nn.Module, rank: int, alpha: float, num_tasks: int,
                          *, focus_group_ranks: Mapping[str, int] | None = None,
                          focus_admission_ratio: float = 0.25,
                          focus_importance_decay: float = 0.99,
                          focus_orthogonality_weight: float = 1e-3) -> InjectionReport:
    grouped = _focus_matches(model)
    group_ranks, reference_budget = _focus_rank_plan(model, rank, focus_group_ranks)
    model.requires_grad_(False)
    replaced: list[str] = []
    module_groups: dict[str, list[str]] = {}
    for group, matches in grouped.items():
        module_groups[group] = []
        for name, module in matches:
            adapter = ImportancePoolFocusLoRALinear(
                module, rank=group_ranks[group], alpha=alpha, num_tasks=num_tasks,
                function_group=group, admission_ratio=focus_admission_ratio,
                importance_decay=focus_importance_decay,
                orthogonality_weight=focus_orthogonality_weight,
            )
            adapter.to(device=module.weight.device)
            _replace_linear(model, name, adapter)
            replaced.append(name)
            module_groups[group].append(name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    persistent = sum(p.numel() for module in model.modules()
                     if isinstance(module, ImportancePoolFocusLoRALinear)
                     for p in module.adapter_parameters())
    if persistent > reference_budget:
        raise RuntimeError(
            f"FOCUS-v2 adapter budget {persistent} exceeds reference {reference_budget}")
    total = sum(p.numel() for p in model.parameters())
    return InjectionReport("focus_lora_v2", replaced, trainable, total,
                           persistent, module_groups, group_ranks)


def _inject_focus_lora_v3(model: nn.Module, rank: int, alpha: float, num_tasks: int,
                          *, focus_group_ranks: Mapping[str, int] | None = None,
                          focus_top_k: int = 4,
                          focus_orthogonality_weight: float = 1e-3) -> InjectionReport:
    grouped = _focus_matches(model)
    group_ranks, reference_budget = _focus_rank_plan(model, rank, focus_group_ranks)
    insufficient = {group: value for group, value in group_ranks.items()
                    if value < num_tasks}
    if insufficient:
        raise ValueError(
            f"FOCUS-v3 needs at least one rank per stream stage; insufficient {insufficient}")
    model.requires_grad_(False)
    replaced: list[str] = []
    module_groups: dict[str, list[str]] = {}
    for group, matches in grouped.items():
        module_groups[group] = []
        for name, module in matches:
            adapter = CapacityPartitionFocusLoRALinear(
                module, rank=group_ranks[group], alpha=alpha, num_tasks=num_tasks,
                function_group=group, top_k=focus_top_k,
                orthogonality_weight=focus_orthogonality_weight,
            )
            adapter.to(device=module.weight.device)
            _replace_linear(model, name, adapter)
            replaced.append(name)
            module_groups[group].append(name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    persistent = sum(p.numel() for module in model.modules()
                     if isinstance(module, CapacityPartitionFocusLoRALinear)
                     for p in module.adapter_parameters())
    if persistent > reference_budget:
        raise RuntimeError(
            f"FOCUS-v3 adapter budget {persistent} exceeds reference {reference_budget}")
    total = sum(p.numel() for p in model.parameters())
    return InjectionReport("focus_lora_v3", replaced, trainable, total,
                           persistent, module_groups, group_ranks)


def _inject_task_decoupled_lora(model: nn.Module, rank: int, alpha: float, num_tasks: int,
                                *, focus_group_ranks: Mapping[str, int] | None = None,
                                focus_orthogonality_weight: float = 1e-3) -> InjectionReport:
    """Inject fixed-budget functional adapters with one shared multi-anchor route."""
    grouped = _focus_matches(model)
    group_ranks, reference_budget = _focus_rank_plan(model, rank, focus_group_ranks)
    insufficient = {group: value for group, value in group_ranks.items()
                    if value < num_tasks}
    if insufficient:
        raise ValueError(
            f"task-decoupled LoRA needs one rank per anchor; insufficient {insufficient}")
    source_name, source_module = grouped["visual"][0]
    router = MultiAnchorTaskRouter(source_module.in_features, num_tasks)
    model.requires_grad_(False)
    replaced: list[str] = []
    module_groups: dict[str, list[str]] = {}
    for group, matches in grouped.items():
        module_groups[group] = []
        for name, module in matches:
            adapter = MultiAnchorTaskDecoupledLoRALinear(
                module, rank=group_ranks[group], alpha=alpha, num_tasks=num_tasks,
                function_group=group,
                orthogonality_weight=focus_orthogonality_weight,
            )
            adapter.attach_anchor_router(router, source=name == source_name)
            adapter.to(device=module.weight.device)
            _replace_linear(model, name, adapter)
            replaced.append(name)
            module_groups[group].append(name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    persistent = sum(p.numel() for module in model.modules()
                     if isinstance(module, MultiAnchorTaskDecoupledLoRALinear)
                     for p in module.adapter_parameters())
    if persistent > reference_budget:
        raise RuntimeError(
            f"task-decoupled adapter budget {persistent} exceeds reference {reference_budget}")
    total = sum(p.numel() for p in model.parameters())
    return InjectionReport("task_decoupled_lora", replaced, trainable, total,
                           persistent, module_groups, group_ranks)


def _inject_task_decoupled_mlp(model: nn.Module, rank: int, alpha: float, num_tasks: int,
                               *, focus_group_ranks: Mapping[str, int] | None = None,
                               focus_orthogonality_weight: float = 1e-3,
                               object_consistent: bool = False,
                               shared_core_residual: bool = False,
                               unified_object: bool = False,
                               consensus_complement: bool = False,
                               compositional_anchor: bool = False) -> InjectionReport:
    grouped = _focus_matches(model)
    group_ranks, reference_budget = _focus_rank_plan(model, rank, focus_group_ranks)
    visual_pattern = re.compile(r"\.blocks\.(\d+)\.attn\.")
    visual_blocks = sorted({int(match.group(1)) for name, _ in grouped["visual"]
                            if (match := visual_pattern.search(name))})
    if not visual_blocks:
        raise RuntimeError("could not locate SAM 3 visual trunk blocks for MLP routing")
    # As in SAMCL, the frozen first half produces a stable domain descriptor;
    # task-specific visual capacity starts only after this routing cut.
    routing_block = visual_blocks[len(visual_blocks) // 2]
    grouped["visual"] = [
        (name, module) for name, module in grouped["visual"]
        if int(visual_pattern.search(name).group(1)) >= routing_block
    ]
    source_candidates = [(name, module) for name, module in grouped["visual"]
                         if f".blocks.{routing_block}.attn.qkv" in name]
    source_name, source_module = (source_candidates or grouped["visual"])[0]
    router_cls = (CompositionalPromptMLPTaskRouter
                  if compositional_anchor else
                  ObjectConsistentSkillRouter
                  if (object_consistent or shared_core_residual or unified_object or
                      consensus_complement)
                  else PromptMLPTaskRouter)
    router = router_cls(source_module.in_features, num_tasks,
                        embeddings_per_task=300,
                        routing_block=routing_block)

    # Pay for the selectors by deterministically reducing the largest functional
    # rank cost until the ordinary rank-R LoRA budget is met.
    costs = {g: sum(m.in_features + m.out_features for _, m in matches)
             for g, matches in grouped.items()}
    selector_parameters = sum(p.numel() for p in router.parameters())
    ranks = dict(group_ranks)
    while sum(costs[g] * ranks[g] for g in ranks) + selector_parameters > reference_budget:
        candidates = [g for g in ranks if ranks[g] > num_tasks]
        if not candidates:
            raise RuntimeError("cannot fit MLP selectors into the fixed LoRA budget")
        ranks[max(candidates, key=lambda g: costs[g])] -= 1

    model.requires_grad_(False)
    replaced: list[str] = []
    module_groups: dict[str, list[str]] = {}
    for group, matches in grouped.items():
        module_groups[group] = []
        for name, module in matches:
            adapter_cls = (ConsensusComplementLoRALinear
                           if consensus_complement else MLPTaskDecoupledLoRALinear)
            adapter = adapter_cls(
                module, rank=ranks[group], alpha=alpha, num_tasks=num_tasks,
                function_group=group,
                orthogonality_weight=focus_orthogonality_weight,
            )
            adapter.attach_anchor_router(router, source=name == source_name)
            adapter.to(device=module.weight.device)
            _replace_linear(model, name, adapter)
            replaced.append(name)
            module_groups[group].append(name)
    routing_module_name = f"backbone.vision_backbone.trunk.blocks.{routing_block}"
    routing_module = model.get_submodule(routing_module_name)
    def observe_visual(_module, args):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("visual routing block must receive its activation first")
        router.prepare_visual(args[0], training=_module.training)
    # Re-injection restores this hook when loading a checkpoint. The source qkv
    # retains a fallback for wrappers which bypass the trunk block's forward.
    handle = routing_module.register_forward_pre_hook(observe_visual)
    object.__setattr__(router, "_visual_hook_handle", handle)
    module_groups["selector"] = [f"{routing_module_name}::<input_hook>"]

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    persistent = sum(p.numel() for module in model.modules()
                     if isinstance(module, MLPTaskDecoupledLoRALinear)
                     for p in module.adapter_parameters())
    if persistent > reference_budget:
        raise RuntimeError(
            f"MLP task-decoupled budget {persistent} exceeds reference {reference_budget}")
    total = sum(p.numel() for p in model.parameters())
    method_name = ("compositional_anchor_lora" if compositional_anchor else
                   "consensus_complement_lora" if consensus_complement else
                   "unified_object_lora" if unified_object else
                   "shared_core_residual_lora" if shared_core_residual else
                   "object_consistent_lora" if object_consistent else
                   "task_decoupled_mlp")
    return InjectionReport(method_name, replaced, trainable, total,
                           persistent, module_groups, ranks)


def _inject_samcl_slora(model: nn.Module, rank: int, alpha: float,
                        num_tasks: int) -> InjectionReport:
    """Port the original prompt-free SLoRA/selector path to SAM 3's visual trunk."""
    if rank <= 0:
        raise ValueError("SAMCL SLoRA rank must be positive")
    visual = _focus_matches(model)["visual"]
    visual_pattern = re.compile(r"\.blocks\.(\d+)\.attn\.")
    visual_blocks = sorted({int(match.group(1)) for name, _ in visual
                            if (match := visual_pattern.search(name))})
    if not visual_blocks:
        raise RuntimeError("could not locate SAM 3 visual trunk blocks for SAMCL")
    # Official SAMCL extracts a frozen intermediate feature and adapts only the
    # subsequent image-encoder blocks.
    routing_block = visual_blocks[len(visual_blocks) // 2]
    visual = [(name, module) for name, module in visual
              if int(visual_pattern.search(name).group(1)) >= routing_block]
    source_candidates = [(name, module) for name, module in visual
                         if f".blocks.{routing_block}.attn.qkv" in name]
    source_name, source_module = (source_candidates or visual)[0]
    widths = {module.in_features for _, module in visual}
    if widths != {source_module.in_features}:
        raise RuntimeError(
            f"SAMCL cross-layer shared A requires one visual width, found {sorted(widths)}")
    router = PromptMLPTaskRouter(
        source_module.in_features, num_tasks, embeddings_per_task=300,
        routing_block=routing_block)
    shared_a = SAMCLSharedA(source_module.in_features, rank, num_tasks)

    model.requires_grad_(False)
    replaced: list[str] = []
    for name, module in visual:
        adapter = SAMCLSLoRALinear(
            module, rank=rank, alpha=alpha, num_tasks=num_tasks,
            shared_a=shared_a, function_group="visual")
        adapter.attach_samcl_state(
            router, shared_a, source=name == source_name)
        # Retain FP32 adapter masters, matching the official implementation.
        adapter.to(device=module.weight.device)
        _replace_linear(model, name, adapter)
        replaced.append(name)

    routing_module_name = f"backbone.vision_backbone.trunk.blocks.{routing_block}"
    routing_module = model.get_submodule(routing_module_name)

    def observe_visual(_module, args):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("SAMCL routing block must receive its activation first")
        router.prepare_visual(args[0], training=_module.training)

    handle = routing_module.register_forward_pre_hook(observe_visual)
    object.__setattr__(router, "_visual_hook_handle", handle)
    module_groups = {
        "visual": replaced,
        "shared_A": [source_name + "::shared_A"],
        "selector": [f"{routing_module_name}::<input_hook>"],
    }
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    persistent = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, SAMCLSLoRALinear)
        for p in module.adapter_parameters()
    )
    total = sum(p.numel() for p in model.parameters())
    return InjectionReport("samcl_slora", replaced, trainable, total,
                           persistent, module_groups, {"visual": rank})


def _inject_independent_capacity_lora(
        model: nn.Module, rank: int, alpha: float, num_tasks: int,
        *, focus_group_ranks: Mapping[str, int] | None = None,
        focus_orthogonality_weight: float = 1e-3,
        visual_mlp: bool = False) -> InjectionReport:
    """Inject parameter-matched, full per-task capacity across all functions."""
    grouped = _focus_matches(model)
    if visual_mlp:
        grouped["visual"].extend(_linear_matches(model, VISUAL_MLP_PATTERNS))
    attn_pattern = re.compile(r"\.blocks\.(\d+)\.attn\.")
    block_pattern = re.compile(r"\.blocks\.(\d+)\.")
    visual_blocks = sorted({int(match.group(1)) for name, _ in grouped["visual"]
                            if (match := attn_pattern.search(name))})
    if not visual_blocks:
        raise RuntimeError("could not locate SAM 3 visual trunk blocks for independent capacity")
    routing_block = visual_blocks[len(visual_blocks) // 2]
    grouped["visual"] = [
        (name, module) for name, module in grouped["visual"]
        if int(block_pattern.search(name).group(1)) >= routing_block
    ]
    if focus_group_ranks is None:
        # With the standard reference rank 16 this gives 5/4/3/3 per task.
        # Across five tasks that is 5.024M parameters, within 0.34% of the
        # rank-10 SAMCL-SLoRA baseline's 5.041M parameters on SAM 3.
        ratios = {"visual": 5 / 16, "semantic": 4 / 16,
                  "box": 3 / 16, "mask": 3 / 16}
        ranks = {group: max(1, int(round(rank * ratio)))
                 for group, ratio in ratios.items()}
    else:
        missing = set(FOCUS_TARGET_GROUPS) - set(focus_group_ranks)
        if missing:
            raise ValueError(f"independent capacity ranks are missing groups: {sorted(missing)}")
        ranks = {group: int(focus_group_ranks[group]) for group in FOCUS_TARGET_GROUPS}
        if any(value < 1 for value in ranks.values()):
            raise ValueError("every independent task/function rank must be positive")

    source_candidates = [(name, module) for name, module in grouped["visual"]
                         if f".blocks.{routing_block}.attn.qkv" in name]
    source_name, source_module = (source_candidates or grouped["visual"])[0]
    router = CompositionalPromptMLPTaskRouter(
        source_module.in_features, num_tasks, embeddings_per_task=300,
        routing_block=routing_block)
    model.requires_grad_(False)
    replaced: list[str] = []
    module_groups: dict[str, list[str]] = {}
    for group, matches in grouped.items():
        module_groups[group] = []
        for name, module in matches:
            adapter = IndependentCapacityTaskLoRALinear(
                module, rank=ranks[group], alpha=alpha, num_tasks=num_tasks,
                function_group=group,
                orthogonality_weight=focus_orthogonality_weight)
            adapter.attach_anchor_router(router, source=name == source_name)
            adapter.to(device=module.weight.device)
            _replace_linear(model, name, adapter)
            replaced.append(name)
            module_groups[group].append(name)

    routing_module_name = f"backbone.vision_backbone.trunk.blocks.{routing_block}"
    routing_module = model.get_submodule(routing_module_name)

    def observe_visual(_module, args):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("independent-capacity routing block needs an activation")
        router.prepare_visual(args[0], training=_module.training)

    handle = routing_module.register_forward_pre_hook(observe_visual)
    object.__setattr__(router, "_visual_hook_handle", handle)
    module_groups["selector"] = [f"{routing_module_name}::<input_hook>"]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    persistent = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, IndependentCapacityTaskLoRALinear)
        for p in module.adapter_parameters()
    )
    total = sum(p.numel() for p in model.parameters())
    method_name = ("independent_visual_mlp_lora" if visual_mlp
                   else "independent_capacity_lora")
    return InjectionReport(method_name, replaced, trainable, total,
                           persistent, module_groups, ranks)


def _inject_object_coupled_capacity_lora(
        model: nn.Module, rank: int, alpha: float, num_tasks: int,
        *, focus_group_ranks: Mapping[str, int] | None = None,
        focus_orthogonality_weight: float = 1e-3,
        object_core_rank: int = 2,
        energy_decomposed: bool = False,
        semantic_protected: bool = False,
        compositional_router: bool = False,
        visual_mlp: bool = False,
        energy_ablation: str | None = None,
        lora_scope: str = "all",
        visual_insert: str = "late",
        general_object_rank: int = 1,
        energy_calibration_lambda: float = 3.0,
        independent_object_core: bool = False,
        cqsm_enabled: bool = False,
        cqsm_shared: bool = True,
        cqsm_private: bool = True,
        cqsm_bank_mode: str = "all",
        cqsm_basis: str = "covariance",
        cqsm_memory_rank: int = 2,
        cqsm_energy_calibration: bool | None = None,
        cqsm_restore_legacy: bool = False,
        cqsm_shared_layout: str = "late_parallel",
        cqsm_global_rank: int = 2,
        cqsm_global_gating: bool = False,
        cqsm_global_gate_init: float = 0.1) -> InjectionReport:
    """Inject independent capacity coupled through shared or per-head Query bases."""
    if energy_ablation not in {None, "general_only", "isolated_only", "random_core"}:
        raise ValueError(f"unknown energy ablation: {energy_ablation}")
    if lora_scope not in {"all", "encoder", "decoder"}:
        raise ValueError(f"unknown lora scope: {lora_scope}")
    if visual_insert not in {"late", "mlp_only", "early_mlp"}:
        raise ValueError(f"unknown visual insert: {visual_insert}")
    if independent_object_core and semantic_protected:
        raise ValueError("semantic protection currently requires one shared object core")
    if cqsm_shared_layout not in {"late_parallel", "visual_prefix", "visual_residual"}:
        raise ValueError("unknown shared layout")
    if cqsm_global_gating and (not cqsm_enabled or not cqsm_shared or
                               cqsm_shared_layout != "visual_residual"):
        raise ValueError("Global gating requires enabled Global in visual_residual CQSM")
    if cqsm_enabled:
        if cqsm_memory_rank not in (1, 2, 4) or lora_scope != "all" or visual_insert != "late":
            raise ValueError("CQSM uses rank-2 memory and the preregistered all/late scope")
        if semantic_protected or independent_object_core or compositional_router:
            raise ValueError("CQSM does not include legacy semantic/compositional variants")
    if energy_ablation in {"general_only", "random_core"}:
        visual_mlp = True
        general_object_rank = object_core_rank
    elif energy_ablation == "isolated_only":
        visual_mlp = True
        general_object_rank = 0
    if visual_insert in {"mlp_only", "early_mlp"}:
        visual_mlp = True
    if lora_scope != "all":
        visual_mlp = True
        energy_decomposed = True
    grouped = _focus_matches(model)
    if visual_mlp:
        grouped["visual"].extend(_linear_matches(model, VISUAL_MLP_PATTERNS))
    # The frozen SAM 3 open-vocabulary scorer reads the same decoder Query as
    # the box and mask heads.  Adapt its Query projection only in this method,
    # leaving the frozen text projection as a stable semantic reference.
    scorer_name = "dot_prod_scoring.hs_proj"
    try:
        scorer = model.get_submodule(scorer_name)
    except AttributeError:
        scorer = None
    if isinstance(scorer, nn.Linear):
        grouped["semantic"].append((scorer_name, scorer))

    attn_pattern = re.compile(r"\.blocks\.(\d+)\.attn\.")
    block_pattern = re.compile(r"\.blocks\.(\d+)\.")
    visual_blocks = sorted({int(match.group(1)) for name, _ in grouped["visual"]
                            if (match := attn_pattern.search(name))})
    if not visual_blocks:
        raise RuntimeError("could not locate visual routing blocks for object coupling")
    routing_block = visual_blocks[len(visual_blocks) // 2]

    def _visual_block_id(name: str) -> int:
        match = block_pattern.search(name)
        if match is None:
            raise RuntimeError(f"visual adapter {name} has no trunk block id")
        return int(match.group(1))

    source_candidates = [(name, module) for name, module in grouped["visual"]
                         if f".blocks.{routing_block}.attn.qkv" in name]
    if visual_insert == "mlp_only":
        grouped["visual"] = [
            (name, module) for name, module in grouped["visual"]
            if ".mlp." in name and _visual_block_id(name) >= routing_block]
    elif visual_insert == "early_mlp":
        grouped["visual"] = [
            (name, module) for name, module in grouped["visual"]
            if ".mlp." in name and _visual_block_id(name) < routing_block]
    else:
        grouped["visual"] = [
            (name, module) for name, module in grouped["visual"]
            if _visual_block_id(name) >= routing_block and
            (not semantic_protected or _visual_block_id(name) > routing_block)
        ]
    source_name, source_module = (source_candidates or grouped["visual"])[0]
    prefix_blocks = sorted({_visual_block_id(n) for n, _ in grouped["visual"]})
    prefix_blocks = prefix_blocks[:max(1, len(prefix_blocks)//2)]
    if lora_scope == "encoder":
        grouped = {"visual": grouped["visual"], "semantic": [], "box": [], "mask": []}
    elif lora_scope == "decoder":
        grouped = {group: (matches if group != "visual" else [])
                   for group, matches in grouped.items()}
    if focus_group_ranks is None:
        if semantic_protected:
            # Reallocate one late visual block and one visual rank into the
            # Query-facing heads. This funds a 2G+2I object core while keeping
            # the final persistent parameter count within SAMCL's budget.
            ranks = {"visual": 4, "semantic": 6, "box": 5, "mask": 5}
        else:
            ratios = {"visual": 5 / 16, "semantic": 4 / 16,
                      "box": 3 / 16, "mask": 3 / 16}
            ranks = {group: max(1, int(round(rank * ratio)))
                     for group, ratio in ratios.items()}
    else:
        missing = set(FOCUS_TARGET_GROUPS) - set(focus_group_ranks)
        if missing:
            raise ValueError(
                f"object-coupled ranks are missing groups: {sorted(missing)}")
        ranks = {group: int(focus_group_ranks[group])
                 for group in FOCUS_TARGET_GROUPS}
    if object_core_rank < 1:
        raise ValueError("object-core rank must be positive")
    need_object_core = lora_scope != "encoder"
    if need_object_core:
        max_core = min(ranks[group] for group in ("semantic", "box", "mask")) - 1
        if max_core < 1:
            raise ValueError("semantic/box/mask ranks must exceed the object-core rank")
        if cqsm_enabled and max_core < 2:
            raise ValueError("CQSM private allocation requires the original rank 2 reservation")
        if not cqsm_enabled and object_core_rank > max_core:
            object_core_rank = max_core
        if general_object_rank > object_core_rank:
            general_object_rank = object_core_rank

    bbox_sources = [(name, module) for name, module in grouped["box"]
                    if name == "transformer.decoder.bbox_embed.layers.0"]
    if need_object_core and not bbox_sources:
        raise RuntimeError("could not locate the SAM 3 Query-facing box head")
    object_width = bbox_sources[0][1].in_features if bbox_sources else source_module.in_features
    def make_object_core() -> TaskObjectCore:
        if cqsm_enabled:
            return CovarianceQuerySubspaceMemory(
                object_width, object_core_rank, num_tasks, basis=cqsm_basis,
                energy_calibration=(cqsm_restore_legacy if cqsm_energy_calibration is None
                                    else cqsm_restore_legacy and cqsm_energy_calibration),
                calibration_lambda=energy_calibration_lambda,
                allow_experimental_rank=cqsm_memory_rank != 2)
        if semantic_protected:
            return SemanticProtectedObjectCore(
                object_width, object_core_rank, num_tasks,
                general_rank=general_object_rank,
                calibration_lambda=energy_calibration_lambda)
        if energy_decomposed:
            return EnergyDecomposedObjectCore(
                object_width, object_core_rank, num_tasks,
                general_rank=general_object_rank,
                calibration_lambda=energy_calibration_lambda,
                random_directions=energy_ablation == "random_core")
        return TaskObjectCore(
            object_width, object_core_rank, num_tasks,
            orthogonality_weight=focus_orthogonality_weight)

    adapter_cls = (EnergyObjectSubspaceLoRALinear
                   if energy_decomposed or semantic_protected or not need_object_core
                   else ObjectCoupledTaskLoRALinear)
    if cqsm_enabled:
        adapter_cls = CQSMDualLoRALinear
    object_core = (None if not need_object_core or independent_object_core
                   else make_object_core())

    core_patterns = (
        re.compile(r"^transformer\.decoder\.layers\.\d+\.linear1$"),
        re.compile(r"^transformer\.decoder\.bbox_embed\.layers\.0$"),
        re.compile(r"^segmentation_head\.mask_predictor\.mask_embed\.layers\.0$"),
        re.compile(r"^dot_prod_scoring\.hs_proj$"),
    )

    def uses_core(name: str, module: nn.Linear) -> bool:
        return (need_object_core and module.in_features == object_width and
                any(pattern.fullmatch(name) for pattern in core_patterns))

    def object_core_family(name: str) -> str:
        if re.fullmatch(r"transformer\.decoder\.layers\.\d+\.linear1", name):
            return "query_ffn"
        if name == "transformer.decoder.bbox_embed.layers.0":
            return "box_head"
        if name == "segmentation_head.mask_predictor.mask_embed.layers.0":
            return "mask_head"
        if name == "dot_prod_scoring.hs_proj":
            return "open_vocabulary_head"
        raise RuntimeError(f"Query-facing module has no independent core family: {name}")

    remaining_names = {name for matches in grouped.values() for name, _ in matches}
    if need_object_core and lora_scope == "all" and source_name in remaining_names and ".attn.qkv" in source_name:
        object_core_source_name = source_name
    elif need_object_core and bbox_sources:
        object_core_source_name = bbox_sources[0][0]
    else:
        object_core_source_name = None
    independent_core_sources: dict[str, str] = {}
    independent_cores: dict[str, TaskObjectCore] = {}
    if independent_object_core:
        for matches in grouped.values():
            for name, module in matches:
                if uses_core(name, module):
                    family = object_core_family(name)
                    independent_core_sources.setdefault(family, name)
        independent_cores = {
            family: make_object_core() for family in independent_core_sources
        }
        if set(independent_cores) != {
                "query_ffn", "box_head", "mask_head", "open_vocabulary_head"}:
            raise RuntimeError(
                "independent Query bases require Query FFN, box, mask, and OV heads")
    adapter_names = [name for matches in grouped.values() for name, _ in matches]
    router_source_name = (
        source_name if source_name in adapter_names else
        (adapter_names[0] if adapter_names else None))
    use_compositional = compositional_router or not (energy_decomposed or semantic_protected)
    router_cls = (CompositionalPromptMLPTaskRouter if use_compositional
                  else PromptMLPTaskRouter)
    router = router_cls(
        source_module.in_features, num_tasks, embeddings_per_task=300,
        routing_block=routing_block)
    model.requires_grad_(False)
    replaced: list[str] = []
    module_groups: dict[str, list[str]] = {}
    core_modules: list[str] = []
    for group, matches in grouped.items():
        module_groups[group] = []
        for name, module in matches:
            coupled = uses_core(name, module)
            core_family = object_core_family(name) if coupled else None
            adapter_core = (independent_cores[core_family]
                            if independent_object_core and core_family is not None
                            else object_core)
            # Sweep only memory capacity: preserve every private tensor and alpha/rank scale.
            private_reservation = 2 if cqsm_enabled else object_core_rank
            private_rank = ranks[group] - private_reservation if coupled else ranks[group]
            cqsm_kwargs = {}
            if cqsm_enabled:
                # Residual mode adds Global while preserving every private map.
                # Prefix mode remains available to reproduce schema-3 runs.
                in_prefix = (group == "visual" and _visual_block_id(name) in prefix_blocks)
                if cqsm_shared_layout in {"visual_prefix", "visual_residual"}:
                    shared_here = cqsm_shared and in_prefix and name.endswith(".attn.qkv")
                    private_here = cqsm_private and (cqsm_shared_layout == "visual_residual" or not in_prefix)
                else:
                    shared_here = cqsm_shared and name.startswith((
                        "transformer.decoder.", "dot_prod_scoring.", "segmentation_head."))
                    private_here = cqsm_private
                cqsm_kwargs = {"shared_enabled": shared_here,
                               "shared_rank": cqsm_global_rank,
                               "shared_gating": cqsm_global_gating,
                               "shared_gate_init": cqsm_global_gate_init,
                               "shared_qv_only": cqsm_shared_layout in {"visual_prefix", "visual_residual"},
                               "private_enabled": private_here,
                               "bank_mode": cqsm_bank_mode,
                               "restore_legacy": cqsm_restore_legacy}
            adapter = adapter_cls(
                module, rank=private_rank, alpha=alpha, num_tasks=num_tasks,
                function_group=group, effective_rank=ranks[group],
                object_core=adapter_core, uses_object_core=coupled,
                orthogonality_weight=focus_orthogonality_weight, **cqsm_kwargs)
            adapter.attach_anchor_router(router, source=name == router_source_name)
            if adapter_core is not None:
                core_source = (
                    name == independent_core_sources[core_family]
                    if independent_object_core and core_family is not None
                    else name == object_core_source_name)
                adapter.attach_object_core(
                    adapter_core, source=core_source)
            adapter.to(device=module.weight.device)
            _replace_linear(model, name, adapter)
            replaced.append(name)
            module_groups[group].append(name)
            if coupled:
                core_modules.append(name)

    routing_module_name = f"backbone.vision_backbone.trunk.blocks.{routing_block}"
    routing_module = model.get_submodule(routing_module_name)

    def observe_visual(_module, args):
        if _SHARED_ONLY_FORWARD.get():
            return
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("object-coupled routing block needs an activation")
        router.prepare_visual(args[0], training=_module.training)

    handle = routing_module.register_forward_pre_hook(observe_visual)
    object.__setattr__(router, "_visual_hook_handle", handle)
    module_groups["selector"] = [f"{routing_module_name}::<input_hook>"]
    module_groups["object_core"] = core_modules
    if cqsm_enabled:
        module_groups["global_shared"] = [
            name for name in replaced
            if isinstance(model.get_submodule(name), CQSMDualLoRALinear)
            and model.get_submodule(name).shared_enabled]
    if semantic_protected:
        if not isinstance(scorer, nn.Linear):
            raise RuntimeError("semantic protection needs SAM 3's frozen Query projection")
        prompt_projection_name = "dot_prod_scoring.prompt_proj"
        prompt_projection = model.get_submodule(prompt_projection_name)

        def observe_text(_module, _args, value):
            if not torch.is_tensor(value):
                raise RuntimeError("semantic prompt projection must return a tensor")
            object_core.observe_semantic_text(value, scorer.weight)

        semantic_handle = prompt_projection.register_forward_hook(observe_text)
        object.__setattr__(object_core, "_semantic_hook_handle", semantic_handle)
        module_groups["semantic_anchor"] = [
            f"{prompt_projection_name}::<frozen_output_hook>"]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    persistent = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, ObjectCoupledTaskLoRALinear)
        for p in module.adapter_parameters()
    )
    total = sum(p.numel() for p in model.parameters())
    if cqsm_enabled:
        method_name = "cqsm_dual_lora"
    elif semantic_protected:
        method_name = "semantic_protected_subspace_lora"
    elif lora_scope == "encoder":
        method_name = "energy_visual_mlp_encoder"
    elif lora_scope == "decoder":
        method_name = "energy_visual_mlp_decoder"
    elif visual_insert == "mlp_only":
        method_name = "energy_visual_mlp_mlp_only"
    elif visual_insert == "early_mlp":
        method_name = "energy_visual_mlp_early_mlp"
    elif energy_ablation == "random_core":
        method_name = "energy_visual_mlp_random_core"
    elif energy_ablation == "general_only":
        method_name = ("energy_visual_mlp_independent_core"
                       if independent_object_core
                       else "energy_visual_mlp_general_only")
    elif energy_ablation == "isolated_only":
        method_name = "energy_visual_mlp_isolated_only"
    elif visual_mlp:
        method_name = "energy_visual_mlp_lora"
    elif compositional_router:
        method_name = "energy_object_compositional_lora"
    elif energy_decomposed:
        method_name = "energy_object_subspace_lora"
    else:
        method_name = "object_coupled_capacity_lora"
    return InjectionReport(method_name, replaced, trainable, total,
                           persistent, module_groups, ranks)


def inject_continual_lora(model: nn.Module, method: str, rank: int = 16, alpha: float = 8,
                          num_tasks: int = 5, target_patterns=DEFAULT_TARGETS, **kwargs) -> InjectionReport:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose from {sorted(METHODS)}")
    if method == "cqsm_dual_lora":
        # The default preserves the original release; restoration is explicit.
        if not kwargs.get("cqsm_restore_legacy", False):
            kwargs.pop("energy_calibration_lambda", None)
            kwargs["focus_orthogonality_weight"] = 0.0
        kwargs["object_core_rank"] = int(kwargs.get("cqsm_memory_rank", 2))
        kwargs["general_object_rank"] = kwargs["object_core_rank"]
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True,
            visual_mlp=True, energy_ablation="general_only", cqsm_enabled=True,
            **kwargs)
    if method == "focus_lora":
        return _inject_focus_lora(model, rank, alpha, num_tasks, **kwargs)
    if method == "focus_lora_v2":
        return _inject_focus_lora_v2(model, rank, alpha, num_tasks, **kwargs)
    if method == "focus_lora_v3":
        return _inject_focus_lora_v3(model, rank, alpha, num_tasks, **kwargs)
    if method == "task_decoupled_lora":
        return _inject_task_decoupled_lora(model, rank, alpha, num_tasks, **kwargs)
    if method == "task_decoupled_mlp":
        return _inject_task_decoupled_mlp(model, rank, alpha, num_tasks, **kwargs)
    if method == "compositional_anchor_lora":
        return _inject_task_decoupled_mlp(
            model, rank, alpha, num_tasks, compositional_anchor=True, **kwargs)
    if method == "object_consistent_lora":
        return _inject_task_decoupled_mlp(
            model, rank, alpha, num_tasks, object_consistent=True, **kwargs)
    if method == "shared_core_residual_lora":
        return _inject_task_decoupled_mlp(
            model, rank, alpha, num_tasks, shared_core_residual=True, **kwargs)
    if method == "unified_object_lora":
        return _inject_task_decoupled_mlp(
            model, rank, alpha, num_tasks, unified_object=True, **kwargs)
    if method == "consensus_complement_lora":
        return _inject_task_decoupled_mlp(
            model, rank, alpha, num_tasks, consensus_complement=True, **kwargs)
    if method == "samcl_slora":
        return _inject_samcl_slora(model, rank, alpha, num_tasks)
    if method == "independent_capacity_lora":
        return _inject_independent_capacity_lora(
            model, rank, alpha, num_tasks, **kwargs)
    if method == "independent_visual_mlp_lora":
        return _inject_independent_capacity_lora(
            model, rank, alpha, num_tasks, visual_mlp=True, **kwargs)
    if method == "object_coupled_capacity_lora":
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, **kwargs)
    if method == "energy_object_subspace_lora":
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True, **kwargs)
    if method == "energy_object_compositional_lora":
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True,
            compositional_router=True, **kwargs)
    if method == "energy_visual_mlp_lora":
        kwargs.setdefault("general_object_rank", 2)
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True,
            visual_mlp=True, **kwargs)
    if method == "energy_visual_mlp_general_only":
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True,
            visual_mlp=True, energy_ablation="general_only", **kwargs)
    if method == "energy_visual_mlp_independent_core":
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True,
            visual_mlp=True, energy_ablation="general_only",
            independent_object_core=True, **kwargs)
    if method == "energy_visual_mlp_isolated_only":
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True,
            visual_mlp=True, energy_ablation="isolated_only", **kwargs)
    if method == "energy_visual_mlp_encoder":
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True,
            visual_mlp=True, lora_scope="encoder", **kwargs)
    if method == "energy_visual_mlp_decoder":
        kwargs.setdefault("general_object_rank", 2)
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True,
            visual_mlp=True, lora_scope="decoder", **kwargs)
    if method == "energy_visual_mlp_mlp_only":
        kwargs.setdefault("general_object_rank", 2)
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True,
            visual_mlp=True, visual_insert="mlp_only", **kwargs)
    if method == "energy_visual_mlp_early_mlp":
        kwargs.setdefault("general_object_rank", 2)
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True,
            visual_mlp=True, visual_insert="early_mlp", **kwargs)
    if method == "energy_visual_mlp_random_core":
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True,
            visual_mlp=True, energy_ablation="random_core", **kwargs)
    if method == "semantic_protected_subspace_lora":
        return _inject_object_coupled_capacity_lora(
            model, rank, alpha, num_tasks, energy_decomposed=True,
            semantic_protected=True, object_core_rank=4,
            general_object_rank=2, **kwargs)
    if method in {"inflora", "bilora", "rank1_expert", "rank1_expert_pm"} and rank < num_tasks:
        raise ValueError(f"{method} needs rank >= num_tasks for disjoint fixed-budget allocation")
    model.requires_grad_(False)
    matches = _linear_matches(model, target_patterns)
    cls = METHODS[method]
    replaced=[]
    for name, module in matches:
        adapter = cls(module, rank=rank, alpha=alpha, num_tasks=num_tasks, **kwargs)
        adapter.to(device=module.weight.device, dtype=module.weight.dtype)
        _replace_linear(model, name, adapter)
        replaced.append(name)
    if not replaced:
        raise RuntimeError("No SAM 3 linear layers matched LoRA target patterns")
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    persistent=sum(p.numel() for module in model.modules()
                   if isinstance(module, ContinualAdapterLinear)
                   for p in module.adapter_parameters())
    return InjectionReport(method,replaced,trainable,total,persistent)


class AdapterController:
    def __init__(self, model: nn.Module, method: str, ewc_lambda: float = 100.0):
        self.model, self.method, self.ewc_lambda = model, method, ewc_lambda
        self.ewc_means: dict[str, torch.Tensor] = {}
        self.ewc_fisher: dict[str, torch.Tensor] = {}

    def adapters(self):
        return [m for m in self.model.modules() if isinstance(m, ContinualAdapterLinear)]

    def focus_adapters(self, group: str | None = None) -> list[ContinualAdapterLinear]:
        focus_types = (FocusLoRALinear, ImportancePoolFocusLoRALinear,
                       CapacityPartitionFocusLoRALinear,
                       MultiAnchorTaskDecoupledLoRALinear,
                       MLPTaskDecoupledLoRALinear, SAMCLSLoRALinear,
                       IndependentCapacityTaskLoRALinear,
                       ObjectCoupledTaskLoRALinear)
        adapters = [m for m in self.model.modules() if isinstance(m, focus_types)]
        return adapters if group is None else [m for m in adapters if m.function_group == group]

    def focus_parameters(self, group: str | None = None,
                         *, trainable_only: bool = True) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for adapter in self.focus_adapters(group):
            for parameter in adapter.adapter_parameters():
                if not trainable_only or parameter.requires_grad:
                    params.append(parameter)
        return params

    def consensus_complement_parameters(self) -> list[
            tuple[nn.Parameter, torch.Tensor, torch.Tensor, torch.Tensor]]:
        entries = []
        for adapter in self.focus_adapters():
            if isinstance(adapter, ConsensusComplementLoRALinear):
                entries.extend(adapter.role_parameter_masks())
        return entries

    def begin_task(self, task_id: int) -> None:
        router = self.anchor_router()
        if router is not None:
            router.begin_task(task_id)
        for m in self.adapters(): m.begin_task(task_id)

    def anchor_router(self) -> MultiAnchorTaskRouter | PromptMLPTaskRouter | None:
        for module in self.model.modules():
            if isinstance(module, (MultiAnchorTaskRouter, PromptMLPTaskRouter)):
                return module
        return None

    def energy_object_core(self) -> EnergyDecomposedObjectCore | None:
        cores = self.energy_object_cores()
        return cores[0] if cores else None

    def energy_object_cores(self) -> list[EnergyDecomposedObjectCore]:
        cores: list[EnergyDecomposedObjectCore] = []
        seen: set[int] = set()
        for module in self.model.modules():
            if isinstance(module, EnergyDecomposedObjectCore) and id(module) not in seen:
                cores.append(module)
                seen.add(id(module))
        return cores

    def start_energy_covariance_collection(self) -> None:
        cores = self.energy_object_cores()
        if not cores:
            raise RuntimeError("energy object core was not injected")
        for core in cores:
            core.start_collection()

    def finalize_energy_object_subspaces(
            self, task_id: int) -> dict[str, object]:
        cores = self.energy_object_cores()
        if not cores:
            raise RuntimeError("energy object core was not injected")
        reports = [core.finalize_subspaces(task_id) for core in cores]
        if len(reports) == 1:
            return reports[0]
        numeric_keys = set.intersection(*(
            {key for key, value in report.items()
             if isinstance(value, (int, float))} for report in reports))
        aggregate: dict[str, object] = {
            "object_core_count": len(cores),
            "per_core": reports,
        }
        for key in sorted(numeric_keys):
            aggregate[f"mean_{key}"] = sum(
                float(report[key]) for report in reports) / len(reports)
        return aggregate

    def synchronize_anchor_statistics(self) -> None:
        router = self.anchor_router()
        if router is not None:
            router.synchronize_active()

    def begin_anchor_batch(self) -> None:
        router = self.anchor_router()
        if router is not None:
            router.begin_batch()

    def selector_loss(self) -> torch.Tensor:
        router = self.anchor_router()
        if router is not None and hasattr(router, "selector_loss"):
            return router.selector_loss()
        params = [p for p in self.model.parameters() if p.requires_grad]
        return params[0].sum() * 0.0 if params else torch.tensor(0.0)

    def selector_parameters(self) -> list[nn.Parameter]:
        router = self.anchor_router()
        return ([p for p in router.selector.parameters() if p.requires_grad]
                if isinstance(router, PromptMLPTaskRouter) else [])

    def fit_selector(self) -> dict[str, float | int]:
        router = self.anchor_router()
        if isinstance(router, PromptMLPTaskRouter):
            return router.fit_selector()
        return {}

    def end_task(self) -> dict:
        reports = [report for m in self.adapters() if (report := m.end_task()) is not None]
        if not reports:
            return {}
        by_group: dict[str, list[float]] = {}
        for report in reports:
            by_group.setdefault(str(report["group"]), []).append(
                float(report["relative_compression_error"]))
        summary = {
            "layers": len(reports),
            "mean_relative_compression_error": sum(
                float(x["relative_compression_error"]) for x in reports) / len(reports),
            "group_relative_compression_error": {
                group: sum(values) / len(values) for group, values in by_group.items()
            },
        }
        if all("allocated_rank" in report for report in reports):
            summary.update(
                mean_allocated_rank=sum(int(x["allocated_rank"]) for x in reports) / len(reports),
                mean_active_rank=sum(int(x["active_rank"]) for x in reports) / len(reports),
            )
        if all("recycled_rank" in report for report in reports):
            summary["recycled_rank_atoms"] = sum(
                int(x["recycled_rank"]) for x in reports)
        if all("mean_importance" in report for report in reports):
            summary["mean_rank_importance"] = sum(
                float(x["mean_importance"]) for x in reports) / len(reports)
        if all("remaining_rank" in report for report in reports):
            summary["remaining_rank_atoms"] = sum(
                int(x["remaining_rank"]) for x in reports)
        return summary

    def reset_plastic_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        if self.method != "focus_lora":
            return
        for adapter in self.focus_adapters():
            for parameter in adapter.plastic_parameters():
                optimizer.state.pop(parameter, None)

    def accumulate_focus_importance(self) -> None:
        if self.method != "focus_lora_v2":
            return
        for adapter in self.focus_adapters():
            adapter.accumulate_importance()

    def restore_focus_protected_parameters(self) -> None:
        if self.method not in {"focus_lora_v2", "focus_lora_v3", "task_decoupled_lora",
                               "task_decoupled_mlp", "compositional_anchor_lora",
                               "object_consistent_lora",
                               "shared_core_residual_lora", "unified_object_lora",
                               "consensus_complement_lora", "samcl_slora",
                               "independent_capacity_lora",
                               "independent_visual_mlp_lora",
                               "object_coupled_capacity_lora",
                               "energy_visual_mlp_encoder",
                               *ENERGY_SUBSPACE_METHODS, *CQSM_METHODS}:
            return
        for adapter in self.focus_adapters():
            adapter.restore_protected_parameters()
        router = self.anchor_router()
        if isinstance(router, CompositionalPromptMLPTaskRouter):
            router.restore_protected_composition()

    def reset_focus_active_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        if self.method not in {"focus_lora_v2", "focus_lora_v3", "task_decoupled_lora",
                               "task_decoupled_mlp", "compositional_anchor_lora",
                               "object_consistent_lora",
                               "shared_core_residual_lora", "unified_object_lora",
                               "consensus_complement_lora", "samcl_slora",
                               "independent_capacity_lora",
                               "independent_visual_mlp_lora",
                               "object_coupled_capacity_lora",
                               "energy_visual_mlp_encoder",
                               *ENERGY_SUBSPACE_METHODS, *CQSM_METHODS}:
            return
        for adapter in self.focus_adapters():
            adapter.reset_active_optimizer_state(optimizer)

    def restore_focus_trainability(self) -> None:
        for adapter in self.focus_adapters():
            adapter.restore_trainability()

    def set_focus_capacity_routing(self, mode: str, *, blocks: int = 1,
                                   oracle_stage: int | None = None) -> None:
        if self.method != "focus_lora_v3":
            raise ValueError("capacity routing overrides require focus_lora_v3")
        for adapter in self.focus_adapters():
            adapter.set_routing(mode, blocks=blocks, oracle_stage=oracle_stage)

    def regularization_loss(self) -> torch.Tensor:
        params=[p for p in self.model.parameters() if p.requires_grad]
        loss=params[0].new_zeros(()) if params else torch.tensor(0.0)
        for m in self.adapters(): loss = loss + m.regularization_loss()
        if self.method == "lora_ewc" and self.ewc_means:
            current=dict(self.model.named_parameters())
            for name, mean in self.ewc_means.items():
                if name in current:
                    loss = loss + self.ewc_lambda * (self.ewc_fisher[name] * (current[name]-mean).square()).sum()
        return loss

    @torch.no_grad()
    def consolidate_ewc(self, fisher: dict[str, torch.Tensor]) -> None:
        if self.method != "lora_ewc": return
        for name,p in self.model.named_parameters():
            if p.requires_grad and name in fisher:
                f=fisher[name].detach().clone()
                if name in self.ewc_fisher: f=f+self.ewc_fisher[name]
                self.ewc_fisher[name]=f
                self.ewc_means[name]=p.detach().clone()

    def state_dict(self) -> dict:
        return {"method":self.method,"ewc_lambda":self.ewc_lambda,
                "ewc_means":self.ewc_means,"ewc_fisher":self.ewc_fisher}

    def load_state_dict(self, state: dict) -> None:
        self.ewc_means=state.get("ewc_means",{}); self.ewc_fisher=state.get("ewc_fisher",{})
