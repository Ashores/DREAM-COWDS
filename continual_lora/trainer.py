"""SAM 3 continual detection+instance-segmentation training utilities."""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .methods import (AdapterController, ConsensusComplementLoRALinear, CQSM_METHODS, CQSMDualLoRALinear,
                      ContinualAdapterLinear, ENERGY_SUBSPACE_METHODS,
                      INDEPENDENT_CAPACITY_METHODS, LORA_LOCATION_METHODS,
                      FocusLoRALinear,
                      ObjectConsistentSkillRouter, inject_continual_lora)


FOCUS_METHODS = {"focus_lora", "focus_lora_v2", "focus_lora_v3",
                 "task_decoupled_lora", "task_decoupled_mlp",
                 "compositional_anchor_lora",
                 "object_consistent_lora", "shared_core_residual_lora",
                 "unified_object_lora", "consensus_complement_lora",
                 "independent_capacity_lora", "independent_visual_mlp_lora",
                 "object_coupled_capacity_lora",
                 *LORA_LOCATION_METHODS,
                 *ENERGY_SUBSPACE_METHODS}
OBJECT_ROUTER_METHODS = {"object_consistent_lora", "shared_core_residual_lora",
                         "unified_object_lora", "consensus_complement_lora"}
SAMCL_METHODS = {"samcl_slora"}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    split: str = "train"


DEFAULT_STREAM = (
    TaskSpec("PASCAL_VOC_2012"),
    TaskSpec("HRSID"),
    TaskSpec("Kvasir-Instrument"),
    TaskSpec("Cityscapes"),
    TaskSpec("TrashCan-Instance"),
)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def enable_lora_training_compatibility() -> None:
    """Route ViT MLP through ``nn.Module.forward`` so LoRA wrappers work.

    SAM 3's fused ``addmm_act`` reads ``linear.weight`` / ``linear.bias``
    directly. Continual adapters replace those ``nn.Linear`` modules and do
    not expose ``.bias``, so both training and evaluation must call the
    wrapped module instead of the fused kernel.
    """
    import sam3.model.vitdet as vitdet
    import sam3.perflib.fused as fused

    def module_addmm_act(activation, linear, x):
        return activation()(linear(x))

    vitdet.addmm_act = module_addmm_act
    fused.addmm_act = module_addmm_act


def build_loader(data_root: str | Path, task: TaskSpec, resolution: int = 1008,
                 batch_size: int = 1, workers: int = 0, category_chunk_size: int = 16,
                 rank: int = 0, world_size: int = 1, limit_ids: int | None = None):
    from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
    from sam3.train.transforms.basic_for_api import (NormalizeAPI, PadToSizeAPI,
        RandomResizeAPI, ToTensorAPI)
    from sam3.train.transforms.segmentation import DecodeRle

    root = Path(data_root) / task.name
    ann = root / "annotations" / f"instances_{task.split}.json"
    if not ann.is_file():
        raise FileNotFoundError(f"Missing prepared COCO annotation: {ann}")
    if limit_ids is not None:
        annotation_meta=json.loads(ann.read_text(encoding="utf-8"))
        chunks=max(math.ceil(len(annotation_meta["categories"])/category_chunk_size),1)
        limit_ids=limit_ids*chunks
    transforms = [
        DecodeRle(),
        RandomResizeAPI([resolution], consistent_transform=False,
                        max_size=resolution, square=True),
        PadToSizeAPI(resolution, consistent_transform=False, bottom_right=True),
        ToTensorAPI(),
        NormalizeAPI([0.5] * 3, [0.5] * 3),
    ]
    loader_cls = partial(COCO_FROM_JSON, include_negatives=True,
                         category_chunk_size=category_chunk_size)
    dataset = Sam3ImageDataset(
        img_folder=str(root), ann_file=str(ann), transforms=transforms,
        max_ann_per_img=200, multiplier=1, training=True,
        load_segmentation=True, max_train_queries=category_chunk_size,
        use_caching=False, coco_json_loader=loader_cls,
        limit_ids=limit_ids,
    )
    collate = partial(collate_fn_api, dict_key="all", with_seg_masks=True, repeats=0)
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=True, seed=42) if world_size > 1 else None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None,
                      sampler=sampler, num_workers=workers,
                      collate_fn=collate, pin_memory=True, drop_last=False)


def replay_memory_profile(data_root: str | Path, stream, images_per_task: int) -> dict:
    profile={"images_per_task":images_per_task,"datasets":{},"total_images":0,"total_bytes":0}
    for task in stream:
        root=Path(data_root)/task.name
        ann=root/"annotations"/f"instances_{task.split}.json"
        data=json.loads(ann.read_text(encoding="utf-8"))
        images=sorted(data["images"],key=lambda x:int(x["id"]))[:images_per_task]
        size=sum((root/x["file_name"]).stat().st_size for x in images)
        profile["datasets"][task.name]={"images":len(images),"bytes":size}
        profile["total_images"]+=len(images); profile["total_bytes"]+=size
    return profile


def build_joint_loss(device: torch.device):
    from .numerics import enable_stable_presence_loss
    enable_stable_presence_loss()
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
    matcher = BinaryHungarianMatcherV2(cost_class=2, cost_bbox=5, cost_giou=2,
                                       focal=True, alpha=.25, gamma=2, stable=False)
    o2m = BinaryOneToManyMatcher(alpha=.3, threshold=.4, topk=4)
    losses = [
        Boxes(weight_dict={"loss_bbox": 5., "loss_giou": 2.}),
        IABCEMdetr(weight_dict={"loss_ce": 20., "presence_loss": 20.},
                   pos_weight=10., alpha=.25, gamma=2, use_presence=True,
                   pos_focal=False, pad_n_queries=200, pad_scale_pos=1., weak_loss=False),
        Masks(weight_dict={"loss_mask": 200., "loss_dice": 10.},
              focal_alpha=.25, focal_gamma=2., compute_aux=False),
    ]
    return Sam3LossWrapper(loss_fns_find=losses, normalization="local", matcher=matcher,
                           o2m_matcher=o2m, o2m_weight=2.,
                           use_o2m_matcher_on_o2m_aux=False,
                           scale_by_find_batch_size=True).to(device)


class UnifiedObjectBridge:
    """Training-only Text--Query--Box--Mask object representation bridge.

    SAM 3 predicts masks by comparing decoder queries with 256-D instance
    pixel embeddings. This bridge pools the same pixel embedding once inside
    the matched query's box and once under its soft mask, then aligns all three
    views. In ``semantic_query_teacher`` mode, the native open-vocabulary
    scorer also supplies projected text and query views.  The text view is a
    stop-gradient semantic anchor, preventing geometry alignment from erasing
    the language-sensitive part of the object query.  The bridge owns no
    parameter and its hooks are inactive outside the main training forward, so
    checkpoints and inference remain unchanged.
    """

    def __init__(self, model: torch.nn.Module, *, max_instances: int = 32,
                 temperature: float = 0.1, mode: str = "symmetric"):
        raw_model = (model.module if isinstance(model, DistributedDataParallel)
                     else model)
        segmentation_head = getattr(raw_model, "segmentation_head", None)
        instance_head = getattr(segmentation_head, "instance_seg_head", None)
        if instance_head is None:
            raise RuntimeError("unified object learning needs the SAM 3 instance pixel head")
        self.max_instances = int(max_instances)
        self.temperature = float(temperature)
        if mode not in {"symmetric", "query_teacher", "semantic_query_teacher"}:
            raise ValueError(f"unknown unified object mode: {mode}")
        self.mode = mode
        self._captures: list[torch.Tensor] = []
        self._semantic_query_captures: list[torch.Tensor] = []
        self._semantic_text_captures: list[torch.Tensor] = []
        self._collecting = False
        self._hook = instance_head.register_forward_hook(self._observe)
        self._semantic_hooks = []
        if mode == "semantic_query_teacher":
            scorer = getattr(raw_model, "dot_prod_scoring", None)
            query_projection = getattr(scorer, "hs_proj", None)
            text_projection = getattr(scorer, "prompt_proj", None)
            if query_projection is None or text_projection is None:
                raise RuntimeError(
                    "semantic object learning needs SAM 3's native dot-product scorer")
            self._semantic_hooks = [
                query_projection.register_forward_hook(self._observe_semantic_query),
                text_projection.register_forward_hook(self._observe_semantic_text),
            ]

    def _observe(self, _module, _args, value: torch.Tensor) -> None:
        if self._collecting:
            self._captures.append(value)

    def _observe_semantic_query(self, _module, _args, value: torch.Tensor) -> None:
        if self._collecting:
            self._semantic_query_captures.append(value)

    def _observe_semantic_text(self, _module, _args, value: torch.Tensor) -> None:
        if self._collecting:
            self._semantic_text_captures.append(value)

    def begin_batch(self) -> None:
        self._captures.clear()
        self._semantic_query_captures.clear()
        self._semantic_text_captures.clear()
        self._collecting = True

    @staticmethod
    def _steps(output) -> list[dict]:
        return [prediction for stage in output.output for prediction in stage]

    @staticmethod
    def _zero(steps: list[dict]) -> torch.Tensor:
        return steps[0]["queries"].sum().float() * 0.0

    def loss(self, output) -> tuple[torch.Tensor, dict[str, float]]:
        self._collecting = False
        steps = self._steps(output)
        captures = self._captures[:]
        semantic_query_captures = self._semantic_query_captures[:]
        semantic_text_captures = self._semantic_text_captures[:]
        self._captures.clear()
        self._semantic_query_captures.clear()
        self._semantic_text_captures.clear()
        if len(steps) != len(captures):
            raise RuntimeError(
                f"object bridge observed {len(captures)} pixel maps for {len(steps)} predictions")

        semantic_mode = self.mode == "semantic_query_teacher"
        if semantic_mode and (len(steps) != len(semantic_query_captures) or
                              len(steps) != len(semantic_text_captures)):
            raise RuntimeError(
                "semantic object bridge observed "
                f"{len(semantic_query_captures)} query and "
                f"{len(semantic_text_captures)} text projections for "
                f"{len(steps)} predictions")

        queries, box_views, mask_views, reliabilities = [], [], [], []
        semantic_alignments, semantic_contrastives = [], []
        for step_id, (prediction, pixel_embedding) in enumerate(zip(steps, captures)):
            indices = prediction.get("indices")
            if not isinstance(indices, (tuple, list)) or len(indices) < 2:
                continue
            batch_index, query_index = indices[:2]
            if batch_index.numel() == 0:
                continue
            batch_index = batch_index.to(pixel_embedding.device, dtype=torch.long)
            query_index = query_index.to(pixel_embedding.device, dtype=torch.long)
            confidence = prediction["pred_logits"][batch_index, query_index, 0].sigmoid()
            if len(query_index) > self.max_instances:
                keep = confidence.topk(self.max_instances, sorted=False).indices
                batch_index, query_index = batch_index[keep], query_index[keep]
                confidence = confidence[keep]
            query = prediction["queries"][batch_index, query_index].float()
            boxes = prediction["pred_boxes"][batch_index, query_index].detach().float()
            masks = prediction["pred_masks"][batch_index, query_index].detach().sigmoid()
            height, width = masks.shape[-2:]
            grid_y = ((torch.arange(height, device=masks.device, dtype=torch.float32) + .5)
                      / height).view(1, height, 1)
            grid_x = ((torch.arange(width, device=masks.device, dtype=torch.float32) + .5)
                      / width).view(1, 1, width)
            cx, cy, bw, bh = boxes.unbind(-1)
            x1, x2 = (cx - bw / 2).view(-1, 1, 1), (cx + bw / 2).view(-1, 1, 1)
            y1, y2 = (cy - bh / 2).view(-1, 1, 1), (cy + bh / 2).view(-1, 1, 1)
            box_weight = ((grid_x >= x1) & (grid_x <= x2) &
                          (grid_y >= y1) & (grid_y <= y2)).to(masks.dtype)

            pooled_box = torch.empty((len(query), pixel_embedding.shape[1]),
                                     device=query.device, dtype=torch.float32)
            pooled_mask = torch.empty_like(pooled_box)
            for image_index in batch_index.unique(sorted=True):
                select = torch.where(batch_index == image_index)[0]
                pixels = pixel_embedding[int(image_index)].to(
                    dtype=box_weight.dtype)
                box_local, mask_local = box_weight[select], masks[select]
                pooled_box[select] = (torch.einsum("nhw,chw->nc", box_local, pixels) /
                                      box_local.sum((-2, -1), keepdim=False).clamp_min(1.)[:, None])
                pooled_mask[select] = (torch.einsum("nhw,chw->nc", mask_local, pixels) /
                                       mask_local.sum((-2, -1), keepdim=False).clamp_min(1.)[:, None])

            intersection = (masks * box_weight).sum((-2, -1))
            union = (masks + box_weight - masks * box_weight).sum((-2, -1)).clamp_min(1e-6)
            agreement = (intersection / union).detach()
            reliability = .5 + .5 * (confidence.detach() * agreement).clamp_min(0).sqrt()
            queries.append(F.normalize(query, dim=-1))
            box_views.append(F.normalize(pooled_box, dim=-1))
            mask_views.append(F.normalize(pooled_mask, dim=-1))
            reliabilities.append(reliability.float())

            if semantic_mode:
                # These are the exact projections used by SAM 3's native
                # open-vocabulary dot-product scorer during this forward.  Text
                # is detached as a fixed language anchor; gradients update the
                # matched object query and its semantic LoRA path only.
                projected_queries = semantic_query_captures[step_id]
                projected_text = semantic_text_captures[step_id]
                if projected_queries.dim() != 4 or projected_text.dim() != 2:
                    raise RuntimeError(
                        "unexpected semantic projection shapes: "
                        f"query={tuple(projected_queries.shape)}, "
                        f"text={tuple(projected_text.shape)}")
                semantic_query = F.normalize(
                    projected_queries[-1, batch_index, query_index].float(), dim=-1)
                semantic_text_bank = F.normalize(projected_text.detach().float(), dim=-1)
                positive_text = semantic_text_bank[batch_index]
                semantic_alignments.append(1 - (semantic_query * positive_text).sum(-1))
                if len(semantic_text_bank) > 1:
                    logits = semantic_query @ semantic_text_bank.T / self.temperature
                    semantic_contrastives.append(
                        F.cross_entropy(logits, batch_index, reduction="none"))
                else:
                    semantic_contrastives.append(
                        semantic_query.new_zeros(len(semantic_query)))

        if not queries:
            zero = self._zero(steps)
            return zero, {"instances": 0.0, "alignment": 0.0,
                          "contrastive": 0.0, "semantic_alignment": 0.0,
                          "semantic_contrastive": 0.0, "reliability": 0.0}
        query = torch.cat(queries)
        box_view = torch.cat(box_views)
        mask_view = torch.cat(mask_views)
        reliability = torch.cat(reliabilities)

        target_query = F.normalize(box_view.detach() + mask_view.detach(), dim=-1)
        if self.mode in {"query_teacher", "semantic_query_teacher"}:
            # Fine-grained mask pixels and coarse box pixels jointly teach the
            # common decoder query. They are stop-gradient evidence, preventing
            # the box view from smoothing the mask representation itself.
            per_instance = 1 - (query * target_query).sum(-1)
        else:
            # Leave-one-view-out symmetric alignment retained as an ablation.
            target_box = F.normalize(query.detach() + mask_view.detach(), dim=-1)
            target_mask = F.normalize(query.detach() + box_view.detach(), dim=-1)
            per_instance = ((1 - (query * target_query).sum(-1)) +
                            (1 - (box_view * target_box).sum(-1)) +
                            (1 - (mask_view * target_mask).sum(-1))) / 3
        alignment = (per_instance * reliability).sum() / reliability.sum().clamp_min(1e-6)

        contrastive = alignment.new_zeros(())
        if len(query) > 1:
            labels = torch.arange(len(query), device=query.device)
            query_to_pixel = query @ target_query.T / self.temperature
            if self.mode in {"query_teacher", "semantic_query_teacher"}:
                contrastive = F.cross_entropy(query_to_pixel, labels)
            else:
                pixel_object = F.normalize((box_view + mask_view) * .5, dim=-1)
                pixel_to_query = pixel_object @ query.detach().T / self.temperature
                contrastive = .5 * (F.cross_entropy(query_to_pixel, labels) +
                                    F.cross_entropy(pixel_to_query, labels))
        semantic_alignment = alignment.new_zeros(())
        semantic_contrastive = alignment.new_zeros(())
        if semantic_mode:
            semantic_alignment_values = torch.cat(semantic_alignments)
            semantic_contrastive_values = torch.cat(semantic_contrastives)
            semantic_alignment = ((semantic_alignment_values * reliability).sum() /
                                  reliability.sum().clamp_min(1e-6))
            semantic_contrastive = ((semantic_contrastive_values * reliability).sum() /
                                    reliability.sum().clamp_min(1e-6))
        # Keep the original geometry objective intact.  The semantic anchor has
        # half its strength so it protects open-vocabulary identity without
        # dominating box/mask evidence or introducing another CLI hyperparameter.
        value = (alignment + .25 * contrastive +
                 .5 * semantic_alignment + .125 * semantic_contrastive)
        return value, {
            "instances": float(len(query)),
            "alignment": float(alignment.detach()),
            "contrastive": float(contrastive.detach()),
            "semantic_alignment": float(semantic_alignment.detach()),
            "semantic_contrastive": float(semantic_contrastive.detach()),
            "reliability": float(reliability.mean().detach()),
        }


def _cycle(loader: Iterable):
    epoch = 0
    while True:
        if isinstance(getattr(loader, "sampler", None), DistributedSampler):
            loader.sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


@torch.no_grad()
def _initialize_energy_object_subspaces(model, loader, device,
                                        controller: AdapterController,
                                        task_id: int, batches: int) -> dict:
    """Estimate object energy without retaining examples or Query tokens."""
    from sam3.model.utils.misc import copy_data_to_device
    router = controller.anchor_router()
    if router is None or not hasattr(router, "inference_oracle_stage"):
        raise RuntimeError("energy decomposition needs the shared MLP selector")
    previous_oracle = router.inference_oracle_stage
    was_training = model.training
    iterator = iter(loader)
    controller.start_energy_covariance_collection()
    model.eval()
    router.inference_oracle_stage = int(task_id)
    completed = 0
    try:
        for _ in range(batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if len(batch) != 1:
                raise ValueError(
                    f"Expected one collator stream, got keys {list(batch)}")
            controller.begin_anchor_batch()
            packed = copy_data_to_device(next(iter(batch.values())), device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                model(packed)
            completed += 1
        report = controller.finalize_energy_object_subspaces(task_id)
        report["covariance_batches_per_worker"] = completed
        return report
    finally:
        router.inference_oracle_stage = previous_oracle
        model.train(was_training)


def _runtime_adapter_state(model) -> dict:
    prefixes = {name for name, module in model.named_modules()
                if isinstance(module, ContinualAdapterLinear)}
    return {name: value.detach().clone() for name, value in model.state_dict().items()
            if any(name == p or name.startswith(p + ".") for p in prefixes)
            and ".base." not in name}


@torch.no_grad()
def _load_runtime_adapter_state(model, state: dict) -> None:
    model.load_state_dict(state, strict=False)


def _prediction_distillation(student, teacher) -> torch.Tensor:
    """Distill semantic, box and mask outputs from the previous stage."""
    loss = None
    weights = {"pred_logits": 1.0, "presence_logit_dec": 1.0,
               "pred_boxes": 2.0, "pred_masks": 2.0}
    student_steps = [stage[-1] for stage in student.output]
    teacher_steps = [stage[-1] for stage in teacher.output]
    for s, t in zip(student_steps, teacher_steps):
        for key, weight in weights.items():
            if key not in s or key not in t or s[key].shape != t[key].shape: continue
            sv, tv = s[key].float(), t[key].detach().float()
            if key in {"pred_logits", "presence_logit_dec", "pred_masks"}:
                value = torch.nn.functional.mse_loss(sv.sigmoid(), tv.sigmoid())
            else:
                value = torch.nn.functional.smooth_l1_loss(sv, tv)
            loss = value * weight if loss is None else loss + value * weight
    if loss is not None: return loss
    first = next(iter(student_steps[0].values()))
    return first.new_zeros(())


def _weighted_functional_losses(loss_map: dict, core: torch.Tensor,
                                batch_scale: float = 1.0) -> dict[str, torch.Tensor]:
    """Reconstruct SAM 3's weighted semantic/box/mask objectives by function."""
    groups = {name: core * 0.0 for name in ("semantic", "box", "mask")}
    definitions = (
        ("loss_bbox", "box", 5.0),
        ("loss_giou", "box", 2.0),
        ("loss_ce", "semantic", 20.0),
        ("presence_loss", "semantic", 20.0),
        ("loss_mask", "mask", 200.0),
        ("loss_dice", "mask", 10.0),
    )
    for key, value in loss_map.items():
        if not torch.is_tensor(value) or key == "core_loss":
            continue
        for prefix, group, weight in definitions:
            if key.startswith(prefix):
                # Sam3LossWrapper has already multiplied logged o2m values by
                # its o2m_weight.  Only the per-loss weight and optional find
                # batch scaling remain to reproduce core_loss.
                groups[group] = groups[group] + value * weight * batch_scale
                break
    groups["reconstruction_gap"] = core - sum(groups.values())
    return groups


def _focus_mask_box_consistency(output) -> torch.Tensor:
    """Penalize predicted mask probability outside its paired normalized box.

    This uses the model's existing unified object queries; no dataset-specific
    category, threshold, or routing signal is introduced.
    """
    total = None
    count = 0
    for stage in output.output:
        prediction = stage[-1]
        boxes, masks = prediction.get("pred_boxes"), prediction.get("pred_masks")
        if not torch.is_tensor(boxes) or not torch.is_tensor(masks):
            continue
        if masks.ndim == boxes.ndim + 2 and masks.shape[-3] == 1:
            masks = masks.squeeze(-3)
        if boxes.ndim < 2 or masks.ndim < 3:
            continue
        if tuple(boxes.shape[:-1]) != tuple(masks.shape[:-2]):
            continue
        height, width = masks.shape[-2:]
        dtype, device = masks.dtype, masks.device
        cx, cy, bw, bh = boxes.float().unbind(-1)
        x1, x2 = (cx - bw / 2).clamp(0, 1), (cx + bw / 2).clamp(0, 1)
        y1, y2 = (cy - bh / 2).clamp(0, 1), (cy + bh / 2).clamp(0, 1)
        gx = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
        gy = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
        sharpness = 40.0
        inside_x = (torch.sigmoid(sharpness * (gx - x1.unsqueeze(-1))) *
                    torch.sigmoid(sharpness * (x2.unsqueeze(-1) - gx)))
        inside_y = (torch.sigmoid(sharpness * (gy - y1.unsqueeze(-1))) *
                    torch.sigmoid(sharpness * (y2.unsqueeze(-1) - gy)))
        inside = inside_y.unsqueeze(-1) * inside_x.unsqueeze(-2)
        probability = masks.float().sigmoid()
        ratio = (probability * (1.0 - inside)).sum(dim=(-2, -1)) / (
            probability.sum(dim=(-2, -1)) + 1e-6)
        logits = prediction.get("pred_logits")
        if torch.is_tensor(logits) and tuple(logits.shape[:-1]) == tuple(ratio.shape):
            weights = logits.float().sigmoid().amax(dim=-1).detach()
            value = (ratio * weights).sum() / (weights.sum() + 1e-6)
        else:
            value = ratio.mean()
        total = value if total is None else total + value
        count += 1
    if total is not None:
        return (total / count).to(dtype)
    first = next(iter(output.output[0][-1].values()))
    return first.sum() * 0.0


def _object_bidirectional_box_mask_consistency(output) -> torch.Tensor:
    """Bidirectionally align a query's box extent and mask support."""
    total = None
    count = 0
    for stage in output.output:
        prediction = stage[-1]
        boxes, masks = prediction.get("pred_boxes"), prediction.get("pred_masks")
        if not torch.is_tensor(boxes) or not torch.is_tensor(masks):
            continue
        if masks.ndim == boxes.ndim + 2 and masks.shape[-3] == 1:
            masks = masks.squeeze(-3)
        if boxes.ndim < 2 or masks.ndim < 3:
            continue
        if tuple(boxes.shape[:-1]) != tuple(masks.shape[:-2]):
            continue
        height, width = masks.shape[-2:]
        cx, cy, bw, bh = boxes.float().unbind(-1)
        x1, x2 = (cx - bw / 2).clamp(0, 1), (cx + bw / 2).clamp(0, 1)
        y1, y2 = (cy - bh / 2).clamp(0, 1), (cy + bh / 2).clamp(0, 1)
        gx = (torch.arange(width, device=masks.device, dtype=torch.float32) + 0.5) / width
        gy = (torch.arange(height, device=masks.device, dtype=torch.float32) + 0.5) / height
        sharpness = 40.0
        inside_x = (torch.sigmoid(sharpness * (gx - x1.unsqueeze(-1))) *
                    torch.sigmoid(sharpness * (x2.unsqueeze(-1) - gx)))
        inside_y = (torch.sigmoid(sharpness * (gy - y1.unsqueeze(-1))) *
                    torch.sigmoid(sharpness * (y2.unsqueeze(-1) - gy)))
        inside = inside_y.unsqueeze(-1) * inside_x.unsqueeze(-2)
        probability = masks.float().sigmoid()

        # Box -> mask: predicted foreground should remain in its paired box.
        outside = (probability * (1.0 - inside)).sum(dim=(-2, -1)) / (
            probability.sum(dim=(-2, -1)) + 1e-6)
        # Mask -> box: horizontal/vertical mask projections should span the
        # predicted box. This is shape tolerant but constrains all four extents.
        projection_x = probability.amax(dim=-2)
        projection_y = probability.amax(dim=-1)
        dice_x = 1.0 - (2.0 * (projection_x * inside_x).sum(dim=-1) + 1e-6) / (
            projection_x.sum(dim=-1) + inside_x.sum(dim=-1) + 1e-6)
        dice_y = 1.0 - (2.0 * (projection_y * inside_y).sum(dim=-1) + 1e-6) / (
            projection_y.sum(dim=-1) + inside_y.sum(dim=-1) + 1e-6)
        value = outside + 0.5 * (dice_x + dice_y)
        logits = prediction.get("pred_logits")
        if torch.is_tensor(logits) and tuple(logits.shape[:-1]) == tuple(value.shape):
            weights = logits.float().sigmoid().amax(dim=-1).detach()
            value = (value * weights).sum() / (weights.sum() + 1e-6)
        else:
            value = value.mean()
        total = value if total is None else total + value
        count += 1
    if total is not None:
        return (total / count).to(masks.dtype)
    first = next(iter(output.output[0][-1].values()))
    return first.sum() * 0.0


def _grad_dot(left, right) -> torch.Tensor:
    values = [(a.float() * b.float()).sum() for a, b in zip(left, right)
              if a is not None and b is not None]
    if values:
        return sum(values)
    exemplar = next((x for x in (*left, *right) if x is not None), None)
    return torch.tensor(0.0, device=exemplar.device if exemplar is not None else "cpu")


def _project_functional_gradients(gradients):
    """Symmetric PCGrad over semantic, box, and mask gradient lists."""
    projected = [[None if value is None else value.clone() for value in group]
                 for group in gradients]
    cosines: dict[str, float] = {}
    names = ("semantic", "box", "mask")
    norms = [_grad_dot(group, group).clamp_min(1e-12) for group in gradients]
    for i in range(len(gradients)):
        for j in range(i + 1, len(gradients)):
            dot = _grad_dot(gradients[i], gradients[j])
            cosine = dot / (norms[i].sqrt() * norms[j].sqrt())
            cosines[f"{names[i]}_{names[j]}"] = float(cosine.detach())
    for i in range(len(gradients)):
        for j in range(len(gradients)):
            if i == j:
                continue
            dot = _grad_dot(projected[i], gradients[j])
            if float(dot.detach()) >= 0.0:
                continue
            coefficient = dot / norms[j]
            for index, reference in enumerate(gradients[j]):
                if projected[i][index] is not None and reference is not None:
                    projected[i][index].sub_(reference * coefficient.to(reference.dtype))
    combined = []
    raw = []
    for values in zip(*projected):
        present = [x for x in values if x is not None]
        combined.append(sum(present) if present else None)
    for values in zip(*gradients):
        present = [x for x in values if x is not None]
        raw.append(sum(present) if present else None)
    return combined, raw, cosines


def _private_focus_backward(total_loss: torch.Tensor, functional: dict[str, torch.Tensor],
                    controller: AdapterController, apply_conflict: bool,
                    *, strict_finite: bool = False) -> dict[str, float]:
    shared = controller.focus_parameters("visual", trainable_only=True)
    if getattr(controller, "method", None) in CQSM_METHODS:
        global_ids = {id(p) for n,p in controller.model.named_parameters()
                      if n.endswith(("global_A", "global_B", "global_gate_logit")) or ".sdo_bridge." in n}
        shared = [p for p in shared if id(p) not in global_ids]
    if not apply_conflict or not shared:
        total_loss.backward()
        return {}
    gradients = [torch.autograd.grad(functional[name], shared, retain_graph=True,
                                     allow_unused=True)
                 for name in ("semantic", "box", "mask")]
    if not all(bool(torch.isfinite(value).all()) for group in gradients
               for value in group if value is not None):
        if strict_finite:
            raise FloatingPointError("nonfinite CQSM functional gradient before projection")
        # Fall back to the unmodified finite part of the joint objective.  The
        # caller sanitizes its gradients before the optimizer can see them.
        total_loss.backward()
        return {"projection_skipped_nonfinite": 1.0}
    projected, raw, cosines = _project_functional_gradients(gradients)
    if not all(bool(torch.isfinite(value).all()) for value in projected if value is not None):
        if strict_finite:
            raise FloatingPointError("nonfinite CQSM functional gradient after projection")
        total_loss.backward()
        return {"projection_skipped_nonfinite": 1.0}
    total_loss.backward()
    # Keep gradients from consistency, regularization, and any criterion term
    # not reconstructed above; only replace the three functional components.
    for parameter, replacement, original in zip(shared, projected, raw):
        if replacement is None or original is None:
            continue
        delta = replacement - original
        if parameter.grad is None:
            parameter.grad = delta.to(parameter)
        else:
            parameter.grad.add_(delta.to(parameter.grad))
    return cosines


def _focus_backward(total_loss: torch.Tensor, functional: dict[str, torch.Tensor],
                    controller: AdapterController, apply_conflict: bool,
                    *, strict_finite: bool = False, global_conflict: bool = False,
                    global_world_size: int = 1) -> dict[str, float]:
    from .global_coordination import functional_correction, apply_correction
    params, corrections, report = [], [], {}
    if global_conflict and apply_conflict:
        params, corrections, report = functional_correction(
            controller.model, functional, _project_functional_gradients, global_world_size)
    elif global_conflict:
        report = {"global_functional_projection_applied": 0}
    private_report = _private_focus_backward(total_loss, functional, controller, apply_conflict,
                                              strict_finite=strict_finite)
    apply_correction(params, corrections)
    return {**private_report, **report}


def _object_consensus_backward(total_loss: torch.Tensor,
                               functional: dict[str, torch.Tensor],
                               controller: AdapterController) -> dict[str, float]:
    """Write only box/mask agreement into shared object-anchor parameters.

    Detection- and segmentation-specific head adapters retain their ordinary
    gradients. In the shared visual/semantic anchor, coordinates whose task
    gradients disagree are withheld instead of allowing one output to overwrite
    the other. Semantic, consistency, and regularization gradients are retained.
    """
    shared = (controller.focus_parameters("visual", trainable_only=True) +
              controller.focus_parameters("semantic", trainable_only=True))
    if not shared:
        total_loss.backward()
        return {}
    box = torch.autograd.grad(functional["box"], shared, retain_graph=True,
                              allow_unused=True)
    mask = torch.autograd.grad(functional["mask"], shared, retain_graph=True,
                               allow_unused=True)
    if not all(bool(torch.isfinite(value).all()) for group in (box, mask)
               for value in group if value is not None):
        total_loss.backward()
        return {"consensus_skipped_nonfinite": 1.0}
    replacements, originals = [], []
    agreed_values, total_values = 0, 0
    for box_gradient, mask_gradient in zip(box, mask):
        if box_gradient is None and mask_gradient is None:
            replacements.append(None); originals.append(None); continue
        if box_gradient is None:
            replacements.append(mask_gradient); originals.append(mask_gradient); continue
        if mask_gradient is None:
            replacements.append(box_gradient); originals.append(box_gradient); continue
        agreement = box_gradient.float() * mask_gradient.float() >= 0
        replacement = 0.5 * (box_gradient.float() + mask_gradient.float())
        replacement = replacement * agreement
        replacements.append(replacement.to(box_gradient))
        originals.append(box_gradient + mask_gradient)
        agreed_values += int(agreement.sum().item())
        total_values += agreement.numel()
    dot = _grad_dot(box, mask)
    norm_box = _grad_dot(box, box).clamp_min(1e-12)
    norm_mask = _grad_dot(mask, mask).clamp_min(1e-12)
    cosine = float((dot / (norm_box.sqrt() * norm_mask.sqrt())).detach())
    total_loss.backward()
    for parameter, replacement, original in zip(shared, replacements, originals):
        if replacement is None or original is None:
            continue
        delta = replacement - original
        if parameter.grad is None:
            parameter.grad = delta.to(parameter)
        else:
            parameter.grad.add_(delta.to(parameter.grad))
    return {
        "box_mask_shared_cosine": cosine,
        "box_mask_gradient_agreement": agreed_values / max(total_values, 1),
    }


def _shared_core_residual_backward(total_loss: torch.Tensor,
                                   functional: dict[str, torch.Tensor],
                                   controller: AdapterController) -> dict[str, float]:
    """Use the full shared rank for a balanced joint-object direction.

    Visual and semantic adapters form the object core. Detection and mask
    gradients are norm-balanced there; conflicting global directions undergo
    a symmetric projection instead of coordinate deletion. Box and mask head
    adapters are explicitly restricted to their own functional residuals.
    Consistency and regularization gradients from ``total_loss`` are retained.
    """
    shared = (controller.focus_parameters("visual", trainable_only=True) +
              controller.focus_parameters("semantic", trainable_only=True))
    box_head = controller.focus_parameters("box", trainable_only=True)
    mask_head = controller.focus_parameters("mask", trainable_only=True)
    managed = shared + box_head + mask_head
    if not managed:
        total_loss.backward()
        return {"shared_core_routing_missing": 1.0}

    box = torch.autograd.grad(functional["box"], managed, retain_graph=True,
                              allow_unused=True)
    mask = torch.autograd.grad(functional["mask"], managed, retain_graph=True,
                               allow_unused=True)
    head_parameters = box_head + mask_head
    semantic_head = torch.autograd.grad(
        functional["semantic"], head_parameters, retain_graph=True,
        allow_unused=True) if head_parameters else ()
    if not all(bool(torch.isfinite(value).all())
               for group in (box, mask, semantic_head)
               for value in group if value is not None):
        total_loss.backward()
        return {"shared_core_skipped_nonfinite": 1.0}

    shared_count = len(shared)
    box_shared, mask_shared = box[:shared_count], mask[:shared_count]
    box_norm = _grad_dot(box_shared, box_shared).clamp_min(1e-12).sqrt()
    mask_norm = _grad_dot(mask_shared, mask_shared).clamp_min(1e-12).sqrt()
    raw_dot = _grad_dot(box_shared, mask_shared)
    raw_cosine = raw_dot / (box_norm * mask_norm)
    target_norm = (box_norm * mask_norm).sqrt()
    box_scale = target_norm / box_norm
    mask_scale = target_norm / mask_norm
    balanced_dot = raw_dot * box_scale * mask_scale
    balanced_norm_sq = target_norm.square().clamp_min(1e-12)
    conflict = bool((balanced_dot < 0).detach())
    projection_coefficient = (balanced_dot / balanced_norm_sq
                              if conflict else balanced_dot.new_zeros(()))

    shared_replacements: list[torch.Tensor | None] = []
    shared_originals: list[torch.Tensor | None] = []
    shared_energy = raw_energy = 0.0
    for parameter, gb, gm in zip(shared, box_shared, mask_shared):
        if gb is None and gm is None:
            shared_replacements.append(None); shared_originals.append(None); continue
        gb32 = (gb.float() if gb is not None
                else torch.zeros_like(parameter, dtype=torch.float32))
        gm32 = (gm.float() if gm is not None
                else torch.zeros_like(parameter, dtype=torch.float32))
        balanced_box = gb32 * box_scale
        balanced_mask = gm32 * mask_scale
        if conflict:
            projected_box = balanced_box - projection_coefficient * balanced_mask
            projected_mask = balanced_mask - projection_coefficient * balanced_box
        else:
            projected_box, projected_mask = balanced_box, balanced_mask
        core = 0.5 * (projected_box + projected_mask)
        original = gb32 + gm32
        shared_replacements.append(core.to(parameter))
        shared_originals.append(original.to(parameter))
        shared_energy += float(core.square().sum())
        raw_energy += float(original.square().sum())

    # Each private head receives only its matching functional gradient. The
    # total backward below still contributes box-mask consistency and adapter
    # regularization, which are deliberately not removed here.
    box_offset = shared_count
    mask_offset = box_offset + len(box_head)
    semantic_box = semantic_head[:len(box_head)]
    semantic_mask = semantic_head[len(box_head):]
    head_replacements: list[torch.Tensor] = []
    head_originals: list[torch.Tensor] = []
    for parameter, gs, gb, gm in zip(
            box_head, semantic_box, box[box_offset:mask_offset],
            mask[box_offset:mask_offset]):
        zero = torch.zeros_like(parameter, dtype=torch.float32)
        gs32 = gs.float() if gs is not None else zero
        gb32 = gb.float() if gb is not None else zero
        gm32 = gm.float() if gm is not None else zero
        head_replacements.append(gb32.to(parameter))
        head_originals.append((gs32 + gb32 + gm32).to(parameter))
    for parameter, gs, gb, gm in zip(
            mask_head, semantic_mask, box[mask_offset:], mask[mask_offset:]):
        zero = torch.zeros_like(parameter, dtype=torch.float32)
        gs32 = gs.float() if gs is not None else zero
        gb32 = gb.float() if gb is not None else zero
        gm32 = gm.float() if gm is not None else zero
        head_replacements.append(gm32.to(parameter))
        head_originals.append((gs32 + gb32 + gm32).to(parameter))

    total_loss.backward()
    for parameter, replacement, original in zip(
            shared + box_head + mask_head,
            shared_replacements + head_replacements,
            shared_originals + head_originals):
        if replacement is None or original is None:
            continue
        delta = replacement - original
        if parameter.grad is None:
            parameter.grad = delta.to(parameter)
        else:
            parameter.grad.add_(delta.to(parameter.grad))
    return {
        "box_mask_shared_cosine": float(raw_cosine.detach()),
        "shared_conflict_projected": float(conflict),
        "box_gradient_balance_scale": float(box_scale.detach()),
        "mask_gradient_balance_scale": float(mask_scale.detach()),
        "shared_core_energy_ratio": shared_energy / max(raw_energy, 1e-12),
        "box_private_parameters": float(len(box_head)),
        "mask_private_parameters": float(len(mask_head)),
    }


def _consensus_complement_backward(total_loss: torch.Tensor,
                                   functional: dict[str, torch.Tensor],
                                   controller: AdapterController) -> dict[str, float]:
    """Route common and residual gradients into fixed object/box/mask atoms."""
    entries = controller.consensus_complement_parameters()
    if not entries:
        total_loss.backward()
        return {"complement_routing_missing": 1.0}
    shared = [entry[0] for entry in entries]
    semantic = torch.autograd.grad(functional["semantic"], shared, retain_graph=True,
                                   allow_unused=True)
    box = torch.autograd.grad(functional["box"], shared, retain_graph=True,
                              allow_unused=True)
    mask = torch.autograd.grad(functional["mask"], shared, retain_graph=True,
                               allow_unused=True)
    gradients = (semantic, box, mask)
    if not all(bool(torch.isfinite(value).all()) for group in gradients
               for value in group if value is not None):
        total_loss.backward()
        return {"complement_routing_skipped_nonfinite": 1.0}

    replacements, originals = [], []
    agreed_values = total_values = 0
    box_energy = mask_energy = object_energy = 0.0
    for (parameter, object_role, box_role, mask_role), gs, gb, gm in zip(
            entries, semantic, box, mask):
        zero = torch.zeros_like(parameter, dtype=torch.float32)
        gs32 = gs.float() if gs is not None else zero
        gb32 = gb.float() if gb is not None else zero
        gm32 = gm.float() if gm is not None else zero
        if gb is not None and gm is not None:
            agreement = gb32 * gm32 >= 0
            consensus = 0.5 * (gb32 + gm32) * agreement
            agreed_values += int(agreement.sum().item())
            total_values += agreement.numel()
        else:
            consensus = zero
        box_complement = gb32 - consensus
        mask_complement = gm32 - consensus
        desired = ((gs32 + consensus) * object_role.to(gs32) +
                   box_complement * box_role.to(gs32) +
                   mask_complement * mask_role.to(gs32))
        original = gs32 + gb32 + gm32
        replacements.append(desired.to(parameter))
        originals.append(original.to(parameter))
        object_energy += float(((gs32 + consensus) * object_role.to(gs32)).square().sum())
        box_energy += float((box_complement * box_role.to(gs32)).square().sum())
        mask_energy += float((mask_complement * mask_role.to(gs32)).square().sum())

    dot = _grad_dot(box, mask)
    norm_box = _grad_dot(box, box).clamp_min(1e-12)
    norm_mask = _grad_dot(mask, mask).clamp_min(1e-12)
    cosine = float((dot / (norm_box.sqrt() * norm_mask.sqrt())).detach())
    total_loss.backward()
    for parameter, replacement, original in zip(shared, replacements, originals):
        delta = replacement - original
        if parameter.grad is None:
            parameter.grad = delta.to(parameter)
        else:
            parameter.grad.add_(delta.to(parameter.grad))
    energy = max(object_energy + box_energy + mask_energy, 1e-12)
    return {
        "box_mask_shared_cosine": cosine,
        "box_mask_gradient_agreement": agreed_values / max(total_values, 1),
        "object_gradient_energy_fraction": object_energy / energy,
        "box_complement_energy_fraction": box_energy / energy,
        "mask_complement_energy_fraction": mask_energy / energy,
    }


def _all_reduce_gradients(model: torch.nn.Module, world_size: int) -> None:
    """Manual DDP-equivalent gradient averaging used by FOCUS multi-backward."""
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(world_size)


def _sanitize_gradients(model: torch.nn.Module) -> int:
    """Prevent a single non-finite batch from corrupting all future steps."""
    replaced = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        invalid = ~torch.isfinite(parameter.grad)
        if bool(invalid.any()):
            replaced += int(invalid.sum().item())
            parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    return replaced


def _forward_loss(model, criterion, batch, device, controller=None,
                  teacher_state: dict | None = None, distill_weight: float = 1.0):
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY
    if len(batch) != 1:
        raise ValueError(f"Expected one collator stream, got keys {list(batch)}")
    packed = copy_data_to_device(next(iter(batch.values())), device)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    teacher_output = None
    if teacher_state is not None:
        current_state = _runtime_adapter_state(raw_model)
        _load_runtime_adapter_state(raw_model, teacher_state)
        raw_model.eval()
        with torch.no_grad(): teacher_output = raw_model(packed)
        _load_runtime_adapter_state(raw_model, current_state)
        model.train()
    if controller is not None:
        controller.begin_anchor_batch()
    output = model(packed)
    targets = [raw_model.back_convert(x) for x in packed.find_targets]
    loss_map = criterion(output, targets)
    core = loss_map[CORE_LOSS_KEY]
    reg = controller.regularization_loss() if controller is not None else core.new_zeros(())
    kd = (_prediction_distillation(output, teacher_output) * distill_weight
          if teacher_output is not None else core.new_zeros(()))
    terms={k: float(v.detach()) for k, v in loss_map.items() if torch.is_tensor(v)}
    terms["distillation_loss"] = float(kd.detach())
    return core + reg + kd, terms


def _forward_focus_loss(model, criterion, batch, device, controller,
                        consistency_weight: float,
                        unified_bridge: UnifiedObjectBridge | None = None,
                        unified_object_weight: float = 0.0):
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY
    if len(batch) != 1:
        raise ValueError(f"Expected one collator stream, got keys {list(batch)}")
    packed = copy_data_to_device(next(iter(batch.values())), device)
    controller.begin_anchor_batch()
    if unified_bridge is not None:
        unified_bridge.begin_batch()
    output = model(packed)
    targets = [model.back_convert(x) for x in packed.find_targets]
    loss_map = criterion(output, targets)
    core = loss_map[CORE_LOSS_KEY]
    batch_sizes = [int(target["num_boxes"].shape[0]) for target in targets]
    batch_scale = math.sqrt(batch_sizes[0]) if batch_sizes and len(set(batch_sizes)) == 1 else 1.0
    functional = _weighted_functional_losses(loss_map, core, batch_scale=batch_scale)
    consistency_fn = (_object_bidirectional_box_mask_consistency
                      if controller.method in OBJECT_ROUTER_METHODS
                      else _focus_mask_box_consistency)
    consistency = consistency_fn(output) * consistency_weight
    regularization = controller.regularization_loss()
    selector = controller.selector_loss()
    if unified_bridge is not None:
        unified_object, unified_report = unified_bridge.loss(output)
    else:
        unified_object = core.new_zeros(())
        unified_report = {"instances": 0.0, "alignment": 0.0,
                          "contrastive": 0.0, "semantic_alignment": 0.0,
                          "semantic_contrastive": 0.0, "reliability": 0.0}
    weighted_unified_object = unified_object * unified_object_weight
    total = core + consistency + regularization + selector + weighted_unified_object
    sdo_report = {}
    if getattr(controller, "online_deconflict", False):
        from .online_deconflict import online_deconflict_loss
        sdo_loss, sdo_report = online_deconflict_loss(controller.model)
        total = total + sdo_loss
    terms = {key: float(value.detach()) for key, value in loss_map.items()
             if torch.is_tensor(value)}
    terms.update({
        "focus_semantic_loss": float(functional["semantic"].detach()),
        "focus_box_loss": float(functional["box"].detach()),
        "focus_mask_loss": float(functional["mask"].detach()),
        "focus_reconstruction_gap": float(functional["reconstruction_gap"].detach()),
        "focus_consistency_loss": float(consistency.detach()),
        "focus_regularization_loss": float(regularization.detach()),
        "focus_selector_loss": float(selector.detach()),
        "unified_object_loss": float(unified_object.detach()),
        "unified_object_weighted_loss": float(weighted_unified_object.detach()),
        **{f"unified_object_{key}": value for key, value in unified_report.items()},
        **sdo_report,
    })
    return total, terms, functional


@torch.no_grad()
def _object_candidate_components(model, criterion, batch, device,
                                 controller: AdapterController,
                                 candidate: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one counterfactual route using existing training objectives."""
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY
    router = controller.anchor_router()
    if not isinstance(router, ObjectConsistentSkillRouter):
        raise TypeError("object candidate evaluation needs ObjectConsistentSkillRouter")
    router.force_stage(candidate)
    controller.begin_anchor_batch()
    packed = copy_data_to_device(next(iter(batch.values())), device)
    output = model(packed)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    targets = [raw_model.back_convert(x) for x in packed.find_targets]
    loss_map = criterion(output, targets)
    core = loss_map[CORE_LOSS_KEY]
    batch_sizes = [int(target["num_boxes"].shape[0]) for target in targets]
    batch_scale = math.sqrt(batch_sizes[0]) if batch_sizes and len(set(batch_sizes)) == 1 else 1.0
    functional = _weighted_functional_losses(loss_map, core, batch_scale=batch_scale)
    consistency = _object_bidirectional_box_mask_consistency(output)
    components = torch.stack((functional["semantic"].float(),
                              functional["box"].float(),
                              functional["mask"].float(),
                              consistency.float())).detach()
    descriptor = router._visual_descriptor
    if descriptor is None:
        raise RuntimeError("object selector descriptor was not captured")
    return components, descriptor.detach()


def _rank_normalized_utility(components: torch.Tensor) -> torch.Tensor:
    """Average component-wise ranks so no branch-loss scale is hand tuned."""
    if components.ndim != 2:
        raise ValueError("candidate components must be [candidates, objectives]")
    # A single unstable counterfactual forward must never become an attractive
    # selector target.  Rank it last for every affected objective instead.
    components = torch.nan_to_num(components, nan=1e12, posinf=1e12, neginf=1e12)
    candidates = len(components)
    if candidates <= 1:
        return components.new_zeros((candidates,))
    order = components.argsort(dim=0, stable=True)
    ranks = torch.empty_like(components)
    rank_values = torch.arange(candidates, device=components.device,
                               dtype=components.dtype).unsqueeze(1).expand_as(order)
    ranks.scatter_(0, order, rank_values)
    return (ranks / (candidates - 1)).mean(dim=1)


@torch.no_grad()
def _select_object_anchor_parent(model, criterion, loader, device,
                                 controller: AdapterController, task_id: int,
                                 samples: int) -> dict[str, object]:
    """Select base or an earlier skill by joint box/mask counterfactual utility."""
    router = controller.anchor_router()
    if not isinstance(router, ObjectConsistentSkillRouter):
        return {}
    if task_id == 0:
        router.set_parent(0, -1, score=0.0, base_score=0.0)
        return {"parent": -1, "base_score": 0.0, "candidate_scores": {"base": 0.0},
                "samples": 0}
    candidates = [-1, *range(task_id)]
    sums = torch.zeros(len(candidates), 4, device=device)
    count = torch.zeros((), device=device)
    was_training = model.training
    # SAM 3 attaches Hungarian match indices only on its training forward;
    # the loss object needs them even though this counterfactual probe is
    # gradient-free.
    model.train()
    for index, batch in enumerate(loader):
        if index >= samples:
            break
        values = []
        for candidate in candidates:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                components, _ = _object_candidate_components(
                    model, criterion, batch, device, controller, candidate)
            values.append(components)
        sums.add_(torch.stack(values))
        count.add_(1)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sums); dist.all_reduce(count)
    if int(count.item()) == 0:
        raise RuntimeError("object parent probing received no samples")
    mean_components = sums / count
    utility = _rank_normalized_utility(mean_components)
    base_score = float(utility[0].item())
    old_index = int(utility[1:].argmin().item()) + 1
    parent = candidates[old_index] if float(utility[old_index]) < base_score else -1
    selected_score = (float(utility[old_index].item()) if parent >= 0 else base_score)
    router.set_parent(task_id, parent, score=selected_score, base_score=base_score)
    router.force_stage(None)
    model.train(was_training)
    return {
        "parent": int(parent),
        "base_score": base_score,
        "selected_score": selected_score,
        "candidate_scores": {
            ("base" if candidate < 0 else str(candidate)): float(score)
            for candidate, score in zip(candidates, utility.tolist())
        },
        "samples": int(count.item()),
    }


@torch.no_grad()
def _collect_object_selector_targets(model, criterion, loader, device,
                                     controller: AdapterController, task_id: int,
                                     samples: int) -> dict[str, object]:
    """Create feature/utility rehearsal pairs without retaining raw samples."""
    router = controller.anchor_router()
    if not isinstance(router, ObjectConsistentSkillRouter):
        return {}
    candidates = list(range(task_id + 1))
    was_training = model.training
    # Keep SAM's training output contract (notably ``indices``) while
    # no_grad prevents the calibration pass from building a graph.
    model.train()
    stored = 0
    utility_sum = torch.zeros(len(candidates), device=device)
    batches = 0
    for index, batch in enumerate(loader):
        if index >= samples:
            break
        values, descriptor = [], None
        for candidate in candidates:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                components, observed = _object_candidate_components(
                    model, criterion, batch, device, controller, candidate)
            values.append(components)
            if descriptor is None:
                descriptor = observed
        utility = _rank_normalized_utility(torch.stack(values))
        utility_sum.add_(utility)
        target = utility.unsqueeze(0).expand(len(descriptor), -1)
        stored += router.store_utility(descriptor, target)
        batches += 1
    router.force_stage(None)
    model.train(was_training)
    return {
        "local_samples": stored,
        "local_batches": batches,
        "mean_candidate_utility": {
            str(candidate): float(value) / max(batches, 1)
            for candidate, value in zip(candidates, utility_sum.tolist())
        },
    }


def estimate_fisher(model, criterion, loader, device, batches: int = 8):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    names = {n: p for n, p in raw_model.named_parameters() if p.requires_grad}
    fisher = {n: torch.zeros_like(p) for n, p in names.items()}
    model.train()
    iterator = _cycle(loader)
    for _ in range(batches):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss, _ = _forward_loss(model, criterion, next(iterator), device)
        loss.backward()
        for n, p in names.items():
            if p.grad is not None: fisher[n].add_(p.grad.detach().square() / batches)
    model.zero_grad(set_to_none=True)
    return fisher


def adapter_state_dict(model) -> dict:
    prefixes = {name for name, module in model.named_modules()
                if isinstance(module, ContinualAdapterLinear)}
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if any(name == p or name.startswith(p + ".") for p in prefixes)
            and ".base." not in name}


def load_adapter_checkpoint(model, checkpoint_path: str | Path, device="cpu"):
    """Inject the recorded method and load an adapter-only continual checkpoint."""
    enable_lora_training_compatibility()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload["config"]
    injection_kwargs = {}
    if config["method"] in CQSM_METHODS:
        schema = config.get("cqsm_schema_version")
        restored = config.get("cqsm_restore_legacy", False)
        if schema not in (1, 2, 3, 4, 5, 6) or (schema in (2, 3, 4, 5, 6)) != bool(restored):
            raise ValueError("CQSM checkpoint requires schema 1, or schema 2/3/4/5 with explicit restoration")
        if schema == 4 and config.get("cqsm_shared_layout") != "visual_residual":
            raise ValueError("schema 4 requires the capacity-preserving visual_residual layout")
        if config.get("cqsm_shared_layout") == "visual_residual" and restored and schema not in (4, 5, 6):
            raise ValueError("restored visual_residual checkpoints require schema 4 or 5")
        if (schema == 6) != bool(config.get("global_deconflict", False)):
            raise ValueError("schema 6 exclusively denotes online Global/Private deconflicting")
        if schema == 6 and (not config.get("cqsm_shared") or not config.get("cqsm_private")
                or config.get("cqsm_shared_layout") != "visual_residual" or config.get("cqsm_global_gating")):
            raise ValueError("schema 6 requires ungated co-located Global and Private")
        gated = bool(config.get("cqsm_global_gating", False))
        if schema == 5 and (not gated or config.get("cqsm_shared_layout") != "visual_residual"
                            or not config.get("cqsm_shared")):
            raise ValueError("schema 5 requires gated Global in visual_residual CQSM")
        if gated and schema != 5:
            raise ValueError("gated Global checkpoints require schema 5")
        injection_kwargs = {
            "focus_group_ranks": config["focus_group_ranks"],
            "cqsm_shared": config["cqsm_shared"],
            "cqsm_private": config["cqsm_private"],
            "cqsm_bank_mode": config["cqsm_bank_mode"],
            "cqsm_basis": config["cqsm_basis"],
            "cqsm_memory_rank": config.get("object_core_rank_per_task", 2),
            "cqsm_energy_calibration": config.get("energy_calibration", bool(restored)),
            "cqsm_restore_legacy": restored,
            "cqsm_shared_layout": config.get("cqsm_shared_layout", "late_parallel"),
            "cqsm_global_rank": config.get("global_rank", 2),
            "cqsm_global_gating": gated,
            "cqsm_global_gate_init": config.get("cqsm_global_gate_init", 0.1),
        }
        if restored:
            injection_kwargs.update(
                energy_calibration_lambda=config["energy_calibration_lambda"],
                focus_orthogonality_weight=config["focus_orthogonality_weight"])
    elif config["method"] == "focus_lora":
        injection_kwargs = {
            "focus_group_ranks": config["focus_group_ranks"],
            "focus_stable_ratio": config["focus_stable_ratio"],
            "focus_merge_gamma": config["focus_merge_gamma"],
            "focus_orthogonality_weight": config["focus_orthogonality_weight"],
        }
    elif config["method"] == "focus_lora_v2":
        injection_kwargs = {
            "focus_group_ranks": config["focus_group_ranks"],
            "focus_admission_ratio": config["focus_admission_ratio"],
            "focus_importance_decay": config["focus_importance_decay"],
            "focus_orthogonality_weight": config["focus_orthogonality_weight"],
        }
    elif config["method"] == "focus_lora_v3":
        injection_kwargs = {
            "focus_group_ranks": config["focus_group_ranks"],
            "focus_top_k": config["focus_top_k"],
            "focus_orthogonality_weight": config["focus_orthogonality_weight"],
        }
    elif config["method"] == "task_decoupled_lora":
        injection_kwargs = {
            "focus_group_ranks": config["focus_group_ranks"],
            "focus_orthogonality_weight": config["focus_orthogonality_weight"],
        }
    elif config["method"] in {"task_decoupled_mlp", "compositional_anchor_lora",
                             *INDEPENDENT_CAPACITY_METHODS, "object_coupled_capacity_lora",
                             *ENERGY_SUBSPACE_METHODS, *LORA_LOCATION_METHODS,
                             *OBJECT_ROUTER_METHODS}:
        injection_kwargs = {
            "focus_group_ranks": config["focus_group_ranks"],
            "focus_orthogonality_weight": config["focus_orthogonality_weight"],
        }
        if config["method"] in ENERGY_SUBSPACE_METHODS:
            injection_kwargs["energy_calibration_lambda"] = config.get(
                "energy_calibration_lambda", 3.0)
        if "general_object_rank_per_task" in config:
            injection_kwargs["general_object_rank"] = config[
                "general_object_rank_per_task"]
        if "object_core_rank_per_task" in config and config["object_core_rank_per_task"]:
            injection_kwargs["object_core_rank"] = config["object_core_rank_per_task"]
    report = inject_continual_lora(model, config["method"], rank=config["rank"],
                                   alpha=config["alpha"],
                                   num_tasks=len(config["stream"]),
                                   **injection_kwargs)
    if config.get("global_deconflict", False):
        from .online_deconflict import attach_online_deconflict
        report = attach_online_deconflict(model, report, hidden=config["sdo_hidden"],
            max_correction=config["sdo_max_correction"], conflict_weight=config["sdo_conflict_weight"],
            preservation_weight=config["sdo_preservation_weight"])
    # InfLoRA's activation subspace grows from [0, d] to [history_rank, d].
    # Resize the registered buffers before strict shape checking in load_state_dict.
    for key, value in payload["adapter"].items():
        if key.endswith(".history_basis"):
            module_name = key.rsplit(".", 1)[0]
            module = model.get_submodule(module_name)
            module._buffers["history_basis"] = torch.empty_like(
                value, device=module.base.weight.device)
    if config["method"] in CQSM_METHODS:
        expected = set(adapter_state_dict(model))
        actual = set(payload["adapter"])
        if expected != actual:
            raise RuntimeError(
                f"Incomplete CQSM checkpoint: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}")
    missing, unexpected = model.load_state_dict(payload["adapter"], strict=False)
    unexpected = [x for x in unexpected if ".base." not in x]
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {unexpected}")
    controller = AdapterController(model, config["method"])
    controller.online_deconflict = bool(config.get("global_deconflict", False))
    controller.load_state_dict(payload.get("controller", {}))
    controller.restore_focus_trainability()
    return controller, report, payload


def train_stream(*, method: str, checkpoint: str, bpe_path: str, data_root: str,
                 output_dir: str, rank: int = 16, alpha: float = 8., lr: float = 1e-4,
                 weight_decay: float = .01, steps_per_task: int = 1000,
                 fisher_batches: int = 8, resolution: int = 1008,
                 category_chunk_size: int = 16, seed: int = 42,
                 stream=DEFAULT_STREAM, device: str = "cuda",
                 memory_per_task: int = 20, replay_every: int = 4,
                 distill_weight: float = 1.0,
                 focus_stable_ratio: float = 0.75,
                 focus_merge_gamma: float = 0.75,
                 focus_orthogonality_weight: float = 1e-3,
                 focus_admission_ratio: float = 0.25,
                 focus_importance_decay: float = 0.99,
                  focus_top_k: int = 4,
                  focus_conflict_every: int = 4,
                  focus_consistency_weight: float = 0.1,
                  global_retention_weight: float = 1.0,
                  cqsm_shared_layout: str = "visual_residual",
                  cqsm_global_rank: int = 10,
                  shared_lr: float = 1e-4,
                  shared_retention_ratio: float = 0.2,
                  global_functional_conflict: bool = False,
                  global_independent_clip: bool = False,
                  cqsm_global_gating: bool = False,
                  cqsm_global_gate_init: float = 0.1,
                  global_deconflict: bool = False,
                  sdo_conflict_weight: float = 0.05,
                  sdo_preservation_weight: float = 1.0,
                  sdo_lr: float = 1e-3,
                  sdo_hidden: int = 32,
                  sdo_max_correction: float = 0.25,
                  sdo_warmup_steps: int = 50,
                  object_utility_samples: int = 32,
                  unified_object_weight: float = 0.1,
                  unified_object_mode: str = "query_teacher",
                  energy_covariance_batches: int = 16,
                  energy_calibration_lambda: float = 3.0,
                  cqsm_shared: bool = True,
                  cqsm_private: bool = True,
                  cqsm_bank_mode: str = "all",
                  cqsm_basis: str = "covariance",
                  cqsm_restore_legacy: bool = False,
                  cqsm_memory_rank: int = 2,
                  cqsm_energy_calibration: bool = True) -> list[dict]:
    from sam3.model_builder import build_sam3_image_model
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank_id = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    is_main = rank_id == 0
    seed_everything(seed)
    if cqsm_restore_legacy and method not in CQSM_METHODS:
        raise ValueError("--cqsm-restore-legacy is only supported by CQSM")
    restored_cqsm = method in CQSM_METHODS and cqsm_restore_legacy
    if global_deconflict and (not restored_cqsm or cqsm_shared_layout != "visual_residual"
            or not cqsm_shared or not cqsm_private or cqsm_global_gating):
        raise ValueError("Online deconflicting requires restored ungated Global/Private visual_residual CQSM")
    if min(sdo_conflict_weight, sdo_preservation_weight, sdo_warmup_steps) < 0 or sdo_lr <= 0 or sdo_hidden < 1 or sdo_max_correction <= 0:
        raise ValueError("Invalid online deconflicting hyperparameters")
    if (global_functional_conflict or global_independent_clip or cqsm_global_gating) and (
            not restored_cqsm or cqsm_shared_layout != "visual_residual" or not cqsm_shared):
        raise ValueError("Global coordination requires restored visual_residual CQSM with Global enabled")
    if restored_cqsm:
        unified_object_mode = "semantic_query_teacher"
    if focus_conflict_every < 1:
        raise ValueError("focus_conflict_every must be >= 1")
    if object_utility_samples < 1:
        raise ValueError("object_utility_samples must be >= 1")
    if unified_object_weight < 0:
        raise ValueError("unified_object_weight must be non-negative")
    if energy_covariance_batches < 1:
        raise ValueError("energy_covariance_batches must be >= 1")
    if energy_calibration_lambda <= 0:
        raise ValueError("energy_calibration_lambda must be positive")
    if unified_object_mode not in {"symmetric", "query_teacher", "semantic_query_teacher"}:
        raise ValueError(
            "unified_object_mode must be symmetric, query_teacher, or "
            "semantic_query_teacher")
    if resolution != 1008:
        raise ValueError("This SAM 3 checkpoint has fixed RoPE embeddings and requires --resolution 1008")
    dev = torch.device(device)
    builder_device = "cuda" if distributed else device
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint,
        load_from_HF=False, enable_segmentation=True, device=builder_device, eval_mode=False)
    enable_lora_training_compatibility()
    injection_kwargs = {}
    if method in CQSM_METHODS:
        injection_kwargs = {"cqsm_shared": cqsm_shared,
                            "cqsm_private": cqsm_private,
                            "cqsm_bank_mode": cqsm_bank_mode,
                            "cqsm_basis": cqsm_basis,
                            "cqsm_memory_rank": cqsm_memory_rank,
                            "cqsm_energy_calibration": cqsm_energy_calibration,
                            "cqsm_restore_legacy": restored_cqsm,
                            "cqsm_shared_layout": cqsm_shared_layout,
                            "cqsm_global_rank": cqsm_global_rank,
                            "cqsm_global_gating": cqsm_global_gating,
                            "cqsm_global_gate_init": cqsm_global_gate_init,
                            "energy_calibration_lambda": energy_calibration_lambda,
                            "focus_orthogonality_weight": focus_orthogonality_weight}
    elif method == "focus_lora":
        injection_kwargs = {
            "focus_stable_ratio": focus_stable_ratio,
            "focus_merge_gamma": focus_merge_gamma,
            "focus_orthogonality_weight": focus_orthogonality_weight,
        }
    elif method == "focus_lora_v2":
        injection_kwargs = {
            "focus_admission_ratio": focus_admission_ratio,
            "focus_importance_decay": focus_importance_decay,
            "focus_orthogonality_weight": focus_orthogonality_weight,
        }
    elif method == "focus_lora_v3":
        injection_kwargs = {
            "focus_top_k": focus_top_k,
            "focus_orthogonality_weight": focus_orthogonality_weight,
        }
    elif method == "task_decoupled_lora":
        injection_kwargs = {
            "focus_orthogonality_weight": focus_orthogonality_weight,
        }
    elif method in {"task_decoupled_mlp", "compositional_anchor_lora",
                   *INDEPENDENT_CAPACITY_METHODS, "object_coupled_capacity_lora",
                   *ENERGY_SUBSPACE_METHODS, *LORA_LOCATION_METHODS,
                   *OBJECT_ROUTER_METHODS}:
        injection_kwargs = {
            "focus_orthogonality_weight": focus_orthogonality_weight,
        }
        if method in ENERGY_SUBSPACE_METHODS:
            injection_kwargs["energy_calibration_lambda"] = energy_calibration_lambda
    report = inject_continual_lora(model, method, rank=rank, alpha=alpha,
                                   num_tasks=len(stream), **injection_kwargs)
    if global_deconflict:
        from .online_deconflict import attach_online_deconflict
        report = attach_online_deconflict(model, report, hidden=sdo_hidden,
            max_correction=sdo_max_correction, conflict_weight=sdo_conflict_weight,
            preservation_weight=sdo_preservation_weight)
    controller = AdapterController(model, method)
    controller.online_deconflict = bool(global_deconflict)
    unified_bridge = (UnifiedObjectBridge(model, mode=unified_object_mode)
                      if method in {"unified_object_lora",
                                    "object_coupled_capacity_lora",
                                    *ENERGY_SUBSPACE_METHODS, *LORA_LOCATION_METHODS}
                      or restored_cqsm else None)
    # FOCUS performs multiple functional gradient queries on one graph.  It
    # therefore averages gradients explicitly instead of invoking DDP reducer
    # hooks multiple times.  The optimization is otherwise equivalent.
    manual_distributed = distributed and method in (FOCUS_METHODS | CQSM_METHODS)
    train_model = (DistributedDataParallel(model, device_ids=[local_rank],
                   output_device=local_rank, broadcast_buffers=True,
                   # Distillation/replay objectives do not exercise every adapter
                   # branch on every iteration. Let DDP discover those branches
                   # so LwF and DER++ can advance past task 1 safely.
                   find_unused_parameters=True)
                   if distributed and not manual_distributed else model)
    seed_everything(seed + rank_id)
    criterion = build_joint_loss(dev)
    # The SAMCL-style selector has its own replay optimizer and must not be
    # updated by detection/segmentation gradients.
    selector_parameter_ids = {id(p) for p in controller.selector_parameters()}
    residual_mode = method in CQSM_METHODS and cqsm_shared_layout == "visual_residual"
    prefix_mode = method in CQSM_METHODS and cqsm_shared_layout in {"visual_prefix", "visual_residual"}
    from .global_coordination import global_parameters, clip_adapter_gradients, global_diagnostics
    shared_parameters = global_parameters(model) if prefix_mode else []
    shared_ids = {id(p) for p in shared_parameters}
    shared_groups = [{"params": [p for n, p in model.named_parameters()
                                 if id(p) in shared_ids and n.endswith("global_B")],
                      "weight_decay": weight_decay}]
    gate_parameters = [p for n, p in model.named_parameters()
                       if id(p) in shared_ids and n.endswith("global_gate_logit")]
    if gate_parameters:
        shared_groups.append({"params": gate_parameters, "weight_decay": 0.0})
    sdo_parameters = [p for name,p in model.named_parameters() if ".sdo_bridge." in name and p.requires_grad]
    if sdo_parameters:
        shared_groups.append({"params": sdo_parameters, "weight_decay": 0.0, "lr": sdo_lr})
    shared_optimizer = (torch.optim.AdamW(shared_groups, lr=shared_lr)
                        if shared_parameters else None)
    optimizer = torch.optim.AdamW((p for p in model.parameters()
                                   if p.requires_grad and id(p) not in selector_parameter_ids and id(p) not in shared_ids),
                                  lr=lr, weight_decay=weight_decay)
    total_parameters = sum(p.numel() for p in model.parameters())
    frozen_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    config = dict(method=method, checkpoint=checkpoint, bpe_path=bpe_path,
                  data_root=data_root, rank=rank, alpha=alpha, lr=lr,
                  weight_decay=weight_decay, steps_per_task=steps_per_task,
                  fisher_batches=fisher_batches, resolution=resolution,
                  category_chunk_size=category_chunk_size, seed=seed,
                  memory_per_task=memory_per_task, replay_every=replay_every,
                  distill_weight=distill_weight,
                   unified_object_weight=unified_object_weight,
                   unified_object_mode=unified_object_mode,
                   energy_covariance_batches=energy_covariance_batches,
                   energy_calibration_lambda=energy_calibration_lambda,
                  stream=[asdict(x) for x in stream], injected_modules=report.modules,
                  total_parameters=total_parameters,
                  frozen_parameters=frozen_parameters,
                  trainable_parameters=report.trainable_parameters,
                  persistent_adapter_parameters=report.persistent_adapter_parameters,
                  trainable_percent=100.0*report.trainable_parameters/max(total_parameters,1))
    if method in CQSM_METHODS:
        config.update(
            cqsm_schema_version=(6 if global_deconflict else 5 if cqsm_global_gating else 4 if residual_mode else 3 if prefix_mode else 2) if restored_cqsm else 1,
            cqsm_restore_legacy=bool(restored_cqsm),
            method_mainline="continually_updated_global_private_lora_with_ungated_CQSM",
            cqsm_shared=bool(cqsm_shared), cqsm_bank_mode=cqsm_bank_mode,
            cqsm_private=bool(cqsm_private), cqsm_basis=cqsm_basis,
            global_rank=cqsm_global_rank if cqsm_shared else 0,
            cqsm_shared_layout=cqsm_shared_layout,
            cqsm_global_gating=bool(cqsm_global_gating), cqsm_global_gate_init=cqsm_global_gate_init,
            global_functional_conflict=bool(global_functional_conflict),
            global_independent_clip=bool(global_independent_clip),
            global_functional_gradient_scope=("global_B_and_online_operator" if global_deconflict else "global_B_and_layer_gates") if global_functional_conflict else "disabled",
            global_functional_gradient_averaging="before_projection_across_training_ranks" if global_functional_conflict else "disabled",
            global_gate_weight_decay=0.0,
            shared_optimizer="AdamW" if prefix_mode else "joint_AdamW", shared_lr=shared_lr if prefix_mode else lr,
            shared_retention_ratio=shared_retention_ratio,
            global_deconflict=bool(global_deconflict), sdo_conflict_weight=sdo_conflict_weight,
            sdo_preservation_weight=sdo_preservation_weight, sdo_lr=sdo_lr,
            sdo_hidden=sdo_hidden, sdo_max_correction=sdo_max_correction,
            sdo_warmup_steps=sdo_warmup_steps,
            sdo_extra_parameters=sum(p.numel() for p in sdo_parameters),
            sdo_conditioning="live detached output signatures in fixed orthonormal Global input coordinates",
            sdo_private_conditioning="B_private @ sqrt(A_private @ A_private.T)",
            sdo_conflict_objective="scale-normalized SDO polar overlap relative to each live uncorrected pair",
            sdo_structural_gradient_scope="Global_B_and_operator_only_private_signatures_detached",
            sdo_shared_only_retention="unchanged_raw_Global_path_operator_excluded",
            sdo_training_scope="from_first_task_joint_current_Private_CQSM_Global_and_operator",
            shared_retention_max_multiplier=1e6,
            global_projection="visual_fused_QV_only" if prefix_mode else "full_linear",
            shared_retention_scope=("previous_shared_only_visual_feature" if residual_mode else
                                    "previous_shared_prefix_full_feature" if prefix_mode else "local_mapping"),
            global_modules=report.module_groups.get("global_shared", []),
            global_update=("visual_residual_shared_only_feature_retention_gradient_balance" if residual_mode else
                           "visual_prefix_feature_retention_gradient_balance" if prefix_mode
                           else "fixed_orthogonal_down_projection_with_local_output_retention"),
            global_down_projection="fixed_random_orthogonal", global_retention_weight=global_retention_weight,
            global_retention_scope=("disabled_in_favor_of_shared_prefix_feature_retention" if prefix_mode
                                    else "previous_stage_local_global_output_on_detached_current_inputs"),
            focus_group_ranks=report.group_ranks,
            focus_module_groups=report.module_groups,
            object_core_rank_per_task=controller.energy_object_core().rank,
            general_object_rank_per_task=controller.energy_object_core().rank,
            isolated_object_rank_per_task=0,
            object_core_width=controller.energy_object_core().in_features,
            object_core_consumers=report.module_groups.get("object_core", []),
            query_statistic="mean_per_consumer_uncentered_second_moment_trace_d",
            historical_statistic="sum_of_task_normalized_second_moments",
            covariance_sampling="training_split_only_fixed_parameter_prepass",
            energy_calibration=bool(restored_cqsm and cqsm_energy_calibration),
            cqsm_memory_rank_sweep_private_allocation="fixed_final_recipe_rank2_reservation",
            energy_calibration_lambda=energy_calibration_lambda if restored_cqsm else None,
            joint_training_objectives=(["native_SAM3_joint_loss", "box_mask_consistency",
                "private_factor_orthogonality", "semantic_query_teacher"] if restored_cqsm
                else ["native_SAM3_joint_loss"]),
            focus_orthogonality_weight=focus_orthogonality_weight if restored_cqsm else 0.0,
            focus_consistency_weight=focus_consistency_weight if restored_cqsm else 0.0,
            functional_gradient_conflict_removal=bool(restored_cqsm),
            focus_conflict_every=focus_conflict_every if restored_cqsm else None,
            functional_gradient_scope="private_visual_adapters" if restored_cqsm else "disabled",
            nonfinite_gradient_policy="fail_fast_no_sanitization_or_skipped_projection",
            unified_object_weight=unified_object_weight if restored_cqsm else 0.0,
            unified_object_mode=unified_object_mode if restored_cqsm else "disabled",
            stores_historical_samples=False, stores_historical_features=True,
            historical_features_per_task=300,
            historical_feature_format="fp16_mid_visual_embedding",
            selector_training_epochs=25,
            selector_optimizer="SGD(lr=0.01,batch_size=24)",
            anchor_router="samcl_four_layer_visual_mlp_hard_top1",
            uses_task_id_at_training=True, uses_stage_boundary_for_training=True,
            uses_task_id_at_inference=False, uses_dataset_id_at_inference=False,
            prompt_conditioned_routing=False,
            single_shared_route_across_all_functions=True,
            historical_module_update="private_A_B_and_bank_readouts_strictly_frozen",
            private_initialization="orthogonal_to_frozen_historical_input_subspaces",
            private_residual_scale=1.0, global_residual_scale="sigmoid_layer_gate" if cqsm_global_gating else 1.0,
            bank_residual_scale=1.0,
            checkpoint_selection="last_fixed_step_per_task_no_validation_selection",
        )
    if method in FOCUS_METHODS:
        config.update(
            focus_group_ranks=report.group_ranks,
            focus_module_groups=report.module_groups,
            focus_orthogonality_weight=focus_orthogonality_weight,
            focus_conflict_every=focus_conflict_every,
            focus_consistency_weight=focus_consistency_weight,
            uses_task_id_at_inference=False,
            uses_dataset_id_at_inference=False,
            task_specific_modules=False,
        )
        if method == "focus_lora":
            config.update(focus_stable_ratio=focus_stable_ratio,
                          focus_merge_gamma=focus_merge_gamma)
        elif method == "task_decoupled_lora":
            config.update(stores_historical_samples=False,
                          stores_historical_features=False,
                          stores_anchor_statistics=True,
                          anchor_count=len(stream),
                          anchor_router="diagonal_gaussian_softmax",
                          anchor_temperature=None,
                          anchor_top_k=None,
                          uses_stage_boundary_for_training=True,
                          uses_task_id_at_training=True,
                          recycles_allocated_capacity=False,
                          shared_routing_across_functions=True,
                          input_conditioned_inference=True)
        elif method in {"task_decoupled_mlp", "compositional_anchor_lora",
                       *INDEPENDENT_CAPACITY_METHODS, "object_coupled_capacity_lora",
                       *ENERGY_SUBSPACE_METHODS, *LORA_LOCATION_METHODS,
                       *OBJECT_ROUTER_METHODS}:
            config.update(stores_historical_samples=False,
                          stores_historical_features=True,
                          historical_feature_format="fp16_mid_visual_embedding",
                          historical_features_per_task=300,
                          selector_training_epochs=25,
                          selector_optimizer="SGD(lr=0.01,batch_size=24)",
                          stores_anchor_statistics=False,
                          anchor_count=len(stream),
                          anchor_router="samcl_four_layer_visual_mlp_hard_top1",
                          selector_rehearsal="embedding_buffer",
                          selector_hidden_rule="D->D->D/4->D/4->tasks",
                          uses_stage_boundary_for_training=True,
                          uses_task_id_at_training=True,
                          recycles_allocated_capacity=False,
                          single_shared_route_across_all_functions=True,
                          shared_routing_across_detection_and_segmentation=True,
                          prompt_conditioned_routing=False,
                           input_conditioned_inference=True)
            if method in {"compositional_anchor_lora", *INDEPENDENT_CAPACITY_METHODS,
                          "object_coupled_capacity_lora",
                          "energy_object_compositional_lora"}:
                config.update(
                    anchor_training="independent_residual_plus_learned_history_composition",
                    anchor_composition="shared_signed_lower_triangular_residual",
                    anchor_composition_initialization="zero_exact_independent_baseline",
                    anchor_composition_constraint="tanh_bounded_signed",
                    anchor_composition_shared_across=["visual", "semantic", "box", "mask"],
                    anchor_composition_parameter_growth=len(stream) * len(stream),
                    anchor_parent_selection="continuous_end_to_end_not_hard_parent",
                    freezes_historical_anchor_atoms=True,
                )
            if method in {*INDEPENDENT_CAPACITY_METHODS, "object_coupled_capacity_lora",
                          *ENERGY_SUBSPACE_METHODS, *LORA_LOCATION_METHODS}:
                samcl_reference_parameters = 5041079
                config.update(
                    capacity_mode="full_disjoint_module_per_task_and_function",
                    task_specific_modules=True,
                    grows_module_per_task=True,
                    per_task_group_ranks=report.group_ranks,
                    effective_final_group_ranks={
                        group: value * len(stream)
                        for group, value in report.group_ranks.items()
                    },
                    independent_capacity_scaling=1.0,
                    independent_capacity_initialization=(
                        "orthogonal_to_frozen_historical_input_subspaces"),
                    historical_module_update="strictly_frozen",
                    samcl_reference_adapter_parameters=samcl_reference_parameters,
                    samcl_parameter_budget_ratio=(
                        report.persistent_adapter_parameters / samcl_reference_parameters),
                    parameter_matching_target="SAMCL-SLoRA_rank10",
                    shared_routing_across_detection_and_segmentation=True,
                    joint_training_objectives=[
                        "open_vocabulary_semantic", "box", "mask",
                        "box_mask_consistency", "functional_gradient_conflict_removal"
                    ],
                )
                if method == "object_coupled_capacity_lora":
                    config.update(
                        method_mainline="decouple_tasks_couple_objects",
                        object_core_rank_per_task=2,
                        object_core_width=256,
                        object_core_consumers=report.module_groups.get("object_core", []),
                        object_core_semantics=(
                            "one_task_private_query_subspace_shared_by_semantic_box_mask"),
                        stable_semantic_residual=(
                            "frozen_SAM3_text_projection_plus_frozen_base_query_projection"),
                        object_consensus_training=(
                            "query_box_mask_pixel_alignment_on_inference_path_core"),
                    )
                elif method == "energy_object_subspace_lora":
                    config.update(
                        method_mainline=(
                            "task_isolated_object_decomposed_prediction_coupled"),
                        anchor_training="independent_task_residual",
                        anchor_composition="none",
                        object_core_rank_per_task=2,
                        general_object_rank_per_task=1,
                        isolated_object_rank_per_task=1,
                        object_core_width=256,
                        object_core_consumers=report.module_groups.get("object_core", []),
                        object_subspace_estimation=(
                            "query_covariance_top_eigen_plus_generalized_energy_ratio"),
                        object_down_projection="data_derived_fixed_after_task_prepass",
                        historical_state="normalized_256x256_covariance_only",
                        general_object_activation="always_on_without_selector",
                        isolated_object_activation="single_MLP_hard_top1_route",
                        task_id_used_for_subspace_estimation_only=True,
                        task_id_used_at_inference=False,
                        energy_recalibration=(
                            "closed_form_current_energy_over_historical_energy"),
                        object_consensus_training=(
                            "semantic_query_box_mask_alignment_plus_PCGrad"),
                    )
                elif method == "energy_object_compositional_lora":
                    config.update(
                        method_mainline=(
                            "energy_object_plus_independent_compositional_router"),
                        object_core_rank_per_task=2,
                        general_object_rank_per_task=1,
                        isolated_object_rank_per_task=1,
                        object_core_width=256,
                        object_core_consumers=report.module_groups.get("object_core", []),
                        object_subspace_estimation=(
                            "query_covariance_top_eigen_plus_generalized_energy_ratio"),
                        object_down_projection="data_derived_fixed_after_task_prepass",
                        historical_state="normalized_256x256_covariance_only",
                        general_object_activation="always_on_without_selector",
                        isolated_object_activation="compositional_leaf_route",
                        task_id_used_for_subspace_estimation_only=True,
                        task_id_used_at_inference=False,
                        energy_recalibration=(
                            "closed_form_current_energy_over_historical_energy"),
                        object_consensus_training=(
                            "semantic_query_box_mask_alignment_plus_PCGrad"),
                    )
                elif method == "energy_visual_mlp_lora":
                    core = controller.energy_object_core()
                    config.update(
                        method_mainline="energy_object_always_on_general_plus_late_vit_mlp_lora",
                        visual_mlp_targets="blocks.N.mlp.fc1_and_fc2_after_routing_cut",
                        object_core_rank_per_task=(core.rank if core else 0),
                        general_object_rank_per_task=(
                            core.general_rank if core else 0),
                        isolated_object_rank_per_task=(
                            core.isolated_rank if core else 0),
                        object_core_width=256,
                        object_core_consumers=report.module_groups.get("object_core", []),
                        object_subspace_estimation="query_covariance_top_eigen_only",
                        object_down_projection="data_derived_fixed_after_task_prepass",
                        historical_state="normalized_256x256_covariance_only",
                        general_object_activation="always_on_without_selector",
                        isolated_object_activation="disabled",
                        task_id_used_for_subspace_estimation_only=True,
                        task_id_used_at_inference=False,
                        energy_recalibration=(
                            "closed_form_current_energy_over_historical_energy"),
                        object_consensus_training=(
                            "semantic_query_box_mask_alignment_plus_PCGrad"),
                    )
                elif method == "energy_visual_mlp_general_only":
                    config.update(
                        method_mainline="energy_visual_mlp_ablate_isolated_energy",
                        visual_mlp_targets="blocks.N.mlp.fc1_and_fc2_after_routing_cut",
                        object_core_rank_per_task=2,
                        general_object_rank_per_task=2,
                        isolated_object_rank_per_task=0,
                        object_core_width=256,
                        object_core_consumers=report.module_groups.get("object_core", []),
                        object_subspace_estimation="query_covariance_top_eigen_only",
                        general_object_activation="always_on_without_selector",
                        isolated_object_activation="disabled",
                        task_id_used_for_subspace_estimation_only=True,
                        task_id_used_at_inference=False,
                    )
                elif method == "energy_visual_mlp_independent_core":
                    config.update(
                        method_mainline="energy_visual_mlp_independent_per_head_query_bases",
                        visual_mlp_targets="blocks.N.mlp.fc1_and_fc2_after_routing_cut",
                        object_core_rank_per_task=2,
                        general_object_rank_per_task=2,
                        isolated_object_rank_per_task=0,
                        object_core_width=256,
                        object_core_count=len(controller.energy_object_cores()),
                        object_core_consumers=report.module_groups.get("object_core", []),
                        query_basis_sharing="independent_for_query_ffn_ov_box_and_mask",
                        object_subspace_estimation="per_head_query_covariance_top_eigen",
                        general_object_activation="always_on_without_selector",
                        isolated_object_activation="disabled",
                        task_id_used_for_subspace_estimation_only=True,
                        task_id_used_at_inference=False,
                    )
                elif method == "energy_visual_mlp_isolated_only":
                    config.update(
                        method_mainline="energy_visual_mlp_ablate_always_on_general",
                        visual_mlp_targets="blocks.N.mlp.fc1_and_fc2_after_routing_cut",
                        object_core_rank_per_task=2,
                        general_object_rank_per_task=0,
                        isolated_object_rank_per_task=2,
                        object_core_width=256,
                        object_core_consumers=report.module_groups.get("object_core", []),
                        object_subspace_estimation="generalized_energy_ratio_only",
                        general_object_activation="disabled",
                        isolated_object_activation="single_MLP_hard_top1_route",
                        task_id_used_for_subspace_estimation_only=True,
                        task_id_used_at_inference=False,
                    )
                elif method == "independent_visual_mlp_lora":
                    config.update(
                        method_mainline="independent_capacity_plus_late_vit_mlp_lora",
                        visual_mlp_targets="blocks.N.mlp.fc1_and_fc2_after_routing_cut",
                    )
                elif method == "energy_visual_mlp_encoder":
                    config.update(
                        method_mainline="energy_visual_mlp_lora_encoder_only",
                        lora_scope="encoder_vit_attn_and_mlp",
                        visual_mlp_targets="blocks.N.mlp.fc1_and_fc2_after_routing_cut",
                        object_core_rank_per_task=0,
                        decoder_adapters=False,
                    )
                elif method == "energy_visual_mlp_decoder":
                    config.update(
                        method_mainline="energy_visual_mlp_lora_decoder_only",
                        lora_scope="decoder_semantic_box_mask",
                        visual_mlp_targets="disabled",
                        object_core_rank_per_task=2,
                        general_object_rank_per_task=2,
                        isolated_object_rank_per_task=0,
                        object_core_consumers=report.module_groups.get("object_core", []),
                        general_object_activation="always_on_without_selector",
                        isolated_object_activation="disabled",
                    )
                elif method == "energy_visual_mlp_mlp_only":
                    core = controller.energy_object_core()
                    config.update(
                        method_mainline="energy_visual_mlp_late_mlp_without_attn",
                        visual_insert="mlp_only",
                        visual_mlp_targets="blocks.N.mlp.fc1_and_fc2_after_routing_cut",
                        object_core_rank_per_task=(core.rank if core else 0),
                        general_object_rank_per_task=(core.general_rank if core else 0),
                        isolated_object_rank_per_task=(core.isolated_rank if core else 0),
                        object_core_consumers=report.module_groups.get("object_core", []),
                        general_object_activation="always_on_without_selector",
                        isolated_object_activation="disabled",
                    )
                elif method == "energy_visual_mlp_early_mlp":
                    core = controller.energy_object_core()
                    config.update(
                        method_mainline="energy_visual_mlp_early_mlp_without_attn",
                        visual_insert="early_mlp",
                        visual_mlp_targets="blocks.N.mlp.fc1_and_fc2_before_routing_cut",
                        object_core_rank_per_task=(core.rank if core else 0),
                        general_object_rank_per_task=(core.general_rank if core else 0),
                        isolated_object_rank_per_task=(core.isolated_rank if core else 0),
                        object_core_consumers=report.module_groups.get("object_core", []),
                        general_object_activation="always_on_without_selector",
                        isolated_object_activation="disabled",
                    )
                elif method == "energy_visual_mlp_random_core":
                    core = controller.energy_object_core()
                    config.update(
                        method_mainline="energy_visual_mlp_random_orthonormal_object_core",
                        visual_mlp_targets="blocks.N.mlp.fc1_and_fc2_after_routing_cut",
                        object_core_rank_per_task=(core.rank if core else 0),
                        general_object_rank_per_task=(core.general_rank if core else 0),
                        isolated_object_rank_per_task=(core.isolated_rank if core else 0),
                        object_subspace_estimation="random_orthonormal",
                        object_core_consumers=report.module_groups.get("object_core", []),
                        general_object_activation="always_on_without_selector",
                        isolated_object_activation="disabled",
                    )
                elif method == "semantic_protected_subspace_lora":
                    config.update(
                        method_mainline=(
                            "semantic_protected_multihead_object_decomposition"),
                        anchor_training="independent_task_residual",
                        anchor_composition="none",
                        object_core_rank_per_task=4,
                        general_object_rank_per_task=2,
                        isolated_object_rank_per_task=2,
                        object_core_width=256,
                        object_core_consumers=report.module_groups.get("object_core", []),
                        object_general_subspace=(
                            "harmonic_consensus_of_semantic_box_mask_covariances"),
                        semantic_protection=(
                            "frozen_prompt_covariance_pulled_back_through_frozen_query_projection"),
                        object_isolated_subspace=(
                            "generalized_current_to_historical_arithmetic_energy"),
                        object_down_projection="data_derived_fixed_after_task_prepass",
                        historical_state=(
                            "consensus_and_specific_256x256_covariances_only"),
                        general_object_activation="always_on_without_selector",
                        isolated_object_activation="single_MLP_hard_top1_route",
                        visual_capacity_reallocation=(
                            "rank5_to_rank4_and_drop_earliest_adapted_late_block"),
                        task_id_used_for_subspace_estimation_only=True,
                        task_id_used_at_inference=False,
                        object_consensus_training=(
                            "semantic_query_box_mask_alignment_plus_PCGrad"),
                    )
            if method in OBJECT_ROUTER_METHODS:
                config.update(
                    anchor_router="object_utility_mlp_hard_top1",
                    anchor_training="box_mask_consensus_residual",
                    anchor_parent_selection="disabled_independent_multi_anchor",
                    anchor_composition="none",
                    anchor_shared_gradient="elementwise_box_mask_agreement",
                    anchor_private_residuals=["box", "mask"],
                    box_mask_consistency="bidirectional_extent_and_containment",
                    selector_supervision="anchor_ownership_ce_plus_counterfactual_utility_ranking",
                    selector_uses_stage_labels="anchor_ownership_only",
                    selector_uses_task_id_at_training=False,
                    selector_uses_task_id_at_inference=False,
                    historical_feature_format="fp16_dense_owner_embeddings_plus_joint_utility_hard_negatives",
                    object_utility_samples=object_utility_samples,
                )
                if method == "consensus_complement_lora":
                    config.update(
                        anchor_training="consensus_complement_object_anchor",
                        anchor_shared_gradient="object_consensus_plus_box_mask_complements",
                        anchor_rank_roles={"visual": ["object", "box", "mask"],
                                           "semantic": ["object"]},
                        complement_parameter_growth=0,
                    )
                elif method == "shared_core_residual_lora":
                    config.update(
                        anchor_training="full_capacity_shared_core_head_residual",
                        anchor_shared_gradient="norm_balanced_symmetric_projection",
                        anchor_private_residuals={"box_head": "box_only",
                                                  "mask_head": "mask_only"},
                        anchor_rank_roles="all_active_ranks_available_to_object_core",
                        complement_parameter_growth=0,
                    )
                elif method == "unified_object_lora":
                    config.update(
                        anchor_training="query_box_mask_unified_object_representation",
                        unified_object_views=["decoder_query", "box_pooled_pixel_embedding",
                                              "mask_pooled_pixel_embedding"],
                        unified_object_matching="existing_hungarian_positive_queries",
                        unified_object_objective="leave_one_view_out_cosine_plus_instance_contrastive",
                        unified_object_mode=unified_object_mode,
                        unified_object_semantic_anchor=(
                            "native_projected_text_stop_gradient" if
                            unified_object_mode == "semantic_query_teacher" else "disabled"),
                        unified_object_semantic_objective=(
                            "matched_text_cosine_plus_in_batch_prompt_contrastive" if
                            unified_object_mode == "semantic_query_teacher" else "disabled"),
                        unified_object_max_instances_per_step=32,
                        unified_object_temperature=0.1,
                        unified_object_training_only=True,
                        unified_object_inference_parameter_growth=0,
                    )
        else:
            config.update(stores_historical_samples=False,
                          stores_historical_features=False,
                          performs_rank_truncation=False)
            if method == "focus_lora_v2":
                config.update(focus_admission_ratio=focus_admission_ratio,
                              focus_importance_decay=focus_importance_decay)
            else:
                config.update(focus_top_k=focus_top_k,
                              uses_stage_boundary_for_capacity=True,
                              uses_task_id_at_training=True,
                              recycles_allocated_capacity=False,
                              input_conditioned_inference=True)
    if method in SAMCL_METHODS:
        config.update(
            samcl_reference_repository="INV-WZQ/SAMCL",
            samcl_reference_commit="a16f164e0b0bf5a2968c9291d0d4814aef93cbf7",
            samcl_port="SAM3_detection_instance_segmentation",
            samcl_components=["cross_layer_shared_A_SLoRA", "four_layer_MLP_selector"],
            samcl_prompt_augmentation=(
                "disabled_no_interactive_point_input_to_avoid_ground_truth_leakage"),
            prompt_augmentation_requires_unavailable_input=(
                "official_points_json_is_generated_from_ground_truth_masks"),
            protocol_adaptation=(
                "image_text_only_joint_detection_segmentation_with_native_SAM3_loss"),
            slora_rank_per_task=rank,
            slora_residual_scale=1.0,
            slora_cross_layer_shared_down_projection=True,
            slora_task_specific_up_projection=True,
            task_specific_modules=True,
            grows_module_per_task=True,
            adapted_function_groups=["visual_encoder"],
            stores_historical_samples=False,
            stores_historical_features=True,
            historical_feature_format="fp16_mid_visual_embedding",
            historical_features_per_task=300,
            selector_training_epochs=25,
            selector_optimizer="SGD(lr=0.01,batch_size=24)",
            selector_rehearsal="embedding_buffer",
            selector_hidden_rule="D->D->D/4->D/4->tasks",
            anchor_router="samcl_four_layer_visual_mlp_hard_top1",
            uses_stage_boundary_for_training=True,
            uses_task_id_at_training=True,
            uses_task_id_at_inference=False,
            uses_dataset_id_at_inference=False,
            input_conditioned_inference=True,
        )
    config.update(world_size=world_size, per_gpu_batch_size=1,
                  global_batch_size=world_size)
    if method in {"er_lora","derpp_lora"}:
        config["replay_memory"] = replay_memory_profile(data_root,stream,memory_per_task)
    if is_main:
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    config['presence_loss_numerics'] = 'gamma_zero_equivalent_bce_fp32_v1'
    if global_deconflict:
        config['joint_training_objectives'].extend(['online_private_conditioned_global_rewriting',
            'normalized_output_subspace_deconflicting','rewritten_global_geometry_preservation'])
    if is_main:
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    from continual_lora.global_retention import GlobalRetention
    global_retention = GlobalRetention(model, weight=0.0 if prefix_mode else global_retention_weight)
    from continual_lora.shared_prefix import SharedPrefixRetention, SharedOnlyPrefixRetention
    retention_class = SharedOnlyPrefixRetention if residual_mode else SharedPrefixRetention
    prefix_retention = (retention_class(model, gradient_ratio=shared_retention_ratio)
                        if prefix_mode else None)
    if prefix_retention is not None:
        config.update(shared_prefix_blocks=prefix_retention.block_ids,
                      shared_private_boundary=(prefix_retention.block_ids[-1]
                                               if prefix_retention.block_ids and not residual_mode else None),
                      private_visual_blocks=sorted({int(n.split(".blocks.")[1].split(".")[0])
                          for n, m in model.named_modules() if ".blocks." in n
                          and isinstance(m, CQSMDualLoRALinear) and m.private_enabled}),
                      shared_retention_excludes_private=bool(residual_mode),
                      global_retention_weight=0.0,
                      selector_descriptor="frozen_prefix_before_first_shared_block")
        if cqsm_shared and shared_retention_ratio > 0:
            config["joint_training_objectives"].append("shared_prefix_feature_retention_gradient_balance")
        if is_main: (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history=[]; teacher_state=None
    for task_id, task in enumerate(stream):
        task_started = time.time()
        if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
        global_retention.begin_task(task_id)
        if prefix_retention is not None: prefix_retention.begin_task(task_id)
        controller.begin_task(task_id)
        if global_deconflict:
            from .online_deconflict import set_deconflict_strength
            set_deconflict_strength(model, 0.0)
        controller.reset_focus_active_optimizer_state(optimizer)
        if distributed and method in {"focus_lora_v2", "focus_lora_v3",
                                      "task_decoupled_lora", "task_decoupled_mlp",
                                      "compositional_anchor_lora",
                                      *INDEPENDENT_CAPACITY_METHODS,
                                      "object_coupled_capacity_lora",
                                      *ENERGY_SUBSPACE_METHODS, *LORA_LOCATION_METHODS,
                                      *OBJECT_ROUTER_METHODS, *SAMCL_METHODS, *CQSM_METHODS}:
            # Atom admission uses random residual directions.  Synchronize only
            # the small adapter tensors so both manual-DDP workers start from
            # exactly the same pool without broadcasting the frozen SAM model.
            for parameter in controller.focus_parameters(trainable_only=False):
                dist.broadcast(parameter.data, src=0)
        loader = build_loader(data_root, task, resolution=resolution,
                              category_chunk_size=category_chunk_size,
                              rank=rank_id, world_size=world_size)
        subspace_initialization = {}
        if method in (ENERGY_SUBSPACE_METHODS | CQSM_METHODS):
            subspace_initialization = _initialize_energy_object_subspaces(
                train_model, loader, dev, controller, task_id,
                energy_covariance_batches)
            if is_main:
                print(json.dumps({"task": task.name,
                                  "energy_subspace": subspace_initialization}),
                      flush=True)
        parent_selection = {}
        if method in OBJECT_ROUTER_METHODS:
            # Keep anchors independent. Hard parent composition produced
            # negative transfer when a related domain was only superficially
            # close (notably Cityscapes -> TrashCan).
            router = controller.anchor_router()
            if isinstance(router, ObjectConsistentSkillRouter):
                router.set_parent(task_id, -1, score=0.0, base_score=0.0)
            parent_selection = {
                "parent": -1,
                "mode": "independent_multi_anchor",
                "reason": "hard_parent_composition_disabled",
            }
        iterator = _cycle(loader); train_model.train()
        replay_iters=[]
        if method in {"er_lora", "derpp_lora"} and task_id > 0:
            for old_task in stream[:task_id]:
                old_loader=build_loader(data_root, old_task, resolution=resolution,
                    category_chunk_size=category_chunk_size, rank=rank_id,
                    world_size=world_size, limit_ids=memory_per_task)
                replay_iters.append(_cycle(old_loader))
        cumulative_iters=[]
        if method == "cumulative_lora":
            for seen_task in stream[:task_id + 1]:
                seen_loader=build_loader(data_root, seen_task, resolution=resolution,
                    category_chunk_size=category_chunk_size, rank=rank_id,
                    world_size=world_size)
                cumulative_iters.append(_cycle(seen_loader))
        running = 0.
        for step in range(steps_per_task):
            use_replay = bool(replay_iters) and (step + 1) % replay_every == 0
            if method == "cumulative_lora":
                batch = next(cumulative_iters[step % len(cumulative_iters)])
            elif use_replay:
                batch = next(replay_iters[(step // replay_every) % len(replay_iters)])
            else:
                batch = next(iterator)
            kd_state = None
            if teacher_state is not None and (method == "lwf_lora" or
                                               (method == "derpp_lora" and use_replay)):
                kd_state = teacher_state
            global_retention.begin_batch()
            if prefix_retention is not None: prefix_retention.begin_batch()
            if shared_optimizer is not None: shared_optimizer.zero_grad(set_to_none=True)
            prefix_gradients = None
            optimizer.zero_grad(set_to_none=True)
            if global_deconflict:
                strength = min(1.0, max(0.0, (step + 1 - sdo_warmup_steps) / max(sdo_warmup_steps, 1)))
                set_deconflict_strength(model, strength)
            if method in FOCUS_METHODS or restored_cqsm:
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                    enabled=dev.type == "cuda"):
                    loss, terms, functional = _forward_focus_loss(
                        train_model, criterion, batch, dev, controller,
                        consistency_weight=focus_consistency_weight,
                        unified_bridge=unified_bridge,
                        unified_object_weight=unified_object_weight)
                retention = global_retention.loss()
                loss = loss + retention
                terms["global_retention_loss"] = float(retention.detach())
                if prefix_retention is not None:
                    feature_loss, prefix_gradients = prefix_retention.loss_and_gradients()
                    terms["shared_feature_retention_loss"] = float(feature_loss) if feature_loss is not None else 0.0
                apply_conflict = (step + 1) % focus_conflict_every == 0
                if method == "shared_core_residual_lora" and apply_conflict:
                    cosines = _shared_core_residual_backward(
                        loss, functional, controller)
                elif method == "consensus_complement_lora" and apply_conflict:
                    cosines = _consensus_complement_backward(loss, functional, controller)
                elif method == "object_consistent_lora" and apply_conflict:
                    cosines = _object_consensus_backward(loss, functional, controller)
                elif method == "unified_object_lora" and apply_conflict:
                    cosines = _object_consensus_backward(loss, functional, controller)
                else:
                    cosines = _focus_backward(loss, functional, controller, apply_conflict,
                                             strict_finite=restored_cqsm,
                                             global_conflict=global_functional_conflict,
                                             global_world_size=world_size)
                terms.update({f"focus_gradient_cosine_{key}": value
                              for key, value in cosines.items() if not key.startswith("global_")})
                terms.update({key: value for key, value in cosines.items() if key.startswith("global_")})
                terms["focus_conflict_applied"] = int(apply_conflict)
                if manual_distributed:
                    _all_reduce_gradients(model, world_size)
            else:
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                    enabled=dev.type == "cuda"):
                    loss, terms = _forward_loss(train_model, criterion, batch, dev, controller,
                                                teacher_state=kd_state,
                                                distill_weight=distill_weight)
                retention = global_retention.loss().to(loss.device)
                loss = loss + retention
                terms["global_retention_loss"] = float(retention.detach())
                if prefix_retention is not None:
                    feature_loss, prefix_gradients = prefix_retention.loss_and_gradients()
                    terms["shared_feature_retention_loss"] = float(feature_loss) if feature_loss is not None else 0.0
                loss.backward()
                if method in CQSM_METHODS and manual_distributed:
                    _all_reduce_gradients(model, world_size)
            if prefix_retention is not None:
                terms.update(prefix_retention.apply(prefix_gradients, world_size=world_size))
            if method in CQSM_METHODS:
                if not bool(torch.isfinite(loss.detach())):
                    raise FloatingPointError("nonfinite CQSM native training loss")
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                       for p in model.parameters()):
                    raise FloatingPointError("nonfinite CQSM gradient; refusing silent replacement")
            nonfinite_gradients = 0 if method in CQSM_METHODS else _sanitize_gradients(model)
            if method in FOCUS_METHODS or restored_cqsm:
                terms["focus_nonfinite_gradients_replaced"] = nonfinite_gradients
            if method in FOCUS_METHODS:
                controller.accumulate_focus_importance()
            terms.update(clip_adapter_gradients(model, controller, global_independent_clip))
            optimizer.step()
            if shared_optimizer is not None:
                shared_optimizer.step()
                terms["shared_lr"] = shared_optimizer.param_groups[0]["lr"]
            controller.restore_focus_protected_parameters()
            running += float(loss.detach())
            if is_main and (step == 0 or step + 1 == steps_per_task or (step + 1) % 50 == 0 or
                            (restored_cqsm and step + 1 == focus_conflict_every)):
                if global_functional_conflict or cqsm_global_gating:
                    terms.update(global_diagnostics(model))
                if global_deconflict and sdo_parameters:
                    terms["sdo_parameter_norm"] = float(torch.stack([p.detach().float().square().sum() for p in sdo_parameters]).sum().sqrt())
                print(json.dumps({"task": task.name, "step": step + 1,
                                  "loss": float(loss.detach()), **terms}), flush=True)
        if method == "lora_ewc" and task_id + 1 < len(stream):
            fisher = estimate_fisher(train_model, criterion, loader, dev, fisher_batches)
            if distributed:
                for value in fisher.values():
                    dist.all_reduce(value); value.div_(world_size)
            controller.consolidate_ewc(fisher)
        consolidation = controller.end_task()
        selector_calibration = {}
        if method in OBJECT_ROUTER_METHODS:
            selector_calibration = _collect_object_selector_targets(
                train_model, criterion, loader, dev, controller, task_id,
                samples=object_utility_samples)
        if method in {"task_decoupled_lora", "task_decoupled_mlp",
                      "compositional_anchor_lora",
                      *INDEPENDENT_CAPACITY_METHODS, "object_coupled_capacity_lora",
                      *ENERGY_SUBSPACE_METHODS, *LORA_LOCATION_METHODS,
                      *OBJECT_ROUTER_METHODS, *SAMCL_METHODS, *CQSM_METHODS}:
            controller.synchronize_anchor_statistics()
            router = controller.anchor_router()
            if router is not None:
                if method in {"task_decoupled_mlp", "compositional_anchor_lora",
                              *INDEPENDENT_CAPACITY_METHODS, "object_coupled_capacity_lora",
                              *ENERGY_SUBSPACE_METHODS, *LORA_LOCATION_METHODS,
                              *OBJECT_ROUTER_METHODS, *SAMCL_METHODS, *CQSM_METHODS}:
                    consolidation["selector_fit"] = controller.fit_selector()
                consolidation["anchor_router"] = router.diagnostics()
        if parent_selection:
            consolidation["parent_selection"] = parent_selection
        if selector_calibration:
            consolidation["selector_calibration"] = selector_calibration
        if method == "focus_lora":
            controller.reset_plastic_optimizer_state(optimizer)
        if distributed:
            for value in model.state_dict().values():
                if torch.is_tensor(value):
                    if value.is_contiguous():
                        dist.broadcast(value, src=0)
                    else:
                        contiguous = value.contiguous()
                        dist.broadcast(contiguous, src=0)
                        value.copy_(contiguous)
            running_tensor=torch.tensor(running,device=dev)
            dist.all_reduce(running_tensor); running=float(running_tensor/world_size)
            dist.barrier()
        if method in {"lwf_lora", "derpp_lora"}:
            teacher_state = _runtime_adapter_state(model)
        record={"task_id": task_id, "task": task.name,
                "mean_train_loss": running / steps_per_task,
                "training_seconds": time.time()-task_started,
                "seconds_per_step": (time.time()-task_started)/steps_per_task,
                "peak_gpu_memory_bytes": (torch.cuda.max_memory_allocated(dev)
                                           if dev.type == "cuda" else 0)}
        if consolidation:
            record["focus_consolidation"] = consolidation
        if subspace_initialization:
            record["energy_subspace_initialization"] = subspace_initialization
        if method in CQSM_METHODS:
            from .cqsm_accounting import cqsm_parameter_accounting
            record["cqsm_parameter_accounting"] = cqsm_parameter_accounting(
                model, controller, task_id)
        if global_deconflict:
            record["sdo_final_diagnostics"] = {key:value for key,value in terms.items() if key.startswith("sdo_")}
        history.append(record)
        checkpoint_file = out / f"task_{task_id:02d}_{task.name}.pt"
        if is_main:
            torch.save({"format": 1, "config": config, "completed_task": task_id,
                        "adapter": adapter_state_dict(model),
                        "controller": controller.state_dict(), "history": history},
                       checkpoint_file)
            record["checkpoint_bytes"] = os.path.getsize(checkpoint_file)
            (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if distributed: dist.barrier()
    if distributed: dist.destroy_process_group()
    return history
