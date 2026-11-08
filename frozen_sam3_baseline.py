#!/usr/bin/env python3
"""Frozen SAM 3 phrase-conditioned COCO box/mask baseline.

One image embedding is reused for every category prompt. Per-image prediction
shards make long runs restartable without repeating completed inference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from continual_lora.eval_protocol import checkpoint_stage_map, sha256_file


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
    tmp.replace(path)


def encode_mask(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def evaluate(coco: COCO, predictions: list[dict], image_ids: list[int], iou_type: str) -> dict:
    if not predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0}
    dt = coco.loadRes(predictions)
    ev = COCOeval(coco, dt, iou_type)
    ev.params.imgIds = image_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]),
            "AP75": float(ev.stats[2]), "APs": float(ev.stats[3]),
            "APm": float(ev.stats[4]), "APl": float(ev.stats[5]),
            "AR100": float(ev.stats[8])}


def concept_metrics(coco: COCO, image_ids: list[int], max_scores: dict[tuple[int, int], float], threshold: float) -> dict:
    gt_positive = {(int(a["image_id"]), int(a["category_id"])) for a in coco.dataset["annotations"]
                   if int(a["image_id"]) in set(image_ids)}
    pairs = [(iid, cid) for iid in image_ids for cid in coco.getCatIds()]
    tp = fp = fn = tn = 0
    for pair in pairs:
        pred = max_scores.get(pair, 0.0) >= threshold
        gt = pair in gt_positive
        if pred and gt: tp += 1
        elif pred: fp += 1
        elif gt: fn += 1
        else: tn += 1
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "negative_fpr": fp / max(fp + tn, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--annotation", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--adapter-checkpoint", type=Path, default=None)
    ap.add_argument("--bpe-path", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--max-per-prompt", type=int, default=100)
    ap.add_argument("--limit-images", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--focus-routing-mode",
                    choices=("atom_topk","block_energy","oracle_block"),default=None,
                    help="Optional FOCUS-v3 inference-only routing diagnostic")
    ap.add_argument("--focus-routing-blocks",type=int,default=1)
    ap.add_argument("--focus-oracle-stage",type=int,default=None)
    ap.add_argument("--anchor-oracle-stage",type=int,default=None,
                    help="Evaluation-only diagnostic that forces one anchor; normal task-free evaluation leaves this unset")
    ap.add_argument("--anchor-expected-stage", type=int, default=None,
                    help="Ground-truth domain stage used only to inject a controlled routing error")
    ap.add_argument("--anchor-route-error-rate", type=float, default=None,
                    help="Evaluation-only probability of replacing the expected stage by a wrong stage")
    ap.add_argument("--anchor-route-error-seed", type=int, default=2026)
    ap.add_argument("--anchor-num-stages", type=int, default=5)
    ap.add_argument("--disable-core-stage", type=int, action="append", default=[],
                    help="Evaluation-only removal of one completed Query Core; repeat to remove several")
    ap.add_argument("--adapter-lesion", choices=("none", "zero_core", "zero_private"),
                    default="none",
                    help="Evaluation-only lesion of the object core or private LoRA")
    ap.add_argument("--keep-predictions", action="store_true",
                    help="Keep COCO predictions and per-image restart shards after metrics are computed")
    args = ap.parse_args()
    if (args.anchor_expected_stage is None) != (args.anchor_route_error_rate is None):
        raise ValueError("--anchor-expected-stage and --anchor-route-error-rate must be set together")
    if args.anchor_route_error_rate is not None:
        if args.anchor_oracle_stage is not None:
            raise ValueError("controlled routing errors cannot be combined with --anchor-oracle-stage")
        if not 0.0 <= args.anchor_route_error_rate <= 1.0:
            raise ValueError("--anchor-route-error-rate must be in [0, 1]")
        if not 0 <= args.anchor_expected_stage < args.anchor_num_stages:
            raise ValueError("--anchor-expected-stage is outside the configured stage range")

    signature = {"annotation_sha256": sha256_file(args.annotation),
                 "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
                 "base_checkpoint_sha256": sha256_file(args.checkpoint),
                 "base_checkpoint": str(args.checkpoint),
                 "threshold": args.threshold, "limit_images": args.limit_images,
                 "max_per_prompt": args.max_per_prompt,
                 "anchor_oracle_stage": args.anchor_oracle_stage,
                 "anchor_expected_stage": args.anchor_expected_stage,
                 "anchor_route_error_rate": args.anchor_route_error_rate,
                 "anchor_route_error_seed": args.anchor_route_error_seed,
                 "focus_routing_mode": args.focus_routing_mode,
                 "focus_routing_blocks": args.focus_routing_blocks,
                 "focus_oracle_stage": args.focus_oracle_stage,
                 "adapter_lesion": args.adapter_lesion,
                 "disable_core_stage": sorted(args.disable_core_stage)}
    signature_path = args.output_dir / "run_signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise RuntimeError("Evaluation provenance mismatch; use a new output directory")
    elif ((args.output_dir / "metrics.json").exists() or
          any((args.output_dir / "shards").glob("*.json"))):
        raise RuntimeError("Legacy cached predictions lack provenance; use a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(signature_path, signature)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    coco = COCO(str(args.annotation))
    image_ids = sorted(coco.getImgIds())
    if args.limit_images > 0:
        image_ids = image_ids[:args.limit_images]
    categories = sorted(coco.dataset["categories"], key=lambda x: int(x["id"]))

    model = build_sam3_image_model(
        bpe_path=str(args.bpe_path),
        checkpoint_path=str(args.checkpoint), load_from_HF=False,
        enable_segmentation=True, device=args.device, eval_mode=True)
    adapter_report = None
    adapter_controller = None
    adapter_payload = None
    if args.adapter_checkpoint is not None:
        from continual_lora.trainer import (
            enable_lora_training_compatibility, load_adapter_checkpoint)
        enable_lora_training_compatibility()
        adapter_controller, adapter_report, adapter_payload = load_adapter_checkpoint(
            model, args.adapter_checkpoint, device=args.device)
        if args.anchor_oracle_stage is not None:
            router = adapter_controller.anchor_router()
            if router is not None and hasattr(router, "force_stage"):
                router.force_stage(int(args.anchor_oracle_stage))
            elif router is not None and hasattr(router, "inference_oracle_stage"):
                router.inference_oracle_stage = int(args.anchor_oracle_stage)
            else:
                raise ValueError("checkpoint does not contain an oracle-capable anchor router")
        if args.focus_routing_mode is not None:
            adapter_controller.set_focus_capacity_routing(
                args.focus_routing_mode,blocks=args.focus_routing_blocks,
                oracle_stage=args.focus_oracle_stage)
        if args.adapter_lesion != "none":
            from continual_lora.methods import EnergyObjectSubspaceLoRALinear
            with torch.no_grad():
                for module in model.modules():
                    if not isinstance(module, EnergyObjectSubspaceLoRALinear):
                        continue
                    if args.adapter_lesion == "zero_core" and module.uses_object_core:
                        module.core_B.zero_()
                    elif args.adapter_lesion == "zero_private":
                        module.lora_A.zero_()
                        module.lora_B.zero_()
        if args.disable_core_stage:
            from continual_lora.methods import EnergyObjectSubspaceLoRALinear
            with torch.no_grad():
                for module in model.modules():
                    if not (isinstance(module, EnergyObjectSubspaceLoRALinear) and
                            module.uses_object_core):
                        continue
                    for stage in args.disable_core_stage:
                        if not 0 <= stage < module.core_B.shape[0]:
                            raise ValueError(f"disabled Core stage {stage} is out of range")
                        module.core_B[stage].zero_()
    model.requires_grad_(False)
    model.eval()
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)
    # SAM 3's official examples run the full image/text inference path under
    # bfloat16 autocast. Some backbone blocks intentionally emit BF16 tensors.
    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    autocast_context.__enter__()

    started = time.time(); completed = 0
    anchor_routes = []
    forced_stage_by_image = {}
    for index, iid in enumerate(image_ids, 1):
        shard = shard_dir / f"{iid}.json"
        if shard.exists():
            completed += 1; continue
        meta = coco.loadImgs([iid])[0]
        image_path = args.dataset_dir / meta["file_name"]
        image = Image.open(image_path).convert("RGB")
        forced_anchor_stage = None
        if args.anchor_route_error_rate is not None:
            # Stateless integer mixing makes the perturbation deterministic and
            # restart-safe: the same image receives the same forced route even
            # when evaluation resumes from existing shards.
            mixed = ((int(iid) * 1103515245 + args.anchor_route_error_seed * 12345)
                     & 0x7FFFFFFF)
            inject_error = (mixed / float(0x80000000)) < args.anchor_route_error_rate
            forced_anchor_stage = int(args.anchor_expected_stage)
            if inject_error and args.anchor_num_stages > 1:
                offset = 1 + ((mixed >> 8) % (args.anchor_num_stages - 1))
                forced_anchor_stage = int(
                    (args.anchor_expected_stage + offset) % args.anchor_num_stages)
            router = adapter_controller.anchor_router() if adapter_controller else None
            if router is not None and hasattr(router, "force_stage"):
                router.force_stage(forced_anchor_stage)
            elif router is not None and hasattr(router, "inference_oracle_stage"):
                router.inference_oracle_stage = forced_anchor_stage
            else:
                raise ValueError("checkpoint does not support controlled anchor routing")
        # A visual selector route is valid for one image only. Without this
        # reset, all later images silently reuse the first image's anchor.
        if adapter_controller is not None:
            adapter_controller.begin_anchor_batch()
        state = processor.set_image(image)
        image_anchor_routes = []
        visual_anchor_route = None
        if adapter_controller is not None:
            router = adapter_controller.anchor_router()
            route = (getattr(router, "_visual_posterior", None)
                     if router is not None else None)
            if route is None and router is not None:
                route = getattr(router, "_route", None)
            if torch.is_tensor(route):
                visual_anchor_route = route.detach().float().mean(0).cpu().tolist()
        records = []; concept_scores = {}
        for cat in categories:
            state = processor.set_text_prompt(str(cat.get("canonical_name", cat["name"])), state)
            if adapter_controller is not None:
                router = adapter_controller.anchor_router()
                route = (getattr(router, "_posterior", None)
                         if router is not None else None)
                if route is None and router is not None:
                    route = getattr(router, "_route", None)
                if torch.is_tensor(route):
                    image_anchor_routes.extend(route.detach().float().cpu().tolist())
            scores = state["scores"].detach().float().cpu().numpy()
            boxes = state["boxes"].detach().float().cpu().numpy()
            masks = state["masks"].detach().cpu().numpy()
            order = np.argsort(-scores)[:args.max_per_prompt]
            concept_scores[str(cat["id"])] = float(scores.max()) if scores.size else 0.0
            for j in order:
                x1, y1, x2, y2 = boxes[j].tolist()
                mask = masks[j, 0]
                area = float(mask.sum())
                if area <= 0 or x2 <= x1 or y2 <= y1:
                    continue
                records.append({"image_id": int(iid), "category_id": int(cat["id"]),
                    "score": float(scores[j]), "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "segmentation": encode_mask(mask), "area": area})
        write_json(shard, {"image_id": int(iid), "predictions": records,
                           "concept_scores": concept_scores,
                           "anchor_routes": image_anchor_routes,
                           "visual_anchor_route": visual_anchor_route,
                           "forced_anchor_stage": forced_anchor_stage})
        completed += 1
        elapsed = time.time() - started
        print(json.dumps({"progress": f"{completed}/{len(image_ids)}", "image_id": iid,
                          "predictions": len(records), "elapsed_sec": round(elapsed, 1),
                          "sec_per_image": round(elapsed / max(completed, 1), 2)}), flush=True)

    predictions=[]; max_scores={}; visual_anchor_routes=[]
    for iid in image_ids:
        d=json.load((shard_dir/f"{iid}.json").open())
        predictions.extend(d["predictions"])
        if d.get("anchor_routes"):
            anchor_routes.extend(np.asarray(x, dtype=np.float32)
                                 for x in d["anchor_routes"])
        elif d.get("anchor_route") is not None:
            anchor_routes.append(np.asarray(d["anchor_route"], dtype=np.float32))
        if d.get("visual_anchor_route") is not None:
            visual_anchor_routes.append(np.asarray(d["visual_anchor_route"],dtype=np.float32))
        if d.get("forced_anchor_stage") is not None:
            forced_stage_by_image[int(iid)] = int(d["forced_anchor_stage"])
        for cid,score in d["concept_scores"].items(): max_scores[(int(iid),int(cid))]=float(score)
    write_json(args.output_dir/"predictions.json", predictions)
    total_parameters=sum(p.numel() for p in model.parameters())
    adapter_parameters=adapter_report.trainable_parameters if adapter_report else 0
    routing_metrics = None
    if anchor_routes:
        route_matrix = np.stack(anchor_routes)
        selected = route_matrix.argmax(axis=1)
        source = coco.dataset.get("info",{}).get("source_dataset")
        expected_map = checkpoint_stage_map(adapter_payload)
        expected = expected_map.get(source)
        routing_metrics = {
            "available_anchors": int(route_matrix.shape[1]),
            "mean_posterior": route_matrix.mean(axis=0).tolist(),
            "selected_counts": {str(i): int((selected == i).sum())
                                for i in range(route_matrix.shape[1])},
            "mean_entropy": float(
                -(route_matrix * np.log(np.clip(route_matrix, 1e-12, 1))).sum(axis=1).mean()),
            "expected_anchor": expected,
            "selection_accuracy": (float((selected == expected).mean())
                                   if expected is not None and expected < route_matrix.shape[1]
                                   else None),
        }
    visual_routing_metrics = None
    if visual_anchor_routes:
        matrix=np.stack(visual_anchor_routes); selected=matrix.argmax(axis=1)
        source=coco.dataset.get("info",{}).get("source_dataset")
        expected=checkpoint_stage_map(adapter_payload).get(source)
        visual_routing_metrics={
            "mean_posterior":matrix.mean(axis=0).tolist(),
            "selected_counts":{str(i):int((selected==i).sum()) for i in range(matrix.shape[1])},
            "mean_entropy":float(-(matrix*np.log(np.clip(matrix,1e-12,1))).sum(1).mean()),
            "expected_anchor":expected,
            "selection_accuracy":(float((selected==expected).mean())
                                  if expected is not None and expected<matrix.shape[1] else None),
        }
    metrics={"dataset":coco.dataset.get("info",{}).get("source_dataset"),
             "annotation_sha256": signature["annotation_sha256"],
             "adapter_checkpoint_sha256": signature["adapter_checkpoint_sha256"],
             "base_checkpoint_sha256": signature["base_checkpoint_sha256"],
             "annotation":str(args.annotation),"checkpoint":str(args.checkpoint),
             "adapter_checkpoint":str(args.adapter_checkpoint) if args.adapter_checkpoint else None,
             "focus_routing":{"mode":args.focus_routing_mode,
                              "blocks":args.focus_routing_blocks,
                              "oracle_stage":args.focus_oracle_stage},
              "anchor_oracle_stage":args.anchor_oracle_stage,
              "routing_stress":({
                  "expected_stage": args.anchor_expected_stage,
                  "requested_error_rate": args.anchor_route_error_rate,
                  "seed": args.anchor_route_error_seed,
                  "forced_images": len(forced_stage_by_image),
                  "applied_error_rate": (
                      float(np.mean([stage != args.anchor_expected_stage
                                     for stage in forced_stage_by_image.values()]))
                      if forced_stage_by_image else None),
              } if args.anchor_route_error_rate is not None else None),
              "disabled_core_stages": sorted(set(args.disable_core_stage)),
             "multi_anchor_routing":routing_metrics,
             "visual_selector_routing":visual_routing_metrics,
             "parameters":{"total":total_parameters,"adapter":adapter_parameters,
                           "adapter_percent":100.0*adapter_parameters/max(total_parameters,1),
                           "injected_modules":len(adapter_report.modules) if adapter_report else 0},
             "threshold":args.threshold,"images":len(image_ids),"predictions":len(predictions),
             "box":evaluate(coco,predictions,image_ids,"bbox"),
             "mask":evaluate(coco,predictions,image_ids,"segm"),
             "concept":concept_metrics(coco,image_ids,max_scores,args.threshold),
             "runtime_sec":time.time()-started,
             "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    write_json(args.output_dir/"metrics.json",metrics)
    print(json.dumps(metrics,indent=2),flush=True)
    if not args.keep_predictions:
        (args.output_dir / "predictions.json").unlink(missing_ok=True)
        import shutil
        shutil.rmtree(shard_dir, ignore_errors=True)
        print(json.dumps({"cleanup": "temporary predictions removed",
                          "retained": str(args.output_dir / "metrics.json")}), flush=True)


if __name__ == "__main__":
    main()
