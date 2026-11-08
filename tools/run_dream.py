"""Portable entry point for the recorded DREAM recipe and its evaluation."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"PASCAL_VOC_2012": "val", "HRSID": "test", "Kvasir-Instrument": "val",
          "Cityscapes": "val", "TrashCan-Instance": "val"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["train", "frozen", "evaluate"])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="Original SAM 3 checkpoint")
    p.add_argument("--bpe-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method-dir", type=Path)
    p.add_argument("--frozen-root", type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "configs/dream_routed.json")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    gpus = [x.strip() for x in a.gpus.split(",") if x.strip()]
    if not gpus or len(gpus) != len(set(gpus)) or not all(x.isdigit() for x in gpus):
        p.error("--gpus must contain distinct physical GPU indices")
    if not a.dry_run:
        for name in ["data_root", "checkpoint", "bpe_path"]:
            if not getattr(a, name).exists(): p.error(f"Missing {name}: {getattr(a, name)}")
    data, ckpt, bpe, output = [str(x.resolve()) for x in
                              [a.data_root, a.checkpoint, a.bpe_path, a.output_dir]]

    def run(cmd, selected=None):
        print(shlex.join(cmd), flush=True)
        if a.dry_run: return
        env = os.environ.copy()
        if selected is not None: env["CUDA_VISIBLE_DEVICES"] = selected
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    if a.action == "train":
        config = json.loads(a.config.read_text())
        if len(gpus) != config["training_processes"]:
            p.error("The recorded training recipe requires exactly two GPUs")
        args = list(config["train_args"])
        args[args.index("--seed") + 1] = str(a.seed)
        run([sys.executable, "-m", "torch.distributed.run", "--standalone",
             "--nproc_per_node=2", str(ROOT / "train_continual_lora.py"), *args,
             "--data-root", data, "--checkpoint", ckpt, "--bpe-path", bpe,
             "--output-dir", output], ",".join(gpus))
    elif a.action == "frozen":
        tasks = list(SPLITS.items())
        def worker(gpu, items):
            for name, split in items:
                dataset = Path(data) / name
                run([sys.executable, str(ROOT / "frozen_sam3_baseline.py"),
                     "--dataset-dir", str(dataset), "--annotation",
                     str(dataset / "annotations" / f"instances_{split}.json"),
                     "--checkpoint", ckpt, "--bpe-path", bpe,
                     "--output-dir", str(Path(output) / name), "--device", "cuda"], gpu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu, tasks[i::len(gpus)]) for i, gpu in enumerate(gpus)]
            for f in futures: f.result()
    else:
        if a.method_dir is None or a.frozen_root is None:
            p.error("evaluate requires --method-dir and --frozen-root")
        run([sys.executable, str(ROOT / "evaluate_continual_lora.py"),
             "--method-dir", str(a.method_dir.resolve()), "--evaluation-dir", output,
             "--data-root", data, "--frozen-root", str(a.frozen_root.resolve()),
             "--base-checkpoint", ckpt, "--bpe-path", bpe, "--gpus", ",".join(gpus)])


if __name__ == "__main__":
    main()
