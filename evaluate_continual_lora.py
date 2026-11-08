#!/usr/bin/env python3
"""Evaluate every continual checkpoint and summarize full CL metrics."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from continual_lora.eval_protocol import sha256_file


EVAL_SPLITS = {
    "PASCAL_VOC_2012": "val",
    "HRSID": "test",
    "Kvasir-Instrument": "val",
    "Cityscapes": "val",
    "TrashCan-Instance": "val",
}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.replace(path)


def validate_cached_metrics(metrics, *, annotation_sha256, base_checkpoint_sha256,
                            adapter_checkpoint_sha256, expected_images, location):
    """Reject results from different weights, splits, or incomplete coverage."""
    expected = {
        "annotation_sha256": annotation_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "adapter_checkpoint_sha256": adapter_checkpoint_sha256,
        "threshold": 0.3,
        "images": expected_images,
    }
    mismatches = [key for key, value in expected.items()
                  if key not in metrics or metrics[key] != value]
    if adapter_checkpoint_sha256 is None and metrics.get("adapter_checkpoint") is not None:
        mismatches.append("adapter_checkpoint")
    if mismatches:
        raise RuntimeError(
            f"Stale or unverified metrics at {location}: {', '.join(mismatches)}; "
            "use a new output directory")


def cl_summary(matrix, frozen):
    r=np.asarray(matrix,dtype=float); r0=np.asarray(frozen,dtype=float); n=len(r)
    final=float(r[-1].mean())
    forgetting=float(np.mean([r[j:,j].max()-r[-1,j] for j in range(n-1)]))
    bwt=float(np.mean([r[-1,j]-r[j,j] for j in range(n-1)]))
    fwt=float(np.mean([r[j-1,j]-r0[j] for j in range(1,n)]))
    aia=float(np.mean([r[i,:i+1].mean() for i in range(n)]))
    plasticity=float(np.mean([r[j,j]-r0[j] for j in range(n)]))
    retention=float(np.mean(r[-1]/np.maximum(r0,1e-12)))
    return {"ACC":final,"Forgetting":forgetting,"BWT":bwt,"FWT":fwt,
            "AIA":aia,"Plasticity":plasticity,"FrozenRetentionRatio":retention}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--method-dir",type=Path,required=True)
    p.add_argument("--evaluation-dir",type=Path,default=None,
                   help="Independent derived evaluation directory; never overwrite legacy results")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--frozen-root",type=Path,required=True)
    p.add_argument("--base-checkpoint",required=True)
    p.add_argument("--bpe-path",required=True)
    p.add_argument("--baseline-script",default=str(Path(__file__).with_name("frozen_sam3_baseline.py")))
    p.add_argument("--python",default=sys.executable)
    p.add_argument("--gpus",default="0,1",
                   help="Comma-separated physical GPU ids used as parallel evaluation workers")
    p.add_argument("--limit-images",type=int,default=0,
                   help="Integration-test limit; 0 evaluates complete datasets")
    a=p.parse_args()
    config=json.loads((a.method_dir/"config.json").read_text())
    stream_names=[task["name"] for task in config["stream"]]
    if len(stream_names)!=len(EVAL_SPLITS) or set(stream_names)!=set(EVAL_SPLITS):
        raise RuntimeError(
            "Checkpoint stream must contain every evaluation dataset exactly once: "
            f"{stream_names}")
    datasets=[(name,EVAL_SPLITS[name]) for name in stream_names]
    checkpoints=sorted(a.method_dir.glob("task_*.pt"))
    if len(checkpoints)!=len(datasets):
        raise RuntimeError(f"Expected {len(datasets)} task checkpoints, found {len(checkpoints)}")
    eval_root=a.evaluation_dir if a.evaluation_dir is not None else a.method_dir/"evaluation"
    base_checkpoint_sha256=sha256_file(a.base_checkpoint)
    annotation_info={}
    for dataset,split in datasets:
        annotation=a.data_root/dataset/"annotations"/f"instances_{split}.json"
        total_images=len(json.loads(annotation.read_text())["images"])
        annotation_info[dataset]=(sha256_file(annotation),
                                  min(a.limit_images,total_images) if a.limit_images>0 else total_images)
    checkpoint_sha256={str(ckpt):sha256_file(ckpt) for ckpt in checkpoints}
    jobs=[]
    for stage,ckpt in enumerate(checkpoints):
        for dataset,split in datasets:
            out=eval_root/f"stage_{stage:02d}"/dataset
            if not (out/"metrics.json").is_file():
                jobs.append((stage,ckpt,dataset,split,out))
            else:
                existing=json.loads((out/"metrics.json").read_text())
                annotation_sha256,expected_images=annotation_info[dataset]
                validate_cached_metrics(existing,annotation_sha256=annotation_sha256,
                    base_checkpoint_sha256=base_checkpoint_sha256,
                    adapter_checkpoint_sha256=checkpoint_sha256[str(ckpt)],
                    expected_images=expected_images,location=out)
    gpus=[x.strip() for x in a.gpus.split(",") if x.strip()]
    if not gpus: raise ValueError("--gpus must contain at least one GPU id")
    def evaluate_job(job,gpu):
        stage,ckpt,dataset,split,out=job
        out.mkdir(parents=True,exist_ok=True)
        cmd=[a.python,a.baseline_script,"--dataset-dir",str(a.data_root/dataset),
             "--annotation",str(a.data_root/dataset/"annotations"/f"instances_{split}.json"),
             "--checkpoint",a.base_checkpoint,"--adapter-checkpoint",str(ckpt),
             "--bpe-path",a.bpe_path,"--output-dir",str(out),"--device","cuda"]
        if a.limit_images: cmd += ["--limit-images",str(a.limit_images)]
        env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpu
        with (out/"evaluation.log").open("a",encoding="utf-8") as log:
            subprocess.run(cmd,check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        done={"stage":stage,"dataset":dataset,"gpu":gpu}
        print(json.dumps({"evaluation_complete":done}),flush=True)
        return done
    def gpu_worker(gpu,gpu_jobs):
        return [evaluate_job(job,gpu) for job in gpu_jobs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(gpu_worker,gpu,jobs[i::len(gpus)])
                 for i,gpu in enumerate(gpus)]
        for future in futures: future.result()
    frozen=[json.loads((a.frozen_root/d/"metrics.json").read_text()) for d,_ in datasets]
    for (dataset, split), metrics in zip(datasets, frozen):
        annotation_sha256,expected_images=annotation_info[dataset]
        validate_cached_metrics(metrics,annotation_sha256=annotation_sha256,
            base_checkpoint_sha256=base_checkpoint_sha256,adapter_checkpoint_sha256=None,
            expected_images=expected_images,location=a.frozen_root/dataset)
    rows=[]
    for stage in range(len(datasets)):
        rows.append([json.loads((eval_root/f"stage_{stage:02d}"/d/"metrics.json").read_text())
                     for d,_ in datasets])
        for (dataset,_),metrics in zip(datasets,rows[-1]):
            annotation_sha256,expected_images=annotation_info[dataset]
            validate_cached_metrics(metrics,annotation_sha256=annotation_sha256,
                base_checkpoint_sha256=base_checkpoint_sha256,
                adapter_checkpoint_sha256=checkpoint_sha256[str(checkpoints[stage])],
                expected_images=expected_images,location=eval_root/f"stage_{stage:02d}"/dataset)
    result={"datasets":stream_names,"stages":stream_names,
            "parameter_profile":config,
            "training_profile":json.loads((a.method_dir/"history.json").read_text()),
            "metrics":{}}
    for name,path in (("box_AP",("box","AP")),("box_AP50",("box","AP50")),
                      ("mask_AP",("mask","AP")),("mask_AP50",("mask","AP50")),
                      ("concept_F1",("concept","f1"))):
        mat=[[cell[path[0]][path[1]] for cell in row] for row in rows]
        base=[cell[path[0]][path[1]] for cell in frozen]
        result["metrics"][name]={"frozen":base,"matrix":mat,
                                  "continual":cl_summary(mat,base)}
    summary_root=eval_root if a.evaluation_dir is not None else a.method_dir
    dump(summary_root/"continual_metrics.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
