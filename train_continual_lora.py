#!/usr/bin/env python3
import argparse
from continual_lora.trainer import DEFAULT_STREAM, train_stream
from continual_lora.methods import METHODS


def main():
    p=argparse.ArgumentParser(description="Fixed-budget continual LoRA training for SAM 3")
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bpe-path", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rank", type=int, default=16); p.add_argument("--alpha", type=float, default=8.)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--steps-per-task", type=int, default=1000); p.add_argument("--fisher-batches", type=int, default=8)
    p.add_argument("--resolution", type=int, default=1008); p.add_argument("--category-chunk-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", default="cuda")
    p.add_argument("--memory-per-task", type=int, default=20)
    p.add_argument("--replay-every", type=int, default=4)
    p.add_argument("--distill-weight", type=float, default=1.0)
    p.add_argument("--focus-stable-ratio", type=float, default=0.75)
    p.add_argument("--focus-merge-gamma", type=float, default=0.75)
    p.add_argument("--focus-orthogonality-weight", type=float, default=1e-3)
    p.add_argument("--focus-admission-ratio", type=float, default=0.25,
                   help="FOCUS-v2 fraction of each function rank pool admitted per stage")
    p.add_argument("--focus-importance-decay", type=float, default=0.99,
                   help="FOCUS-v2 retention factor for global rank importance")
    p.add_argument("--focus-top-k", type=int, default=4,
                   help="FOCUS-v3 input-conditioned active atoms per token")
    p.add_argument("--focus-conflict-every", type=int, default=4,
                   help="Apply semantic/box/mask gradient conflict removal every K steps")
    p.add_argument("--focus-consistency-weight", type=float, default=0.1)
    p.add_argument("--global-retention-weight", type=float, default=1.0,
                   help="Previous-stage local Global output MSE weight; 0 disables retention.")
    p.add_argument("--cqsm-shared-layout", choices=["late_parallel", "visual_prefix", "visual_residual"],
                   default="visual_residual", help="visual_residual preserves all private LoRA maps")
    p.add_argument("--cqsm-global-rank", type=int, default=10)
    p.add_argument("--cqsm-memory-rank", type=int, default=2, choices=[1,2,4],
                   help="Memory rank sweep; private ranks and scaling remain at the final recipe")
    p.add_argument("--cqsm-energy-calibration", action=argparse.BooleanOptionalAction, default=True,
                   help="Ablate only task-end memory calibration, keeping restored training objectives")
    p.add_argument("--shared-lr", type=float, default=1e-4)
    p.add_argument("--shared-retention-ratio", type=float, default=0.2)
    p.add_argument("--global-deconflict", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sdo-conflict-weight", type=float, default=.05)
    p.add_argument("--sdo-preservation-weight", type=float, default=1.)
    p.add_argument("--sdo-lr", type=float, default=1e-3)
    p.add_argument("--sdo-hidden", type=int, default=32)
    p.add_argument("--sdo-max-correction", type=float, default=.25)
    p.add_argument("--sdo-warmup-steps", type=int, default=50)
    p.add_argument("--global-functional-conflict", action=argparse.BooleanOptionalAction, default=False,
                   help="Project Global semantic/box/mask gradients separately after rank averaging")
    p.add_argument("--global-independent-clip", action=argparse.BooleanOptionalAction, default=False,
                   help="Clip Global and private/CQSM groups independently to unit norm")
    p.add_argument("--cqsm-global-gating", action=argparse.BooleanOptionalAction, default=False,
                   help="One sigmoid residual gate per Global Q/V module")
    p.add_argument("--cqsm-global-gate-init", type=float, default=0.1)
    p.add_argument("--object-utility-samples", type=int, default=32,
                   help="Per-GPU current samples used for parent probing and utility selector labels")
    p.add_argument("--unified-object-weight", type=float, default=0.1,
                   help="Training-only Query-Box-Mask representation loss weight")
    p.add_argument("--unified-object-mode",
                   choices=["query_teacher", "semantic_query_teacher", "symmetric"],
                   default="query_teacher",
                   help="Asymmetric evidence-to-query distillation or symmetric ablation")
    p.add_argument("--energy-covariance-batches", type=int, default=16,
                   help="Per-GPU current-task batches for Query covariance estimation")
    p.add_argument("--energy-calibration-lambda", type=float, default=3.0,
                   help="Current-energy weight in closed-form general-subspace calibration")
    p.add_argument("--cqsm-shared", action=argparse.BooleanOptionalAction, default=True,
                   help="CQSM-only ablation: enable the continually updated global LoRA")
    p.add_argument("--cqsm-private", action=argparse.BooleanOptionalAction, default=True,
                   help="CQSM-only ablation: enable routed private adapters")
    p.add_argument("--cqsm-basis", choices=("covariance", "random"), default="covariance",
                   help="CQSM basis; random uses fixed seed 42+1009*stage, not test tuning")
    p.add_argument("--cqsm-bank-mode", choices=("all", "off", "current", "routed"),
                   default="all", help="CQSM-only bank availability; current means latest learned entry")
    p.add_argument("--cqsm-restore-legacy", action="store_true",
                   help="Restore energy calibration, semantic Query teacher, consistency, private orthogonality and periodic visual gradient conflict removal; keep finite-gradient checks and the corrected presence loss")
    p.add_argument("--max-tasks", type=int, default=0,
                   help="Limit task count for integration tests; 0 runs the full stream")
    p.add_argument("--stream-order", default="",
                   help="Comma-separated permutation of the five dataset names")
    args=vars(p.parse_args()); max_tasks=args.pop("max_tasks")
    stream_order=args.pop("stream_order")
    stream=DEFAULT_STREAM
    if stream_order:
        names=[name.strip() for name in stream_order.split(",") if name.strip()]
        by_name={task.name:task for task in DEFAULT_STREAM}
        if len(names)!=len(DEFAULT_STREAM) or set(names)!=set(by_name):
            raise ValueError(
                "--stream-order must contain every default dataset exactly once; "
                f"received {names}")
        stream=tuple(by_name[name] for name in names)
    if max_tasks: stream=stream[:max_tasks]
    if stream!=DEFAULT_STREAM or max_tasks: args["stream"]=stream
    train_stream(**args)


if __name__ == "__main__": main()
