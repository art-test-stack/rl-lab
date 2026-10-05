"""Evaluate ARC-Easy and ARC-Challenge using nanochat's next-token answer-label scoring."""

import argparse
import os

from rl_lab.eval import evaluate_categorical as evaluate
from rl_lab.eval.report import EvaluationReport
from rl_lab.tasks.arc import ARC
from rl_lab.distributed import init_distributed, cleanup_dist_groups
from rl_lab.utils.common import detect_device_type
from rl_lab.utils.huggingface import load_hf_model

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default=MODEL_ID)
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default=None)
    parser.add_argument("--max-per-task", type=int, default=-1, help="Examples to score; -1 means all")
    parser.add_argument("--subset", choices=["ARC-Easy", "ARC-Challenge"], default=None, help="Default: run both subsets")
    args = parser.parse_args()
    if args.max_per_task != -1 and args.max_per_task < 1:
        parser.error("--max-per-task must be -1 or a positive integer")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run this script in a single process")
    dist_ctx = init_distributed(args.device_type or detect_device_type())
    try:
        model, tokenizer = load_hf_model(args.hf_path, dist_ctx)
        subsets = [args.subset] if args.subset else ["ARC-Easy", "ARC-Challenge"]
        for subset in subsets:
            print(f"Evaluating {args.hf_path} on {subset}", flush=True)
            report = EvaluationReport(subset, args.hf_path, {
                **vars(args), "device": str(dist_ctx.device), "split": "test",
                "evaluation": "highest next-token answer-label logit; no generation",
            })
            result = evaluate(
                ARC(subset=subset, split="test"), model, tokenizer,
                args.max_per_task, on_result=report.record,
            )
            print(f"{subset}: accuracy={result.score:.4f}, num_examples={result.num_examples}")
            print(f"Report: {report.write(result)}")
    finally:
        cleanup_dist_groups()


if __name__ == "__main__":
    main()
