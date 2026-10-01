"""Evaluate gpqa with one greedy completion per example and its existing scorer."""

import argparse
import os

from rl_lab.engines.transformers import TransformersEngine
from rl_lab.eval import evaluate_task
from rl_lab.eval.report import EvaluationReport
from rl_lab.tasks.gpqa import GPQA
from rl_lab.distributed import init_distributed, cleanup_dist_groups
from rl_lab.utils.common import detect_device_type
from rl_lab.utils.huggingface import load_hf_model

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default=MODEL_ID)
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default=None)
    parser.add_argument("--max-per-task", type=int, default=-1, help="Examples to score; -1 means all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--subset", choices=GPQA.subsets, default="gpqa_diamond")
    args = parser.parse_args()
    if args.max_per_task != -1 and args.max_per_task < 1:
        parser.error("--max-per-task must be -1 or a positive integer")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run this script in a single process")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--batch-size and --max-new-tokens must be positive")
    dist_ctx = init_distributed(args.device_type or detect_device_type())
    try:
        model, tokenizer = load_hf_model(args.hf_path, dist_ctx)
        with TransformersEngine(model, tokenizer) as engine:
            task = GPQA(subset=args.subset)
            print(f"Evaluating {args.hf_path} on gpqa", flush=True)
            report = EvaluationReport(f"GPQA-{args.subset}", args.hf_path, {
                **vars(args), "device": str(dist_ctx.device),
                "split": "train", "temperature": 0.0,
                "system_prompt": None,
                "evaluation": "one greedy completion; existing task scorer",
            })
            result = evaluate_task(
                task, engine, batch_size=args.batch_size,
                limit=None if args.max_per_task == -1 else args.max_per_task,
                max_new_tokens=args.max_new_tokens,
                on_result=report.record,
            )
            print(f"accuracy={result.score:.4f}, num_examples={result.num_examples}")
            print(f"Report: {report.write(result)}")
    finally:
        cleanup_dist_groups()


if __name__ == "__main__":
    main()
