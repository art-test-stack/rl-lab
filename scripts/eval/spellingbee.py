"""Evaluate spellingbee with one greedy completion per example and its existing scorer."""

import argparse
import os

from rl_lab.engines.transformers import TransformersEngine
from rl_lab.eval import evaluate_task
from rl_lab.eval.report import EvaluationReport
from rl_lab.tasks.spellingbee import SpellingBee, SimpleSpelling
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
    parser.add_argument("--task", choices=["spellingbee", "simple_spelling"], default="spellingbee")
    parser.add_argument("--size", type=int, default=1000)
    args = parser.parse_args()
    if args.max_per_task != -1 and args.max_per_task < 1:
        parser.error("--max-per-task must be -1 or a positive integer")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run this script in a single process")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--batch-size and --max-new-tokens must be positive")
    if args.size < 1:
        parser.error("--size must be positive")
    task_class = SpellingBee if args.task == "spellingbee" else SimpleSpelling
    system_prompt = (
        "End with #### followed by the count."
        if args.task == "spellingbee"
        else "Reply only in the format word:w,o,r,d, spelling the requested word."
    )
    dist_ctx = init_distributed(args.device_type or detect_device_type())
    try:
        model, tokenizer = load_hf_model(args.hf_path, dist_ctx)
        with TransformersEngine(model, tokenizer) as engine:
            task = task_class(size=args.size, split="test")
            print(f"Evaluating {args.hf_path} on {args.task}", flush=True)
            report = EvaluationReport(args.task, args.hf_path, {
                **vars(args), "device": str(dist_ctx.device),
                "split": "test", "temperature": 0.0,
                "system_prompt": system_prompt,
                "evaluation": "one greedy completion; existing task scorer",
            })
            result = evaluate_task(
                task, engine, batch_size=args.batch_size,
                limit=None if args.max_per_task == -1 else args.max_per_task,
                max_new_tokens=args.max_new_tokens,
                on_result=report.record,
                system_prompt=system_prompt,
            )
            print(f"accuracy={result.score:.4f}, num_examples={result.num_examples}")
            print(f"Report: {report.write(result)}")
    finally:
        cleanup_dist_groups()


if __name__ == "__main__":
    main()
