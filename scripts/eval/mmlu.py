"""Evaluate MMLU using nanochat's next-token answer-label scoring."""

import argparse
import os

import torch

from rl_lab.eval import EvalResult
from rl_lab.eval.report import EvaluationReport
from rl_lab.tasks.mmlu import MMLU
from rl_lab.distributed import init_distributed, cleanup_dist_groups
from rl_lab.utils.common import detect_device_type
from rl_lab.utils.huggingface import load_hf_model

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def evaluate(task, model, tokenizer, max_per_task=-1, on_result=None):
    """Choose the highest next-token logit among the task's single-token labels."""
    count = len(task) if max_per_task == -1 else min(len(task), max_per_task)
    if count == 0:
        raise ValueError("Cannot evaluate an empty task")
    model.eval()
    correct = 0
    with torch.inference_mode():
        for index in range(count):
            example = task[index]
            labels = task.evaluation_candidates(example)
            label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
            if any(len(ids) != 1 for ids in label_ids):
                raise ValueError("Each answer label must encode to exactly one token")
            prompt = task.build_eval_input(example)
            inputs = tokenizer.apply_chat_template(
                list(prompt.messages), tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            ).to(next(model.parameters()).device)
            logits = model(**inputs, use_cache=False).logits[0, -1]
            prediction = labels[logits[[ids[0] for ids in label_ids]].argmax().item()]
            score = int(task.evaluate(example, prediction))
            correct += score
            if on_result is not None:
                on_result(index, example, prediction, score)
            print(f"\r{correct}/{index + 1} correct", end="", flush=True)
    print()
    return EvalResult(num_examples=count, score=correct / count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default=MODEL_ID)
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default=None)
    parser.add_argument("--max-per-task", type=int, default=-1, help="Examples to score; -1 means all")
    args = parser.parse_args()
    if args.max_per_task != -1 and args.max_per_task < 1:
        parser.error("--max-per-task must be -1 or a positive integer")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run this script in a single process")
    dist_ctx = init_distributed(args.device_type or detect_device_type())
    try:
        model, tokenizer = load_hf_model(args.hf_path, dist_ctx)
        print(f"Evaluating {args.hf_path} on MMLU", flush=True)
        report = EvaluationReport("MMLU", args.hf_path, {
            **vars(args), "device": str(dist_ctx.device), "subset": "all", "split": "test",
            "evaluation": "highest next-token answer-label logit; no generation",
        })
        result = evaluate(
            MMLU(subset="all", split="test"), model, tokenizer,
            args.max_per_task, on_result=report.record,
        )
        print(f"MMLU: accuracy={result.score:.4f}, num_examples={result.num_examples}")
        print(f"Report: {report.write(result)}")
    finally:
        cleanup_dist_groups()


if __name__ == "__main__":
    main()
