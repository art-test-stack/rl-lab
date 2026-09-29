"""Run with python -m rl_lab.eval; one offline job, no local server."""

import argparse
import json
from dataclasses import asdict

from rl_lab.engines import GenerationConfig, create_engine

from .evaluator import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=["gsm8k", "mmlu"], required=True)
    parser.add_argument(
        "--engine", choices=["transformers", "vllm", "openai"], default="transformers"
    )
    parser.add_argument("--device", help="Transformers model device, e.g. cpu or cuda")
    parser.add_argument("--base-url", help="Existing OpenAI-compatible endpoint")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.task == "mmlu" and args.engine != "transformers":
        parser.error(
            "MMLU requires loglikelihood, currently implemented by transformers"
        )
    if args.device and args.engine != "transformers":
        parser.error("--device is only supported by transformers")
    if args.base_url and args.engine != "openai":
        parser.error("--base-url is only supported by openai")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if args.tensor_parallel_size != 1 and args.engine != "vllm":
        parser.error("--tensor-parallel-size is only supported by vllm")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")
    config = GenerationConfig(max_new_tokens=args.max_new_tokens)
    if args.task == "gsm8k":
        from rl_lab.tasks.gsm8k import GSM8K

        task = GSM8K(subset="main", split="test")
    else:
        from rl_lab.tasks.mmlu import MMLU

        task = MMLU(subset="all", split="test")
    kwargs = {}
    if args.engine == "transformers":
        kwargs.update(device=args.device, batch_size=args.batch_size)
    elif args.engine == "vllm":
        kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    elif args.base_url:
        kwargs["base_url"] = args.base_url
    with create_engine(args.engine, model_name=args.model, **kwargs) as engine:
        result = evaluate(
            task,
            engine,
            config=config,
            batch_size=args.batch_size,
            limit=args.limit,
            system_prompt=(
                "Solve the problem and end your answer with #### followed by the numerical answer."
                if args.task == "gsm8k"
                else None
            ),
        )
    print(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
