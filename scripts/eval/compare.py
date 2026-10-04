"""Compare named Hugging Face models/checkpoints on the scored evaluation suite."""

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import re
import warnings

import torch
from transformers import set_seed

from rl_lab.engines.transformers import TransformersEngine
from rl_lab.eval import evaluate_task
from rl_lab.eval.evaluator import _prepare_prompt
from rl_lab.tasks.arc import ARC
from rl_lab.tasks.gpqa import GPQA
from rl_lab.tasks.gsm8k import GSM8K
from rl_lab.tasks.humaneval import HumanEval
from rl_lab.tasks.mmlu import MMLU
from rl_lab.tasks.spellingbee import SimpleSpelling, SpellingBee
from rl_lab.utils.common import detect_device_type
from scripts.eval.mmlu import evaluate as evaluate_categorical


TASKS = {
    "gsm8k": lambda: GSM8K(subset="main", split="test"),
    "mmlu": lambda: MMLU(subset="all", split="test"),
    "arc_easy": lambda: ARC(subset="ARC-Easy", split="test"),
    "arc_challenge": lambda: ARC(subset="ARC-Challenge", split="test"),
    "gpqa": lambda: GPQA(subset="gpqa_diamond"),
    "humaneval": HumanEval,
    "spellingbee": lambda: SpellingBee(split="test"),
    "simple_spelling": lambda: SimpleSpelling(split="test"),
    "core": None,
}
SYSTEM_PROMPTS = {
    "gsm8k": "Solve the problem and end with #### followed by the numerical answer.",
    "humaneval": "Write the complete Python function, including its signature. Return only Python code.",
    "spellingbee": "End with #### followed by the count.",
    "simple_spelling": "Reply only in the format word:w,o,r,d, spelling the requested word.",
}
# Same plain role format as the SFT recipe; no TRL dependency for evaluation.
FALLBACK_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] | capitalize }}: "
    "{{ message['content'] }}{% if message['role'] == 'assistant' %}"
    "{{ eos_token }}{% else %}{{ '\n' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}Assistant:{% endif %}"
)


def select_tasks(value):
    if value is None or value.lower() in {"none", "all"}:
        return list(TASKS)
    names = []
    for name in value.split(","):
        name = name.strip().lower()
        for expanded in (["arc_easy", "arc_challenge"] if name == "arc" else [name]):
            if expanded not in TASKS:
                raise ValueError(f"Unknown/unscored task {expanded!r}; choose {','.join(TASKS)}")
            if expanded not in names:
                names.append(expanded)
    return names


class CoreTokenizer:
    """Honor CORE's explicit BOS prefix with a Hugging Face tokenizer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_bos_token_id(self):
        return self.tokenizer.bos_token_id

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def __call__(self, prompts, prepend=None):
        rows = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        return {"input_ids": [[prepend] + row if prepend is not None else row for row in rows]}


def check_context(task, engine, name, limit, max_new_tokens):
    context = getattr(engine.model.config, "max_position_embeddings", None)
    if context is None:
        return
    count = len(task) if limit == -1 else min(len(task), limit)
    for index in range(count):
        prompt = _prepare_prompt(task.build_eval_input(task[index]), SYSTEM_PROMPTS.get(name))
        ids = engine.tokenizer.apply_chat_template(
            list(prompt.messages), tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        budget = max_new_tokens if task.eval_type == "generative" else 0
        if len(ids) + budget > context:
            raise ValueError(f"{name} example {index}: prompt ({len(ids)}) + generation "
                             f"budget ({budget}) exceeds context ({context}); no examples were discarded")


def plot_results(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    names = list(dict.fromkeys(name for model in results["models"] for name in model["tasks"]))
    for name in names:
        records = [(model["label"], model["tasks"][name]) for model in results["models"]
                   if name in model["tasks"]]
        fig, ax = plt.subplots(figsize=(max(6, 2 * len(records)), 4.5), layout="constrained")
        bars = ax.bar(range(len(records)), [record["score"] for _, record in records],
                      color=["#4477AA", "#228833", "#CCBB44", "#AA3377"])
        ax.set_xticks(range(len(records)), [label for label, _ in records], rotation=15, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Accuracy")
        ax.set_title(name)
        ax.bar_label(bars, labels=[f"{r['score']:.1%}\nn={r['num_examples']}" for _, r in records], padding=4)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for extension in ("png", "pdf"):
            slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
            fig.savefig(plots / f"{slug}.{extension}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--tasks", default=None, help="Comma-separated tasks; omitted, None or all runs all scored tasks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-type", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--max-per-task", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        names = select_tasks(args.tasks)
    except ValueError as error:
        parser.error(str(error))
    models = []
    for value in args.model:
        label, separator, path = value.partition("=")
        if not separator or not label.strip() or not path.strip():
            parser.error("Each --model must be LABEL=PATH with a nonempty label and path")
        models.append((label.strip(), path.strip()))
    if len({label for label, _ in models}) != len(models):
        parser.error("Model labels must be unique")
    if args.max_per_task != -1 and args.max_per_task < 1:
        parser.error("--max-per-task must be -1 or positive")
    if "core" in names and args.max_per_task != -1 and args.max_per_task < 11:
        parser.error("CORE requires --max-per-task >= 11 for its 10-shot tasks, or -1")
    if min(args.batch_size, args.max_new_tokens) < 1:
        parser.error("--batch-size and --max-new-tokens must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This script supports single-process evaluation")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty --output-dir to preserve previous results")
    # Fail before expensive evaluation if the optional plotting dependency is absent.
    try:
        import matplotlib
    except ImportError:
        parser.error("Run with uv run --locked --with matplotlib python -m scripts.eval.compare ...")
    device = args.device_type or detect_device_type()
    set_seed(args.seed)
    tasks = {name: TASKS[name]() for name in names if name != "core"}
    output.mkdir(parents=True, exist_ok=True)
    results = {"settings": {**vars(args), "output_dir": str(output), "tasks": names,
                            "device_type": device, "temperature": 0.0,
                            "system_prompts": SYSTEM_PROMPTS}, "models": []}
    for model_index, (label, path) in enumerate(models):
        set_seed(args.seed)
        entry = {"label": label, "path": path, "tasks": {}, "complete": False}
        results["models"].append(entry)
        with TransformersEngine.load(path, device=device) as engine:
            engine.model.eval()
            if engine.tokenizer.pad_token_id is None:
                if engine.tokenizer.eos_token_id is None:
                    raise ValueError(f"{label}: tokenizer requires a PAD or EOS token")
                engine.tokenizer.pad_token = engine.tokenizer.eos_token
            if not engine.tokenizer.chat_template:
                warnings.warn(f"{label}: no chat template; using the SFT recipe's plain role format")
                engine.tokenizer.chat_template = FALLBACK_TEMPLATE
            entry["chat_template"] = engine.tokenizer.chat_template
            for name in names:
                print(f"Evaluating {label} on {name}", flush=True)
                if name == "core":
                    from rl_lab.tasks.core import evaluate_core
                    # Preserve CORE's existing continuation scoring and left-cropping protocol.
                    engine.model.max_seq_len = getattr(engine.model.config, "max_position_embeddings", None)
                    core = evaluate_core(engine.model, CoreTokenizer(engine.tokenizer),
                                         torch.device(device), max_per_task=args.max_per_task)
                    entry["core_metrics"] = {key: value for key, value in core.items() if key != "all_core_results"}
                    for task_name, metrics in core["all_core_results"].items():
                        entry["tasks"][f"core_{task_name}"] = {"score": metrics["accuracy"], **metrics}
                else:
                    task = tasks[name]
                    check_context(task, engine, name, args.max_per_task, args.max_new_tokens)
                    predictions = output / f"model-{model_index}-{name}.jsonl"
                    with predictions.open("x", encoding="utf-8") as stream:
                        def record(index, example, prediction, score):
                            stream.write(json.dumps({"index": index, "example": example,
                                                     "reference": example["messages"][-1]["content"],
                                                     "prediction": prediction, "score": score}) + "\n")
                        if task.eval_type == "categorical":
                            result = evaluate_categorical(task, engine.model, engine.tokenizer,
                                                          args.max_per_task, on_result=record)
                        else:
                            result = evaluate_task(task, engine, batch_size=args.batch_size,
                                                   limit=None if args.max_per_task == -1 else args.max_per_task,
                                                   max_new_tokens=args.max_new_tokens, on_result=record,
                                                   system_prompt=SYSTEM_PROMPTS.get(name))
                    entry["tasks"][name] = {**asdict(result), "predictions": predictions.name}
                (output / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            entry["complete"] = True
            (output / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    plot_results(results, output)
    print(f"Results: {output / 'evaluation.json'}\nPlots: {output / 'plots'}")


if __name__ == "__main__":
    main()
