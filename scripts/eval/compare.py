"""Compare named Hugging Face models/checkpoints on the scored evaluation suite."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import re
import warnings

import torch
from transformers import set_seed
from transformers.utils import get_json_schema

from rl_lab.engines.transformers import TransformersEngine
from rl_lab.eval import EvalResult, evaluate_categorical, evaluate_task
from rl_lab.eval.evaluator import _prepare_prompt
from rl_lab.tasks.arc import ARC
from rl_lab.tasks.gpqa import GPQA
from rl_lab.tasks.gsm8k import GSM8K
from rl_lab.tasks.humaneval import HumanEval
from rl_lab.tasks.mmlu import MMLU
from rl_lab.tasks.spellingbee import SimpleSpelling, SpellingBee
from rl_lab.tools.exec import make_python_tool
from rl_lab.utils.common import detect_device_type
from rl_lab.utils.default import MODELS_FOLDER, REPORTS_FOLDER
from rl_lab.tokenizer import QWEN_RESPONSE_TEMPLATE


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
TOOL_INSTRUCTION = (
    "You may use the python tool for calculations. Print the result in your code, "
    "then use the tool response to answer the user."
)


def resolve_model(reference):
    """Resolve HF_ID:METHOD[:RUN] in the training cache; preserve other paths/IDs."""
    if Path(reference).is_dir() or ":" not in reference:
        return reference
    parts = reference.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("Cached models use HF_ID:METHOD[:RUN]")
    model_id, method = parts[:2]
    run = parts[2] if len(parts) == 3 else "default"
    if (len(model_id.split("/")) > 2 or not all(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part)
            for part in [*model_id.split("/"), method, run])):
        raise ValueError("Invalid cached model ID, method or run name")
    path = MODELS_FOLDER / model_id.replace("/", "--") / method / run
    if not (path / "config.json").is_file():
        raise ValueError(f"No cached checkpoint for {reference}: expected {path}")
    return str(path)


def select_tool_models(tools, value, labels):
    """Resolve explicit model labels; tools and selectors accept case-insensitive none."""
    if tools is None or tools.strip().lower() == "none":
        return []
    if tools.strip().lower() != "python":
        raise ValueError("--tools supports python or none")
    if value is None or value.strip().lower() == "none":
        return []
    if value.strip().lower() == "all":
        return labels
    selected = list(dict.fromkeys(label.strip() for label in value.split(",")))
    if any(label not in labels for label in selected):
        raise ValueError("--tool-models must contain model labels from --model, or all/none")
    return selected


def tool_response_template(engine, tools):
    """Use tokenizer-native parsing, or the existing Qwen2 tool response schema."""
    tokenizer = engine.tokenizer
    if not tokenizer.chat_template:
        raise ValueError("Native tool calling requires a tool-aware chat template; GPT-2 is unsupported")
    messages = [{"role": "user", "content": "Calculate a number."}]
    plain = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    rendered = tokenizer.apply_chat_template(
        messages, tools=[get_json_schema(tool) for tool in tools],
        tokenize=False, add_generation_prompt=True,
    )
    if plain == rendered:
        raise ValueError("The chat template does not render tool descriptions; native tools are unsupported")
    template = getattr(tokenizer, "response_template", None)
    if template is not None:
        return template
    if (engine.model.config.model_type == "qwen2"
            and "<tool_call>" in rendered and "<|im_start|>" in rendered):
        return QWEN_RESPONSE_TEMPLATE
    raise ValueError("No native tool response parser is available for this model; "
                     "use a tokenizer response_template or a Qwen2 tool-aware model")


def tool_system_prompt(name):
    return " ".join(filter(None, [SYSTEM_PROMPTS.get(name), TOOL_INSTRUCTION]))


def evaluate_with_tools(task, engine, name, args, tools, response_template, on_result):
    """Score final assistant answers; preserve the full native tool exchange."""
    count = len(task) if args.max_per_task == -1 else min(len(task), args.max_per_task)
    if count == 0:
        raise ValueError("Cannot evaluate an empty task")
    total = 0.0
    for start in range(0, count, args.batch_size):
        examples = [task[index] for index in range(start, min(start + args.batch_size, count))]
        conversations = [list(_prepare_prompt(
            task.build_eval_input(example), tool_system_prompt(name),
        ).messages) for example in examples]
        histories = engine.complete(
            conversations, tools=tools, response_template=response_template,
            max_new_tokens=args.max_new_tokens, max_tool_rounds=args.max_tool_rounds,
        )
        if len(histories) != len(examples):
            raise RuntimeError("Engine returned an unexpected number of tool conversations")
        for offset, (example, history) in enumerate(zip(examples, histories, strict=True)):
            final = history[-1]
            if final.get("role") != "assistant" or final.get("tool_calls"):
                raise ValueError("Tool evaluation requires a final assistant answer without tool calls")
            prediction = final["content"]
            score = float(task.evaluate(example, prediction))
            if not math.isfinite(score):
                raise ValueError("Task returned a non-finite score")
            total += score
            on_result(start + offset, example, prediction, score, history=history)
    return EvalResult(num_examples=count, score=total / count)


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


def check_context(task, engine, name, limit, max_new_tokens, tools=()):
    context = getattr(engine.model.config, "max_position_embeddings", None)
    if context is None:
        return
    count = len(task) if limit == -1 else min(len(task), limit)
    for index in range(count):
        prompt = _prepare_prompt(task.build_eval_input(task[index]),
                                 tool_system_prompt(name) if tools else SYSTEM_PROMPTS.get(name))
        template_kwargs = {"tools": [get_json_schema(tool) for tool in tools]} if tools else {}
        ids = engine.tokenizer.apply_chat_template(
            list(prompt.messages), tokenize=True, add_generation_prompt=True, return_dict=False,
            **template_kwargs,
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
        ax.set_xticks(range(len(records)), [
            f"{label}\n[tools: {','.join(record.get('tools', [])) or 'none'}]"
            for label, record in records
        ], rotation=15, ha="right")
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("Accuracy")
        ax.set_title(name)
        ax.bar_label(bars, labels=[f"{r['score']:.1%}\nn={r['num_examples']}" for _, r in records], padding=4)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for extension in ("png", "pdf"):
            slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
            fig.savefig(plots / f"{slug}.{extension}", dpi=180)
        plt.close(fig)

    benchmarks = [name for name in names if not name.startswith("core_")]
    if any("core/core" in model.get("core_metrics", {}) for model in results["models"]):
        benchmarks.append("core")
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(benchmarks)), 5), layout="constrained")
    width = 0.8 / len(results["models"])
    for index, model in enumerate(results["models"]):
        scores = [model.get("core_metrics", {}).get("core/core", float("nan"))
                  if name == "core" else model["tasks"].get(name, {}).get("score", float("nan"))
                  for name in benchmarks]
        positions = [position - 0.4 + width * (index + 0.5) for position in range(len(benchmarks))]
        bars = ax.bar(positions, scores, width=width, label=model["label"])
        ax.bar_label(bars, labels=[f"{score:.1%}" if math.isfinite(score) else "" for score in scores],
                     padding=3, fontsize=8)
    ax.set_xticks(range(len(benchmarks)), [
        "CORE\n(centered accuracy)" if name == "core" else name for name in benchmarks
    ], rotation=20, ha="right")
    ax.set_ylim(min(0, ax.get_ylim()[0]) * 1.4, max(1.15, ax.get_ylim()[1]))
    ax.set_ylabel("Score (accuracy; CORE: centered accuracy)")
    ax.set_title("Model comparison across benchmarks")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()
    for extension in ("png", "pdf"):
        fig.savefig(plots / f"overview.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=MODEL",
                        help="HF ID, local checkpoint path, or HF_ID:METHOD[:RUN]")
    parser.add_argument("--tasks", default=None, help="Comma-separated tasks; omitted, None or all runs all scored tasks")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory; default: CACHE_DIR/reports/compare-<UTC timestamp>")
    parser.add_argument("--device-type", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--max-per-task", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tools", default="none", help="python or none (case-insensitive); default: none")
    parser.add_argument("--tool-models", default="all", help="Comma-separated model labels, all or none; default: all")
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument("--python-timeout", type=float, default=5.0, help="Seconds per Python call")
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
        try:
            models.append((label.strip(), resolve_model(path.strip())))
        except ValueError as error:
            parser.error(str(error))
    if len({label for label, _ in models}) != len(models):
        parser.error("Model labels must be unique")
    try:
        tool_models = select_tool_models(args.tools, args.tool_models, [label for label, _ in models])
    except ValueError as error:
        parser.error(str(error))
    if args.max_tool_rounds < 1 or not math.isfinite(args.python_timeout) or args.python_timeout <= 0:
        parser.error("--max-tool-rounds and --python-timeout must be positive and finite")
    if args.max_per_task != -1 and args.max_per_task < 1:
        parser.error("--max-per-task must be -1 or positive")
    if "core" in names and args.max_per_task != -1 and args.max_per_task < 11:
        parser.error("CORE requires --max-per-task >= 11 for its 10-shot tasks, or -1")
    if min(args.batch_size, args.max_new_tokens) < 1:
        parser.error("--batch-size and --max-new-tokens must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This script supports single-process evaluation")
    output = args.output_dir or REPORTS_FOLDER / f"compare-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    print(f"Evaluation output: {output}", flush=True)
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
                            "system_prompts": SYSTEM_PROMPTS, "tool_models": tool_models,
                            "tool_instruction": TOOL_INSTRUCTION}, "models": []}
    for model_index, (label, path) in enumerate(models):
        set_seed(args.seed)
        entry = {"label": label, "path": path, "tasks": {}, "complete": False}
        results["models"].append(entry)
        with TransformersEngine.load(path, device=device) as engine:
            engine.model.eval()
            tools = [make_python_tool(timeout=args.python_timeout)] if label in tool_models else []
            response_template = None
            if tools and any(task.eval_type == "generative" for task in tasks.values()):
                try:
                    response_template = tool_response_template(engine, tools)
                except ValueError as error:
                    raise ValueError(f"{label}: {error}") from error
                entry["response_template"] = response_template
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
                        entry["tasks"][f"core_{task_name}"] = {"score": metrics["accuracy"], **metrics, "tools": []}
                else:
                    task = tasks[name]
                    task_tools = tools if task.eval_type == "generative" else []
                    check_context(task, engine, name, args.max_per_task, args.max_new_tokens, task_tools)
                    predictions = output / f"model-{model_index}-{name}.jsonl"
                    with predictions.open("x", encoding="utf-8") as stream:
                        def record(index, example, prediction, score, **extra):
                            stream.write(json.dumps({"index": index, "example": example,
                                                     "reference": example["messages"][-1]["content"],
                                                     "prediction": prediction, "score": score,
                                                     "tools": [tool.__name__ for tool in task_tools], **extra}) + "\n")
                        if task.eval_type == "categorical":
                            result = evaluate_categorical(task, engine.model, engine.tokenizer,
                                                          args.max_per_task, on_result=record)
                        elif task_tools:
                            result = evaluate_with_tools(task, engine, name, args, task_tools,
                                                         response_template, record)
                        else:
                            result = evaluate_task(task, engine, batch_size=args.batch_size,
                                                   limit=None if args.max_per_task == -1 else args.max_per_task,
                                                   max_new_tokens=args.max_new_tokens, on_result=record,
                                                   system_prompt=SYSTEM_PROMPTS.get(name))
                    entry["tasks"][name] = {**asdict(result), "predictions": predictions.name,
                                            "tools": [tool.__name__ for tool in task_tools]}
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
