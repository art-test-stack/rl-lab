"""Single-device SFT mixing short-request VPO spelling demonstrations and SmolTalk."""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re

from datasets import Dataset
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.utils import get_json_schema
from trl import SFTConfig, SFTTrainer
from trl.chat_template_utils import add_response_schema, get_training_chat_template, supports_tool_calling

from rl_lab.optim import load_optimizer_config, setup_optimizer
from rl_lab.tasks.smoltalk import SmolTalk
from rl_lab.tasks.spellingbee import SimpleSpelling
from rl_lab.tasks.spelling_vpo import ANSWER_INSTRUCTION
from rl_lab.tools.exec import make_python_tool
from rl_lab.utils.default import MODELS_FOLDER
from rl_lab.utils.report import get_report


SYSTEM_PROMPT = "You are a helpful assistant. Use available tools when they help answer the user's request."
SOURCE = "smol-magpie-ultra-short"
TRAIN_REQUESTS = (
    "Spell the word: {word}",
    "Spell '{word}' character by character.",
    "List the letters of '{word}' in order.",
    "How do you spell '{word}'?",
)
EVAL_REQUESTS = (
    "Break '{word}' into individual letters.",
    "Show each character in '{word}', in its original order.",
)


@dataclass
class RecipeArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    train_limit: int = 128  # Examples per source, giving an equal mix.
    eval_limit: int = 16
    optimizer_config: str | None = None


def spelling_demonstration(example, request_template):
    word = example["messages"][0]["content"].removeprefix("Spell the word: ")
    answer = example["messages"][-1]["content"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT + ANSWER_INSTRUCTION},
        {"role": "user", "content": request_template.format(word=word)},
        {"role": "assistant", "content": (
            f"Spell {word!r} without correcting it. "
            "I'll use Python to extract every character in order."
        ), "tool_calls": [{"type": "function", "function": {
            "name": "python", "arguments": {
                "code": f"word = {word!r}\nprint(word + ':' + ','.join(word))",
            },
        }}]},
        {"role": "tool", "name": "python", "content": json.dumps({
            "success": True, "stdout": answer + "\n", "stderr": "", "error": None,
            "timeout": False, "memory_exceeded": False,
        })},
        {"role": "assistant", "content": (
            f"<answer_1>{answer}</answer_1>\n<answer_2>{answer}</answer_2>"
        )},
    ]


def export_datasets(split, limit, tokenizer, max_length, tools):
    """Keep complete conversations within the context limit; never truncate demonstrations."""
    spelling = []
    task = SimpleSpelling(size=limit, split=split)
    requests = TRAIN_REQUESTS if split == "train" else EVAL_REQUESTS
    for index in range(len(task)):
        messages = spelling_demonstration(task[index], requests[index % len(requests)])
        if len(tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, return_dict=False)) > max_length:
            raise ValueError("A spelling demonstration exceeds --max-length")
        spelling.append({"messages": messages, "tools": tools})
    general = []
    for row in SmolTalk(split).ds:
        if row["source"] != SOURCE:
            continue
        conversation = row["messages"]
        for index in range(0, len(conversation) - 1, 2):
            exchange = conversation[index:index + 2]
            if [message["role"] for message in exchange] != ["user", "assistant"]:
                continue
            if any(not isinstance(m["content"], str) or not m["content"].strip() for m in exchange):
                continue
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *exchange]
            if len(tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, return_dict=False)) > max_length:
                continue
            general.append({"messages": messages, "tools": tools})
            if len(general) == limit:
                break
        if len(general) == limit:
            break
    if len(general) != limit:
        raise ValueError(f"Could not find {limit} short SmolTalk exchanges fitting --max-length")
    return {"spelling": Dataset.from_list(spelling), "general": Dataset.from_list(general)}


def main():
    parser = HfArgumentParser((RecipeArguments, SFTConfig), description=__doc__)
    parser.set_defaults(
        num_train_epochs=1, learning_rate=1e-5, max_length=512,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, gradient_checkpointing=True,
        assistant_only_loss=True, completion_only_loss=False, packing=False, padding_free=False,
        bf16=False, fp16=False, optim="adamw_torch", max_grad_norm=1.0,
        weight_decay=0.0, lr_scheduler_type="linear", warmup_steps=0,
        save_strategy="no", logging_steps=1, report_to="none", eval_strategy="no",
        dataloader_pin_memory=False, shuffle_dataset=True,
    )
    arguments, _ = parser.parse_known_args()
    run_name = arguments.run_name or "default"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        parser.error("Choose a run name containing only letters, digits, '.', '_' or '-'")
    model_path = Path(arguments.model_name_or_path)
    model_name = model_path.name if model_path.is_dir() else arguments.model_name_or_path.replace("/", "--")
    parser.set_defaults(output_dir=str(MODELS_FOLDER / model_name / "sft" / run_name), run_name=run_name)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This recipe requires one process and device")
    recipe, config = parser.parse_args_into_dataclasses()
    if config.world_size != 1 or config.fsdp or config.deepspeed or config.parallelism_config:
        parser.error("This recipe requires one device without distributed training")
    if min(recipe.train_limit, recipe.eval_limit) < 1 or config.max_length is None or config.max_length < 2:
        parser.error("Sample limits and --max-length must be positive")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not config.assistant_only_loss or config.completion_only_loss or config.packing or config.padding_free:
        parser.error("This recipe requires assistant-only loss without packing or padding-free training")
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty --output-dir to preserve previous runs")

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(recipe.model_name_or_path)
    if not tokenizer.chat_template or not supports_tool_calling(tokenizer):
        parser.error("Choose a tokenizer with a native tool-aware chat template")
    if getattr(tokenizer, "response_template", None) is None:
        tokenizer = add_response_schema(tokenizer)
    tokenizer.chat_template = get_training_chat_template(tokenizer) or tokenizer.chat_template
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            parser.error("The tokenizer requires a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tools = [get_json_schema(make_python_tool())]
    train_parts = export_datasets("train", recipe.train_limit, tokenizer, config.max_length, tools)
    held_out = export_datasets("test", recipe.eval_limit, tokenizer, config.max_length, tools)
    train_words = {row["messages"][-1]["content"] for row in train_parts["spelling"]}
    if train_words & {row["messages"][-1]["content"] for row in held_out["spelling"]}:
        parser.error("Training and held-out spelling words overlap; choose different sample limits")
    train = Dataset.from_list([
        row for pair in zip(train_parts["spelling"], train_parts["general"], strict=True) for row in pair
    ]).shuffle(seed=config.seed)
    trainer = SFTTrainer(
        model=recipe.model_name_or_path, args=config, processing_class=tokenizer,
        train_dataset=train, eval_dataset=held_out,
    )
    if config.max_length > trainer.model.config.max_position_embeddings:
        parser.error("--max-length exceeds the model context window")
    optimizer_recipe = load_optimizer_config(recipe.optimizer_config) if recipe.optimizer_config else None
    if optimizer_recipe is not None:
        width_scale = (trainer.model.config.hidden_size / 768) ** -0.5
        for group in optimizer_recipe["groups"]:
            group["hparams"]["lr"] *= config.learning_rate / 0.004
            if group["optimizer"] == "adamw":
                group["hparams"]["lr"] *= width_scale
    groups = setup_optimizer(trainer, optimizer_recipe)
    report = get_report(output)
    train.to_json(output / "train.jsonl")
    for name, dataset in held_out.items():
        dataset.to_json(output / f"{name}-test.jsonl")
    report.write_json("recipe.json", {
        **asdict(recipe), "training": config.to_dict(), "system_prompt": SYSTEM_PROMPT,
        "spelling_system_prompt": SYSTEM_PROMPT + ANSWER_INSTRUCTION,
        "train_request_templates": TRAIN_REQUESTS, "eval_request_templates": EVAL_REQUESTS,
        "model_dtype": str(next(trainer.model.parameters()).dtype),
        "general_source": SOURCE, "train_examples_per_source": recipe.train_limit,
        "tools": tools, "optimizer_recipe": optimizer_recipe, "optimizer_groups": groups,
        "train_fingerprint": train._fingerprint,
        "eval_fingerprints": {name: dataset._fingerprint for name, dataset in held_out.items()},
    })
    before = trainer.evaluate()
    report.write_json("evaluation.json", {"before": before}, echo=True)
    trainer.train()
    after = trainer.evaluate()
    trainer.save_model()
    trainer.save_state()
    report.write_json("evaluation.json", {"before": before, "after": after}, echo=True)


if __name__ == "__main__":
    main()
