"""SFT on short, one-turn SmolTalk exchanges; evaluate with rl_lab tasks."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import warnings

from datasets import Dataset
import torch
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from trl import SFTConfig, SFTTrainer

from rl_lab.engines.transformers import TransformersEngine
from rl_lab.eval import evaluate_task
from rl_lab.tasks.gpqa import GPQA
from rl_lab.tasks.gsm8k import GSM8K
from rl_lab.tasks.smoltalk import SmolTalk
from rl_lab.tasks.spellingbee import SimpleSpelling, SpellingBee
from rl_lab.utils.default import MODELS_FOLDER


SOURCE = "smol-magpie-ultra-short"
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] | capitalize }}: {{ message['content'] }}"
    "{% if message['role'] == 'assistant' %}{{ eos_token }}{% else %}{{ '\\n' }}{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"
)
EVAL_TASKS = {
    "gsm8k": lambda: GSM8K(subset="main", split="test"),
    "gpqa": lambda: GPQA(),
    "spellingbee": lambda: SpellingBee(split="test"),
    "simple_spelling": lambda: SimpleSpelling(split="test"),
}


@dataclass
class RecipeArguments:
    model_name_or_path: str = "openai-community/gpt2"
    train_limit: int = 1024
    eval_limit: int = 64
    eval_tasks: str = "smoltalk"
    max_new_tokens: int = 128


def export_dataset(task, tokenizer, limit, max_length, *, split_turns=False, max_new_tokens=0):
    """Expand training pairs; keep validation conversations whole, with their final reply as target."""
    examples = []
    discarded = 0
    for row in task.ds:
        if row["source"] != SOURCE:
            continue
        conversation = row["messages"]
        exchanges = (
            [conversation[index:index + 2] for index in range(0, len(conversation), 2)]
            if split_turns else [conversation]
        )
        for messages in exchanges:
            if split_turns and [m["role"] for m in messages] != ["user", "assistant"]:
                continue
            if any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
                continue
            prompt = tokenizer.apply_chat_template(
                task.build_eval_input({"messages": messages}).messages,
                tokenize=False, add_generation_prompt=True,
            )
            completion = " " + messages[-1]["content"] + tokenizer.eos_token
            if (len(tokenizer.encode(prompt + completion)) > max_length
                    or len(tokenizer.encode(prompt)) + max_new_tokens > max_length):
                discarded += 1
                continue
            examples.append({"prompt": prompt, "completion": completion})
            if len(examples) == limit:
                break
        if len(examples) == limit:
            break
    if not examples:
        raise ValueError(f"No {SOURCE} examples fit max_length={max_length}")
    if not split_turns:
        print(f"Validation: kept {len(examples)} whole conversations; skipped {discarded} "
              "oversized conversations while selecting this subset.", flush=True)
    return Dataset.from_list(examples)


def evaluate_tasks(engine, names, smoltalk, limit, max_new_tokens):
    """Score existing generative tasks; SmolTalk supplies qualitative samples only."""
    results = {}
    for name in names:
        if name == "smoltalk":
            samples = []
            for example in smoltalk.select(range(min(3, len(smoltalk)))):
                inputs = engine.tokenizer(example["prompt"], return_tensors="pt")
                outputs, _ = engine.generate(
                    inputs.input_ids, inputs.attention_mask, max_new_tokens=max_new_tokens,
                )
                samples.append({
                    "prompt": example["prompt"],
                    "reference": example["completion"],
                    "generated": outputs[0][0],
                })
            results[name] = {"samples": samples}
        else:
            task = EVAL_TASKS[name]()
            # Refuse oversized contexts instead of silently changing benchmark prompts.
            for index in range(min(limit, len(task))):
                prompt = task.build_eval_input(task[index])
                ids = engine.tokenizer.apply_chat_template(
                    prompt.messages, tokenize=True, add_generation_prompt=True,
                )
                if len(ids) + max_new_tokens > engine.model.config.max_position_embeddings:
                    raise ValueError(f"{name} example {index} exceeds the model context window")
            results[name] = asdict(evaluate_task(
                task, engine, batch_size=1, limit=limit, max_new_tokens=max_new_tokens,
            ))
    return results


def main():
    parser = HfArgumentParser((RecipeArguments, SFTConfig), description=__doc__)
    parser.set_defaults(
        max_length=512, num_train_epochs=1,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        bf16=False, gradient_checkpointing=False, completion_only_loss=True,
        loss_type="nll", save_strategy="epoch", save_total_limit=1,
        prediction_loss_only=True, report_to="none",
    )
    arguments, _ = parser.parse_known_args()
    model_path = Path(arguments.model_name_or_path)
    model_name = model_path.name if model_path.is_dir() else arguments.model_name_or_path.replace("/", "--")
    parser.set_defaults(output_dir=str(MODELS_FOLDER / model_name / "sft"))
    recipe, config = parser.parse_args_into_dataclasses()
    names = [name.strip() for name in recipe.eval_tasks.split(",")]
    if any(name not in {"smoltalk", *EVAL_TASKS} for name in names):
        parser.error(f"--eval-tasks supports smoltalk,{','.join(EVAL_TASKS)}")
    if min(recipe.train_limit, recipe.eval_limit, recipe.max_new_tokens) < 1:
        parser.error("Sample limits and --max-new-tokens must be positive")
    if config.max_length is None or config.max_length < 2:
        parser.error("--max-length must be at least 2")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This first recipe supports single-process training and evaluation")
    if config.packing or config.padding_free or not config.completion_only_loss:
        parser.error("This recipe requires answer-only loss with packing and padding-free disabled")
    output = Path(config.output_dir)
    print(f"Training output: {output}", flush=True)
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty --output-dir to preserve existing runs")

    set_seed(config.seed)
    context_length = AutoConfig.from_pretrained(recipe.model_name_or_path).max_position_embeddings
    train_max_length = config.max_length
    if train_max_length > context_length:
        parser.error("--max-length exceeds the model context window")
    if recipe.max_new_tokens >= context_length:
        parser.error("--max-new-tokens must leave room for a prompt in the model context window")
    # ponytail: validation excludes oversized conversations; windowed evaluation can cover them later.
    warnings.warn(
        f"Validation keeps only whole conversations that fit the model's {context_length}-token "
        "context and generation budget; oversized conversations will be discarded. "
        "Metrics describe this filtered subset, not the full test split.",
        stacklevel=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(recipe.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE
    train = export_dataset(
        SmolTalk("train"), tokenizer, recipe.train_limit, train_max_length, split_turns=True,
    )
    held_out = export_dataset(
        SmolTalk("test"), tokenizer, recipe.eval_limit, context_length,
        max_new_tokens=recipe.max_new_tokens if "smoltalk" in names else 0,
    )
    # Training is already length-bounded; preserve the longer, whole validation inputs.
    config.max_length = None
    trainer = SFTTrainer(
        model=recipe.model_name_or_path, args=config, processing_class=tokenizer,
        train_dataset=train, eval_dataset=held_out,
    )
    output.mkdir(parents=True, exist_ok=True)
    train.to_json(output / "train.jsonl")
    held_out.to_json(output / "smoltalk-test.jsonl")
    (output / "recipe.json").write_text(json.dumps({
        **asdict(recipe), "source": SOURCE, "training": config.to_dict(),
        "training_max_length": train_max_length, "validation_context_length": context_length,
        "train_examples": len(train), "eval_examples": len(held_out),
        "train_fingerprint": train._fingerprint, "eval_fingerprint": held_out._fingerprint,
    }, indent=2), encoding="utf-8")

    before = {"smoltalk_loss": trainer.evaluate()["eval_loss"]}
    with TransformersEngine(trainer.model, tokenizer) as engine:
        before["tasks"] = evaluate_tasks(
            engine, names, held_out, recipe.eval_limit, recipe.max_new_tokens,
        )
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    after = {"smoltalk_loss": trainer.evaluate()["eval_loss"]}
    device = config.device
    del trainer
    with TransformersEngine.load(str(output), device=device, dtype=torch.float32) as engine:
        after["tasks"] = evaluate_tasks(
            engine, names, held_out, recipe.eval_limit, recipe.max_new_tokens,
        )
    results = {"before": before, "after": after}
    (output / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
