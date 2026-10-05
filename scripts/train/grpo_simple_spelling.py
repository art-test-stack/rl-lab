"""Single-device GRPO on SimpleSpelling, with exact-match task rewards."""

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import re
import warnings

from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from trl import GRPOConfig, GRPOTrainer

from rl_lab.engines.transformers import TransformersEngine
from rl_lab.eval import evaluate_task
from rl_lab.tasks.spellingbee import SimpleSpelling
from rl_lab.utils.default import MODELS_FOLDER
from rl_lab.utils.report import get_report


SYSTEM_PROMPT = (
    "Spell the requested word as word:comma-separated-letters. "
    "For example, cat:c,a,t. Respond only with that answer."
)


@dataclass
class RecipeArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    train_limit: int = 128
    eval_limit: int = 32


def export_dataset(task, tokenizer, context_length, max_completion_length):
    """Export prompts and reward references; reject context overflow."""
    rows = []
    for index in range(len(task)):
        example = task[index]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *task.build_eval_input(example).messages,
        ]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(ids) + max_completion_length > context_length:
            raise ValueError(f"{task.split} example {index} exceeds the model context window")
        rows.append({"prompt": messages, "example": example})
    return Dataset.from_list(rows)


def main():
    parser = HfArgumentParser((RecipeArguments, GRPOConfig), description=__doc__)
    parser.set_defaults(
        max_steps=32, learning_rate=1e-6,
        per_device_train_batch_size=4, num_generations=4,
        max_completion_length=64, beta=0.0, loss_type="grpo",
        bf16=False, fp16=False, gradient_checkpointing=False, disable_dropout=True,
        save_strategy="steps", save_steps=32, save_total_limit=1,
        logging_steps=1, report_to="none", eval_strategy="no",
    )
    arguments, _ = parser.parse_known_args()
    run_name = arguments.run_name or "default"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        parser.error("--run-name must start with a letter or digit and contain only letters, digits, '.', '_' or '-'")
    model_path = Path(arguments.model_name_or_path)
    model_name = model_path.name if model_path.is_dir() else arguments.model_name_or_path.replace("/", "--")
    parser.set_defaults(output_dir=str(MODELS_FOLDER / model_name / "grpo" / run_name), run_name=run_name)
    # Reject launchers before GRPOConfig initializes distributed execution.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This recipe requires a single process and device")
    recipe, config = parser.parse_args_into_dataclasses()
    if config.world_size != 1 or config.fsdp or config.deepspeed or config.parallelism_config:
        parser.error("This recipe requires a single process and device without FSDP or DeepSpeed")
    if config.use_vllm or config.use_transformers_continuous_batching or config.use_transformers_paged:
        parser.error("This recipe uses ordinary Transformers generation on the training device")
    if config.remove_unused_columns:
        parser.error("Reward references require --remove-unused-columns false")
    if config.eval_strategy != "no":
        parser.error("This recipe evaluates fixed greedy task scores before and after training")
    if min(recipe.train_limit, recipe.eval_limit, config.max_completion_length) < 1:
        parser.error("Sample limits and --max-completion-length must be positive")
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty --output-dir to preserve existing runs")

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(recipe.model_name_or_path, padding_side="left")
    if not tokenizer.chat_template:
        parser.error("Choose a model/tokenizer with a chat template")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            parser.error("The tokenizer requires a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    context_length = AutoConfig.from_pretrained(recipe.model_name_or_path).max_position_embeddings
    train_task = SimpleSpelling(size=recipe.train_limit, split="train")
    eval_task = SimpleSpelling(size=recipe.eval_limit, split="test")
    train = export_dataset(train_task, tokenizer, context_length, config.max_completion_length)
    held_out = export_dataset(eval_task, tokenizer, context_length, config.max_completion_length)

    def spelling_reward(completions, example, **kwargs):
        return [
            train_task.reward(reference, completion[0]["content"])
            for reference, completion in zip(example, completions, strict=True)
        ]

    trainer = GRPOTrainer(
        model=recipe.model_name_or_path, args=config, processing_class=tokenizer,
        train_dataset=train, reward_funcs=spelling_reward,
    )
    report = get_report(output)
    train.to_json(output / "train.jsonl")
    held_out.to_json(output / "simple-spelling-test.jsonl")
    report.write_json("recipe.json", {
        **asdict(recipe), "training": config.to_dict(), "system_prompt": SYSTEM_PROMPT,
        "reward": "SimpleSpelling exact match (0/1)", "context_length": context_length,
        "train_fingerprint": train._fingerprint, "eval_fingerprint": held_out._fingerprint,
    })
    warnings.warn(
        "Exact-match rewards are sparse: groups with identical rewards give no policy-learning signal. "
        "Start from a model that sometimes spells correctly; inspect reward and reward_std logs.",
        stacklevel=1,
    )
    with TransformersEngine(trainer.model, tokenizer) as engine:
        before = asdict(evaluate_task(
            eval_task, engine, batch_size=1, max_new_tokens=config.max_completion_length,
            system_prompt=SYSTEM_PROMPT,
        ))
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    device = config.device
    del trainer
    with TransformersEngine.load(str(output), device=device) as engine:
        after = asdict(evaluate_task(
            eval_task, engine, batch_size=1, max_new_tokens=config.max_completion_length,
            system_prompt=SYSTEM_PROMPT,
        ))
    report.write_json("evaluation.json", {"before": before, "after": after}, echo=True)


if __name__ == "__main__":
    main()
