"""Single-device GRPO on SimpleSpelling with a Python tool."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os
import re
import warnings

from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from transformers.utils import get_json_schema
from trl import GRPOConfig, GRPOTrainer
from trl.chat_template_utils import add_response_schema, get_training_chat_template, supports_tool_calling

from rl_lab.engines.transformers import TransformersEngine
from rl_lab.optim import load_optimizer_config, setup_optimizer
from rl_lab.tasks.spellingbee import SimpleSpelling
from rl_lab.tasks.spelling_rewards import REWARD_NAMES
from rl_lab.tasks.spelling_vpo import ANSWER_INSTRUCTION, WEIGHT_SAMPLES, WHITESPACE_PENALTY, make_vpo_reward, parse_candidates
from rl_lab.tools.exec import make_python_tool
from rl_lab.utils.default import MODELS_FOLDER
from rl_lab.utils.report import get_report


SYSTEM_PROMPT = (
    "Extract every character of the supplied word and separate the characters with commas. "
    "Preserve the word exactly, even if it is unfamiliar or misspelled. "
    "Do not check or correct its spelling. The required answer is the original word, "
    "a colon, and its comma-separated characters. For example, cat:c,a,t. "
    "Use the python tool: assign the supplied word to word, then run "
    "print(word + ':' + ','.join(word)). "
    "Reply with only the printed answer from stdout, without quotes or explanation."
)

WORKED_EXAMPLE = [
    {"role": "user", "content": "Spell the word: cat"},
    {"role": "assistant", "content": "", "tool_calls": [{
        "type": "function", "function": {"name": "python", "arguments": {
            "code": "word = 'cat'\nprint(word + ':' + ','.join(word))",
        }},
    }]},
    {"role": "tool", "name": "python", "content": json.dumps({
        "success": True, "stdout": "cat:c,a,t\n", "stderr": "", "error": None,
        "timeout": False, "memory_exceeded": False,
    })},
    {"role": "assistant", "content": "cat:c,a,t"},
]


def build_prompt(task, example, system_prompt=SYSTEM_PROMPT, use_worked_example=True):
    return [{"role": "system", "content": system_prompt},
            *(WORKED_EXAMPLE if use_worked_example else []),
            *task.build_eval_input(example).messages]


@dataclass
class RecipeArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    train_limit: int = 128
    eval_limit: int = 32
    system_prompt: str = SYSTEM_PROMPT
    use_worked_example: bool = True
    reward_mode: str = "exact"
    optimizer_config: str = str(Path(__file__).resolve().parents[2] / "configs/optim/qwen-nanochat.yaml")


def configure_optimizer(trainer, path, learning_rate):
    """Use the trainer optimizer for an empty path; otherwise scale grouped rates."""
    if not path:
        return setup_optimizer(trainer, None)
    recipe = load_optimizer_config(path)
    width_scale = (trainer.model.config.hidden_size / 768) ** -0.5
    for group in recipe["groups"]:
        group["hparams"]["lr"] *= learning_rate / 0.004
        if group["optimizer"] == "adamw":
            group["hparams"]["lr"] *= width_scale
    return setup_optimizer(trainer, recipe)


def export_dataset(task, tokenizer, context_length, max_completion_length, tools,
                   system_prompt=SYSTEM_PROMPT, use_worked_example=True):
    """Export prompts and reward references; reject context overflow."""
    rows = []
    for index in range(len(task)):
        example = task[index]
        messages = build_prompt(task, example, system_prompt, use_worked_example)
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False,
            tools=tools,
        )
        if len(ids) + max_completion_length > context_length:
            raise ValueError(f"{task.split} example {index} exceeds the model context window")
        rows.append({"prompt": messages, "example": example})
    return Dataset.from_list(rows)


def final_answer(messages):
    """Reward only a final text answer, never a tool call or tool output."""
    final = messages[-1] if messages else {}
    content = final.get("content")
    return content if (final.get("role") == "assistant" and not final.get("tool_calls")
                       and isinstance(content, str)) else ""


def score_model(model, tokenizer, task, max_completion_length, tools, max_tool_rounds, samples_path,
                system_prompt=SYSTEM_PROMPT, use_worked_example=True, reward_mode="exact"):
    samples = []
    with TransformersEngine(model, tokenizer) as engine:
        for index in range(len(task)):
            example = task[index]
            messages = build_prompt(task, example, system_prompt, use_worked_example)
            history = engine.complete(
                [messages], tools=tools, max_new_tokens=max_completion_length,
                max_tool_rounds=max_tool_rounds,
            )[0]
            prediction = final_answer(history)
            candidates = parse_candidates(prediction) if reward_mode == "vpo" else [prediction]
            samples.append({"example": example, "history": history, "prediction": prediction,
                            "candidates": candidates,
                            "score": max(float(task.evaluate(example, candidate)) for candidate in candidates)})
    with samples_path.open("w") as stream:
        for sample in samples:
            stream.write(json.dumps(sample) + "\n")
    return {"num_examples": len(samples), "score": sum(row["score"] for row in samples) / len(samples)}


def main():
    parser = HfArgumentParser((RecipeArguments, GRPOConfig), description=__doc__)
    parser.set_defaults(
        max_steps=-1, num_train_epochs=1, learning_rate=0.0002,
        per_device_train_batch_size=1, gradient_accumulation_steps=256,
        generation_batch_size=16, num_generations=16, num_iterations=1,
        temperature=1.0, top_k=50, top_p=1.0,
        max_completion_length=256, max_tool_calling_iterations=1,
        beta=0.0, loss_type="dapo", scale_rewards="none",
        lr_scheduler_type="linear", warmup_steps=0, weight_decay=0.0, max_grad_norm=0.0,
        optim="adamw_torch", bf16=False, fp16=False, gradient_checkpointing=True, disable_dropout=True,
        dataloader_pin_memory=False, shuffle_dataset=False,
        save_strategy="steps", save_steps=60, save_total_limit=1,
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
    if recipe.reward_mode not in {"exact", "vpo"}:
        parser.error("--reward-mode must be exact or vpo")
    if recipe.reward_mode == "vpo":
        if recipe.use_worked_example:
            parser.error("VPO requires --use-worked-example false; the existing example is single-answer")
        if config.scale_rewards != "none" or config.multi_objective_aggregation != "sum_then_normalize":
            parser.error("VPO requires --scale-rewards none and sum_then_normalize aggregation")
        if config.reward_weights is not None and config.reward_weights != [1.0]:
            parser.error("VPO requires a single unweighted advantage callback")
        recipe.system_prompt += ANSWER_INSTRUCTION
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
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if config.max_tool_calling_iterations is None or config.max_tool_calling_iterations < 1:
        parser.error("--max-tool-calling-iterations must be positive")
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty --output-dir to preserve existing runs")

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(recipe.model_name_or_path, padding_side="left")
    if not tokenizer.chat_template:
        parser.error("Choose a model/tokenizer with a chat template")
    if not supports_tool_calling(tokenizer):
        parser.error("Choose a tokenizer with a native tool-aware chat template")
    if getattr(tokenizer, "response_template", None) is None:
        tokenizer = add_response_schema(tokenizer)
    tokenizer.chat_template = get_training_chat_template(tokenizer) or tokenizer.chat_template
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            parser.error("The tokenizer requires a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    context_length = AutoConfig.from_pretrained(recipe.model_name_or_path).max_position_embeddings
    tools = [make_python_tool()]
    schemas = [get_json_schema(tool) for tool in tools]
    train_task = SimpleSpelling(size=recipe.train_limit, split="train")
    eval_task = SimpleSpelling(size=recipe.eval_limit, split="test")
    train = export_dataset(train_task, tokenizer, context_length, config.max_completion_length, schemas,
                           recipe.system_prompt, recipe.use_worked_example)
    held_out = export_dataset(eval_task, tokenizer, context_length, config.max_completion_length, schemas,
                              recipe.system_prompt, recipe.use_worked_example)

    def spelling_reward(completions, example, **kwargs):
        return [
            train_task.reward(reference, final_answer(completion))
            for reference, completion in zip(example, completions, strict=True)
        ]

    report = get_report(output)
    train.to_json(output / "train.jsonl")
    held_out.to_json(output / "simple-spelling-test.jsonl")
    settings = {
        **asdict(recipe), "training": config.to_dict(),
        "system_prompt": recipe.system_prompt,
        "worked_example": WORKED_EXAMPLE if recipe.use_worked_example else [],
        "tools": schemas, "response_template": tokenizer.response_template,
        "reward": "SimpleSpelling exact match (0/1)", "context_length": context_length,
        "train_fingerprint": train._fingerprint, "eval_fingerprint": held_out._fingerprint,
    }
    reward_func = spelling_reward
    if recipe.reward_mode == "vpo":
        reward_func = make_vpo_reward(config.num_generations, config.seed, output / "vpo-rollouts.jsonl")
        settings.update(reward="VPO normalized set advantage; see vpo/set_reward for raw rewards",
                        evaluation="best-of-two exact spelling",
                        vpo={"candidates_per_rollout": 2, "weight_samples": WEIGHT_SAMPLES,
                             "reward_names": REWARD_NAMES, "dirichlet_alpha": 1.0,
                             "whitespace_penalty": WHITESPACE_PENALTY})
    warnings.warn(
        ("VPO requires varying set rewards: inspect vpo/group_std and vpo/parsed_fraction."
         if recipe.reward_mode == "vpo" else
         "Exact-match rewards are sparse: groups with identical rewards give no policy-learning signal. "
         "Tools do not guarantee mixed rewards; inspect reward and reward_std logs."),
        stacklevel=1,
    )
    trainer = GRPOTrainer(
        model=recipe.model_name_or_path, args=config, processing_class=tokenizer,
        train_dataset=train, reward_funcs=reward_func,
        tools=tools,
    )
    settings["grpo_optimizer_groups"] = configure_optimizer(trainer, recipe.optimizer_config, config.learning_rate)
    report.write_json("recipe.json", settings)
    before = score_model(trainer.model, tokenizer, eval_task, config.max_completion_length,
                         tools, config.max_tool_calling_iterations, output / "before.jsonl",
                         recipe.system_prompt, recipe.use_worked_example, recipe.reward_mode)
    report.write_json("evaluation.json", {"before": before}, echo=True)
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    device = config.device
    del trainer
    with TransformersEngine.load(str(output), device=device) as engine:
        engine.tokenizer.response_template = tokenizer.response_template
        after = score_model(engine.model, engine.tokenizer, eval_task, config.max_completion_length,
                            tools, config.max_tool_calling_iterations, output / "after.jsonl",
                            recipe.system_prompt, recipe.use_worked_example, recipe.reward_mode)
    report.write_json("evaluation.json", {"before": before, "after": after}, echo=True)


if __name__ == "__main__":
    main()
