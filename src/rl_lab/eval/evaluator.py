"""Run scalar-scored conversation tasks through an existing inference engine."""

import math
from dataclasses import dataclass

import torch

from rl_lab.engines import (
    ChatPrompt,
    CompletionEngine,
    TextPrompt,
)
from rl_lab.tasks._base import Task


@dataclass(frozen=True)
class EvalResult:
    num_examples: int
    score: float


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _prepare_prompt(prompt, system_prompt):
    if isinstance(prompt, ChatPrompt):
        if not prompt.messages:
            raise ValueError("Chat inputs must contain at least one message")
        messages = []
        for message in prompt.messages:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("Only system, user and assistant input roles are supported")
            if not isinstance(message.get("content"), str):
                raise ValueError("Only text input messages are supported")
            messages.append(dict(message))
        if system_prompt is not None:
            if any(message["role"] == "system" for message in messages):
                raise ValueError("Task input already contains a system message")
            messages.insert(0, {"role": "system", "content": system_prompt})
        return ChatPrompt(messages)
    if not isinstance(prompt, (str, TextPrompt)):
        raise TypeError("Task inputs must be str, TextPrompt or ChatPrompt")
    if system_prompt is not None:
        raise ValueError("system_prompt requires ChatPrompt inputs")
    return prompt


def _encode_batch(tokenizer, prompts):
    sequences = []
    for prompt in prompts:
        if isinstance(prompt, ChatPrompt):
            ids = tokenizer.apply_chat_template(
                list(prompt.messages), tokenize=True, add_generation_prompt=True,
                return_dict=False,
            )
        else:
            ids = tokenizer.encode(
                prompt.text if isinstance(prompt, TextPrompt) else prompt,
                add_special_tokens=True,
            )
        sequences.append(list(ids))
    if any(not ids for ids in sequences):
        bos = tokenizer.bos_token_id
        if bos is None:
            raise ValueError("Empty prompts require a BOS token")
        sequences = [ids or [bos] for ids in sequences]
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Batch generation needs a pad_token_id or eos_token_id")
    width = max(map(len, sequences))
    input_ids = torch.full((len(sequences), width), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(sequences):
        input_ids[row, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, width - len(ids):] = 1
    return input_ids, attention_mask


def evaluate_task(
    task: Task,
    engine: CompletionEngine,
    *,
    max_new_tokens: int = 51,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: tuple[str, ...] | list[str] | None = None,
    tokenizer=None,
    batch_size: int = 8,
    limit: int | None = None,
    system_prompt: str | None = None,
    on_result=None,
) -> EvalResult:
    """Return the mean per-example score without loading or closing an engine.

    Tasks own input construction and scalar scoring. This runner currently supports
    generative tasks only; categorical scoring is deferred. This is not a CORE,
    agentic, pass@k or distributed benchmark runner.
    ``batch_size`` bounds examples per generation batch.
    Optional ``on_result(index, example, completion, score)`` receives each scored
    example in task-view order. Callback exceptions propagate to the caller.
    """
    _positive_integer("batch_size", batch_size)
    if limit is not None:
        _positive_integer("limit", limit)
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string or None")
    try:
        kind = task.eval_type
    except NotImplementedError as error:
        raise ValueError(f"{type(task).__name__} does not define an evaluation protocol") from error
    if kind != "generative":
        raise ValueError(f"Unsupported task eval_type: {kind!r}")
    engine.require("generation")
    count = len(task) if limit is None else min(len(task), limit)
    if count == 0:
        raise ValueError("Cannot evaluate an empty task")
    if tokenizer is None:
        tokenizer = getattr(engine, "tokenizer", None)
    total = 0.0
    for start in range(0, count, batch_size):
        examples = [task[i] for i in range(start, min(start + batch_size, count))]
        prompts = [
            _prepare_prompt(task.build_eval_input(example), system_prompt)
            for example in examples
        ]
        if any(isinstance(prompt, ChatPrompt) for prompt in prompts):
            engine.require("chat")
        if tokenizer is None:
            raise ValueError("Generative evaluation requires a tokenizer")
        input_ids, attention_mask = _encode_batch(tokenizer, prompts)
        outputs, _ = engine.generate(
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        if len(outputs) != len(examples):
            raise RuntimeError("Engine returned an unexpected number of completions")
        completions = [output[0] for output in outputs]
        for offset, (example, completion) in enumerate(zip(examples, completions, strict=True)):
            score = float(task.evaluate(example, completion))
            if not math.isfinite(score):
                raise ValueError("Task returned a non-finite score")
            total += score
            if on_result is not None:
                on_result(start + offset, example, completion, score)
    return EvalResult(num_examples=count, score=total / count)
