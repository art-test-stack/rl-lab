"""Run scalar-scored conversation tasks through an existing inference engine."""

import math
from dataclasses import dataclass

from rl_lab.engines import (
    ChatPrompt,
    CompletionEngine,
    GenerationConfig,
    LogLikelihoodRequest,
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


def evaluate_task(
    task: Task,
    engine: CompletionEngine,
    *,
    config: GenerationConfig | None = None,
    batch_size: int = 8,
    limit: int | None = None,
    system_prompt: str | None = None,
) -> EvalResult:
    """Return the mean per-example score without loading or closing an engine.

    Tasks own input construction, candidate strings and scalar scoring. Categorical
    predictions maximize summed continuation log likelihood; ties select the first
    candidate. This is not a CORE, agentic, pass@k or distributed benchmark runner.
    ``batch_size`` bounds examples per batch (and thus generated completions), not
    the number of candidate continuations in a categorical batch.
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
    if kind not in {"generative", "categorical"}:
        raise ValueError(f"Unsupported task eval_type: {kind!r}")
    engine.require("generation" if kind == "generative" else "loglikelihood")
    count = len(task) if limit is None else min(len(task), limit)
    if count == 0:
        raise ValueError("Cannot evaluate an empty task")
    config = config if config is not None else GenerationConfig()
    total = 0.0
    for start in range(0, count, batch_size):
        examples = [task[i] for i in range(start, min(start + batch_size, count))]
        prompts = [
            _prepare_prompt(task.build_eval_input(example), system_prompt)
            for example in examples
        ]
        if any(isinstance(prompt, ChatPrompt) for prompt in prompts):
            engine.require("chat")
        if kind == "generative":
            outputs = engine.generate(prompts, config)
            if len(outputs) != len(examples):
                raise RuntimeError("Engine returned an unexpected number of completions")
            completions = [output.text for output in outputs]
        else:
            candidates = []
            requests = []
            for example, prompt in zip(examples, prompts, strict=True):
                choices = task.evaluation_candidates(example)
                if isinstance(choices, str):
                    raise ValueError("Candidates must be a sequence of non-empty strings")
                choices = tuple(choices)
                if not choices or any(not isinstance(c, str) or not c for c in choices):
                    raise ValueError("Candidates must be a sequence of non-empty strings")
                candidates.append(choices)
                requests.extend(LogLikelihoodRequest(prompt, choice) for choice in choices)
            scores = engine.loglikelihood(requests)
            if len(scores) != len(requests):
                raise RuntimeError("Engine returned an unexpected number of scores")
            if any(math.isnan(score) or score == math.inf for score in scores):
                raise ValueError("Engine returned invalid likelihood scores")
            completions = []
            offset = 0
            for choices in candidates:
                option_scores = scores[offset:offset + len(choices)]
                if all(score == -math.inf for score in option_scores):
                    raise ValueError("All candidates have negative-infinite likelihood")
                completions.append(choices[max(range(len(choices)), key=option_scores.__getitem__)])
                offset += len(choices)
        for example, completion in zip(examples, completions, strict=True):
            score = float(task.evaluate(example, completion))
            if not math.isfinite(score):
                raise ValueError("Task returned a non-finite score")
            total += score
    return EvalResult(num_examples=count, score=total / count)
