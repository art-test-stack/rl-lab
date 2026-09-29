"""Evaluation of rl-lab conversation tasks through a backend-neutral engine."""

from dataclasses import dataclass

from rl_lab.engines import ChatPrompt, GenerationConfig, LogLikelihoodRequest


@dataclass(frozen=True)
class EvalResult:
    num_examples: int
    score: float


def evaluate(
    task, engine, *, config=None, batch_size=8, limit=None, system_prompt=None
) -> EvalResult:
    """Score generative tasks or categorical answer letters.

    The final assistant message is the reference answer and is never sent to
    the model. Categorical tasks use summed continuation log likelihoods;
    generative tasks delegate completion scoring to task.evaluate().
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    kind = task.eval_type
    if kind not in {"generative", "categorical"}:
        raise ValueError(f"Unsupported task eval_type: {kind!r}")
    engine.require("chat", "generation" if kind == "generative" else "loglikelihood")
    count = len(task) if limit is None else min(len(task), limit)
    if count == 0:
        raise ValueError("Cannot evaluate an empty task")
    total = 0.0
    config = config or GenerationConfig()
    for start in range(0, count, batch_size):
        examples = [task[i] for i in range(start, min(start + batch_size, count))]
        prompts = []
        for example in examples:
            messages = example["messages"]
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                raise ValueError(
                    "Task examples must end with an assistant reference answer"
                )
            inputs = messages[:-1]
            if system_prompt is not None:
                inputs = [{"role": "system", "content": system_prompt}] + inputs
            if any(not isinstance(m["content"], str) for m in inputs):
                raise ValueError("Only text input messages are supported")
            prompts.append(ChatPrompt(inputs))
        if kind == "generative":
            outputs = engine.generate(prompts, config)
            if len(outputs) != len(examples):
                raise RuntimeError(
                    "Engine returned an unexpected number of completions"
                )
            completions = [o.text for o in outputs]
        else:
            completions = []
            for example, prompt in zip(examples, prompts, strict=True):
                letters = example.get("letters")
                if not letters:
                    raise ValueError("Categorical tasks must provide answer letters")
                scores = engine.loglikelihood(
                    [LogLikelihoodRequest(prompt, letter) for letter in letters]
                )
                if len(scores) != len(letters):
                    raise RuntimeError("Engine returned an unexpected number of scores")
                completions.append(
                    letters[max(range(len(letters)), key=scores.__getitem__)]
                )
        total += sum(
            float(task.evaluate(example, completion))
            for example, completion in zip(examples, completions, strict=True)
        )
    return EvalResult(count, total / count)
