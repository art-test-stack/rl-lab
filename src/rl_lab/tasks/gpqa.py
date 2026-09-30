"""GPQA question/choice schema and randomized answer placement.

Sources:
https://huggingface.co/datasets/Idavidrein/gpqa
https://github.com/idavidrein/gpqa/blob/main/baselines/utils.py
"""

import random
import re

from datasets import load_dataset

from ._base import Task, render_mc


_FINAL_ANSWER = re.compile(r"\bthe correct answer is\s*\(([A-D])\)", re.IGNORECASE)


def extract_answer(completion):
    """Read a bare letter or the final answer marker used by the GPQA baselines."""
    if not isinstance(completion, str):
        raise TypeError("GPQA completion must be text")
    direct = re.fullmatch(r"\s*\(?([A-D])\)?[.]?\s*", completion, re.IGNORECASE)
    if direct:
        return direct.group(1).upper()
    matches = _FINAL_ANSWER.findall(completion)
    return matches[-1].upper() if matches else None


class GPQA(Task):
    """Generative GPQA by default; optional answer-letter likelihood evaluation.

    The Hub publishes its evaluation questions under the ``train`` split; this
    naming does not make them training data. Dataset access may require accepting
    the publisher's terms. No dataset examples are bundled with rl-lab.
    """

    letters = ("A", "B", "C", "D")
    subsets = ("gpqa_main", "gpqa_extended", "gpqa_diamond", "gpqa_experts")

    def __init__(self, subset="gpqa_diamond", split="train", *, evaluation="generative", seed=42, **kwargs):
        super().__init__(**kwargs)
        if subset not in self.subsets:
            raise ValueError(f"GPQA subset must be one of {self.subsets}")
        if split != "train":
            raise ValueError("GPQA is published under the train split")
        if evaluation not in {"generative", "categorical"}:
            raise ValueError("evaluation must be generative or categorical")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed
        self.evaluation = evaluation
        self.ds = load_dataset("Idavidrein/gpqa", subset, split=split).shuffle(seed=seed)

    @property
    def eval_type(self):
        return self.evaluation

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        answers = [row[f"Incorrect Answer {i}"] for i in range(1, 4)] + [row["Correct Answer"]]
        if not isinstance(row["Question"], str) or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError("GPQA questions and answers must be text")
        # Shuffle indices, not strings: even duplicate answer text cannot change
        # which original option is the reference. Physical index makes slicing
        # and repeated/out-of-order access independent of global RNG state.
        order = list(range(4))
        random.Random(self.seed + index).shuffle(order)
        choices = [answers[i] for i in order]
        answer = self.letters[order.index(3)]
        if self.eval_type == "categorical":
            prompt = render_mc(row["Question"], self.letters, choices)
        else:
            prompt = f"Question: {row['Question']}\nChoices:\n"
            prompt += "\n".join(f"({letter}) {choice}" for letter, choice in zip(self.letters, choices, strict=True))
            prompt += '\nGive your reasoning, then end with "The correct answer is (X)", replacing X with the answer letter.'
        return {
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}],
            "letters": self.letters,
        }

    def evaluate(self, conversation, completion):
        predicted = extract_answer(completion)
        return predicted is not None and predicted == conversation["messages"][-1]["content"]

    def reward(self, conversation, completion):
        return float(self.evaluate(conversation, completion))
