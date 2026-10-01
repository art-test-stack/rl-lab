from types import SimpleNamespace

import pytest
from datasets import Dataset

from rl_lab.tasks import arc, humaneval, mmlu


@pytest.mark.parametrize("module,task_class,kwargs,row", [
    (arc, arc.ARC, {"subset": "ARC-Easy", "split": "test"},
     {"question": "Synthetic question", "choices": {"text": ["yes", "no"],
      "label": ["A", "B"]}, "answerKey": "A"}),
    (mmlu, mmlu.MMLU, {"subset": "all", "split": "test"},
     {"question": "Synthetic question", "choices": ["yes", "no", "maybe", "other"],
      "answer": 0, "subject": "synthetic"}),
])
def test_multiple_choice_correctness_reward(monkeypatch, module, task_class, kwargs, row):
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: Dataset.from_list([row]))
    task = task_class(**kwargs)
    example = task[0]
    for answer, expected in (("A", 1.0), ("B", 0.0)):
        reward = task.reward(example, answer)
        assert type(reward) is float
        assert reward == expected


@pytest.mark.parametrize("success", [True, False])
def test_humaneval_correctness_reward(monkeypatch, success):
    row = {"prompt": "def answer():\n", "canonical_solution": "    return 42",
           "entry_point": "answer", "test": "def check(fn):\n    assert fn() == 42"}
    monkeypatch.setattr(humaneval, "load_dataset", lambda *args, **kwargs: Dataset.from_list([row]))
    programs = []

    def execute(program):
        programs.append(program)
        return SimpleNamespace(success=success)

    monkeypatch.setattr(humaneval, "execute_code", execute)
    task = humaneval.HumanEval()
    completion = "def answer():\n    return 42"
    reward = task.reward(task[0], completion)
    assert type(reward) is float
    assert reward == float(success)
    assert len(programs) == 1
    assert completion in programs[0]
    assert row["test"] in programs[0]
    assert "check(answer)" in programs[0]
