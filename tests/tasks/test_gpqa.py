import random

import pytest
from datasets import Dataset

from rl_lab.engines import CompletionEngine, EngineCapabilities
from rl_lab.eval import evaluate_task
from rl_lab.tasks import gpqa


@pytest.fixture
def dataset(monkeypatch):
    # Synthetic fixtures only; do not publish held-out GPQA questions.
    rows = [{"Question": f"Synthetic question {i}", "Correct Answer": "reference",
             "Incorrect Answer 1": "distractor1", "Incorrect Answer 2": "distractor2",
             "Incorrect Answer 3": "distractor3"} for i in range(24)]
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return Dataset.from_list(rows)
    monkeypatch.setattr(gpqa, "load_dataset", load)
    return calls


def test_schema_shuffling_mapping_and_sliced_view(dataset):
    task = gpqa.GPQA()
    assert len(task) == 24
    assert task.eval_type == "generative"
    assert dataset[0] == (("Idavidrein/gpqa", "gpqa_diamond"), {"split": "train"})
    state = random.getstate()
    examples = [task[i] for i in range(len(task))]
    assert random.getstate() == state
    assert len({e["messages"][-1]["content"] for e in examples}) == 4
    for example in examples:
        answer = example["messages"][-1]["content"]
        assert f"({answer}) reference" in example["messages"][0]["content"]
        assert task.evaluate(example, f"Reasoning. The correct answer is ({answer})")
        assert task.reward(example, "not an answer") == 0
        assert len(task.build_eval_input(example).messages) == 1
    assert task[7] == task[7]
    assert gpqa.GPQA(start=3, stop=10, step=2)[2] == task[7]


@pytest.mark.parametrize("text,expected", [("A", "A"), ("(b)", "B"),
    ("The correct answer is (A). After review, the correct answer is (D).", "D"),
    ("A possibility is B, but I am unsure", None), ("The correct answer is (E)", None)])
def test_extract_answer(text, expected):
    assert gpqa.extract_answer(text) == expected


@pytest.mark.parametrize("kwargs", [{"subset": "unknown"}, {"split": "test"},
    {"evaluation": "unsupported"}, {"seed": -1}])
def test_bad_configuration_before_download(dataset, kwargs):
    with pytest.raises(ValueError):
        gpqa.GPQA(**kwargs)
    assert dataset == []


def test_runner_integration(dataset, tokenizer):
    task = gpqa.GPQA(evaluation="generative", stop=2)
    class Engine(CompletionEngine):
        capabilities = EngineCapabilities(generation=True, chat=True)
        def __init__(self):
            self.tokenizer = tokenizer
            self.position = 0
        def generate(self, input_ids, attention_mask, **kwargs):
            outputs = [
                (
                    f"The correct answer is ({task[self.position + i]['messages'][-1]['content']})",
                    None,
                    None,
                    None,
                )
                for i in range(len(input_ids))
            ]
            self.position += len(input_ids)
            return outputs, None
    assert evaluate_task(task, Engine()).score == 1.0
