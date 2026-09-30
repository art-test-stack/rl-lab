from copy import deepcopy

import pytest

from rl_lab.engines import ChatPrompt, CompletionEngine, EngineCapabilities, TextPrompt
from rl_lab.eval import EvalResult, evaluate_task
from rl_lab.tasks._base import Task


class Examples(Task):
    eval_type = "generative"

    def __init__(self, rows, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        return self.rows[index]

    def evaluate(self, example, completion):
        return completion == example["answer"]


class FakeEngine(CompletionEngine):
    capabilities = EngineCapabilities(generation=True, chat=True)

    def __init__(self):
        self.batches = []
        self.closed = False
        self.tokenizer = FakeTokenizer()

    def generate(self, input_ids, attention_mask, **kwargs):
        self.batches.append((input_ids.tolist(), attention_mask.tolist()))
        return [("correct", None, None) for _ in input_ids]

    def close(self):
        self.closed = True


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self):
        self.seen = []

    def encode(self, text, **kwargs):
        self.seen.append(text)
        return [3 + (ord(char) % 7) for char in text] or [self.bos_token_id]

    def apply_chat_template(self, messages, **kwargs):
        self.seen.append(messages)
        return [self.bos_token_id, 4, 5]


def row(answer="correct"):
    return {"messages": [{"role": "user", "content": "question"},
                         {"role": "assistant", "content": "SECRET REFERENCE"}],
            "answer": answer}


def test_batching_limit_slicing_no_reference_leak_or_mutation():
    rows = [row(), row("wrong"), row(), row(), row()]
    rows[0]["messages"][-1]["content"] = [{"type": "text", "text": "SECRET"}]
    before = deepcopy(rows)
    task = Examples(rows, start=0, step=2)
    engine = FakeEngine()
    assert evaluate_task(task, engine, batch_size=2, limit=3, system_prompt="instruction") == EvalResult(3, 1.0)
    assert [len(batch[0]) for batch in engine.batches] == [2, 1]
    assert all(messages == [{"role": "system", "content": "instruction"},
                            {"role": "user", "content": "question"}]
               for messages in engine.tokenizer.seen)
    assert rows == before
    assert not engine.closed


def test_mean_score_and_limit_larger_than_dataset():
    assert evaluate_task(Examples([row(), row("wrong"), row()]), FakeEngine(), limit=99).score == pytest.approx(2 / 3)


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": True}, {"limit": -1}, {"limit": 1.5}])
def test_invalid_bounds(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_task(Examples([row()]), FakeEngine(), **kwargs)


def test_empty_and_unsupported_tasks():
    with pytest.raises(ValueError, match="empty"):
        evaluate_task(Examples([]), FakeEngine())
    task = Examples([row()])
    task.eval_type = "agentic"
    with pytest.raises(ValueError, match="Unsupported"):
        evaluate_task(task, FakeEngine())
    with pytest.raises(ValueError, match="does not define"):
        evaluate_task(Task(), FakeEngine())


@pytest.mark.parametrize("messages,match", [
    ([{"role": "user", "content": "q"}], "reference"),
    ([{"role": "user", "content": []}, {"role": "assistant", "content": "a"}], "text input"),
    ([{"role": "tool", "content": "q"}, {"role": "assistant", "content": "a"}], "roles"),
])
def test_invalid_inputs(messages, match):
    with pytest.raises(ValueError, match=match):
        evaluate_task(Examples([dict(row(), messages=messages)]), FakeEngine())


def test_system_conflict_and_chat_capability():
    example = row()
    example["messages"].insert(0, {"role": "system", "content": "existing"})
    with pytest.raises(ValueError, match="already contains"):
        evaluate_task(Examples([example]), FakeEngine(), system_prompt="new")
    engine = FakeEngine()
    engine.capabilities = EngineCapabilities(chat=False)
    with pytest.raises(ValueError, match="chat"):
        evaluate_task(Examples([row()]), engine)
    assert engine.batches == []


def test_output_count_mismatch():
    class Broken(FakeEngine):
        def generate(self, input_ids, attention_mask, **kwargs):
            return []

    task = Examples([dict(row(), letters=["A", "B"])])
    with pytest.raises(RuntimeError, match="unexpected number"):
        evaluate_task(task, Broken())


def test_categorical_tasks_are_not_supported_yet():
    task = Examples([row()])
    task.eval_type = "categorical"
    with pytest.raises(ValueError, match="Unsupported task eval_type"):
        evaluate_task(task, FakeEngine())


def test_task_error_propagates_without_closing_engine():
    task = Examples([row()])
    def fail(*args):
        raise RuntimeError("scorer failed")
    task.evaluate = fail
    engine = FakeEngine()
    with pytest.raises(RuntimeError, match="scorer failed"):
        evaluate_task(task, engine)
    assert not engine.closed


def test_gsm8k_generation_scoring(monkeypatch):
    from datasets import Dataset
    from rl_lab.tasks import gsm8k

    monkeypatch.setattr(gsm8k, "load_dataset", lambda *a, **k: Dataset.from_list([
        {"question": "What is 2 + 2?", "answer": "2+2=<<2+2=4>>4\n#### 4"},
    ]))
    engine = FakeEngine()
    engine.generate = lambda input_ids, attention_mask, **kwargs: [
        ("#### 4", None, None) for _ in input_ids
    ]
    assert evaluate_task(gsm8k.GSM8K("main", "test"), engine).score == 1

def test_runner_with_real_attached_engine(model, tokenizer):
    from rl_lab.engines.transformers import TransformersEngine

    class RawTask(Examples):
        def build_eval_input(self, example):
            return TextPrompt("hello ")

        def evaluate(self, example, completion):
            return True

    model.train()
    engine = TransformersEngine(model, tokenizer)
    assert evaluate_task(RawTask([row()]), engine, max_new_tokens=3).score == 1
    assert model.training
    assert engine.model is model
