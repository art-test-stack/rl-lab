from copy import deepcopy

import pytest

from rl_lab.engines import ChatPrompt, CompletionEngine, EngineCapabilities, Generation, TextPrompt
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
    capabilities = EngineCapabilities(generation=True, loglikelihood=True, chat=True)

    def __init__(self):
        self.batches = []
        self.requests = []
        self.closed = False
        self.tokenizer = FakeTokenizer()

    def generate(self, input_ids, attention_mask, **kwargs):
        self.batches.append((input_ids.tolist(), attention_mask.tolist()))
        return [Generation("correct") for _ in input_ids]

    def loglikelihood(self, requests):
        self.requests.append(requests)
        return [-1.0 if r.continuation == "correct" else -5.0 for r in requests]

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


def test_variable_candidate_counts_are_batched_and_aligned():
    rows = [dict(row(), letters=["wrong", "correct"]),
            dict(row(), letters=["correct", "x", "y"])]
    task = Examples(rows)
    task.eval_type = "categorical"
    engine = FakeEngine()
    assert evaluate_task(task, engine).score == 1
    assert len(engine.requests) == 1
    assert [r.continuation for r in engine.requests[0]] == ["wrong", "correct", "correct", "x", "y"]
    assert engine.batches == []


def test_raw_prompt_and_candidate_overrides():
    class RawTask(Examples):
        eval_type = "categorical"

        def build_eval_input(self, example):
            return TextPrompt("raw prefix")

        def evaluation_candidates(self, example):
            return ["correct", "incorrect"]

    engine = FakeEngine()
    engine.capabilities = EngineCapabilities(loglikelihood=True, chat=False)
    assert evaluate_task(RawTask([row()]), engine).score == 1
    with pytest.raises(ValueError, match="requires ChatPrompt"):
        evaluate_task(RawTask([row()]), engine, system_prompt="system")


def test_tie_uses_first_candidate():
    task = Examples([dict(row("first"), letters=["first", "second"])])
    task.eval_type = "categorical"
    assert evaluate_task(task, FakeEngine()).score == 1


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


@pytest.mark.parametrize("kind", ["generative", "categorical"])
def test_output_count_mismatch(kind):
    class Broken(FakeEngine):
        def generate(self, input_ids, attention_mask, **kwargs):
            return []

        def loglikelihood(self, requests):
            return []

    task = Examples([dict(row(), letters=["A", "B"])])
    task.eval_type = kind
    with pytest.raises(RuntimeError, match="unexpected number"):
        evaluate_task(task, Broken())


@pytest.mark.parametrize("choices", [[], "AB", ["A", ""], ["A", 2]])
def test_invalid_candidates(choices):
    task = Examples([dict(row(), letters=choices)])
    task.eval_type = "categorical"
    with pytest.raises(ValueError):
        evaluate_task(task, FakeEngine())


@pytest.mark.parametrize("scores", [[float("nan"), 0], [float("inf"), 0], [-float("inf"), -float("inf")]])
def test_invalid_likelihoods(scores):
    engine = FakeEngine()
    engine.loglikelihood = lambda requests: scores
    task = Examples([dict(row(), letters=["A", "B"])])
    task.eval_type = "categorical"
    with pytest.raises(ValueError, match="likelihood"):
        evaluate_task(task, engine)


def test_task_error_propagates_without_closing_engine():
    task = Examples([row()])
    def fail(*args):
        raise RuntimeError("scorer failed")
    task.evaluate = fail
    engine = FakeEngine()
    with pytest.raises(RuntimeError, match="scorer failed"):
        evaluate_task(task, engine)
    assert not engine.closed


def test_gsm8k_structured_reference_and_mmlu_scoring(monkeypatch):
    from datasets import Dataset
    from rl_lab.tasks import gsm8k, mmlu

    monkeypatch.setattr(gsm8k, "load_dataset", lambda *a, **k: Dataset.from_list([
        {"question": "What is 2 + 2?", "answer": "2+2=<<2+2=4>>4\n#### 4"},
    ]))
    engine = FakeEngine()
    engine.generate = lambda input_ids, attention_mask, **kwargs: [Generation("#### 4") for _ in input_ids]
    assert evaluate_task(gsm8k.GSM8K("main", "test"), engine).score == 1

    monkeypatch.setattr(mmlu, "load_dataset", lambda *a, **k: Dataset.from_list([
        {"question": "question", "choices": ["a", "b", "c", "d"], "answer": 1, "subject": "math"},
    ]))
    engine.loglikelihood = lambda requests: [-5, -1, -5, -5]
    assert evaluate_task(mmlu.MMLU("all", "test"), engine).score == 1


def test_runner_with_real_attached_engine(model, tokenizer):
    from rl_lab.engines.transformers import TransformersEngine

    class RawTask(Examples):
        eval_type = "categorical"

        def build_eval_input(self, example):
            return TextPrompt("hello ")

        def evaluation_candidates(self, example):
            return ["A", "B"]

        def evaluate(self, example, completion):
            return completion in {"A", "B"}

    model.train()
    engine = TransformersEngine(model, tokenizer)
    assert evaluate_task(RawTask([row()]), engine).score == 1
    assert model.training
    assert engine.model is model
