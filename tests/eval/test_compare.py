"""Check comparison orchestration with mocked model loading and benchmark runners."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from rl_lab.eval import EvalResult
from rl_lab.tasks._base import Task


def test_named_models_default_tasks_predictions_and_plots(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from scripts.eval import compare

    class LocalTask(Task):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind

        @property
        def eval_type(self):
            return self.kind

        def num_examples(self):
            return 1

        def get_example(self, index):
            return {"messages": [{"role": "user", "content": "hello"},
                                 {"role": "assistant", "content": "answer"}]}

    monkeypatch.setattr(compare, "TASKS", {
        name: (None if name == "core" else lambda name=name: LocalTask(
            "categorical" if name in {"mmlu", "arc_easy", "arc_challenge"} else "generative"
        )) for name in compare.TASKS
    })
    loads = []
    calls = []

    @contextmanager
    def load(path, device):
        loads.append(path)
        yield SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024), eval=lambda: None),
            tokenizer=SimpleNamespace(pad_token_id=0, chat_template="local", apply_chat_template=lambda *a, **k: [1]),
        )

    def run(task, engine, **kwargs):
        calls.append(("generative", kwargs.get("system_prompt")))
        kwargs["on_result"](0, task[0], "generated", 0.0)
        return EvalResult(num_examples=1, score=0.0)

    def categorical(task, model, tokenizer, limit, on_result):
        calls.append(("categorical", limit))
        on_result(0, task[0], "A", 1.0)
        return EvalResult(num_examples=1, score=1.0)

    monkeypatch.setattr(compare.TransformersEngine, "load", load)
    monkeypatch.setattr(compare, "evaluate_task", run)
    monkeypatch.setattr(compare, "evaluate_categorical", categorical)
    monkeypatch.setattr("rl_lab.tasks.core.evaluate_core", lambda *a, **k: {
        "core/core": 0.25, "all_core_results": {
            "local": {"accuracy": 0.5, "num_examples": 11, "centered_result": 0.25},
        },
    })
    output = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", ["compare", "--model", "After SFT=checkpoint",
                                       "--model", "Baseline=baseline", "--max-per-task", "11",
                                       "--device-type", "cpu", "--output-dir", str(output)])
    compare.main()
    results = json.loads((output / "evaluation.json").read_text())
    assert loads == ["checkpoint", "baseline"]  # No before checkpoint is required.
    assert results["settings"]["tasks"] == compare.select_tasks(None)
    assert compare.select_tasks("None") == compare.select_tasks("all") == list(compare.TASKS)
    assert compare.select_tasks("gsm8k,arc,mmlu") == ["gsm8k", "arc_easy", "arc_challenge", "mmlu"]
    assert len([call for call in calls if call[0] == "categorical"]) == 6
    for index, entry in enumerate(results["models"]):
        assert entry["complete"]
        assert set(entry["tasks"]) == set(compare.TASKS) - {"core"} | {"core_local"}
        assert entry["core_metrics"]["core/core"] == 0.25
        record = json.loads((output / f"model-{index}-gsm8k.jsonl").read_text())
        assert record["reference"] == "answer"
        assert record["prediction"] == "generated"
        assert record["index"] == 0
        assert record["score"] == 0.0
    for name in results["models"][0]["tasks"]:
        assert (output / "plots" / f"{name}.png").stat().st_size > 0
        assert (output / "plots" / f"{name}.pdf").stat().st_size > 0
    with pytest.raises(ValueError, match="Unknown/unscored"):
        compare.select_tasks("smoltalk")
