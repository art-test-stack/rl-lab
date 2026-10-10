"""Check comparison orchestration with mocked model loading and benchmark runners."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from rl_lab.eval import EvalResult
from rl_lab.tasks._base import Task


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("error", [None, ValueError("invalid configuration"), RuntimeError("generation failed")])
def test_tool_generated_text_is_scored_without_retry(monkeypatch, batch_size, error):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from scripts.eval import compare

    class Question(Task):
        def num_examples(self):
            return 2

        def get_example(self, index):
            return {"messages": [{"role": "user", "content": str(index)},
                                 {"role": "assistant", "content": "answer"}]}

        def evaluate(self, example, prediction):
            return float(prediction == "answer")

    calls = []
    def complete(conversations, **kwargs):
        calls.append(conversations)
        if error is not None:
            raise error
        return [messages + [{"role": "assistant", "content":
                            "invalid" if messages[-1]["content"] == "0" else "answer"}]
                for messages in conversations]

    records = []
    args = SimpleNamespace(max_per_task=-1, batch_size=batch_size,
                           max_new_tokens=32, max_tool_rounds=2)
    def run():
        return compare.evaluate_with_tools(
            Question(), SimpleNamespace(complete=complete), "gsm8k", args, [], None,
            lambda index, example, prediction, score, **extra:
                records.append({"index": index, "prediction": prediction, "score": score, **extra}),
        )

    if error is not None:
        with pytest.raises(type(error), match=str(error)):
            run()
        assert len(calls) == 1
        return
    result = run()
    assert result.num_examples == 2 and result.score == 0.5
    assert records[0]["prediction"] == "invalid" and records[0]["score"] == 0.0
    assert records[0]["history"][-1]["content"] == "invalid"
    assert records[1]["score"] == 1 and records[1]["prediction"] == "answer"
    assert len(calls) == 2 // batch_size


def test_named_models_default_tasks_predictions_and_plots(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from scripts.eval import compare
    from matplotlib.figure import Figure

    overview = {}
    savefig = Figure.savefig

    def capture_overview(figure, path, **kwargs):
        if Path(path).name == "overview.png":
            axis = figure.axes[0]
            overview["benchmarks"] = [tick.get_text() for tick in axis.get_xticklabels()]
            overview["scores"] = [[bar.get_height() for bar in bars] for bars in axis.containers]
            overview["models"] = [text.get_text() for text in axis.get_legend().get_texts()]
            overview["limits"] = axis.get_ylim()
        return savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_overview)

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
        "core/core": -0.25, "all_core_results": {
            "local": {"accuracy": 0.5, "num_examples": 11, "centered_result": 0.25},
        },
    })
    monkeypatch.setattr(compare, "REPORTS_FOLDER", tmp_path / "reports")
    monkeypatch.setattr(compare, "MODELS_FOLDER", tmp_path / "models")
    checkpoint = tmp_path / "models" / "owner--model" / "sft" / "default"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    named_checkpoint = checkpoint.parent / "run-2"
    named_checkpoint.mkdir()
    (named_checkpoint / "config.json").write_text("{}")
    assert compare.resolve_model("owner/model:sft:run-2") == str(named_checkpoint)
    assert compare.resolve_model(str(checkpoint)) == str(checkpoint)
    with pytest.raises(ValueError, match="No cached checkpoint"):
        compare.resolve_model("other-owner/model:sft")
    with pytest.raises(ValueError, match="Invalid cached"):
        compare.resolve_model("owner/model:sft:../other")
    monkeypatch.setattr(sys, "argv", ["compare", "--model", "After SFT=owner/model:sft",
                                       "--model", "Baseline=baseline", "--max-per-task", "11",
                                       "--device-type", "cpu"])
    compare.main()
    output, = (tmp_path / "reports").glob("compare-*")
    results = json.loads((output / "evaluation.json").read_text())
    assert loads == [str(checkpoint), "baseline"]  # No before checkpoint is required.
    assert results["settings"]["tasks"] == compare.select_tasks(None)
    assert compare.select_tasks("None") == compare.select_tasks("all") == list(compare.TASKS)
    assert compare.select_tasks("gsm8k,arc,mmlu") == ["gsm8k", "arc_easy", "arc_challenge", "mmlu"]
    assert len([call for call in calls if call[0] == "categorical"]) == 6
    for index, entry in enumerate(results["models"]):
        assert entry["complete"]
        assert set(entry["tasks"]) == set(compare.TASKS) - {"core"} | {"core_local"}
        assert entry["core_metrics"]["core/core"] == -0.25
        record = json.loads((output / f"model-{index}-gsm8k.jsonl").read_text())
        assert record["reference"] == "answer"
        assert record["prediction"] == "generated"
        assert record["index"] == 0
        assert record["score"] == 0.0
    for name in results["models"][0]["tasks"]:
        assert (output / "plots" / f"{name}.png").stat().st_size > 0
        assert (output / "plots" / f"{name}.pdf").stat().st_size > 0
    assert overview["benchmarks"] == list(compare.TASKS)[:-1] + ["CORE\n(centered accuracy)"]
    assert overview["models"] == ["After SFT", "Baseline"]
    expected_scores = [results["models"][0]["tasks"][name]["score"]
                       for name in compare.TASKS if name != "core"] + [-0.25]
    assert overview["scores"] == [expected_scores, expected_scores]
    assert overview["limits"][0] < -0.25
    for extension in ("png", "pdf"):
        assert (output / "plots" / f"overview.{extension}").stat().st_size > 0
    with pytest.raises(ValueError, match="Unknown/unscored"):
        compare.select_tasks("smoltalk")


@pytest.mark.parametrize("selector", ["all", "Qwen", "None", "none"])
def test_native_tool_selection_final_scoring_and_saved_history(monkeypatch, tmp_path, selector):
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from scripts.eval import compare

    class Question(Task):
        eval_type = "generative"

        def num_examples(self):
            return 1

        def get_example(self, index):
            return {"messages": [{"role": "user", "content": "Compute 3 + 4."},
                                 {"role": "assistant", "content": "#### 7"}]}

        def evaluate(self, example, prediction):
            assert prediction == "#### 7"  # Score the final answer, not tool output.
            return 1.0

    monkeypatch.setattr(compare, "TASKS", {"gsm8k": Question})
    native_calls = []
    plain_calls = []

    @contextmanager
    def load(path, device):
        def render(messages, tokenize=False, **kwargs):
            assert all(message["role"] != "assistant" for message in messages)
            if tokenize:
                return [1]
            return "<|im_start|><tool_call> python" if kwargs.get("tools") else "plain"

        def complete(conversations, tools, response_template, max_new_tokens, max_tool_rounds):
            native_calls.append(path)
            assert [tool.__name__ for tool in tools] == ["python"]
            assert response_template == compare.QWEN_RESPONSE_TEMPLATE
            assert max_new_tokens == 32 and max_tool_rounds == 2
            assert conversations[0][0]["role"] == "system"
            assert compare.TOOL_INSTRUCTION in conversations[0][0]["content"]
            assert conversations[0][-1] == {"role": "user", "content": "Compute 3 + 4."}
            return [conversations[0] + [
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": "python", "arguments": {"code": "print(3 + 4)"}}},
                ]},
                {"role": "tool", "name": "python", "content": '{"stdout":"7"}'},
                {"role": "assistant", "content": "#### 7"},
            ]]

        yield SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024, model_type="qwen2"),
                                  eval=lambda: None),
            tokenizer=SimpleNamespace(pad_token_id=0, chat_template="tool-aware", apply_chat_template=render),
            complete=complete,
        )

    def plain(task, engine, **kwargs):
        plain_calls.append(True)
        kwargs["on_result"](0, task[0], "no tools", 0.0)
        return EvalResult(num_examples=1, score=0.0)

    monkeypatch.setattr(compare.TransformersEngine, "load", load)
    monkeypatch.setattr(compare, "evaluate_task", plain)
    output = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", [
        "compare", "--model", "Baseline=baseline", "--model", "Qwen=qwen", "--tasks", "gsm8k",
        "--tools", "python", "--tool-models", selector, "--max-new-tokens", "32",
        "--max-tool-rounds", "2", "--python-timeout", "1", "--device-type", "cpu",
        "--output-dir", str(output),
    ])
    compare.main()
    selected = compare.select_tool_models("python", selector, ["Baseline", "Qwen"])
    results = json.loads((output / "evaluation.json").read_text())
    assert results["settings"]["tool_models"] == selected
    assert len(native_calls) == len(selected)
    assert len(plain_calls) == 2 - len(selected)
    for index, entry in enumerate(results["models"]):
        enabled = entry["label"] in selected
        tools = ["python"] if enabled else []
        assert entry["tasks"]["gsm8k"]["tools"] == tools
        assert entry["tasks"]["gsm8k"]["score"] == float(enabled)
        record = json.loads((output / f"model-{index}-gsm8k.jsonl").read_text())
        assert record["tools"] == tools
        if enabled:
            assert record["prediction"] == "#### 7"
            assert record["history"][-2]["role"] == "tool"
            assert record["history"][-3]["tool_calls"][0]["function"]["name"] == "python"
        else:
            assert "history" not in record
    for value in (None, "None", "none", "NONE"):
        assert compare.select_tool_models(value, "all", ["Qwen"]) == []
    with pytest.raises(ValueError, match="model labels"):
        compare.select_tool_models("python", "missing", ["Qwen"])
    with pytest.raises(ValueError, match="GPT-2 is unsupported"):
        compare.tool_response_template(SimpleNamespace(tokenizer=SimpleNamespace(chat_template=None)), [])
