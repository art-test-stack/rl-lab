import pytest

from rl_lab.engines import CompletionEngine, EngineCapabilities, Generation
from rl_lab.eval import evaluate_task
from rl_lab.tasks import spellingbee


@pytest.fixture
def words(tmp_path, monkeypatch):
    path = tmp_path / "words.txt"
    path.write_text("strawberry\n\nbanana\nmississippi\n")
    monkeypatch.setattr(spellingbee, "download_file_with_lock", lambda *a: str(path))
    return path


def test_inherited_contract_and_exact_scoring(words):
    task = spellingbee.SimpleSpelling(size=12)
    assert task.eval_type == "generative"
    assert len(task) == 12
    example = task[3]
    word = example["messages"][0]["content"].removeprefix("Spell the word: ")
    answer = f"{word}:{','.join(word)}"
    assert example["messages"][-1]["content"] == answer
    assert task.evaluate(example, f"\n{answer}\n")
    assert task.reward(example, answer) == 1.0
    assert not task.evaluate(example, word)
    assert not task.evaluate(example, answer + ",x")
    assert task[3] == example
    assert spellingbee.SimpleSpelling(size=12, start=1, stop=8, step=2)[1] == example


def test_runner(words, tokenizer):
    task = spellingbee.SimpleSpelling(size=7, split="test")
    class SpellEngine(CompletionEngine):
        capabilities = EngineCapabilities(chat=True)
        def __init__(self):
            self.tokenizer = tokenizer
            self.position = 0
        def generate(self, input_ids, attention_mask, **kwargs):
            expected = [task[self.position + i]["messages"][-1]["content"] for i in range(len(input_ids))]
            self.position += len(input_ids)
            return [Generation(value) for value in expected]
    assert evaluate_task(task, SpellEngine(), batch_size=3).score == 1


def test_invalid_size_and_empty_words(words):
    with pytest.raises(ValueError, match="size"):
        spellingbee.SimpleSpelling(size=-1)
    words.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        spellingbee.SimpleSpelling()
