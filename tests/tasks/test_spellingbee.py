import random
from io import BytesIO

import pytest

from rl_lab.tasks import spellingbee
from rl_lab.utils import common


@pytest.fixture
def words(tmp_path, monkeypatch):
    path = tmp_path / "words.txt"
    path.write_text("strawberry\n\nbanana\nmississippi\n")
    monkeypatch.setattr(spellingbee, "download_file_with_lock", lambda *a: str(path))
    return path


def test_counting_references_and_deterministic_views(words):
    task = spellingbee.SpellingBee(size=10)
    assert len(task) == 10
    assert "" not in task.words
    before = random.getstate()
    example = task[4]
    assert random.getstate() == before
    assert task[4] == example
    assert spellingbee.SpellingBee(size=10, start=2, stop=8, step=2)[1] == example
    assert spellingbee.SpellingBee(size=10, split="test")[4] != example
    reference = example["messages"][-1]["content"][-1]["text"]
    assert task.evaluate(example, reference) == 1
    assert task.reward(example, reference) == 1.0
    assert task.evaluate(example, "no numerical answer") == 0
    assert len(task.build_eval_input(example).messages) == 1


def test_missing_reference_is_not_a_correct_answer(words):
    task = spellingbee.SpellingBee()
    example = task[0]
    example["messages"][-1]["content"][-1]["text"] = "missing"
    with pytest.raises(ValueError, match="reference"):
        task.evaluate(example, "missing")


def test_empty_word_list_and_invalid_size(words):
    words.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        spellingbee.SpellingBee()
    with pytest.raises(ValueError, match="size"):
        spellingbee.SpellingBee(size=-1)


def test_fresh_download_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "not-created")
    calls = []
    def fetch(url):
        calls.append(url)
        return BytesIO(b"strawberry\n")
    monkeypatch.setattr(common.urllib.request, "urlopen", fetch)
    path = common.download_file_with_lock("https://example.invalid/words", "words.txt")
    assert open(path).read() == "strawberry\n"
    assert common.download_file_with_lock("https://example.invalid/words", "words.txt") == path
    assert len(calls) == 1
