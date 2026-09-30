import pytest
import torch
from types import SimpleNamespace

from rl_lab.engines.transformers import TransformersEngine


def test_generate_batches_and_returns_text_and_token_counts(model, tokenizer, monkeypatch):
    model.train()
    engine = TransformersEngine(model, tokenizer)
    assert engine.model is model
    assert engine.tokenizer is tokenizer

    def generate(**kwargs):
        assert kwargs["input_ids"].tolist() == [[0, 4], [4, 5]]
        assert kwargs["attention_mask"].tolist() == [[0, 1], [1, 1]]
        assert kwargs["max_new_tokens"] == 3
        assert kwargs["use_cache"] is True
        assert kwargs["return_dict_in_generate"] is True
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 0
        assert "stop_strings" not in kwargs
        assert not model.training
        assert not torch.is_grad_enabled()
        return SimpleNamespace(
            sequences=torch.tensor([[0, 4, 6, 2, 0], [4, 5, 5, 6, 6]]),
            past_key_values=object(),
        )

    monkeypatch.setattr(model, "generate", generate)
    outputs, cache = engine.generate(
        [[0, 4], [4, 5]],
        [[0, 1], [1, 1]],
        max_new_tokens=3,
        temperature=0.7,
        top_p=0.9,
        stop=["answer"],
    )

    assert outputs == [("", 1, 2, [6, 2]), ("world ", 2, 3, [5, 6, 6])]
    assert cache is not None
    assert model.training


def test_batched_greedy_matches_individual_generation(model, tokenizer, encode_inputs):
    engine = TransformersEngine(model, tokenizer)
    prompts = ["hello", "hello world"]
    ids, mask = encode_inputs(tokenizer, *prompts)
    batched, _ = engine.generate(ids, mask, max_new_tokens=3)
    individual = [
        engine.generate(*encode_inputs(tokenizer, prompt), max_new_tokens=3)[0][0]
        for prompt in prompts
    ]
    assert batched == individual


def test_restores_top_level_mode_on_generation_failure(model, tokenizer, monkeypatch):
    model.train()

    def fail(**kwargs):
        assert not torch.is_grad_enabled()
        assert not model.training
        raise RuntimeError("generation failed")

    monkeypatch.setattr(model, "generate", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        TransformersEngine(model, tokenizer).generate(
            [4], [1]
        )


def test_pad_falls_back_to_eos_without_mutating_tokenizer(model, tokenizer, encode_inputs):
    tokenizer.pad_token = None
    engine = TransformersEngine(model, tokenizer)
    outputs, _ = engine.generate(
        *encode_inputs(tokenizer, "hello", "hello world"), max_new_tokens=1
    )
    assert len(outputs) == 2
    assert tokenizer.pad_token_id is None
    assert engine.generate([], []) == ([], None)


def test_load_from_local_checkpoint(model, tokenizer, tmp_path, encode_inputs):
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    engine = TransformersEngine.load(
        str(tmp_path),
        device="cpu",
        local_files_only=True,
        tokenizer_kwargs={"local_files_only": True},
    )
    assert engine.model is not model
    outputs, cache = engine.generate(
        *encode_inputs(engine.tokenizer, "hello"), max_new_tokens=1
    )
    assert outputs[0][1] == 1
    assert outputs[0][3]
    assert cache is not None
    engine.close()


def test_generation_continues_from_cache_with_only_uncached_ids(
    model, tokenizer, encode_inputs, monkeypatch
):
    engine = TransformersEngine(model, tokenizer)
    prompt_ids, prompt_mask = encode_inputs(tokenizer, "hello")
    first, cache = engine.generate(prompt_ids, prompt_mask, max_new_tokens=2)
    generated_ids = first[0][3]
    assert len(generated_ids) == 2
    assert cache.get_seq_length() == prompt_ids.shape[1] + len(generated_ids) - 1

    new_query = tokenizer.encode("world", add_special_tokens=False)
    uncached_ids = torch.tensor([[generated_ids[-1], *new_query]])
    full_history = torch.tensor(
        [[*prompt_ids[0].tolist(), *generated_ids, *new_query]]
    )
    cumulative_mask = torch.ones_like(full_history)
    full_result, _ = engine.generate(
        full_history, cumulative_mask, max_new_tokens=2
    )

    original_generate = model.generate
    observed = {}

    def record_generate(**kwargs):
        observed.update(kwargs)
        return original_generate(**kwargs)

    monkeypatch.setattr(model, "generate", record_generate)
    cached_result, updated_cache = engine.generate(
        uncached_ids,
        cumulative_mask,
        past_key_values=cache,
        max_new_tokens=2,
    )

    assert observed["input_ids"].tolist() == uncached_ids.tolist()
    assert observed["attention_mask"].shape == cumulative_mask.shape
    assert observed["past_key_values"] is cache
    assert cached_result[0][3] == full_result[0][3]
    assert updated_cache is not None
