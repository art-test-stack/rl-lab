import pytest
import torch

from rl_lab.engines import (
    ChatPrompt,
    LogLikelihoodRequest,
)
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
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 0
        assert "stop_strings" not in kwargs
        assert not model.training
        assert not torch.is_grad_enabled()
        return torch.tensor([[0, 4, 6, 2, 0], [4, 5, 5, 6, 6]])

    monkeypatch.setattr(model, "generate", generate)
    outputs = engine.generate(
        [[0, 4], [4, 5]],
        [[0, 1], [1, 1]],
        max_new_tokens=3,
        temperature=0.7,
        top_p=0.9,
        stop=["answer"],
    )

    assert outputs == [("", 1, 2), ("world ", 2, 3)]
    assert model.training


def test_batched_greedy_matches_individual_generation(model, tokenizer, encode_inputs):
    engine = TransformersEngine(model, tokenizer)
    prompts = ["hello", "hello world"]
    ids, mask = encode_inputs(tokenizer, *prompts)
    batched = engine.generate(ids, mask, max_new_tokens=3)
    individual = [
        engine.generate(*encode_inputs(tokenizer, prompt), max_new_tokens=3)[0]
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
    assert (
        len(
            engine.generate(*encode_inputs(tokenizer, "hello", "hello world"), max_new_tokens=1)
        )
        == 2
    )
    assert tokenizer.pad_token_id is None
    assert engine.generate([], []) == []


def test_likelihood_matches_manual_causal_scoring_and_restores_mode(model, tokenizer):
    engine = TransformersEngine(model, tokenizer)
    model.eval()
    with torch.no_grad():
        logprobs = model(torch.tensor([[4, 5, 6]])).logits[0].log_softmax(-1)
        expected = (logprobs[0, 5] + logprobs[1, 6]).item()
    model.train()
    result = engine.loglikelihood([LogLikelihoodRequest("hello", " world answer")])
    assert result[0] == pytest.approx(expected, abs=1e-6)
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    assert engine.loglikelihood([LogLikelihoodRequest("hello", "")]) == [0.0]
    assert engine.loglikelihood([LogLikelihoodRequest("", "world")])[0] < 0


def test_likelihood_rejects_merged_boundary(model, tokenizer):
    with pytest.raises(ValueError, match="boundary"):
        TransformersEngine(model, tokenizer).loglikelihood(
            [LogLikelihoodRequest("hello", "world")]
        )
    assert model.training


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
    assert (
        engine.generate(*encode_inputs(engine.tokenizer, "hello"), max_new_tokens=1)[0][1]
        == 1
    )
    engine.close()


def test_chat_likelihood_and_trailing_whitespace(model, tokenizer):
    engine = TransformersEngine(model, tokenizer)
    context = ChatPrompt([{"role": "user", "content": "hello"}])
    scores = engine.loglikelihood([LogLikelihoodRequest(context, "A")])
    assert len(scores) == 1 and scores[0] < 0
    assert engine.loglikelihood(
        [LogLikelihoodRequest("hello ", "world")]
    ) == engine.loglikelihood(
        [
            LogLikelihoodRequest("hello", " world"),
        ]
    )
