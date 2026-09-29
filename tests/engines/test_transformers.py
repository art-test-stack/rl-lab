import copy

import pytest
import torch

from rl_lab.engines import (
    ChatPrompt,
    GenerationConfig,
    LogLikelihoodRequest,
    TextPrompt,
    TransformersEngine,
)


def test_attach_reuses_objects_and_restores_mixed_modes(model, tokenizer):
    model.train()
    model.transformer.h[0].eval()
    states = [m.training for m in model.modules()]
    weights = [p.detach().clone() for p in model.parameters()]
    tokenizer_settings = (tokenizer.padding_side, tokenizer.pad_token_id)
    generation_config = copy.deepcopy(model.generation_config.to_dict())
    engine = TransformersEngine.attach(model, tokenizer, batch_size=2)
    assert engine.model is model
    assert engine.tokenizer is tokenizer
    outputs = engine.generate(
        ["hello", TextPrompt("hello world")], GenerationConfig(max_new_tokens=2)
    )
    assert len(outputs) == 2
    assert [o.prompt_tokens for o in outputs] == [1, 2]
    assert all(1 <= o.completion_tokens <= 2 for o in outputs)
    assert [m.training for m in model.modules()] == states
    assert (tokenizer.padding_side, tokenizer.pad_token_id) == tokenizer_settings
    assert model.generation_config.to_dict() == generation_config
    assert all(torch.equal(old, new) for old, new in zip(weights, model.parameters()))
    engine.close()
    assert [m.training for m in model.modules()] == states
    model(torch.tensor([[4, 5]])).logits.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate([], GenerationConfig())


def test_batched_greedy_matches_individual_generation(model, tokenizer):
    engine = TransformersEngine.attach(model, tokenizer)
    config = GenerationConfig(max_new_tokens=3)
    prompts = ["hello", "hello world"]
    batched = engine.generate(prompts, config)
    individual = [engine.generate([p], config)[0] for p in prompts]
    assert batched == individual


def test_restores_modes_on_generation_failure(model, tokenizer, monkeypatch):
    model.train()
    model.transformer.h[0].eval()
    states = [m.training for m in model.modules()]

    def fail(**kwargs):
        assert not torch.is_grad_enabled()
        assert not model.training
        raise RuntimeError("generation failed")

    monkeypatch.setattr(model, "generate", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        TransformersEngine.attach(model, tokenizer).generate(
            ["hello"], GenerationConfig()
        )
    assert [m.training for m in model.modules()] == states


def test_stop_criteria_per_row_and_completion_only(model, tokenizer, monkeypatch):
    def generate(**kwargs):
        ids = kwargs["input_ids"]
        assert ids.tolist() == [[0, 4], [4, 5]]  # left padding
        assert kwargs["attention_mask"].tolist() == [[0, 1], [1, 1]]
        assert not kwargs["generation_config"].do_sample
        assert kwargs["stopping_criteria"](ids, None).tolist() == [False, False]
        out = torch.cat([ids, torch.tensor([[6, 2], [5, 6]])], dim=1)
        # One row finishes earlier; the other must continue independently.
        assert kwargs["stopping_criteria"](out[:, :-1], None).tolist() == [True, False]
        return out

    monkeypatch.setattr(model, "generate", generate)
    outputs = TransformersEngine.attach(model, tokenizer).generate(
        ["hello", "hello world"],
        GenerationConfig(stop=["answer"]),
    )
    assert [o.text for o in outputs] == ["", "world "]
    assert [o.completion_tokens for o in outputs] == [1, 2]


def test_chat_template_eos_count_and_sampling(model, tokenizer, monkeypatch):
    def generate(**kwargs):
        assert kwargs["input_ids"].tolist() == [[1, 4, 6]]
        config = kwargs["generation_config"]
        assert config.do_sample and config.temperature == 0.7 and config.top_p == 0.9
        return torch.cat([kwargs["input_ids"], torch.tensor([[5, 2, 0]])], dim=1)

    monkeypatch.setattr(model, "generate", generate)
    outputs = TransformersEngine.attach(model, tokenizer).generate(
        [ChatPrompt([{"role": "user", "content": "hello"}])],
        GenerationConfig(temperature=0.7, top_p=0.9),
    )
    assert outputs[0].text == "world"
    assert outputs[0].completion_tokens == 2


def test_pad_falls_back_to_eos_without_mutating_tokenizer(model, tokenizer):
    tokenizer.pad_token = None
    engine = TransformersEngine.attach(model, tokenizer)
    assert (
        len(
            engine.generate(
                ["hello", "hello world"], GenerationConfig(max_new_tokens=1)
            )
        )
        == 2
    )
    assert tokenizer.pad_token_id is None
    assert engine.generate([], GenerationConfig()) == []
    assert engine.loglikelihood([]) == []


def test_likelihood_matches_manual_causal_scoring_and_restores_mode(model, tokenizer):
    engine = TransformersEngine.attach(model, tokenizer)
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
        TransformersEngine.attach(model, tokenizer).loglikelihood(
            [LogLikelihoodRequest("hello", "world")]
        )
    assert model.training


def test_load_from_local_checkpoint(model, tokenizer, tmp_path):
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
        engine.generate(["hello"], GenerationConfig(max_new_tokens=1))[0].prompt_tokens
        == 1
    )
    engine.close()


def test_chat_likelihood_and_trailing_whitespace(model, tokenizer):
    engine = TransformersEngine.attach(model, tokenizer)
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
