"""Inference on an existing causal LM, or a separately loaded HF model."""

from contextlib import contextmanager
from typing import Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import GenerationConfig as HFGenerationConfig

from .base import (
    ChatPrompt,
    CompletionEngine,
    EngineCapabilities,
    Generation,
    LogLikelihoodRequest,
    text_prompt,
    token_batch,
    trim_stop,
    validate_generation_options,
)


class TransformersEngine(CompletionEngine):
    capabilities = EngineCapabilities(loglikelihood=True, chat=True)

    def __init__(self, model, tokenizer, *, batch_size: int = 8):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not callable(getattr(model, "generate", None)):
            raise TypeError(
                "Expected a causal LM exposing generate(); distributed wrappers need an explicit training-runtime adapter"
            )
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("TransformersEngine supports decoder-only causal LMs")
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    @classmethod
    def attach(cls, model, tokenizer, **kwargs):
        """Borrow the exact objects; do not copy, move or reload weights."""
        return cls(model, tokenizer, **kwargs)

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        device=None,
        batch_size=8,
        tokenizer_kwargs=None,
        **model_kwargs,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, **(tokenizer_kwargs or {})
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device is not None:
            model.to(device)
        return cls(model, tokenizer, batch_size=batch_size)

    from_model = attach
    from_pretrained = load

    def _check_open(self):
        if self.model is None:
            raise RuntimeError("Engine is closed")

    @contextmanager
    def _inference(self):
        self._check_open()
        # Preserve mixed train/eval submodule states, even after an exception.
        states = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with torch.inference_mode():
                yield
        finally:
            for module, training in states:
                module.training = training

    def generate(
        self,
        input_ids: torch.Tensor | list[int] | list[list[int]],
        attention_mask: torch.Tensor | list[int] | list[list[int]],
        *,
        max_new_tokens: int = 51,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Sequence[str] | None = None,
    ) -> list[Generation]:
        self._check_open()
        sequences, masks = token_batch(input_ids, attention_mask)
        stop = validate_generation_options(max_new_tokens, temperature, top_p, stop)
        if not sequences:
            return []
        tokenizer = self.tokenizer
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = tokenizer.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = eos_ids[0] if eos_ids else None
        if pad is None:
            raise ValueError("Generation needs a pad_token_id or eos_token_id")
        results = []
        with self._inference():
            for offset in range(0, len(sequences), self.batch_size):
                rows = sequences[offset : offset + self.batch_size]
                row_masks = masks[offset : offset + self.batch_size]
                live_rows = [
                    [token for token, keep in zip(ids, row_mask, strict=True) if keep]
                    for ids, row_mask in zip(rows, row_masks, strict=True)
                ]
                width = max(map(len, live_rows))
                device = self.model.get_input_embeddings().weight.device
                batch_ids = torch.full(
                    (len(rows), width), pad, dtype=torch.long, device=device
                )
                batch_mask = torch.zeros_like(batch_ids)
                for i, ids in enumerate(live_rows):
                    batch_ids[i, -len(ids):] = torch.tensor(ids, device=device)
                    batch_mask[i, -len(ids):] = 1

                class StopStrings(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        texts = tokenizer.batch_decode(
                            input_ids[:, width:], skip_special_tokens=True
                        )
                        return torch.tensor(
                            [
                                any(stop_string in text for stop_string in stop)
                                for text in texts
                            ],
                            dtype=torch.bool,
                            device=input_ids.device,
                        )

                kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    pad_token_id=pad,
                    eos_token_id=eos,
                    num_beams=1,
                    num_return_sequences=1,
                    return_dict_in_generate=False,
                )
                if temperature > 0:
                    kwargs.update(
                        temperature=temperature, top_p=top_p, top_k=0
                    )
                output = self.model.generate(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    generation_config=HFGenerationConfig(**kwargs),
                    stopping_criteria=StoppingCriteriaList(
                        [StopStrings()] if stop else []
                    ),
                )
                for prompt_ids, prompt_mask, row in zip(rows, row_masks, output, strict=True):
                    generated = row[width:].tolist()
                    # Find the first actual termination; ignore batch padding afterward.
                    consumed = len(generated)
                    for i, token in enumerate(generated):
                        stopped = bool(stop) and any(
                            stop_string
                            in tokenizer.decode(
                                generated[: i + 1], skip_special_tokens=True
                            )
                            for stop_string in stop
                        )
                        if token in eos_ids or stopped:
                            consumed = i + 1
                            break
                    text = tokenizer.decode(
                        generated[:consumed], skip_special_tokens=True
                    )
                    results.append(
                        Generation(
                            trim_stop(text, stop), sum(prompt_mask), consumed
                        )
                    )
        return results

    def loglikelihood(self, requests: Sequence[LogLikelihoodRequest]) -> list[float]:
        """Score joint tokenization; reject boundaries that split a tokenizer token.

        No EOS is appended. An empty context is conditioned on BOS (or EOS).
        Evaluation protocols retain control of their whitespace and normalization.
        """
        scores = []
        with self._inference():
            device = self.model.get_input_embeddings().weight.device
            for request in requests:
                if not request.continuation:
                    scores.append(0.0)
                    continue
                context = (
                    self.tokenizer.apply_chat_template(
                        list(request.context.messages),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if isinstance(request.context, ChatPrompt)
                    else text_prompt(request.context)
                )
                # Match common harness behavior at whitespace-sensitive BPE boundaries.
                trailing = context[len(context.rstrip()) :]
                context = context.rstrip()
                continuation = trailing + request.continuation
                prefix = self.tokenizer.encode(context, add_special_tokens=False)
                tokens = self.tokenizer.encode(
                    context + continuation, add_special_tokens=False
                )
                if tokens[: len(prefix)] != prefix:
                    raise ValueError(
                        "Context/continuation boundary splits a token; move trailing context whitespace into the continuation"
                    )
                start = len(prefix)
                if start == 0:
                    bos = self.tokenizer.bos_token_id
                    if bos is None:
                        bos = self.tokenizer.eos_token_id
                    if bos is None:
                        raise ValueError("Empty contexts require a BOS or EOS token")
                    tokens = [bos] + tokens
                    start = 1
                if start == len(tokens):
                    raise ValueError("Non-empty continuation produced no tokens")
                ids = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = self.model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                ).logits
                log_probs = logits[0, start - 1 : -1].float().log_softmax(-1)
                targets = ids[0, start:]
                scores.append(log_probs.gather(-1, targets[:, None]).sum().item())
        return scores

    def close(self):
        # Only drop our references; the trainer's model, optimizer and groups survive.
        self.model = None
        self.tokenizer = None
