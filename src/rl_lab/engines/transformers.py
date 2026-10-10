"""Inference on an existing causal LM, or a separately loaded HF model."""

from copy import deepcopy
from inspect import signature
from typing import Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema

from .base import (
    CompletionEngine,
    EngineCapabilities,
    GenerationOutput,
    trim_stop,
    validate_generation_options,
)


class TransformersEngine(CompletionEngine):
    capabilities = EngineCapabilities(chat=True)

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        device=None,
        tokenizer_kwargs=None,
        **model_kwargs,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, **(tokenizer_kwargs or {})
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device is not None:
            model.to(device)
        return cls(model, tokenizer)

    from_pretrained = load

    def complete(
        self,
        conversations: Sequence[Sequence[dict]],
        *,
        tools: Sequence[Callable] = (),
        response_template: dict | None = None,
        max_tool_rounds: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[list[dict] | None]:
        """Complete chats in batch rounds, executing registered Python callables.

        Return copied histories or None for failed chats in input order.
        Only unfinished chats enter the next round. Tool calls run sequentially
        after the entire batch returns.
        Each round re-encodes full histories without reusing a generation cache.
        ``max_new_tokens`` limits each assistant turn; ``max_tool_rounds`` limits
        tool-bearing turns per chat, followed by one final generation round.

        Parsing uses the supplied response template or the tokenizer's own.
        The model/chat template must support tool use. Turns without EOS return
        the generated text as a final assistant message without executing tools.
        Malformed responses and exhausted tool rounds return None
        for that chat; configuration and generation failures still raise.
        Unknown tools and callable exceptions become tool error responses.
        Callables must enforce their own timeouts, as the Python wrapper does.
        """
        if self.model is None:
            raise RuntimeError("Engine is closed")
        validate_generation_options(max_new_tokens, temperature, top_p, None)
        if (isinstance(max_tool_rounds, bool)
                or not isinstance(max_tool_rounds, int) or max_tool_rounds < 0):
            raise ValueError("max_tool_rounds must be a non-negative integer")
        if response_template is None:
            response_template = getattr(self.tokenizer, "response_template", None)
        if response_template is None:
            raise ValueError("complete requires a response_template for this tokenizer")
        tools = list(tools)
        schemas = [get_json_schema(tool) for tool in tools]
        registry = {
            schema["function"]["name"]: tool
            for schema, tool in zip(schemas, tools, strict=True)
        }
        if len(registry) != len(tools):
            raise ValueError("Tool names must be unique")
        histories = [deepcopy(list(messages)) for messages in conversations]
        if any(not messages for messages in histories):
            raise ValueError("Conversations must contain at least one message")
        active = list(range(len(histories)))
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("complete requires a pad_token_id or eos_token_id")
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        if not eos_ids:
            raise ValueError("complete requires configured EOS token IDs")

        for round_index in range(max_tool_rounds + 1):
            if not active:
                break
            rows = [list(self.tokenizer.apply_chat_template(
                histories[index], tools=schemas, tokenize=True,
                add_generation_prompt=True, return_dict=False,
            )) for index in active]
            width = max(map(len, rows))
            if any(not row for row in rows):
                raise ValueError("Chat template produced an empty prompt")
            ids = torch.full((len(rows), width), pad, dtype=torch.long)
            masks = torch.zeros_like(ids)
            for row_index, row in enumerate(rows):
                ids[row_index, width - len(row):] = torch.tensor(row)
                masks[row_index, width - len(row):] = 1
            outputs, _ = self.generate(
                ids, masks, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
            parsed = []
            # Validate every turn before executing any tool in this round.
            for index, row, output in zip(active, rows, outputs, strict=True):
                tokens = output[3]
                if not tokens or tokens[-1] not in eos_ids:
                    histories[index].append({"role": "assistant", "content": output[0]})
                    continue
                try:
                    message = self.tokenizer.parse_response(
                        tokens, schema=response_template, prefix=row, tools=schemas,
                    )
                    if message.get("role") != "assistant":
                        raise ValueError("Response template must produce an assistant message")
                    calls = message.get("tool_calls", [])
                    if not isinstance(calls, list):
                        raise ValueError("Parsed tool_calls must be a list")
                    for call in calls:
                        function = call.get("function") if isinstance(call, dict) else None
                        if (not isinstance(function, dict)
                                or not isinstance(function.get("name"), str)
                                or not isinstance(function.get("arguments"), dict)):
                            raise ValueError("Parsed tool calls need a function name and argument dict")
                except ValueError:
                    histories[index] = None
                    continue
                if calls and round_index == max_tool_rounds:
                    histories[index] = None
                    continue
                parsed.append((index, message))
            continuing = []
            for index, message in parsed:
                histories[index].append(message)
                calls = message.get("tool_calls", [])
                if not calls:
                    continue
                for call in calls:
                    function = call["function"]
                    name, arguments = function["name"], function["arguments"]
                    try:
                        if name not in registry:
                            raise ValueError(f"Unknown tool: {name}")
                        tool = registry[name]
                        signature(tool).bind(**arguments)
                        content = str(tool(**arguments))
                    except Exception as error:
                        content = f"{type(error).__name__}: {error}"
                    result = {"role": "tool", "name": name, "content": content}
                    if "id" in call:
                        result["tool_call_id"] = call["id"]
                    histories[index].append(result)
                continuing.append(index)
            active = continuing
        return histories

    def generate(
        self,
        input_ids: torch.Tensor | list[int] | list[list[int]],
        attention_mask: torch.Tensor | list[int] | list[list[int]],
        *,
        past_key_values=None,
        max_new_tokens: int = 51,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Sequence[str] | None = None,
    ) -> GenerationOutput:
        """Generate from uncached token IDs and return their updated KV cache.

        With a cache, ``attention_mask`` covers all cached positions plus the new
        input IDs. Each generation row includes exact sampled token IDs; the final
        sampled token is not in the returned cache and starts the next input.
        """
        if self.model is None:
            raise RuntimeError("Engine is closed")

        stop = validate_generation_options(max_new_tokens, temperature, top_p, stop)
        device = next(self.model.parameters()).device
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
        attention_mask = torch.as_tensor(
            attention_mask, dtype=torch.long, device=device
        )
        if input_ids.numel() == 0 and attention_mask.numel() == 0:
            return [], past_key_values
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be 1D or 2D")
        if input_ids.shape[0] != attention_mask.shape[0]:
            raise ValueError("input_ids and attention_mask must have the same batch size")
        if past_key_values is None and input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids and attention_mask must have the same shape without a cache"
            )
        if past_key_values is not None:
            expected_mask_width = (
                past_key_values.get_seq_length() + input_ids.shape[1]
            )
            if attention_mask.shape[1] != expected_mask_width:
                raise ValueError(
                    "attention_mask must cover cached tokens and new input_ids"
                )
        if input_ids.shape[0] == 0:
            return [], past_key_values
        if input_ids.shape[1] == 0:
            return [], past_key_values
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": pad_token_id,
            "use_cache": True,
            "return_dict_in_generate": True,
        }
        if past_key_values is not None:
            generation_kwargs["past_key_values"] = past_key_values
        if temperature > 0:
            generation_kwargs.update(
                temperature=temperature,
                top_p=top_p,
                top_k=0,
            )

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                outputs = self.model.generate(**generation_kwargs)
        finally:
            self.model.train(was_training)

        if outputs.past_key_values is None:
            raise RuntimeError("Transformers did not return a generation cache")

        eos_token_id = self.model.generation_config.eos_token_id
        eos_token_ids = (
            {eos_token_id}
            if isinstance(eos_token_id, int)
            else set(eos_token_id or [])
        )
        prompt_width = input_ids.shape[1]
        results = []
        for mask, output in zip(attention_mask, outputs.sequences, strict=True):
            prompt_tokens = int(mask.sum().item())
            completion_ids = output[prompt_width:].tolist()
            for index, token_id in enumerate(completion_ids):
                if token_id in eos_token_ids:
                    completion_ids = completion_ids[: index + 1]
                    break
            text = self.tokenizer.decode(
                completion_ids, skip_special_tokens=True
            )
            results.append(
                (
                    trim_stop(text, stop),
                    prompt_tokens,
                    len(completion_ids),
                    completion_ids,
                )
            )
        return results, outputs.past_key_values

    def close(self):
        # Only drop our references; the trainer's model, optimizer and groups survive.
        self.model = None
        self.tokenizer = None
