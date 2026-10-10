"""Inference on an existing causal LM, or a separately loaded HF model."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema
from transformers.generation import ContinuousBatchingConfig

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
        return_token_counts: bool = False,
    ) -> list[list[dict]]:
        """Complete chats with Transformers' continuous batching manager.

        Return copied histories in input order. Tools run in worker threads
        across chats, sequentially within each chat; other chats keep generating.
        Each tool result is appended before submitting that chat's next turn.
        Callables must enforce their own timeouts and tolerate concurrent calls.
        Streaming pauses a chat at its first complete tool call while other chats
        continue generating; that chat resumes as a new request after the tool.

        Parsing uses the supplied response template or the tokenizer's own.
        Tool calls do not require EOS. Malformed responses and requests beyond
        max_tool_rounds return generated text without executing tools from that turn.
        Configuration and generation failures raise; unknown tools and callable
        exceptions become tool error responses. The model must support paged
        attention. max_new_tokens limits each assistant turn.
        return_token_counts adds completion_tokens to each new assistant message,
        counting returned generated IDs (including EOS), not prompt or tool output.
        Extra tokens generated after a tool boundary during cancellation are excluded.
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
        # Schemas describe tools to the model; the registry resolves generated names to callables.
        tools = list(tools)
        schemas = [get_json_schema(tool) for tool in tools]
        registry = {
            schema["function"]["name"]: tool
            for schema, tool in zip(schemas, tools, strict=True)
        }
        if len(registry) != len(tools):
            raise ValueError("Tool names must be unique")
        # Appending replies must not mutate the caller's histories.
        histories = [deepcopy(list(messages)) for messages in conversations]
        if any(not messages for messages in histories):
            raise ValueError("Conversations must contain at least one message")
        if not histories:
            return histories
        # These options apply to every assistant turn without changing the model's defaults.
        generation_config = deepcopy(self.model.generation_config)
        generation_config.update(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
        if temperature > 0:
            generation_config.update(temperature=temperature, top_p=top_p, top_k=0)

        def execute_tools(calls):
            # Preserve call order within a chat; the main loop alone updates histories.
            results = []
            for call in calls:
                function = call["function"]
                name, arguments = function["name"], function["arguments"]
                try:
                    if name not in registry:
                        raise ValueError(f"Unknown tool: {name}")
                    content = str(registry[name](**arguments))
                except Exception as error:
                    # Let the model see tool failures and decide how to respond.
                    content = f"{type(error).__name__}: {error}"
                result = {"role": "tool", "name": name, "content": content}
                if "id" in call:
                    result["tool_call_id"] = call["id"]
                results.append(result)
            return results

        ready = list(range(len(histories)))  # Chat indices needing a new generation request.
        pending = {}  # Request ID -> (chat index, streaming parser, decoded text already fed).
        running_tools = {}  # Chat index -> future producing that turn's tool messages.
        rounds = [0] * len(histories)
        was_training = self.model.training
        try:
            with ThreadPoolExecutor() as workers, self.model.continuous_batching_context_manager(
                generation_config=generation_config,
                continuous_batching_config=ContinuousBatchingConfig(
                    max_requests_per_batch=len(histories),
                ),
                warmup=False,
            ) as manager:
                while ready or pending or running_tools:
                    # Snapshot entries because completed futures are removed during iteration.
                    for index, future in list(running_tools.items()):
                        if future.done():
                            histories[index].extend(future.result())
                            del running_tools[index]
                            # Resume this chat only after its tool results are in the history.
                            ready.append(index)
                    for index in ready:
                        # Submit the updated history; the manager owns batching and cache handling.
                        prompt = list(self.tokenizer.apply_chat_template(
                            histories[index], tools=schemas, tokenize=True,
                            add_generation_prompt=True, return_dict=False,
                        ))
                        if not prompt:
                            raise ValueError("Chat template produced an empty prompt")
                        parser = self.tokenizer.get_response_parser(
                            response_template, prefix=prompt, tools=schemas,
                        )
                        request_id = manager.add_request(prompt, streaming=True)
                        if request_id is None:
                            raise RuntimeError("Continuous batching manager rejected a request")
                        pending[request_id] = (index, parser, "")
                    ready = []
                    if not pending and not running_tools:
                        break
                    # Poll briefly so finished tools can resume while other chats generate.
                    output = manager.get_result(timeout=0.05)
                    if output is None:
                        if not manager.is_running():
                            raise RuntimeError("Continuous batching manager stopped before completion")
                        continue
                    # Cancellation is asynchronous; ignore queued chunks from retired requests.
                    if output.request_id not in pending:
                        continue
                    if output.error is not None:
                        raise RuntimeError(output.error)
                    index, parser, previous_text = pending[output.request_id]
                    tokens = output.generated_tokens
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    finished = output.is_finished()
                    decoded = self.tokenizer.decode(tokens)
                    # Wait for a complete UTF-8 character before feeding the parser.
                    if not finished:
                        decoded = decoded.rstrip("\ufffd")
                    closed_call = False
                    if parser is not None:
                        try:
                            events = parser.feed(decoded[len(previous_text):])
                            closed_call = any(
                                event["type"] == "region_close" and event["field"] == "tool_calls"
                                for event in events
                            )
                        except ValueError:
                            parser = None
                    pending[output.request_id] = (index, parser, decoded)
                    if not finished and not closed_call:
                        continue
                    # Parse at a tool boundary or generation end; neither requires EOS.
                    try:
                        message = self.tokenizer.parse_response(
                            tokens, schema=response_template, prefix=output.prompt_ids, tools=schemas,
                        )
                        if message.get("role") != "assistant":
                            raise ValueError("Response template must produce an assistant message")
                        calls = message.get("tool_calls", [])
                        if not isinstance(calls, list):
                            raise ValueError("Parsed tool_calls must be a list")
                        # Stop at the first call, even if one chunk contains several calls.
                        if calls:
                            message["tool_calls"] = calls = calls[:1]
                        for call in calls:
                            function = call.get("function") if isinstance(call, dict) else None
                            if (not isinstance(function, dict)
                                    or not isinstance(function.get("name"), str)
                                    or not isinstance(function.get("arguments"), dict)):
                                raise ValueError("Parsed tool calls need a function name and argument dict")
                    except ValueError:
                        # Malformed structured output is still a usable text result.
                        message = {"role": "assistant", "content": text}
                    if not finished and not message.get("tool_calls"):
                        continue
                    if not finished:
                        manager.cancel_request(output.request_id)
                    del pending[output.request_id]
                    if message.get("tool_calls") and rounds[index] >= max_tool_rounds:
                        # Keep the request text, but do not execute another tool round.
                        message = {"role": "assistant", "content": text}
                    if return_token_counts:
                        message["completion_tokens"] = len(tokens)
                    histories[index].append(message)
                    calls = message.get("tool_calls", [])
                    if calls:
                        rounds[index] += 1
                        # This chat waits for its worker; other chats remain free to generate.
                        running_tools[index] = workers.submit(execute_tools, calls)
                    # Without tool calls, the chat is finished and is not resubmitted.
        finally:
            # Evaluation must not leave a borrowed training model in eval mode.
            self.model.train(was_training)
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
