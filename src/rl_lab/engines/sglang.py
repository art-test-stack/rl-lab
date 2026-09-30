"""Optional offline SGLang engine for HF-compatible text generation.

API: https://docs.sglang.io/docs/basic_usage/offline_engine_api
"""

from sglang import Engine
from transformers import AutoTokenizer

from .base import (
    ChatPrompt,
    CompletionEngine,
    EngineCapabilities,
    Generation,
    text_prompt,
    trim_stop,
)


class SGLangEngine(CompletionEngine):
    """Own an offline SGLang runtime; use a dedicated process/job for its workers."""

    capabilities = EngineCapabilities(chat=True)

    def __init__(
        self, model: str, *, tensor_parallel_size: int = 1,
        tokenizer_path: str | None = None, tokenizer_mode: str = "auto",
        revision: str | None = None, trust_remote_code: bool = False, **kwargs,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("SGLang needs a model name/path, not live training weights")
        if isinstance(tensor_parallel_size, bool) or not isinstance(tensor_parallel_size, int) or tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be a positive integer")
        if "tp_size" in kwargs:
            raise ValueError("Use tensor_parallel_size instead of tp_size")
        if tokenizer_mode not in {"auto", "slow"} or "tokenizer_backend" in kwargs:
            raise ValueError("This adapter supports HF tokenizers in auto or slow mode")
        if kwargs.get("skip_tokenizer_init", False):
            raise ValueError("Text generation requires SGLang tokenizer initialization")
        # Load only a tokenizer here, never an additional model. Match the runtime's
        # tokenizer path/revision/mode without depending on private manager objects.
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model, revision=revision,
            trust_remote_code=trust_remote_code, use_fast=tokenizer_mode != "slow",
        )
        self.llm = Engine(
            model_path=model, tp_size=tensor_parallel_size,
            tokenizer_path=tokenizer_path or model, tokenizer_mode=tokenizer_mode,
            revision=revision, trust_remote_code=trust_remote_code, **kwargs,
        )

    def generate(self, prompts, config):
        if self.llm is None:
            raise RuntimeError("Engine is closed")
        if not prompts:
            return []
        inputs = []
        for prompt in prompts:
            if isinstance(prompt, ChatPrompt):
                if not prompt.messages or any(
                    message.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str)
                    for message in prompt.messages
                ):
                    raise ValueError("Chat prompts require text system/user/assistant messages")
                ids = self.tokenizer.apply_chat_template(
                    list(prompt.messages), tokenize=True,
                    add_generation_prompt=True, return_dict=False,
                )
            else:
                ids = self.tokenizer.encode(text_prompt(prompt), add_special_tokens=True)
            if not ids:
                if self.tokenizer.bos_token_id is None:
                    raise ValueError("Empty prompts require a BOS token")
                ids = [self.tokenizer.bos_token_id]
            inputs.append(ids)
        outputs = self.llm.generate(
            input_ids=inputs,
            sampling_params={
                "n": 1, "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature, "top_p": config.top_p,
                "top_k": -1, "stop": list(config.stop) if config.stop else None,
                "skip_special_tokens": True, "no_stop_trim": False,
            },
            stream=False,
        )
        if isinstance(outputs, dict) and len(prompts) == 1:
            outputs = [outputs]
        if not isinstance(outputs, list) or len(outputs) != len(prompts):
            raise RuntimeError("SGLang returned an unexpected number of results")
        results = []
        for output, ids in zip(outputs, inputs, strict=True):
            if not isinstance(output, dict) or not isinstance(output.get("text"), str):
                raise RuntimeError("SGLang returned a non-text result")
            metadata = output.get("meta_info", {})
            if not isinstance(metadata, dict):
                raise RuntimeError("SGLang returned invalid token metadata")
            prompt_tokens = metadata.get("prompt_tokens", len(ids))
            completion_tokens = metadata.get("completion_tokens")
            for count in (prompt_tokens, completion_tokens):
                if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
                    raise RuntimeError("SGLang returned an invalid token count")
            results.append(Generation(
                text=trim_stop(output["text"], config.stop),
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            ))
        return results

    def close(self):
        # Use the runtime's public lifecycle method, not private worker internals.
        runtime, self.llm = self.llm, None
        self.tokenizer = None
        if runtime is not None:
            runtime.shutdown()
