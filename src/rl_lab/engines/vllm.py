"""Optional offline vLLM backend. No HTTP server or port management."""

from .base import (
    ChatPrompt,
    CompletionEngine,
    EngineCapabilities,
    Generation,
    text_prompt,
    trim_stop,
)


class VLLMEngine(CompletionEngine):
    capabilities = EngineCapabilities(chat=True)

    def __init__(self, model: str, **kwargs):
        if not isinstance(model, str):
            raise TypeError(
                "vLLM needs a model name/path; use TransformersEngine.attach for live training weights"
            )
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "Install vllm in a compatible inference environment to use this backend"
            ) from exc
        self.llm = LLM(model=model, **kwargs)

    def generate(self, prompts, config):
        from vllm import SamplingParams

        if self.llm is None:
            raise RuntimeError("Engine is closed")
        if not prompts:
            return []
        # Supply token IDs so chat templates do not get special tokens added twice.
        tokenizer = self.llm.get_tokenizer()
        inputs = []
        for prompt in prompts:
            ids = (
                tokenizer.apply_chat_template(
                    list(prompt.messages),
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                if isinstance(prompt, ChatPrompt)
                else tokenizer.encode(text_prompt(prompt), add_special_tokens=True)
            )
            if not ids:
                if tokenizer.bos_token_id is None:
                    raise ValueError("Empty prompts require a BOS token")
                ids = [tokenizer.bos_token_id]
            inputs.append({"prompt_token_ids": ids})
        outputs = self.llm.generate(
            inputs,
            SamplingParams(
                n=1,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=list(config.stop) if config.stop else None,
                include_stop_str_in_output=False,
            ),
            use_tqdm=False,
        )
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM returned an unexpected number of results")
        return [
            Generation(
                trim_stop(output.outputs[0].text, config.stop),
                len(output.prompt_token_ids),
                len(output.outputs[0].token_ids),
            )
            for output in outputs
        ]

    def close(self):
        # vLLM owns worker cleanup through its runtime lifetime. Do not call private
        # shutdown methods or destroy global torch.distributed process groups.
        self.llm = None
