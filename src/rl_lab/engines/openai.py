"""Client for an already-running OpenAI-compatible endpoint."""

from .base import (
    ChatPrompt,
    CompletionEngine,
    EngineCapabilities,
    Generation,
    text_prompt,
    trim_stop,
)


class OpenAIEngine(CompletionEngine):
    capabilities = EngineCapabilities(chat=True)

    def __init__(self, model: str, *, client=None, **client_kwargs):
        self.model = model
        self._owns_client = client is None
        if client is not None and client_kwargs:
            raise ValueError("Pass either a client or client connection options")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Install openai to use the remote engine") from exc
            client = OpenAI(**client_kwargs)
        self.client = client

    def generate(self, prompts, config):
        if self.client is None:
            raise RuntimeError("Engine is closed")
        results = []
        for prompt in prompts:
            kwargs = dict(
                model=self.model,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                n=1,
            )
            if config.stop:
                kwargs["stop"] = list(config.stop)
            if isinstance(prompt, ChatPrompt):
                response = self.client.chat.completions.create(
                    messages=list(prompt.messages), **kwargs
                )
                text = response.choices[0].message.content
                if text is None:
                    raise RuntimeError("Remote engine returned no text completion")
            else:
                response = self.client.completions.create(
                    prompt=text_prompt(prompt), **kwargs
                )
                text = response.choices[0].text
            usage = response.usage
            results.append(
                Generation(
                    trim_stop(text, config.stop),
                    usage.prompt_tokens if usage else None,
                    usage.completion_tokens if usage else None,
                )
            )
        return results

    def close(self):
        if self.client is not None and self._owns_client:
            self.client.close()
        self.client = None
