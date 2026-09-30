"""Async client for an existing llm-d gateway's text/chat completion routes.

API: https://github.com/llm-d/llm-d/blob/main/docs/api-reference/epp-http-apis.md
"""

import asyncio
import math

import httpx

from .async_base import AsyncCompletionEngine
from .base import ChatPrompt, EngineCapabilities, Generation, text_prompt, trim_stop


def _token_count(usage, key):
    value = usage.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise RuntimeError(f"llm-d returned an invalid {key}")
    return value


class LLMDEngine(AsyncCompletionEngine):
    """Connect to llm-d; no model loading, server startup or cluster management.

    ``base_url`` is the API prefix, e.g. ``https://gateway.example/v1``.
    Use this engine in one event loop. Await outstanding calls before closing.
    A supplied AsyncClient is borrowed and is never closed by the adapter.
    """

    capabilities = EngineCapabilities(chat=True)

    def __init__(
        self, model: str, *, base_url: str, api_key: str | None = None,
        max_concurrency: int = 8, timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("llm-d requires a non-empty served model name")
        if not isinstance(base_url, str):
            raise ValueError("base_url must be an HTTP(S) API prefix")
        url = httpx.URL(base_url)
        if (url.scheme not in {"http", "https"} or not url.host or url.query or url.fragment
                or url.username or url.password):
            raise ValueError("base_url must be an HTTP(S) API prefix without credentials, query or fragment")
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("api_key must be a non-empty string or None")
        self.model = model
        self.base_url = str(url).rstrip("/")
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient()

    def _request(self, prompt, config):
        payload = {
            "model": self.model, "max_tokens": config.max_new_tokens,
            "temperature": config.temperature, "top_p": config.top_p,
            "n": 1, "stream": False,
        }
        if config.stop:
            payload["stop"] = list(config.stop)
        if isinstance(prompt, ChatPrompt):
            if not prompt.messages:
                raise ValueError("Chat prompts must contain at least one message")
            messages = []
            for message in prompt.messages:
                if message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str):
                    raise ValueError("llm-d chat prompts require text system/user/assistant messages")
                messages.append(dict(message))
            payload["messages"] = messages
            return "chat/completions", payload
        payload["prompt"] = text_prompt(prompt)
        if not isinstance(payload["prompt"], str):
            raise ValueError("Text prompts must contain a string")
        return "completions", payload

    async def generate(self, prompts, config):
        if self._client is None or self._client.is_closed:
            raise RuntimeError("Engine is closed")
        # Validate the full batch before sending anything to the gateway.
        requests = [self._request(prompt, config) for prompt in prompts]
        if not requests:
            return []
        client = self._client

        async def complete(endpoint, payload):
            async with self._semaphore:
                response = await client.post(
                    f"{self.base_url}/{endpoint}", json=payload,
                    headers=self._headers, timeout=self.timeout,
                )
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError as error:
                    raise RuntimeError("llm-d returned invalid JSON") from error
                if not isinstance(data, dict):
                    raise RuntimeError("llm-d returned an invalid response object")
                choices = data.get("choices")
                if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                    raise RuntimeError("llm-d must return exactly one completion per request")
                choice = choices[0]
                if endpoint == "chat/completions":
                    message = choice.get("message")
                    text = message.get("content") if isinstance(message, dict) else None
                else:
                    text = choice.get("text")
                if not isinstance(text, str):
                    raise RuntimeError("llm-d returned a non-text completion; tool-only responses are unsupported")
                usage = data.get("usage")
                if usage is None:
                    usage = {}
                if not isinstance(usage, dict):
                    raise RuntimeError("llm-d returned invalid usage metadata")
                return Generation(
                    text=trim_stop(text, config.stop),
                    prompt_tokens=_token_count(usage, "prompt_tokens"),
                    completion_tokens=_token_count(usage, "completion_tokens"),
                )

        tasks = [asyncio.create_task(complete(*request)) for request in requests]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            # A failed/cancelled batch must not leave sibling HTTP calls running.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def aclose(self):
        client, self._client = self._client, None
        if client is not None and self._owns_client:
            await client.aclose()
