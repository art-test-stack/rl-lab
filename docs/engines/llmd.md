# llm-d

llm-d is a distributed inference serving stack. This adapter connects to an
existing gateway using its `/v1/completions` and `/v1/chat/completions` routes.
The model name must match a model served by that deployment. No Kubernetes
resources, servers, subprocesses, local models or tokenizers are created.

```bash
uv sync --locked --extra llmd
```

```python
import asyncio
from rl_lab.engines import ChatPrompt, GenerationConfig, create_async_engine

async def main():
    async with create_async_engine(
        "llmd",  # "llm-d" is also accepted
        model_name="your-served-model",
        base_url="https://your-gateway.example/v1",
        # api_key=...,  # optional Bearer token; omit for unauthenticated gateways
        max_concurrency=8,
        timeout=60.0,
    ) as engine:
        outputs = await engine.generate(
            [ChatPrompt([{"role": "user", "content": "What is 2 + 2?"}])],
            GenerationConfig(max_new_tokens=64),
        )
        print(outputs[0].text)

if __name__ == "__main__":
    asyncio.run(main())
```

In an existing event loop, call/await the coroutine directly. The factory is
synchronous and performs no network calls; inference and cleanup are async.
Concrete imports use `from rl_lab.engines.llmd import LLMDEngine`. The factory
lazily imports that module; importing the package root does not import HTTPX.
No OpenAI SDK or separate OpenAI engine is added.

`base_url` includes the API prefix (`/v1`, or a proxy prefix ending in `/v1`).
Text strings / `TextPrompt` use the completion route, and `ChatPrompt` uses the
chat route with server-side templating. Both routes must be enabled by the
operator for mixed batches. The adapter requests one non-streaming text result
per input and preserves input order even if HTTP calls finish out of order.
Tool-only/multimodal responses are unsupported and fail explicitly.

Requests share an instance-wide concurrency limit, including concurrent calls
to `generate`. A request failure or cancellation cancels and awaits its batch's
remaining client calls. It does not guarantee cancellation of remote GPU work.
HTTP errors and timeouts propagate; there are no automatic retries. Timeout is
an HTTPX timeout for each network phase, not an overall batch deadline.

A supplied `client=httpx.AsyncClient(...)` is borrowed: the caller controls its
transport/TLS/proxy setup and lifetime. The adapter still applies its request
timeout and optional Bearer header. Use the engine within one event loop and
await outstanding generation calls before closing it. Closing never shuts down
the remote deployment.

Token usage is copied when reported, otherwise `None`; local stop trimming does
not alter server-reported token counts. Continuation likelihood scoring is not
implemented and `require("loglikelihood")` fails explicitly. The synchronous
`CompletionEngine` / `evaluate_task` interface is not compatible with this async
adapter; a separate async runner is future work.

Sources: [llm-d HTTP API](https://github.com/llm-d/llm-d/blob/main/docs/api-reference/epp-http-apis.md),
[model-server architecture](https://llm-d.ai/docs/dev/architecture/core/model-servers).
Tests use HTTPX mock transports; no live cluster integration test was performed.
