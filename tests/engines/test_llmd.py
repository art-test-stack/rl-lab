import asyncio
import json
import os
import subprocess
import sys

import httpx
import pytest

from rl_lab.engines import ChatPrompt, GenerationConfig, TextPrompt, create_async_engine
from rl_lab.engines.llmd import LLMDEngine


def test_routes_payload_order_auth_usage_and_borrowed_client():
    async def run():
        seen = []
        async def handle(request):
            body = json.loads(request.content)
            seen.append((request, body))
            if "messages" in body:
                return httpx.Response(200, json={"choices": [{"message": {"content": "chat END rest"}}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4}})
            await asyncio.sleep(0.01)
            return httpx.Response(200, json={"choices": [{"text": body["prompt"] + " END rest"}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            engine = create_async_engine("llmd", model_name="served", base_url="https://gateway.test/proxy/v1/",
                                         api_key="test-token", client=client, timeout=17)
            prompt = ChatPrompt([{"role": "user", "content": "question"}])
            config = GenerationConfig(max_new_tokens=7, temperature=0.4, top_p=0.9, stop=["END"])
            async with engine:
                out = await engine.generate([TextPrompt("raw"), prompt, "other"], config)
                assert [o.text for o in out] == ["raw ", "chat ", "other "]
                assert out[0].prompt_tokens is None
                assert (out[1].prompt_tokens, out[1].completion_tokens) == (8, 4)
            assert not client.is_closed
            assert prompt.messages == [{"role": "user", "content": "question"}]
            assert [r.url.path for r, _ in seen] == ["/proxy/v1/completions", "/proxy/v1/chat/completions", "/proxy/v1/completions"]
            for request, body in seen:
                assert request.headers["authorization"] == "Bearer test-token"
                assert request.extensions["timeout"]["read"] == 17
                assert body["model"] == "served"
                assert body["n"] == 1 and body["stream"] is False
                assert body["max_tokens"] == 7 and body["temperature"] == 0.4
                assert body["top_p"] == 0.9 and body["stop"] == ["END"]
            with pytest.raises(RuntimeError, match="closed"):
                await engine.generate([], config)
            await engine.aclose()
    asyncio.run(run())


def test_instance_concurrency_bound_across_batches():
    async def run():
        active = peak = 0
        async def handle(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return httpx.Response(200, json={"choices": [{"text": "ok"}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            async with LLMDEngine("model", base_url="https://gateway.test/v1", client=client, max_concurrency=2) as engine:
                await asyncio.gather(*(engine.generate(["a", "b", "c"], GenerationConfig()) for _ in range(3)))
                assert peak == 2 and active == 0
    asyncio.run(run())


def test_failed_request_cancels_and_awaits_siblings():
    async def run():
        running = asyncio.Event()
        cancelled = asyncio.Event()
        async def handle(request):
            if json.loads(request.content)["prompt"] == "fail":
                await running.wait()
                return httpx.Response(503, json={"error": "unavailable"})
            running.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            engine = LLMDEngine("model", base_url="https://gateway.test/v1", client=client)
            with pytest.raises(httpx.HTTPStatusError):
                await asyncio.wait_for(engine.generate(["fail", "wait"], GenerationConfig()), timeout=2)
            assert cancelled.is_set()
    asyncio.run(run())


def test_caller_cancellation_cleans_up_requests():
    async def run():
        started = asyncio.Event()
        finished = asyncio.Event()
        async def handle(request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            engine = LLMDEngine("model", base_url="https://gateway.test/v1", client=client)
            batch = asyncio.create_task(engine.generate(["wait"], GenerationConfig()))
            await asyncio.wait_for(started.wait(), timeout=2)
            batch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await batch
            assert finished.is_set()
    asyncio.run(run())


@pytest.mark.parametrize("payload", [None, [], {}, {"choices": []}, {"choices": [{"text": "a"}, {"text": "b"}]},
    {"choices": [{"text": None}]}, {"choices": [{"text": "a"}], "usage": "bad"},
    {"choices": [{"text": "a"}], "usage": {"prompt_tokens": -1}}])
def test_malformed_response(payload):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))) as client:
            engine = LLMDEngine("model", base_url="https://gateway.test/v1", client=client)
            with pytest.raises(RuntimeError):
                await engine.generate(["q"], GenerationConfig())
    asyncio.run(run())


def test_non_text_chat_response_rejected():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200,
                json={"choices": [{"message": {"content": None, "tool_calls": []}}]}))) as client:
            engine = LLMDEngine("model", base_url="https://gateway.test/v1", client=client)
            with pytest.raises(RuntimeError, match="non-text"):
                await engine.generate([ChatPrompt([{"role": "user", "content": "q"}])], GenerationConfig())
    asyncio.run(run())


def test_validation_before_network_empty_batch_and_unsupported_likelihood():
    async def run():
        def unexpected(request):
            raise AssertionError("No network expected")
        async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
            engine = LLMDEngine("model", base_url="https://gateway.test/v1", client=client)
            assert await engine.generate([], GenerationConfig()) == []
            with pytest.raises(ValueError, match="text"):
                await engine.generate(["valid", ChatPrompt([{"role": "user", "content": []}])], GenerationConfig())
            with pytest.raises(ValueError, match="loglikelihood"):
                engine.require("loglikelihood")
            with pytest.raises(NotImplementedError):
                await engine.loglikelihood([])
    asyncio.run(run())


def test_timeout_propagates_without_retries():
    async def run():
        calls = 0
        def handle(request):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("timeout", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            engine = LLMDEngine("model", base_url="https://gateway.test/v1", client=client)
            with pytest.raises(httpx.ReadTimeout):
                await engine.generate(["q"], GenerationConfig())
            assert calls == 1
    asyncio.run(run())


def test_owned_client_closed_on_context_error_and_alias(monkeypatch):
    async def run():
        owned = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        monkeypatch.setattr("rl_lab.engines.llmd.httpx.AsyncClient", lambda: owned)
        engine = create_async_engine("llm-d", model_name="model", base_url="http://localhost/v1")
        client = engine._client
        with pytest.raises(RuntimeError, match="user error"):
            async with engine:
                raise RuntimeError("user error")
        assert client.is_closed
        await engine.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("kwargs", [{"model": ""}, {"base_url": "ftp://host/v1"},
    {"base_url": "relative"}, {"base_url": "https://user:secret@host/v1"},
    {"base_url": "https://host/v1?secret=x"}, {"max_concurrency": 0}, {"max_concurrency": True},
    {"timeout": float("inf")}, {"timeout": 0}, {"api_key": ""}])
def test_invalid_construction(kwargs):
    with pytest.raises(ValueError):
        LLMDEngine(**({"model": "model", "base_url": "https://gateway.test/v1"} | kwargs))


def test_factory_and_optional_import_isolation():
    with pytest.raises(ValueError, match="Unknown async"):
        create_async_engine("unknown", model_name="m", base_url="https://gateway.test/v1")
    subprocess.run([sys.executable, "-c", """
import sys
from rl_lab.engines import create_async_engine
assert not ({'httpx', 'torch', 'transformers', 'vllm', 'sglang', 'rl_lab.engines.llmd'} & sys.modules.keys())
sys.modules['httpx'] = None
try:
    create_async_engine('llmd', model_name='model', base_url='https://example.test/v1')
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('Selecting llmd must require httpx')
"""], check=True, env=os.environ | {"PYTHONPATH": "src"})
