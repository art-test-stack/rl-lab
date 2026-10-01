# Transformers tool completion

`TransformersEngine.complete()` accepts a batch of message histories and Python
callables. Transformers derives tool descriptions from their type hints and
Google-style docstrings. The engine generates an assistant turn for each active
chat, parses responses, executes requested tools by name, and generates another
round for chats that called tools. Finished chats leave the batch; returned
histories retain input order and include assistant messages and tool results.
Input histories are copied.

Run `uv run --locked python scripts/agent_python.py` for a small Qwen demo that
prints the requested Python code, execution result, and final answer. The first
run downloads the model; generation uses CUDA when available, otherwise CPU.

```python
from rl_lab.engines.transformers import TransformersEngine
from rl_lab.tools.exec import make_python_tool

engine = TransformersEngine.load("Qwen/Qwen2.5-1.5B-Instruct")

# Qwen2.5 has a tool-aware chat template, but no response_template. Supply
# this declarative schema to Transformers' native parse_response() method.
response_template = {
    "start_anchor": "<|im_start|>assistant\n",
    "defaults": {"role": "assistant", "content": ""},
    "fields": {
        "content": {"close": "<|im_end|>", "repeats": True, "join": ""},
        "tool_calls": {
            "open": "<tool_call>",
            "close": "</tool_call>",
            "content": "json",
            "repeats": True,
            "transform": {"type": "function", "function": "{content}"},
        },
    },
}

histories = engine.complete(
    [[{"role": "user", "content": "Use Python to compute sum(range(100))."}]],
    tools=[make_python_tool(timeout=5.0)],
    response_template=response_template,
    max_new_tokens=512,
    max_tool_rounds=4,
)
print(histories[0][-1]["content"])
engine.close()
```

For tokenizers with their own `response_template`, omit the explicit schema.
Other response formats require a matching schema; the model must have learned
tool use and its chat template must render tool descriptions, calls and results.
Schemas must produce assistant messages with `tool_calls` containing
`{"type": "function", "function": {"name": "...", "arguments": {...}}}`.

To add another tool, pass a callable such as:

```python
def multiply(x: int, y: int) -> int:
    """Multiply two integers.

    Args:
        x: First integer.
        y: Second integer.
    """
    return x * y
```

Tool names must be unique. Unknown names, argument binding errors and exceptions
from callables are returned as tool error strings so the model can respond.
Multiple tool calls in a turn run sequentially, as do calls across samples.
Generic callables own their execution limits; the engine supplies no universal
timeout. `make_python_tool()` exposes only `code` to the model and returns the
existing executor's structured result as JSON, including stdout, stderr, errors
and timeout status. Each invocation has a fresh Python namespace. Its existing
process restrictions are not a security sandbox, and memory limits are skipped
on macOS.

`max_new_tokens` applies to each assistant turn. `max_tool_rounds` bounds the
number of tool-bearing turns per chat, followed by one final assistant turn.
Exhausting that limit or generating a turn without EOS raises `RuntimeError`;
malformed responses raise parsing/validation errors. Every turn in a round is
parsed and checked before any tools execute in that round.

This first version re-encodes full histories each round and does not reuse KV
caches between rounds. Fast samples wait for slower samples at each round
boundary. `generate()` remains the existing token-level API. The task evaluator
continues to call `generate()`; tool completion is currently an explicit engine
API, and does not change evaluation scripts or their scoring behavior.
