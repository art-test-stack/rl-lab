"""Show Qwen requesting Python execution, reading the result, and answering.

Run from the repository root: uv run --locked python scripts/agent_python.py
The first run downloads Qwen2.5-1.5B-Instruct.
"""

import json

import torch

from rl_lab.engines.transformers import TransformersEngine
from rl_lab.tools.exec import make_python_tool


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Qwen2.5's tokenizer lacks a response template for native response parsing.
RESPONSE_TEMPLATE = {
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [
        {"role": "system", "content": (
            "Use the python tool for calculations. Print the result in your code, "
            "then use the tool response to answer the user."
        )},
        {"role": "user", "content": (
            "Use Python to calculate the sum of the squares of the integers "
            "from 1 through 1000, inclusive."
        )},
    ]
    print(f"Loading {MODEL_ID} on {device}", flush=True)
    with TransformersEngine.load(MODEL_ID, device=device) as engine:
        histories = engine.complete(
            [messages],
            tools=[make_python_tool(timeout=5.0)],
            response_template=RESPONSE_TEMPLATE,
            max_new_tokens=512,
            max_tool_rounds=4,
        )

    print(f"\nUser: {messages[-1]['content']}")
    for message in histories[0][len(messages):]:
        if message.get("content"):
            print(f"\n{message['role'].capitalize()}: {message['content']}")
        for call in message.get("tool_calls", []):
            function = call["function"]
            print(f"\nTool call: {function['name']}")
            print(json.dumps(function["arguments"], indent=2))
    if not any(message["role"] == "tool" for message in histories[0]):
        print("\nThe model answered without calling a tool.")


if __name__ == "__main__":
    main()
