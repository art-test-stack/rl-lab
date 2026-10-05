"""Tokenizer response templates."""

# Qwen2.5's tokenizer lacks a response template for native response parsing.
QWEN_RESPONSE_TEMPLATE = {
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
