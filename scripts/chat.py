"""Send one terminal prompt to a Qwen-compatible model with the Python tool.

Run: uv run --locked python scripts/chat.py
Repeat the prompt in one continuous batch with --num-samples 4.
Use --model for a Hub model ID or local checkpoint. The default model downloads
on first use. Replies and tool activity appear after each complete() call returns.
"""

import argparse
import json

import torch

from rl_lab.engines.transformers import TransformersEngine
from rl_lab.tokenizer import QWEN_RESPONSE_TEMPLATE
from rl_lab.tools.exec import make_python_tool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of independently sampled copies of the same prompt')
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error('--num-samples must be positive')
    device = args.device or (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    try:
        text = input('You: ').strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not text:
        return
    messages = [{'role': 'system', 'content': (
        'You are a helpful assistant. Use the python tool when useful for calculations. '
        'Print results in your code, then use the tool response to answer the user.'
    )}, {'role': 'user', 'content': text}]
    tools = [make_python_tool(timeout=5.0)]
    print(f'Loading {args.model} on {device}', flush=True)
    with TransformersEngine.load(args.model, device=device, dtype='auto') as engine:
        template = getattr(engine.tokenizer, 'response_template', None) or QWEN_RESPONSE_TEMPLATE
        histories = engine.complete(
            [messages] * args.num_samples, tools=tools, response_template=template,
            max_new_tokens=512, max_tool_rounds=4,
            temperature=1.0, top_p=0.9,
            return_token_counts=True,
        )
    for index, history in enumerate(histories, 1):
        if args.num_samples > 1:
            print(f'\n--- Sample {index} ---')
        for message in history[len(messages):]:
            if message.get('content'):
                print(f"\n{message['role'].capitalize()}: {message['content']}")
            for call in message.get('tool_calls', []):
                function = call['function']
                print(f"\nTool call: {function['name']}")
                print(json.dumps(function['arguments'], indent=2))
        total_tokens = sum(message.get('completion_tokens', 0) for message in history[len(messages):])
        print(f"\nTokens: final answer {history[-1]['completion_tokens']} | total assistant generation {total_tokens}")


if __name__ == '__main__':
    main()
