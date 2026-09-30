"""Small, deterministic CPU fixtures: no Hub downloads or GPU required."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


@pytest.fixture
def encode_inputs():
    def encode(tokenizer, *prompts):
        from rl_lab.engines import ChatPrompt, TextPrompt

        rows = []
        for prompt in prompts:
            if isinstance(prompt, ChatPrompt):
                ids = tokenizer.apply_chat_template(
                    list(prompt.messages), tokenize=True,
                    add_generation_prompt=True, return_dict=False,
                )
            else:
                ids = tokenizer.encode(
                    prompt.text if isinstance(prompt, TextPrompt) else prompt,
                    add_special_tokens=True,
                )
            rows.append(list(ids))
        width = max(map(len, rows), default=0)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, ids in enumerate(rows):
            input_ids[i, width - len(ids):] = torch.tensor(ids)
            attention_mask[i, width - len(ids):] = 1
        return input_ids, attention_mask

    return encode


@pytest.fixture
def tokenizer():
    backend = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "hello": 4,
                "world": 5,
                "answer": 6,
                "A": 7,
                "B": 8,
                "C": 9,
                "D": 10,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = "{{ bos_token }}{% for m in messages %}{{ m['content'] }} {% endfor %}{% if add_generation_prompt %}answer {% endif %}"
    return tokenizer


@pytest.fixture
def model(tokenizer):
    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
