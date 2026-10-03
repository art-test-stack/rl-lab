# First SFT recipe

Run from the repository root:

```bash
uv run --locked --group train python scripts/train/sft.py \
    --output-dir report/sft-gpt2 \
    --train-limit 1024 --eval-limit 64 \
    --eval-tasks smoltalk,gsm8k
```

The default model is `openai-community/gpt2`. The recipe uses the existing
`SmolTalk` adapter and its deterministic dataset order, selecting source
`smol-magpie-ultra-short` independently from the train and test splits. Only
exactly one user message followed by one assistant message is accepted.
System-prefixed, multi-turn, empty, and oversized examples are excluded.
Limits apply after filtering. No turn splitting or task classification is done.

The tokenizer receives a plain-text chat template:

```text
User: Hello!
Assistant: Hello! How can I help you?<|endoftext|>
```

TRL receives separate prompt and completion strings. Native completion-only loss
masks prompt and padding labels; the assistant response and EOS remain supervised.
Padding uses GPT-2's EOS token. Packing is disabled. The initial recipe uses
full-model training, standard next-token loss, and single-process execution.

The script accepts native `SFTConfig` arguments, including `--learning-rate`,
`--num-train-epochs`, `--max-steps`, `--gradient-accumulation-steps`, and `--seed`.
Defaults are one epoch, batch size 2, maximum sequence length 512, and float32
training without gradient checkpointing. Use `--use-cpu` for a CPU run. GPU
training can enable `--bf16 true` where supported. Generation defaults to 128
new tokens. The selected length and generation budget must fit the model context.
Use a new, empty output directory for each run.

## Evaluation and artifacts

`--eval-tasks` accepts a comma-separated list of `smoltalk`, `gsm8k`, `gpqa`,
`spellingbee`, and `simple_spelling`. Existing task dataset access requirements
still apply. ARC/MMLU candidate scoring, CORE, and agentic environments use
different evaluation protocols and are not included in this recipe.

Held-out SmolTalk answer loss is measured before and after training. It has no
correctness scorer: selecting `smoltalk` also records up to three fixed test
prompts, references, and greedy replies. Other tasks use `rl_lab.eval.evaluate_task`
and their existing scorers, with one greedy completion per example and no extra
system prompt. They measure separate capabilities, not conversational quality.
Oversized benchmark prompts raise an error rather than being truncated.

After training, generation evaluations reload the model and tokenizer from the
saved output directory. SmolTalk loss uses the trainer's trained weights.

The output directory contains:

- Model weights, tokenizer, and chat template for inference or task-evaluation scripts.
- TRL's epoch checkpoint with optimizer/scheduler state and `trainer_state.json`.
- `train.jsonl` and `smoltalk-test.jsonl`: the exact formatted examples used.
- `recipe.json`: recipe/training settings, example counts, and dataset fingerprints.
- `evaluation.json`: before/after losses, task scores, and sample replies.

The lockfile fixes software versions, the seed fixes training randomness, and
the exports preserve the selected examples. Dataset/model Hub revisions are not
pinned by this first recipe; hardware differences can also change results.

## Offline compatibility check

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
uv run --locked --group train pytest tests/train/test_sft.py -q
```

This runs actual TRL optimization on the existing tiny local GPT-2 fixture. Only
Hub dataset loading is replaced with local rows. It checks filtering, native
prompt/padding masks, supervised EOS, a weight update, checkpoint reload, and
both SmolTalk samples and an existing task scorer.
