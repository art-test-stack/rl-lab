# First SFT recipe

Run from the repository root:

```bash
uv run --locked --group train python scripts/train/sft.py \
    --train-limit 1024 --eval-limit 64 \
    --eval-tasks smoltalk,gsm8k
```

The default model is `openai-community/gpt2`. The recipe uses the existing
`SmolTalk` adapter and its deterministic dataset order, selecting source
`smol-magpie-ultra-short` independently from the train and test splits.
Every training conversation is expanded into independent user/assistant exchanges:
a three-exchange conversation supplies three training examples. Earlier exchanges
are not included as context, so follow-up questions may lose useful information.
Empty and oversized training exchanges are excluded. `--train-limit` counts
accepted exchanges after expansion and length filtering. No task classification
is done.

Training artifacts default to `CACHE_DIR/models/openai-community--gpt2/sft/default`.
`CACHE_DIR` is resolved from `RLLAB_CACHE_DIR`, defaulting to `~/.cache/rl_lab`.
Hub model names replace `/` with `--` in the directory name; local model paths
use their directory basename. `--run-name run-2` selects `sft/run-2`, keeping
experiments with the same model and method separate. Run names must start with
a letter or digit and contain only letters, digits, `.`, `_`, or `-`.
Evaluate that run with `--model 'After SFT=openai-community/gpt2:sft:run-2'`.
`--output-dir` overrides the default path. Existing
populated directories are rejected; choose another output path for another run.
Comparison and standalone evaluation reports go under `CACHE_DIR/reports`.

Validation conversations remain whole: all earlier messages form the prompt,
and the final assistant reply is the answer-only loss target. The script warns
at startup that only complete conversations fitting the selected model's context
are evaluated. When sample generation is requested, its prompt plus the generation
budget must also fit. Oversized conversations are skipped, and the script reports
how many it skipped while selecting the requested subset. `--eval-limit` counts
accepted whole conversations. The underlying test split is not modified; metrics
describe the selected subset rather than the full split. Windowed evaluation of
longer conversations is deferred.

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
training without gradient checkpointing. `--max-length` limits training exchanges;
validation uses the chosen model's context window, not the training limit. TRL
truncation is disabled because exports are already bounded and validation must
retain its complete context. Use `--use-cpu` for a CPU run. GPU
training can enable `--bf16 true` where supported. Generation defaults to 128
new tokens. Training lengths and validation/generation inputs must fit the model context.
Use a new, empty output directory for each run.

To test separate AdamW hyperparameters for named parameter groups, add
`--optimizer-config configs/optim/gpt2-adamw.yaml --run-name grouped-adamw-test`.
The configuration path accepts YAML or JSON. Resolved group settings are printed
and saved in `recipe.json`; explicit group values override Trainer defaults.
See [optimizer groups](optimizers.md) for the file format and current limits.

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

Sample replies in `evaluation.json` use `prompt`, `reference` (the dataset answer),
and `generated` (the model reply). Dataset exports retain TRL's `completion` field.

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
The manifest records the training length limit and model-dependent validation
context limit separately.

## Training plots

After a run completes, regenerate its figures with:

```bash
cache_dir="${RLLAB_CACHE_DIR:-$HOME/.cache/rl_lab}"
uv run --locked --with matplotlib python scripts/plot_training.py \
    "$cache_dir/models/openai-community--gpt2/sft/default"
```

Replace the directory with any completed SFT run. The script reads its saved JSON
logs and writes `training-overview` and `training-diagnostics` in PNG and PDF
formats under that run's `plots/` directory. Re-running replaces those figures.
Step axes adapt to the logged run; missing optional metrics are labeled
"Not logged". Matplotlib is provided temporarily by `uv --with`, without changing
the project's dependency files.

## Offline compatibility check

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
uv run --locked --group train pytest tests/train/test_sft.py -q
```

This runs actual TRL optimization on the existing tiny local GPT-2 fixture. Only
Hub dataset loading is replaced with local rows. It checks filtering, native
prompt/padding masks, supervised EOS, expansion of all three training exchanges,
whole validation inputs longer than the training limit, exclusion of oversized
validation conversations, a weight update, checkpoint reload, and both SmolTalk
samples and an existing task scorer.
