# Mixed spelling SFT

`scripts/train/sft_simple_spelling.py` starts from Qwen2.5-0.5B-Instruct and
mixes equal numbers of SimpleSpelling demonstrations and short SmolTalk
exchanges. The default pilot uses 128 examples from each source for one epoch.

The system instruction is generic: be helpful and use tools when useful.
There is no worked example in the input. A spelling demonstration teaches the
assistant to produce a task-specific instruction, call Python, and return its
output in `word:comma-separated-characters` form. The instruction preserves the
supplied word rather than correcting its spelling.

The native Qwen training template applies loss to assistant tokens, including
the instruction, tool call, final answer, and turn-ending tokens. User messages
and Python results are context only. General exchanges use the same system
instruction and tool schema, but retain their original assistant responses.
Conversations exceeding the length limit are never truncated: oversized general
exchanges are skipped, and oversized spelling demonstrations raise an error.

Run:

```sh
HF_DEACTIVATE_ASYNC_LOAD=1 uv run --locked --group train \
  python scripts/train/sft_simple_spelling.py \
  --run-name spelling-mixed-sft \
  --model-init-kwargs '{"dtype":"bfloat16"}'
```

Defaults use one sequence per microbatch, eight microbatches per update,
gradient checkpointing, and gradient clipping at 1.0. The command above loads
bfloat16 model weights to reduce MPS memory use; omitting `--model-init-kwargs`
uses the trainer's default dtype (float32 in the first local run). This changes
update precision as well as memory use; it does not enable Accelerate AMP.
The default optimizer is PyTorch AdamW at `--learning-rate 0.00001`.
Qwen's tied embedding/head weight stays tied.

The first float32 Muon pilot stalled in MPS synchronization after update 1,
with a sampled physical footprint peaking at 16.1 GiB on a 16 GiB laptop.
The bfloat16 Muon restart failed to compile a generated Metal `abs` operation
before completing an update. AdamW was selected for the SFT pilot; neither
failure established learning behavior. The shared Muon implementation was not
changed. An optional `--optimizer-config` can select existing grouped recipes;
their base rates receive stage and AdamW width scaling, as in the GRPO recipe.

The output includes the mixed training data, separate held-out spelling and
general conversations, recipe settings, before/after losses for both sources,
and the final checkpoint. Training refuses nonempty output directories.
Generate answers in a fresh evaluation process after training; the recipe
does not load a second model into the training process.

Use the existing comparison script with the saved checkpoint and original
Instruct weights, matching prompts, tools, sample limits, and token budgets.
Lower held-out loss does not by itself establish better task accuracy or
preservation of general abilities. The 50/50 mix is a small pilot, not a
guarantee against overfitting. No automated tests were added at the user's request.

## Local pilot and held-out generation

The AdamW/bfloat16 pilot completed one epoch (32 updates) with finite logged
losses and gradient norms. Held-out spelling loss fell from 1.792 to 0.063;
held-out general loss fell from 1.320 to 1.276. The 16 spelling evaluation
words do not overlap the 128 spelling training words.

On 2026-10-11, original Qwen2.5-0.5B-Instruct and the saved SFT checkpoint
were evaluated sequentially in a fresh process on identical examples and
input messages. Each task used 16 examples, seed 42, greedy generation,
batch size 1, 256 new tokens per turn, and one Python-tool round for generative
tasks. The spelling system prompt was:

> Reply only in the format word:w,o,r,d, spelling the requested word.
> You may use the python tool for calculations. Print the result in your code,
> then use the tool response to answer the user.

There was no worked example. This comparison prompt differs from the generic
system prompt used in SFT; generation under the exact SFT prompt has not yet
been checked.

| Task | Original Instruct | Mixed SFT |
| --- | --- | --- |
| SimpleSpelling | 0/16 | 0/16 |
| SpellingBee counting | 0/16 | 0/16 |
| GSM8K | 0/16 | 0/16 |
| ARC-Easy | 8/16 | 9/16 |
| MMLU | 2/16 | 3/16 |

Neither model called Python on SimpleSpelling. The checkpoint did not produce
the demonstrated strategy/tool/answer sequence. For example, for
`Spell the word: garishly`, original Qwen answered that the word was spelled
correctly; the SFT checkpoint also checked its spelling and added a definition.
For `agrobiologist`, the checkpoint emitted plain-text code rather than a
native tool call or the required answer.

Tool calling itself worked on other tasks. In counting, original Qwen made
13 calls (7 successful executions), while SFT made 8 (7 successful executions).
Only 3 original and 2 SFT examples had tool stdout exactly matching the
reference count. Neither model emitted the required `####` marker on counting
or GSM8K, so the strict scores include answer-format failures. Successful
execution alone does not establish a correct calculation.

The one-answer gains on ARC-Easy and MMLU do not establish preservation of
general abilities with such small samples. Lower SFT loss did not translate
into the desired spelling behavior under this comparison prompt. GRPO was
not started from this checkpoint; checking generation with the exact SFT
prompt is the next diagnostic before changing training.

The checkpoint is at `.tmp/codex/qwen-mixed-sft/run-adamw-bf16/`. Scores,
full conversation histories, plots, and `behavior-summary.json` are under
`.tmp/codex/qwen-mixed-sft/heldout-comparison/results/`.

## One GRPO update from the SFT checkpoint

On 2026-10-11, a single GRPO optimizer step was run from the SFT checkpoint
using a brief task instruction without a worked example or supplied Python
code. AdamW at `1e-5` and bfloat16 weights avoided the compiled Muon path.
Four prompts with four completions each were accumulated at training
microbatch size 1; generation processed four completions at a time.
Gradient clipping was 1.0, the completion budget was 256 tokens, and the
existing exact-match reward, DAPO loss, and zero KL coefficient were retained.

The system instruction was:

> Extract every character of the supplied word without correcting it.
> The final answer must be the original word, a colon, and its comma-separated
> characters. Use Python if useful.

The run completed, but every sampled reward was zero. `reward_std=0`,
`frac_reward_zero_std=1`, `grad_norm=0`, and tool-call frequency was zero.
Greedy held-out scores were 0/16 before and after, with identical answers and
no Python calls. All 290 saved weight tensors were finite and exactly equal
to the SFT checkpoint: the optimizer step produced no parameter update.

TRL retained the final group's four completions in its Parquet log. For
`pinewood`, these included `pninewood` and
`pinewood, :pinewood,colon, pinewood`. The aggregate logs cover all 16
completions; the saved completion sample covers only the final four.
The logged mean completion length was 10 tokens, and the clipped ratio was
zero. This pilot was blocked by absent reward variation, not long generation
or the bfloat16 Muon compiler failure.

The recipe now accepts `--system-prompt` and `--use-worked-example false`;
its existing prompt defaults remain unchanged. An empty `--optimizer-config`
uses the optimizer selected by TRL instead of the grouped recipe. To repeat
the pilot, choose a fresh output directory:

```sh
HF_DEACTIVATE_ASYNC_LOAD=1 uv run --locked --group train \
  python scripts/train/grpo_simple_spelling.py \
  --model-name-or-path .tmp/codex/qwen-mixed-sft/run-adamw-bf16 \
  --model-init-kwargs '{"dtype":"bfloat16"}' \
  --optimizer-config '' \
  --system-prompt 'Extract every character of the supplied word without correcting it. The final answer must be the original word, a colon, and its comma-separated characters. Use Python if useful.' \
  --use-worked-example false \
  --train-limit 16 --eval-limit 16 --max-steps 1 \
  --learning-rate 0.00001 --max-grad-norm 1 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 16 \
  --generation-batch-size 4 --num-generations 4 \
  --max-completion-length 256 --log-completions true \
  --run-name spelling-mixed-sft-grpo-brief-repeat \
  --output-dir .tmp/codex/qwen-mixed-sft/grpo-brief-repeat
```

Artifacts are under `.tmp/codex/qwen-mixed-sft/grpo-brief-one-update/`,
including the training log, before/after conversation histories, reward logs,
checkpoint, and `weight-comparison.json`. Before another exact-match GRPO
run, sampling must establish groups with both correct and incorrect answers.
