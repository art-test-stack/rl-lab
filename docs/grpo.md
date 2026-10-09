# Single-device GRPO on SimpleSpelling with Python tools

For the experiment history, failures and observed results, see the
[step-by-step SimpleSpelling report](simple-spelling-experiment.md).

Run from the repository root:

```bash
HF_DEACTIVATE_ASYNC_LOAD=1 uv run --locked --group train \
    python scripts/train/grpo_simple_spelling.py \
    --train-limit 128 --eval-limit 32 --learning-rate 0.00002 \
    --run-name spelling-python-rl-only
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct`. The recipe uses TRL's
`GRPOTrainer`, full-model Muon/AdamW training, and ordinary Transformers sampling on
the training device. It requires one process and one device; CPU, CUDA, or MPS
selection follows TRL/Transformers. Add `--use-cpu` for CPU. No vLLM, distributed
training, or new dependencies are needed. Model weights and the existing task's
English word list must be downloaded or cached before an offline run. On the
tested macOS setup, prefix the command with `HF_DEACTIVATE_ASYNC_LOAD=1` to avoid
the MPS crash during threaded weight loading.

The recipe loads the selected pretrained model directly into GRPO. It performs
no SFT and generates no supervised training dataset. Python is exposed through
TRL’s native tool loop; the model must already be able to produce valid tool
calls or correct final answers often enough to obtain mixed rewards.

GRPO defaults follow [nanochat’s RL recipe](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py):
16 prompts × 16 completions per optimizer update, temperature 1, top-k 50,
256 completion tokens, and one epoch (eight updates for 128 training prompts).
TRL runs training microbatch one with 256 accumulation steps.
`generation_batch_size=16` limits each rollout batch to one prompt’s 16
completions; native TRL buffering splits it into training microbatches.
Sixteen rollout batches are consumed before each optimizer update. Accumulation
alone would default to generating all 256 sequences together; the explicit
generation batch avoids that. No custom trainer or sampling implementation is needed.

Python is available through TRL’s native tool loop. The completion budget includes
calls, results and final answers; tool result tokens are excluded from policy loss.
One tool round is allowed by default; `--max-tool-calling-iterations` changes it.
The existing Python executor has a five-second timeout and a 256 MiB memory limit
(the memory limit is skipped on macOS). It is not a security sandbox.
Each call starts a lightweight standalone Python worker, so it does not re-import
the training script or model dependencies. The parent enforces a wall-time limit
in addition to the worker's existing execution guard.
Training uses float32 with dropout disabled and gradient checkpointing enabled.
On supported hardware, `--bf16 true` enables mixed precision. Microbatch one
reduces activation memory, but full-model optimizer state and 16 simultaneous
rollouts still consume memory; this is not a verified MPS memory guarantee.

The objective uses TRL’s token-normalized `loss_type="dapo"`, mean-centered
advantages (`scale_rewards="none"`), zero KL and one on-policy iteration.
TRL’s token normalization is over each generation batch; nanochat averages
its individual microbatch token-normalized losses, so this is an adaptation,
not an identical loss for unequal sequence lengths.
RL rates decay linearly from nanochat’s 0.05 stage multiplier:
Muon `0.001`, AdamW output-head base rate `0.0002`.
[AdamW width scaling](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py)
multiplies rates by `sqrt(768 / hidden_size)`: the default Qwen width 896 gives
a shared embedding/head RL rate ≈`1.85e-4`.
Qwen ties its input embedding to its output head, so both use the lower
output-head rate; the weights remain tied. Qwen-specific biases and learned
norms use the same scaled rate with AdamW betas `(0.8, 0.95)`.
The shared head uses betas `(0.8, 0.96)` and epsilon `1e-10`; Muon uses momentum
0.95, factored second-moment beta 0.9, and five Polar Express iterations.
All groups use zero weight decay; gradient clipping is disabled.
The existing repository Muon is reused; no optimizer implementation is added.

`configs/optim/qwen-nanochat.yaml` defines the base groups. `--learning-rate`
sets the output-head rate before width scaling; all group rates scale
proportionally. `--optimizer-config` selects another group recipe, which
receives the same rate and width scaling.
`--max-steps` overrides the default one-epoch GRPO horizon. Native batch and
sampling flags remain available; generation batches must contain whole
completion groups and align with accumulation to keep sampling on-policy.
Evaluation remains the existing fixed greedy spelling score.

Each prompt contains the task's user question and this system instruction:

```text
Extract every character of the supplied word and separate the characters with commas. Preserve the word exactly, even if it is unfamiliar or misspelled. Do not check or correct its spelling. The required answer is the original word, a colon, and its comma-separated characters. For example, cat:c,a,t. Use the python tool: assign the supplied word to word, then run print(word + ':' + ','.join(word)). Reply with only the printed answer from stdout, without quotes or explanation.
```

Before the actual question, every prompt includes one fixed worked conversation:
a user asks for `cat`; the assistant calls Python with
`word = 'cat'; print(word + ':' + ','.join(word))`; the tool returns the executor's
JSON result with stdout `cat:c,a,t\n`; the assistant answers `cat:c,a,t`.
Training and evaluation use the same example. It is an example in the prompt,
not SFT, and no answer to the current question is inserted. The example and its
tool result are prompt tokens and are excluded from policy loss.

Reference answers are stored separately for the reward callback and never appear in model
prompts. The callback calls the existing `SimpleSpelling.reward` on the final
assistant message, not the tool call or tool output: 1 for
the exact `word:w,o,r,d` target after trimming surrounding whitespace, 0 otherwise.
The model needs a tool-aware chat template and native response parser. Known
templates use TRL's response-parser setup; a tokenizer-native parser is reused
when present. The template is replaced with its TRL training variant for conversation
prefix preservation, and saved with the checkpoint. Prompt length including tool schemas plus
completion budget must fit the model context window;
oversized examples raise an error rather than being truncated.

Exact-match rewards are sparse. When all sampled completions in a group have
the same reward, GRPO has no relative policy-learning signal for that group.
Exposing a tool does not teach tool use by itself. Inspect
`reward`, `reward_std`, and `frac_reward_zero_std` in the training logs. A run
that executes successfully does not establish learning or score improvement.
There is no shaped reward or answer injection into GRPO rollouts.

The worked example produced mixed rewards in natural sampling. A verified
single-update run on Qwen 0.5B used rates 10× below the nanochat defaults:
Muon `0.0001` and AdamW approximately `1.85e-5`, selected by the command above.
With the MPS synchronization boundary enabled, the captured update matched CPU
replay within `2.98e-8`. The reloaded model scored 8/8 in a fresh evaluation
process. This checks one update; learning gains and long-run stability remain
unverified. Keep the worked prompt while validating longer training.

The kernel uses `torch.no_grad()` and the upstream
`torch.compile(dynamic=False, fullgraph=True)` settings. The recompilation limit
is 64 during kernel calls; the caller's compiler limit is restored afterward.
On MPS, the optimizer
synchronizes after each matrix-batch computation before copying results back.
CPU and CUDA do not take this synchronization branch.

All eight Python calls timed out during the original in-process post-training
evaluation under heavy memory pressure. Reevaluating the same checkpoint
in a fresh process produced eight successful calls and an 8/8 score. Inspect
tool errors alongside scores when running on a memory-constrained laptop.

## Evaluation and artifacts

The same fixed `SimpleSpelling(split="test")` examples are evaluated greedily
before training and after reloading the final GRPO checkpoint. All phases use
the same instruction, Python tool and tool-round limit through
`TransformersEngine.complete`. Evaluation uses the selected token budget per
assistant turn; GRPO bounds the entire completion including tool results.
Truncated turns, malformed calls and exhausted tool rounds count as failed
answers and are recorded; other runtime failures propagate.
Training uses `split="train"`; example order and split seeds follow the existing
task. These splits share a vocabulary and can contain the same words: this is
evaluation on a deterministic test view, not proof of generalization to unseen
words. The word-list URL and model Hub revisions are not pinned; hardware can
also affect reproducibility.

Artifacts default to
`CACHE_DIR/models/Qwen--Qwen2.5-0.5B-Instruct/grpo/default`, where `CACHE_DIR`
comes from `RLLAB_CACHE_DIR` and otherwise defaults to `~/.cache/rl_lab`.
`--run-name run-2` selects `grpo/run-2`; `--output-dir` overrides the path.
Populated output directories are rejected to preserve existing runs.

The output contains model weights and tokenizer, GRPO checkpoints and
`trainer_state.json`, exact `train.jsonl` and `simple-spelling-test.jsonl`
exports, `recipe.json` with training settings, worked example, tool schema, response parser and dataset
fingerprints, and
`evaluation.json` with `before` and `after` exact-match scores and
example counts. The baseline score is saved before GRPO starts.
`before.jsonl` and `after.jsonl` preserve evaluation histories,
predictions, references, scores and errors so zero scores can be diagnosed.
Periodic training checkpoints use `--save-steps 60` by default; change `--save-steps` when
changing the run length if you need an optimizer checkpoint. Final inference
weights and trainer state are always saved after training.

## Offline compatibility check

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
uv run --locked --group train pytest tests/tools/test_exec.py tests/train/test_grpo.py tests/tasks/test_simple_spelling.py tests/tasks/test_correctness_rewards.py tests/engines/test_transformers_tools.py -q
```

The integration test uses a local tiny tied-weight Qwen2, byte-level tokenizer,
native Qwen template and word list. It verifies that GRPO starts from the
original checkpoint, uses Muon/AdamW with the expected group rates and tied
weight ownership, and accumulates two generation batches into one optimizer
update. Real generation runs before controlled samples ensure mixed rewards.
Both Python tool loops execute real subprocesses. Checks cover tool-output
masking, a finite weight update, final-answer-only rewards, saved histories,
checkpoint reload, and absence of SFT artifacts. Controlled samples verify
the machinery, not natural-model learning.
