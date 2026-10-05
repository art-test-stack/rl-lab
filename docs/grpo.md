# Single-device GRPO on SimpleSpelling

Run from the repository root:

```bash
uv run --locked --group train python scripts/train/grpo_simple_spelling.py \
    --train-limit 128 --eval-limit 32 --max-steps 32
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct`. The recipe uses TRL's
`GRPOTrainer`, full-model AdamW training, and ordinary Transformers sampling on
the training device. It requires one process and one device; CPU, CUDA, or MPS
selection follows TRL/Transformers. Add `--use-cpu` for CPU. No vLLM, distributed
training, or new dependencies are needed. Model weights and the existing task's
English word list must be downloaded or cached before an offline run.

Defaults are 32 optimizer steps, learning rate `1e-6`, four sampled completions
per prompt, batch size four, and 64 completion tokens. Training uses float32
with dropout disabled. On hardware supporting it, `--bf16 true` enables mixed
precision. Native `GRPOConfig` options such as `--gradient-accumulation-steps`,
`--num-generations`, `--temperature`, and `--seed` remain available. The generation
batch must be divisible by the number of completions per prompt; TRL validates
this. Reduce both batch size and number of generations to two for a smaller run.

The objective is explicitly TRL's `loss_type="grpo"`. `beta=0` disables the KL
penalty and avoids a reference-model copy. There is no reward model or value
model. Setting `--beta` above zero adds TRL's reference model and increases memory
use. This is a bounded first recipe, not a measured hardware sizing guarantee.

Each prompt contains the task's user question and this system instruction:

```text
Spell the requested word as word:comma-separated-letters. For example, cat:c,a,t. Respond only with that answer.
```

Reference answers are stored separately for the reward callback and never appear
in model inputs. The callback calls the existing `SimpleSpelling.reward`: 1 for
the exact `word:w,o,r,d` target after trimming surrounding whitespace, 0 otherwise.
The tokenizer's existing chat template is retained; models without one are
rejected. Prompt length plus completion budget must fit the model context window;
oversized examples raise an error rather than being truncated.

Exact-match rewards are sparse. When all sampled completions in a group have
the same reward, GRPO has no relative policy-learning signal for that group.
Start from an instruction model or spelling-capable checkpoint, and inspect
`reward`, `reward_std`, and `frac_reward_zero_std` in the training logs. A run
that executes successfully does not establish learning or score improvement.
There is no automatic SFT warmup, shaped reward, or answer injection.

## Evaluation and artifacts

The same fixed `SimpleSpelling(split="test")` examples are evaluated greedily
before training and after reloading the saved checkpoint. Both phases use the
same system instruction and completion budget through `evaluate_task`.
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

The output contains model weights and tokenizer, TRL checkpoints and
`trainer_state.json`, exact `train.jsonl` and `simple-spelling-test.jsonl`
exports, `recipe.json` with settings and dataset fingerprints, and
`evaluation.json` with before/after exact-match scores and example counts.
The default saves a training checkpoint at step 32; change `--save-steps` when
changing the run length if you need an optimizer checkpoint. Final inference
weights and trainer state are always saved after training.

## Offline compatibility check

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
uv run --locked --group train pytest tests/train/test_grpo.py tests/tasks/test_simple_spelling.py -q
```

The integration test uses a local tiny GPT-2 and word list. It exercises actual
Transformers generation, then substitutes one correct and one incorrect sampled
answer to guarantee a nonzero advantage. Real TRL loss, backpropagation, and
optimization must update weights; the test also checks exact reward semantics,
reference exclusion, exports, checkpoint reload, before/after evaluation,
context rejection, and run-directory protection. Controlled samples verify the
training machinery, not natural sampling success or pretrained-model gains.
