# Separate task evaluation scripts

## Compare models and checkpoints

The independent comparison script works with checkpoints from any training
method. Labels are supplied explicitly; no before-training model is required:

```bash
cache_dir="${RLLAB_CACHE_DIR:-$HOME/.cache/rl_lab}"
uv run --locked --with matplotlib python -m scripts.eval.compare \
  --model 'GPT-2 (before SFT)=openai-community/gpt2' \
  --model "GPT-2 (after SFT)=$cache_dir/models/openai-community--gpt2/sft" \
  --tasks gsm8k,mmlu \
  --max-per-task 16 --max-new-tokens 64 --device-type mps
```

Add another `--model 'Baseline=MODEL_OR_CHECKPOINT'` to compare a baseline.
Omit the before model to evaluate only the trained checkpoint and/or baselines.
By default each comparison writes to `CACHE_DIR/reports/compare-<UTC timestamp>`.
`CACHE_DIR` uses `RLLAB_CACHE_DIR`, or `~/.cache/rl_lab` when unset.
An explicit `--output-dir` must be a new, empty directory.

Omitting `--tasks`, or passing `--tasks None` or `--tasks all`, evaluates all
implemented scored benchmarks: `gsm8k`, `mmlu`, `arc_easy`, `arc_challenge`,
`gpqa`, `humaneval`, `spellingbee`, `simple_spelling`, and `core`.
The alias `arc` selects both ARC subsets. SmolTalk and CustomJSON have no
scorer; DeepSWE is unimplemented. They are excluded and explicitly requesting
them is an error. All tasks uses GPQA diamond, MMLU all/test, and the other
scripts' default evaluation splits. Synthetic spelling tasks contain 1,000
examples before applying the limit.

The default example limit is `-1` (full datasets). Start with explicit tasks
and a small limit on a laptop. If CORE is selected, a positive limit must be
at least 11 to accommodate its existing few-shot protocol. GPQA may require
dataset access approval. HumanEval uses the existing scorer that executes
generated Python; it is not an isolated security sandbox.

Models load sequentially using Transformers with its default loading dtype.
Each benchmark is loaded once for the comparison, keeping its deterministic
task order identical across models. Each model uses its own tokenizer and
chat template. If its tokenizer lacks a template, the script warns and uses
the plain `User: ...` / `Assistant:` format from the SFT recipe. This is a
prompting convention for evaluating base models, not a claim that they are
instruction tuned. Generation is greedy, and task-specific system prompts
match the standalone scripts. ARC/MMLU reuse next-token answer-label scoring
and require labels to be single tokens. Oversized chat prompts fail instead
of being dropped or truncated. CORE retains its separate continuation-scoring
and context-cropping protocol.

CORE processes the full retained context but computes losses and predictions
only for the suffix containing the answers. Models supporting Hugging Face's
`logits_to_keep` also project only that suffix to the vocabulary; other models
project the full sequence before slicing. This preserves answer scoring and
does not change the few-shot examples or context limit. `--max-new-tokens`
does not limit CORE's continuation scoring.

Outputs:

- `evaluation.json`: model labels and paths, resolved tasks, settings, actual
  chat templates, aggregate scores, counts, and completion status per model.
- `model-N-TASK.jsonl`: each scored example, reference, prediction, and score.
  CORE currently exposes only aggregates, so it has no prediction files.
- `plots/TASK.png` and `.pdf`: accuracy bars for every named model, annotated
  with scores and example counts. CORE gets a plot for each constituent task;
  its centered aggregate metrics are retained separately in the JSON.

Scores are saved after each task. A failure propagates and leaves partial
results for inspection; incomplete models are marked `complete: false`.
Plots are generated only after the entire comparison succeeds. The script
does not merge older runs or assume that equal labels imply comparable
settings. Model and dataset Hub revisions are not pinned by this first
implementation; prediction files retain the actual scored examples.

### Native Python tool calling

Add `--tools python --tool-models 'Qwen2.5-1.5B-Instruct'` to grant Python
access to a particular model label. `--tool-models` accepts comma-separated
labels or `all` (the default selector). `--tools` defaults to `none`;
`--tools none` or `--tool-models none` disables access. None values are
case-insensitive, so `None` and `NONE` work too.

Tool completion reuses `TransformersEngine.complete()`. Models must have a
chat template that renders tool descriptions and a native response parser.
The script accepts the tokenizer's `response_template`, or uses the existing
Qwen2 response schema for Qwen2 tool-aware templates (including Qwen2.5).
GPT-2's plain role template is unsupported: requesting tools for it fails
explicitly rather than silently evaluating without tools. For a mixed
GPT-2/Qwen comparison, select only the Qwen label instead of `all`.

For generative tasks, the model receives the usual task instructions plus
permission to use Python. It may choose to answer without a tool call.
The final assistant answer is passed to the unchanged task scorer; GSM8K
still requires `#### number`. No reference solution enters the tool
conversation. ARC/MMLU and CORE remain without tools even when the model
has been selected for tool access elsewhere.

`--max-new-tokens` applies to each assistant turn in tool mode, rather than
the entire conversation. `--max-tool-rounds` defaults to 4, and
`--python-timeout` defaults to 5 seconds per execution. Each Python call has a
fresh namespace: use `print()` to return a result. The executor is not a
security sandbox, and its memory limit is not enforced on macOS. See
[tool completion documentation](engines/transformers-tools.md) for details.

Initial context checks include tool descriptions. Subsequent turns are
handled by the native engine; model context errors, truncated turns, malformed
calls and exhausted round budgets stop the comparison.
Execution errors and timeouts appear as tool responses, letting the model
attempt recovery. No failed examples are silently excluded from the score.

Each task result and prediction record includes `tools`, the names of tools
available during that evaluation. Tool-enabled prediction records also have
`history`, retaining the full prompt, assistant requests, tool outputs and
final answer. Run settings retain selectors, instructions, limits and timeout;
model metadata retains the response schema. Plot labels distinguish
`[tools: python]` from `[tools: none]`. These scores describe different
evaluation conditions and should be reported with their tool settings.

## Individual task scripts

Run each script from the repository root. The default model is
`Qwen/Qwen2.5-1.5B-Instruct`. Each scored script uses the same
`init_distributed` and `load_hf_model` setup as `scripts/eval/core.py`, but
supports only single-process evaluation. Model placement and loading precision
follow that existing helper. CORE remains a separate, unchanged script.

Start with a small run, then omit `--max-per-task` to score the full dataset:

```bash
uv run --locked python -m scripts.eval.arc --max-per-task 2
uv run --locked python -m scripts.eval.mmlu --max-per-task 2
uv run --locked python -m scripts.eval.gsm8k --max-per-task 2
uv run --locked python -m scripts.eval.humaneval --max-per-task 2
uv run --locked python -m scripts.eval.gpqa --max-per-task 2
uv run --locked python -m scripts.eval.spellingbee --max-per-task 2
uv run --locked python -m scripts.eval.spellingbee --task simple_spelling --max-per-task 2
```

Every scored script accepts `--hf-path`, `--device-type` (`cuda`, `cpu`, or
`mps`), and `--max-per-task` (`-1` means all). Results print the mean accuracy
and number of examples. Models, datasets, and the spelling word list may need
downloads on the first run.

Each completed benchmark run also writes a timestamped Markdown file under
`CACHE_DIR/reports/` and prints its path. ARC writes one report for each evaluated subset.
Reports include the model, run settings, accuracy, and every scored example,
with **wrong answers first**. Each example shows its question/conversation
(including choices), expected answer, model answer, and score. Example numbers
refer to the task's shuffled evaluation order, starting at 1.

For ARC/MMLU, the model answer is the selected label; no reasoning is generated.
For generation tasks, the full response is preserved. Structured reference
answers show their text and Python/tool-output parts without executing them.
HumanEval references contain the canonical solution; correctness still comes
from the existing test execution scorer.

Reports retain earlier runs, are ignored by Git, and are written only after
successful evaluation. This first implementation keeps example results in
memory until the report is written. Previous terminal-only runs cannot be
reconstructed; rerun a benchmark to produce its report.

| Script | Evaluation |
| --- | --- |
| `arc.py` | Both ARC test subsets by default; `--subset` selects one. Highest next-token logit among the example's answer labels. |
| `mmlu.py` | MMLU `all`/`test`; highest next-token logit among A/B/C/D. |
| `gsm8k.py` | `main`/`test`; greedy generation, numerical answer after `####`. |
| `humaneval.py` | Test split; greedy generation, existing Python execution scorer. |
| `gpqa.py` | `gpqa_diamond` by default; `--subset` selects another supported subset. Existing final-answer extraction. Dataset access may require accepting its terms. |
| `spellingbee.py` | Synthetic test examples; `--task` selects letter counting or SimpleSpelling. `--size` defaults to 1000. |

ARC and MMLU follow
[nanochat's categorical evaluation](https://github.com/karpathy/nanochat/blob/master/scripts/chat_eval.py).
They require each answer label to encode to exactly one token without special
tokens. The prompt includes Qwen's assistant generation prefix and excludes
the reference answer. These scripts score answer labels, not the full choice
text, and do not generate reasoning.

Generation scripts use the existing `evaluate_task` runner and task scorers.
They accept `--batch-size` (default 1) and `--max-new-tokens` (default 512).
They produce one greedy completion per question; token-budget exhaustion or
failure to produce the required answer format can cause a wrong score.
HumanEval executes generated Python through the repository's existing
execution helper, which is not a security isolation boundary.

`smoltalk.py`, `customjson.py`, and `deepswe.py` expose explanatory entry points
that exit with an error before loading a model: the first two have no scorer,
and DeepSWE has no task implementation. They do not report benchmark accuracy.
`_base.py` and `__init__.py` are task infrastructure, not benchmarks.
