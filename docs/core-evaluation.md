# CORE evaluation output

The CORE evaluation script writes a CSV to
`$RLLAB_CACHE_DIR/base_eval/<model-slug>.csv` (the default cache directory is
`~/.cache/rl_lab`). Each row represents one task for the evaluated model.

| Column | Meaning |
| --- | --- |
| `model` | Hugging Face model path |
| `task` | CORE task label |
| `accuracy` | Fraction of scored examples answered correctly |
| `num_examples` | Number of scored examples |
| `random_baseline` | Random baseline accuracy as a percentage |
| `centered_result` | Accuracy adjusted for the random baseline |
| `elapsed_time` | Task evaluation time in seconds |
| `core/accuracy` | Mean task accuracy |
| `core/core` | Mean centered task accuracy (CORE metric) |
| `core/max_per_task` | Requested maximum examples per task; `-1` means all |
| `core/step_time_ms` | Total task evaluation time in milliseconds |

Aggregate `core/*` values repeat on each task row. There is no separate summary
row. This replaces the previous `Task, Accuracy, Centered` CSV format. Numeric
values are written without explicit rounding for downstream tracking.

The Markdown report at `report/base-model-evaluation.md` also includes all
aggregate metrics and each task metric under `<task>/<metric>` keys. Its float
values use the report writer's four-decimal display precision.

`--max-per-task` truncates the shuffled dataset before evaluation. Few-shot
demonstrations come from that subset, excluding the scored example; their count
remains the configured shot count, matching upstream nanochat. If the subset has
fewer than `num_fewshot + 1` rows, sampling raises `ValueError`. For the current
10-shot tasks, use `--max-per-task=11` or greater, or `-1` for the full dataset
(the default). The CLI rejects all other values below 11 before device setup or
model loading, with an explanation of the demonstration row requirement.
