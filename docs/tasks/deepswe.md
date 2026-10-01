# DeepSWE

DeepSWE requires a coding agent, a repository environment and the benchmark's
program-based verifier. Its score cannot be inferred from a generated answer or
from equality with a reference patch. The class implements the `Task` data and
scoring methods while declaring `eval_type = "agentic"`.

Obtain the official [task bundle](https://github.com/datacurve-ai/deep-swe) through
the publisher's supported access path. Pass its `tasks/` directory or a single
task directory to `DeepSWE`. Loading is local; it does not fetch gated data or
start containers.

```python
from rl_lab.tasks.deepswe import DeepSWE, DeepSWEResult

task = DeepSWE("/path/to/deep-swe/tasks", start=0, stop=10)
example = task[0]
prompt = task.build_eval_input(example)
# Run the agent and official verifier with Pier/Harbor for example["task_dir"].
# Then associate THAT trial's verifier output with the same task id:
result = DeepSWEResult.from_reward_file(
    example["task_id"], "/path/to/that-trial/verifier/reward.json"
)
score = task.evaluate(example, result)
```

The loader uses sorted task directories, `task.toml` metadata and `instruction.md`.
It excludes solution and test contents from examples and model prompts. Metadata
and task paths are for the runtime; pass only the prompt to the agent. The external
runtime remains responsible for isolating held-out tests and reference solutions.

`evaluate` accepts only a `DeepSWEResult` for the matching task. It uses the
verifier's binary `reward`, not its auxiliary pass fractions. Missing results,
non-binary values and the `-1` infrastructure-failure sentinel raise instead of
silently counting as model failures. The caller binds the task id because the
upstream reward file contains no identity; this class does not authenticate files.

`evaluate_task` deliberately rejects this protocol. Agent execution, container
lifecycle and official verifier invocation remain in Pier/Harbor; no such run is
performed by the unit tests. Tests use synthetic manifests and verifier files.

Sources: [dataset card](https://huggingface.co/datasets/datacurve/deep-swe),
[official format and execution instructions](https://github.com/datacurve-ai/deep-swe/blob/main/README.md),
[task manifest](https://github.com/datacurve-ai/deep-swe/blob/main/tasks/abs-module-cache-flags/task.toml),
[verifier result schema](https://github.com/datacurve-ai/deep-swe/blob/main/tasks/abs-module-cache-flags/tests/grader.py).
