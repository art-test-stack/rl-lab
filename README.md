# rl-lab

RL-lab is a lightweight, educational project for PyTorch-native post-training
recipes, using [Transformers](https://github.com/huggingface/transformers).

For the GPT-2 / one-turn SmolTalk training script, see the [SFT recipe](docs/sft.md).
For single-device GRPO on SimpleSpelling, see the [RL recipe](docs/grpo.md).


## Target architecture

This diagram describes the intended design, not a list of implemented features.
The Transformers inference adapter and TRL SFT and single-device GRPO recipes are implemented;
the other recipes and backends shown below are planned integrations.

```text
                    ┌─────────────────────┐
                    │       Recipe        │
                    │ GRPO / DPO / PPO /  │
                    │    Distillation     │
                    └──────────┬──────────┘
                               │
                  algorithm semantics only
                               │
             ┌─────────────────┴─────────────────┐
             │                                   │
     Experience / Batch                    Objective / Loss
        production                           computation
             │                                   │
     ┌───────┴────────┐                  ┌───────┴────────┐
     │ rollout/scorer │                  │ training model │
     │ reward/teacher │                  │ + autograd     │
     └───────┬────────┘                  └───────┬────────┘
             │                                   │
     Inference backends                    Training backend
     - Transformers                       - plain PyTorch
     - vLLM (planned)                     - DDP/FSDP2
     - SGLang (planned)                   - TorchTitan
     - custom                             - future gpt-lab
```


## Inference engines

The Transformers engine executes inference using attached training weights or a
standalone model. vLLM and SGLang are recognized by the factory but currently
raise `NotImplementedError`.

### Use the current training weights

```python
from rl_lab.engines.transformers import TransformersEngine

# model and tokenizer are the exact objects already used by the trainer.
# Render chat templates before passing token IDs to the engine.
engine = TransformersEngine(model, tokenizer)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2 + 2?"}],
    tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
)
outputs, _cache = engine.generate(
    inputs.input_ids, inputs.attention_mask, max_new_tokens=64,
)
print(outputs[0][0])
```

Each result is `(text, prompt_tokens, completion_tokens, generated_token_ids)`.
The second return value is the Transformers KV cache. To continue a Transformers generation, pass only uncached
token IDs, the cumulative attention mask, and `past_key_values=cache`; retain
generated token IDs to construct the next uncached input. Include the final
sampled token from the previous result because it is not yet in the returned
cache.

Construction with an existing model and tokenizer does not copy weights or move
the model. Generation runs without autograd, switches the model to eval mode,
then restores its top-level train/eval flag. Closing the adapter only drops its
references; the trainer's model, optimizer and process groups remain usable.
`from_pretrained()` is an alias for `load()`.

The attached path currently expects a decoder-only HF-compatible model exposing
`generate()`. DDP/FSDP/TP wrapping and collective
participation remain the training runtime's responsibility: this adapter does
not automatically unwrap models or promise rank-zero-only sharded inference.
Pause training while evaluating a borrowed model; these operations are not
thread-safe against concurrent training or generation on that same object.

### Load a standalone model

```python
from rl_lab.engines import create_engine

engine = create_engine("transformers", model_name="your-model", device="cuda")
inputs = engine.tokenizer(["A raw completion prompt"], return_tensors="pt")
outputs, _cache = engine.generate(
    inputs.input_ids, inputs.attention_mask,
    max_new_tokens=64, temperature=0.0, stop=["END"],
)
```

`generate()` accepts token IDs and an attention mask; tokenize and render chat
templates before calling it. Generation options are direct keyword arguments,
such as `max_new_tokens`, `temperature`, `top_p`, and `stop`. The engine returns
completion text, token counts and generated token IDs in input order.

Results preserve input order. Each result's text contains only the completion,
trimmed at the first requested stop string. Transformers decoding skips special
tokens. Token usage reflects the model's prompt and generated token counts. Greedy generation is selected by
`temperature=0`.

The evaluator currently supports generation-based tasks. Categorical candidate
scoring is deferred until it is needed by on-policy methods.

The `vllm` and `sglang` factory names remain reserved for future backends; calling
either currently raises `NotImplementedError`.

### Tests

```bash
uv sync --locked --group dev
uv run --locked pytest
```

The GitHub Actions worker runs the same locked setup on Python 3.14 for pull
requests and pushes to `main`, and uploads a JUnit report. Tests create a tiny
random GPT-2 and a tokenizer locally, run on CPU with Hub access disabled, and
cover attachment, restoration after errors, batching, stop handling,
local checkpoint loading, factory validation and capabilities. Real GPU
integration runs remain separate from this CI suite.


## Task evaluation

```python
from rl_lab.eval import evaluate_task
from rl_lab.tasks.gsm8k import GSM8K

# Reuse an engine attached to training weights, or one loaded for evaluation.
result = evaluate_task(
    GSM8K(subset="main", split="test"),
    engine,
    batch_size=8,
    limit=128,
    system_prompt="Solve the problem and end with #### followed by the numerical answer.",
)
print(result.num_examples, result.score)
```

Tasks define `build_eval_input(example)` and `evaluate(example, completion)`. The default prompt hook removes the final
assistant reference; subclasses can return `TextPrompt` for raw completions.
The runner owns batching and returns the arithmetic mean of scalar task scores.
It neither creates nor closes the engine, and does not change training ownership.

The runner currently supports generative tasks, using one completion per example.
Categorical task definitions remain in the repository, but evaluation rejects
them until candidate scoring is needed. Chat inputs require chat support.
`limit` selects the first examples of the task's existing view. A supplied
`system_prompt` is only valid for chat prompts without an existing system
message. No input conversations are modified.

Training-only datasets, CORE's specialized scoring and agentic environment
benchmarks need their own evaluation protocols. This runner is single-process;
it does not implement distributed aggregation, pass@k or benchmark-specific
macro averages.

### Compare models and checkpoints

Use [scripts/eval/compare.py](scripts/eval/compare.py) to evaluate models before
and after post-training, or compare any named models on the same task examples:

```bash
uv run --locked --with matplotlib python -m scripts.eval.compare \
  --model 'GPT-2 (before SFT)=openai-community/gpt2' \
  --model 'GPT-2 (after SFT)=openai-community/gpt2:sft' \
  --model 'Qwen2.5-1.5B-Instruct=Qwen/Qwen2.5-1.5B-Instruct' \
  --tasks gsm8k,mmlu \
  --max-per-task 16 --max-new-tokens 512 \
  --device-type mps
```

Each `--model` is `LABEL=MODEL_OR_CHECKPOINT`; labels are arbitrary. Omit the
before-training model if it is not needed, and add more models as baselines.
Cached checkpoints use `HF_ID:METHOD[:RUN]`, for example
`openai-community/gpt2:sft:run-2`. An omitted run selects `default`.
Models load sequentially. Use `cpu` or `cuda` instead of `mps` as appropriate,
and optionally set `--output-dir` to a new, empty directory.

`CACHE_DIR` is resolved from `RLLAB_CACHE_DIR` by `rl_lab.utils.default`,
defaulting to `~/.cache/rl_lab`. SFT defaults to
`CACHE_DIR/models/openai-community--gpt2/sft/default`; `--run-name run-2`
selects the `sft/run-2` directory. Comparisons create a timestamped
directory under `CACHE_DIR/reports`. Checkpoints and comparison reports are
stored separately.

Omit `--tasks`, or pass `--tasks None` or `--tasks all`, to run all implemented
scored benchmarks. Tasks without a scorer are excluded. When CORE is selected,
`--max-per-task` must be at least 11 or `-1` for the full dataset. The default
limit is `-1`; start with explicit tasks and a small limit for a laptop run.

The output directory contains `evaluation.json` with scores and settings,
per-model/task JSONL files with examples, references and predictions, and
`plots/overview.png` and `.pdf` compare all models across benchmarks, using
CORE's aggregate centered accuracy. `plots/` also contains one PNG/PDF
accuracy bar plot per task. CORE exposes aggregate
results only and gets a plot for each constituent task. The script reuses the
suite's generation, categorical and CORE scoring protocols.

Native tool calling is optional. To allow Python for the Qwen model in the
example, add:

```bash
--tools python --tool-models 'Qwen2.5-1.5B-Instruct'
```

`--tool-models all` grants access to every named model. `--tools none` (the
default), or `--tool-models none`, disables it; `None`/`NONE` are also accepted.
Selected models need a tool-aware chat template and response parser; GPT-2's
plain role template is unsupported. Tools apply to generative tasks only;
categorical and CORE evaluation retain their existing protocols. Reports save
full tool conversations, and plot labels show whether Python was available.

See [evaluation documentation](docs/task-evaluation-scripts.md) for supported
tasks, prompting conventions and evaluation limits.
