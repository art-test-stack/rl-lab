# rl-lab

RL-lab is a lightweight, educational project for PyTorch-native post-training
recipes, using [Transformers](https://github.com/huggingface/transformers).


## Target architecture

This diagram describes the intended design, not a list of implemented features.
The inference adapters currently support Transformers, offline vLLM and offline SGLang; the
recipes and other backends shown below are planned integrations.

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
     - vLLM                               - DDP/FSDP2
     - SGLang                             - TorchTitan
     - custom                             - future gpt-lab
```


## Inference engines

An engine executes inference. It can borrow the training model or load its own
model for a standalone run. The factory selects and constructs the adapter;
standalone construction loads the model immediately. No HTTP server is started.
Attaching a Transformers model leaves distributed setup to the trainer; offline
vLLM manages its own runtime and may create workers and process groups.

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
outputs = engine.generate(
    inputs.input_ids, inputs.attention_mask, max_new_tokens=64,
)
print(outputs[0][0])
```

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
from rl_lab.engines import create_engine, LogLikelihoodRequest

engine = create_engine("transformers", model_name="your-model", device="cuda")
inputs = engine.tokenizer(["A raw completion prompt"], return_tensors="pt")
outputs = engine.generate(
    inputs.input_ids, inputs.attention_mask,
    max_new_tokens=64, temperature=0.0, stop=["END"],
)
engine.require("loglikelihood")
scores = engine.loglikelihood([LogLikelihoodRequest("The answer is", " A")])
```

`generate()` accepts token IDs and an attention mask; tokenize and render chat
templates before calling it. Generation options are direct keyword arguments,
such as `max_new_tokens`, `temperature`, `top_p`, and `stop`. The engine returns
completion text and token counts, in input order.

Results preserve input order and return completion text only, trimmed at the
first requested stop string. Transformers decoding skips special tokens; vLLM
uses its runtime's decoding defaults. Token usage reflects backend-consumed
counts, which can include termination tokens; counts are not guaranteed to be
identical across backends. Greedy generation is selected by `temperature=0`.

Likelihood scoring uses joint context/continuation tokenization, moves trailing
context whitespace into the continuation, adds no EOS, and sums natural-log
probabilities only over continuation tokens. Empty contexts use BOS (or EOS).
Boundaries that still split a token fail explicitly instead of silently scoring
a different continuation. This reference implementation scores requests one at
a time and does not truncate overlong contexts.

### Offline vLLM

vLLM is optional and absent from the base lock. This project currently requires
Python >=3.14 and Torch >=2.14.0; a vLLM installation must also support those
requirements and your CUDA environment. No compatible vLLM version is pinned or
GPU-tested by this project. In an environment satisfying both projects' runtime
requirements, install it separately with `uv pip install vllm`. The locked test
setup below does not install vLLM.

Importing `rl_lab.engines` does not import vLLM, SGLang, Torch or Transformers.
The factory imports only the selected engine module. Each engine module imports
its dependencies at module scope, so importing a concrete class directly also
requires that backend's dependencies. Concrete classes are imported from
`rl_lab.engines.transformers`, `rl_lab.engines.vllm` or
`rl_lab.engines.sglang`, not the package root.

```python
from rl_lab.engines import create_engine

# Its own model runtime, no HTTP server, usable as one HF Job.
engine = create_engine("vllm", model_name="your-model", tensor_parallel_size=1)
```

| Backend | Generation/chat | Continuation likelihood | Reuses live training model |
| --- | --- | --- | --- |
| Transformers | Yes | Yes | Yes |
| Offline vLLM | Yes | Not yet | No |
| Offline SGLang | Yes | Not yet | No |

Offline vLLM cleanup follows the runtime's object/process lifetime; `close()`
drops the adapter reference, without private shutdown calls or global distributed
teardown. Prefer a dedicated job for its independent GPU runtime rather than
repeatedly loading it alongside a live trainer.

### Offline SGLang

```python
from rl_lab.engines import create_engine

# Run in a dedicated process/job; the runtime owns its worker processes.
with create_engine("sglang", model_name="your-model", tensor_parallel_size=1) as engine:
    inputs = engine.tokenizer(["A raw completion prompt"], return_tensors="pt")
    outputs = engine.generate(inputs.input_ids, inputs.attention_mask, max_new_tokens=64)
```

SGLang is optional and installed separately in a compatible GPU environment.
The adapter uses its offline Python engine, token-ID inputs and public `shutdown()`
method. It owns a separate model runtime and cannot attach to live training weights.
See [SGLang setup, lifecycle and limitations](docs/engines/sglang.md).

### Tests

```bash
uv sync --locked --group dev
uv run --locked pytest
```

The GitHub Actions worker runs the same locked setup on Python 3.14 for pull
requests and pushes to `main`, and uploads a JUnit report. Tests create a tiny
random GPT-2 and a tokenizer locally, run on CPU with Hub access disabled, and
cover attachment, restoration after errors, batching, stop handling, likelihoods,
local checkpoint loading, factory validation and capabilities. vLLM calls are
mocked; real GPU integration runs remain separate from this CI suite.


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

Tasks define `build_eval_input(example)`, `evaluation_candidates(example)` and
`evaluate(example, completion)`. The default prompt hook removes the final
assistant reference; subclasses can return `TextPrompt` for raw completions.
The runner owns batching and returns the arithmetic mean of scalar task scores.
It neither creates nor closes the engine, and does not change training ownership.

Generative tasks use one completion per example. Categorical tasks (e.g. MMLU
and ARC) score all candidates in an example batch using summed continuation log
likelihoods, without length normalization. They require a likelihood-capable
engine (currently Transformers); chat inputs additionally require chat support.
Ties choose the first candidate. `limit` selects the first examples of the task's
existing view. A supplied `system_prompt` is only valid for chat prompts without
an existing system message. No input conversations are modified.

Training-only datasets, CORE's specialized scoring and agentic environment
benchmarks need their own evaluation protocols. This runner is single-process;
it does not implement distributed aggregation, pass@k or benchmark-specific
macro averages. There is no evaluation CLI.
