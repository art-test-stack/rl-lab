# rl-lab

RL-lab is a light-weight framework in Python for reinforcement learning methods built for PyTorch, and working with [transformers]()


## Architecture
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
model for a standalone run. The factory constructs the adapter without starting
a server or initializing distributed training.

### Use the current training weights

```python
from rl_lab.engines import TransformersEngine, GenerationConfig, ChatPrompt

# model and tokenizer are the exact objects already used by the trainer.
engine = TransformersEngine.attach(model, tokenizer, batch_size=8)
outputs = engine.generate(
    [ChatPrompt([{"role": "user", "content": "What is 2 + 2?"}])],
    GenerationConfig(max_new_tokens=64),
)
print(outputs[0].text)
```

`attach()` does not load/copy weights or move the model. Inference runs without
autograd and restores every submodule's original train/eval flag, including on
failure. Closing the adapter only drops its references; the trainer's model,
optimizer and process groups remain usable. No context manager is required.
`from_model()` and `from_pretrained()` are aliases for `attach()` and `load()`.

The attached path currently expects a decoder-only HF-compatible model exposing
`generate()` and `get_input_embeddings()`. DDP/FSDP/TP wrapping and collective
participation remain the training runtime's responsibility: this adapter does
not automatically unwrap models or promise rank-zero-only sharded inference.
Pause training while evaluating a borrowed model; these operations are not
thread-safe against concurrent training or generation on that same object.

### Load a standalone model

```python
from rl_lab.engines import create_engine, GenerationConfig, LogLikelihoodRequest

engine = create_engine("transformers", model_name="your-model", device="cuda")
outputs = engine.generate(
    ["A raw completion prompt"],
    GenerationConfig(max_new_tokens=64, temperature=0.0, stop=["END"]),
)
engine.require("loglikelihood")
scores = engine.loglikelihood([LogLikelihoodRequest("The answer is", " A")])
```

Strings and `TextPrompt` are raw text; `ChatPrompt` applies the backend's chat
template exactly once. Results preserve input order and contain completion text
without prompt, EOS or stop strings. Token usage reflects backend-consumed
counts, which can include termination tokens; counts are not guaranteed to be
identical across backends. Greedy generation is selected by `temperature=0`.

Likelihood scoring uses joint context/continuation tokenization, moves trailing
context whitespace into the continuation, adds no EOS, and sums natural-log
probabilities only over continuation tokens. Empty contexts use BOS (or EOS).
Boundaries that still split a token fail explicitly instead of silently scoring
a different continuation. This reference implementation scores requests one at
a time and does not truncate overlong contexts.

### Offline vLLM

Install vLLM in a compatible environment (`uv pip install vllm`). It is deliberately
absent from the base lock because it has its own Python/PyTorch/CUDA compatibility
constraints. Importing `rl_lab.engines` does not import vLLM, Torch or Transformers.

```python
# Its own model runtime, no HTTP server, usable as one HF Job.
engine = create_engine("vllm", model_name="your-model", tensor_parallel_size=1)
```

| Backend | Generation/chat | Continuation likelihood | Reuses live training model |
| --- | --- | --- | --- |
| Transformers | Yes | Yes | Yes |
| Offline vLLM | Yes | Not yet | No |

Offline vLLM cleanup follows the runtime's object/process lifetime; `close()`
drops the adapter reference, without private shutdown calls or global distributed
teardown. Prefer a dedicated job for its independent GPU runtime rather than
repeatedly loading it alongside a live trainer.

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
