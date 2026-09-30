# Offline SGLang

The adapter uses SGLang's offline Python `Engine`, without starting an HTTP
server. Construction loads a separate model runtime and may start workers and
initialize distributed resources. Attaching live Transformers training weights
is not supported; pass them to `TransformersEngine(model, tokenizer)` instead.

SGLang is optional and absent from the base lock. Install it separately in an
environment compatible with both SGLang's GPU dependencies and rl-lab's Python
>=3.14 / Torch >=2.14 requirements, for example with `uv pip install sglang`.
No SGLang version is pinned or GPU-tested by this PR; do not assume any arbitrary
SGLang release resolves against rl-lab's lock. Mock-based CI does not install it.

```python
from rl_lab.engines import create_engine

def main():
    with create_engine(
        "sglang",
        model_name="your-model",
        tensor_parallel_size=1,
        dtype="bfloat16",
        # mem_fraction_static=0.8,  # other SGLang Engine arguments pass through
    ) as engine:
        inputs = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2 + 2?"}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        outputs = engine.generate(
            inputs.input_ids,
            inputs.attention_mask,
            max_new_tokens=64,
            temperature=0.0,
        )
        print(outputs[0][0])

if __name__ == "__main__":
    main()
```

`EngineRouter.create(EngineConfig("sglang", "your-model", tensor_parallel_size=2))`
is also supported. The adapter maps `tensor_parallel_size` to SGLang's `tp_size`.
Concrete imports use `from rl_lab.engines.sglang import SGLangEngine`; importing
that module requires SGLang. The factory imports it only when selected, and the
package root never imports SGLang.

`generate()` accepts token IDs and an attention mask; tokenize and render chat
templates before calling it. Options such as `max_new_tokens`, `temperature`,
`top_p`, and `stop` are direct keyword arguments. An HF
`AutoTokenizer` is loaded separately (only the tokenizer, not another model),
with the same `tokenizer_path`, `revision`, `tokenizer_mode` (auto/slow) and
`trust_remote_code` settings as the runtime. Custom non-HF tokenizer backends
and `skip_tokenizer_init=True` are unsupported. Chat needs a configured template.

Batches preserve order and request one non-streaming result per input. Stops are
passed to SGLang and trimmed defensively. Token usage comes from `meta_info`;
missing prompt counts fall back to the input length and missing completion counts
remain `None`. Reported counts can include stop/EOS tokens. Continuation likelihood
scoring is not implemented and `require("loglikelihood")` fails explicitly.

Use a dedicated process/job: the upstream runtime starts child processes and its
public `shutdown()` may terminate descendants of the current process. Do not share
that process with a trainer's unrelated workers. Call `close()` or use the context
manager; repeated close calls are harmless. Construction/generation are synchronous;
use the adapter in a normal script rather than inside a running asyncio event loop.

Sources: [offline engine API](https://docs.sglang.io/docs/basic_usage/offline_engine_api),
[engine implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py),
[sampling parameters](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/sampling/sampling_params.py).
Tests mock the SGLang runtime and use a real tiny local tokenizer; no GPU model
loading or worker lifecycle integration test was performed.
