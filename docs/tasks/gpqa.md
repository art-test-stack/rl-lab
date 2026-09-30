# GPQA

```python
from rl_lab.tasks.gpqa import GPQA
from rl_lab.eval import evaluate_task

task = GPQA(subset="gpqa_diamond", split="train", seed=42)
result = evaluate_task(task, engine, limit=32)
```

All four published configurations are accepted: `gpqa_main`, `gpqa_extended`,
`gpqa_diamond`, `gpqa_experts`. The Hub calls their split `train`; the questions
are evaluation data. Obtain access through the publisher's normal Hub process.

The implementation uses the original Question / Correct Answer / Incorrect
Answer 1–3 schema. Answer positions are shuffled deterministically per example,
independently of iteration order or slicing. Generative evaluation requests the
baseline's `The correct answer is (X)` format and scores the last such marker;
a bare letter is also accepted. Unparseable answers score zero. The task retains
an optional categorical mode for future scoring work, but the current evaluator
accepts generative tasks only. Report the seed and generation settings with
results. This does not claim exact reproduction of the paper's baselines.

Sources: [dataset card](https://huggingface.co/datasets/Idavidrein/gpqa/blob/main/README.md),
[official baseline schema and prompt code](https://github.com/idavidrein/gpqa/blob/main/baselines/utils.py).
Tests use synthetic rows only and do not download or publish benchmark questions.
