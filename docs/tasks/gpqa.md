# GPQA

```python
from rl_lab.tasks.gpqa import GPQA
from rl_lab.eval import evaluate_task

task = GPQA(subset="gpqa_diamond", split="train", seed=42)
result = evaluate_task(task, engine, limit=32)
# Alternative protocol: score answer-letter likelihoods.
task_mc = GPQA(evaluation="categorical")
```

All four published configurations are accepted: `gpqa_main`, `gpqa_extended`,
`gpqa_diamond`, `gpqa_experts`. The Hub calls their split `train`; the questions
are evaluation data. Obtain access through the publisher's normal Hub process.

The implementation uses the original Question / Correct Answer / Incorrect
Answer 1–3 schema. Answer positions are shuffled deterministically per example,
independently of iteration order or slicing. Generative evaluation requests the
baseline's `The correct answer is (X)` format and scores the last such marker;
a bare letter is also accepted. Unparseable answers score zero. The categorical
mode uses rl-lab's common multiple-choice rendering and summed letter likelihood.
These are distinct protocols; report the chosen mode, seed and generation settings
with results. Neither mode claims exact reproduction of the paper's baselines.

Sources: [dataset card](https://huggingface.co/datasets/Idavidrein/gpqa/blob/main/README.md),
[official baseline schema and prompt code](https://github.com/idavidrein/gpqa/blob/main/baselines/utils.py).
Tests use synthetic rows only and do not download or publish benchmark questions.
