# SimpleSpelling

```python
from rl_lab.tasks.spellingbee import SimpleSpelling
from rl_lab.eval import evaluate_task

result = evaluate_task(SimpleSpelling(size=1000, split="test"), engine)
```

Completes `Task.evaluate` for the synthetic spelling task already adapted from
[nanochat](https://github.com/karpathy/nanochat/blob/a5a3a33c63f996c56c0976d45ea47239b3fd74ae/tasks/spellingbee.py).
The target format is `word:w,o,r,d`. Scoring is exact match after trimming only
surrounding whitespace; `reward` returns the same result as a float. This scoring
rule is an rl-lab addition to the upstream training task, not an official external
benchmark metric. Word order, train/test seeds and sliced views stay deterministic.
The train/test generators share a vocabulary and may differ in example seeds.
