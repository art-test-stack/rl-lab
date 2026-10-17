# SimpleSpelling vector rewards

`rl_lab.tasks.spelling_rewards` provides a task-specific scorer compatible
with the `(m, k)` reward-matrix contract in the
[VPO repository's GRPO adapter example](https://github.com/ryanboldi/vpo#using-vpo-outside-verl-trl-openrlhf-your-own-loop).
One row is one candidate solution within a rollout. Columns follow
`REWARD_NAMES`: `exact`, `format`, `letters`, `position`, `order`, `tool_call`.

The scorer itself neither changes training nor sums the criteria. The existing
SimpleSpelling exact-match reward and default GRPO recipe remain unchanged.
The opt-in full VPO training mode described below connects this scorer to TRL.

## Scores

All scores are in `[0, 1]`. Let `n` be the target-word length and `m` the
predicted letter-sequence length.

| Criterion | Definition |
| --- | --- |
| Exact | Existing `word:w,o,r,d` target, ignoring surrounding whitespace only. |
| Format | Correct source-word prefix and colon, followed by single ASCII letters separated by commas without spaces. Letter correctness and length are scored separately. |
| Letters | Multiset F1: twice the number of shared letters divided by `n + m`. Repeated letters count only up to their target frequency. |
| Position | Number of matching letters at the same index, divided by `max(n, m)`. |
| Order | Longest common subsequence length, divided by `max(n, m)`. Missing letters and insertions are allowed in the subsequence but reduce the score. |
| Tool call | One point for at least one native assistant `python` call with a nonempty string `code` argument. No additional credit for repeated calls; execution failure does not remove calling credit. |

Case is preserved. Extra letters are penalized by the denominators rather
than being ignored. This scorer supports nonempty ASCII alphabetic target
words, as used by SimpleSpelling; invalid targets raise an error.

For partial credit, the scorer accepts a bare alphabetic string, optionally
comma-separated, or that sequence after a colon. Spaces/tabs around comma
chunks are accepted for letter credit but can fail format credit. The source
prefix is never included in letter metrics. Prose, code, empty answers,
empty comma chunks, and payloads with additional colons receive zero spelling scores.
It does not search explanations or tool output for a hidden correct answer.
Pass final assistant answer text separately from optional message history.
Code written as text, calls to other tools, and empty/malformed arguments do
not earn tool credit. Native arguments must be parsed dictionaries, as returned
by the engine, rather than raw JSON strings. The scorer does not execute code
or inspect success flags. Tool credit is independent of final-answer quality:
a Python call followed by an invalid answer can earn `[0, 0, 0, 0, 0, 1]`.

Use generated messages where available. Full histories are also accepted:
messages through the latest user turn are ignored for tool credit, excluding
earlier worked examples. No supplied history means zero tool credit.

For example, bare `cat` without a call earns `[0, 0, 1, 1, 1, 0]`, while `cat:d,o,g` earns
`[0, 1, 0, 0, 0, 0]`. Full letter credit is intentionally distinct from solving
the exact-format task. A policy can maximize letter criteria by copying the
word without learning the required format; exact accuracy must remain a
separate success metric.

## VPO matrix interface

```python
from rl_lab.tasks.spelling_rewards import REWARD_NAMES, score_to_matrix

sub_scores = score_to_matrix("cat", ["cat:c,a,t", "cat:t,a,c"])
# [[1, 1, 1, 1, 1, 0], [0, 1, 1, 1/3, 1/3, 0]]

generated = [{"role": "assistant", "tool_calls": [{
    "type": "function", "function": {
        "name": "python", "arguments": {"code": "print('cat:c,a,t')"},
    },
}]}]
sub_scores = score_to_matrix("cat", ["cat:c,a,t"], histories=[generated])
# [[1, 1, 1, 1, 1, 1]]
```

`score_spelling(word, answer, messages=history)` returns a six-component vector.
`score_to_matrix(word, solutions, histories=histories)` returns one vector per supplied solution,
and returns `[]` when no candidate solutions were parsed. The target word
must come from the dataset, not the model's predicted prefix. Histories must
align one-to-one with solutions; a length mismatch raises `ValueError`.

In the upstream `VPOTrainer.score_to_matrix` sketch, this function supplies
the task-specific scoring after candidate parsing. A batch passed to
`vpo_advantage` contains one such matrix per rollout and prompt-group IDs.
The scorer does not normalize advantages; the upstream estimator already
does that.

Scoring each existing single-answer rollout gives a `(1, 6)` matrix. Full
VPO additionally needs multiple candidates within each rollout and a set-level
reward using shared sampled criterion weights across the prompt group.
Independent GRPO completions are not automatically a multi-answer VPO
rollout. The initial scorer step did not implement candidate parsing or trainer
integration; the later full VPO mode does.

## Cached-answer analysis

The initial five spelling criteria were checked on 2026-10-16 without loading
a model or optimizing weights. These cached spelling examples had no generated
Python calls; the added calling reward does not change their scores:

| Cached source | Answers | Exact correct | Nonzero letter credit |
| --- | --- | --- | --- |
| Original Qwen, short comparison prompt | 16 | 0 | 0 |
| Mixed SFT, short comparison prompt | 16 | 0 | 0 |
| Mixed SFT, brief GRPO prompt, greedy evaluation | 16 | 0 | 7 |
| Retained GRPO rollout group | 4 | 0 | 3 |

Only the final group of four sampled GRPO completions was retained in its
Parquet log; this analysis does not reconstruct the other twelve samples.
The original scalar rewards for all four retained answers were zero.

| Candidate for `pinewood` | Exact | Format | Letters | Position | Order |
| --- | --- | --- | --- | --- | --- |
| `pinewood, , :wpinoewood` | 0 | 0 | 0.889 | 0 | 0.800 |
| `pinewood, :pinewood,colon, pinewood` | 0 | 0 | 0.552 | 0.381 | 0.381 |
| `pninewood` | 0 | 0 | 0.941 | 0.222 | 0.889 |
| Multiline `spelling: ... wheelhouse` answer | 0 | 0 | 0 | 0 | 0 |

The vector distinguishes partial mistakes, demonstrating potential reward
variation where exact match had none. It does not establish training progress
or VPO's advantage over a scalar partial-credit baseline. Format and exact
components are still constant zero in these sampled answers.

The temporary analysis script and full rescored answers are under
`.tmp/codex/spelling-vector-scorer/`: `rescore.py`, `rescored.jsonl`, and
`summary.json`.

Validation: `uv run --locked pytest tests/tasks/test_spelling_rewards.py
tests/tasks/test_spellingbee.py` passed all 24 focused CPU checks on 2026-10-17, covering
the matrix contract, duplicate/extra letters, order versus position, malformed
answers, tool-call eligibility, failed and repeated calls, prompt demonstrations,
history alignment, and existing task behavior. No training or trainer integration
was run in this step.

## Offline VPO mock

A subsequent offline mock used the unmodified upstream `vpo_advantage` from
commit `0436e9303ef20e97a61110450ca7d82ac9d8bc55`. Each mocked rollout
contained two candidate answers, each with six scores. The estimator used
1,024 shared Dirichlet(1) weight draws per prompt group with seed 42.
The downloaded source and its license remain under `.tmp/codex/vpo-mock-run/`;
SciPy was provided through `uv run --with scipy`, without changing project
dependencies.

All candidates in the first group had zero exact-match reward:

| Mocked candidate set for `cat` | Set reward | Advantage |
| --- | --- | --- |
| Prose instead of a spelling answer | 0.0000 | -1.3171 |
| Formatted answers with wrong letters | 0.1641 | -0.6176 |
| Formatted answers with missing letters | 0.5175 | +0.8884 |
| Formatted answers with an extra letter | 0.5545 | +1.0462 |

Additional checks confirmed that a set combining a format specialist and a
letter specialist outscored either duplicated specialist set (0.5182 versus
0.1641 and 0.4966). Identical candidate sets received zero advantage. An
independent calculation matched upstream's sampled set rewards and normalized
advantages. A toy categorical policy over the four mocked rollouts produced
a finite gradient with norm 0.5 and the expected improvement direction;
no optimizer step or model weight update was performed.

With the final answer fixed to `cat:c,a`, a native call raised the set reward
from 0.5175 to 0.6883. A group containing no call, a native call, repeated calls,
and a failed execution assigned advantages -1.7320, +0.5773, +0.5773, and
+0.5773 respectively. This checks that attempts earn credit and repetitions
do not amplify it. It does not establish that the tool code is useful or correct.

Repeat the mock with:

```sh
uv run --locked --with scipy python .tmp/codex/vpo-mock-run/run.py
```

Detailed candidates, score matrices, advantages, diagnostics, and source hash
are recorded in `.tmp/codex/vpo-mock-run/results.json`. These checks exercise
the actual upstream estimator on mocked candidate strings. They do not test
LLM generation, multi-answer parsing, tool execution, or TRL integration,
and do not demonstrate improved spelling accuracy.

## Full VPO training mode

`scripts/train/grpo_simple_spelling.py --reward-mode vpo` uses
`rl_lab.tasks.spelling_vpo` to train two candidates within each rollout.
The system prompt gains instructions to put candidates in `<answer_1>` and
`<answer_2>` blocks in the final assistant reply. Missing or duplicated blocks
become empty candidate slots, with zero spelling credit. Genuine native calls
can still earn the rollout-wide tool-call criterion. There is no fallback
that mines prose or code for a correct spelling.

For each prompt group, the callback draws 1,024 Dirichlet(1) weight vectors,
shared by all rollouts in that group. Each rollout's set reward is the mean,
over those weights, of its best candidate's weighted score. Set rewards are
normalized within the group with population standard deviation and epsilon
`1e-6`, matching the pinned upstream estimator. Identical set rewards give
zero advantage.

The callback supplies those normalized advantages through TRL's reward
interface, using `--scale-rewards none`. TRL then only subtracts their already
zero group mean. Additional reward scaling or combining other reward callbacks
would change the algorithm and is disallowed by this mode. The adapter's
advantages matched upstream exactly on controlled inputs; TRL's residual
recentering changed them by at most `5.96e-8`.

TRL's ordinary `reward` log therefore describes normalized advantages, not
raw set quality. Inspect `vpo/set_reward`, `vpo/group_std`,
`vpo/parsed_fraction`, and the per-criterion `vpo/*` metrics. Criterion metrics
report each rollout's best score across its two candidates. Every training
rollout is saved in `vpo-rollouts.jsonl`, including generated messages,
candidates, six-component matrices, set reward, and normalized advantage.

VPO uses the existing single-device/tool-enabled loop; this implementation
does not add veRL, SciPy, or another trainer dependency. The existing worked
example is single-answer, so VPO requires `--use-worked-example false`.
Greedy before/after evaluation reports best-of-two exact spelling accuracy
and retains both candidates in the saved conversation records. This differs
from the earlier single-answer evaluation protocol.

Example one-update pilot (choose a fresh output directory):

```sh
HF_DEACTIVATE_ASYNC_LOAD=1 uv run --locked --group train \
  python scripts/train/grpo_simple_spelling.py \
  --model-name-or-path .tmp/codex/qwen-mixed-sft/run-adamw-bf16 \
  --model-init-kwargs '{"dtype":"bfloat16"}' \
  --reward-mode vpo --optimizer-config '' \
  --system-prompt 'Extract every character of the supplied word without correcting it. Use Python if useful.' \
  --use-worked-example false --train-limit 8 --eval-limit 4 --max-steps 1 \
  --learning-rate 0.00001 --max-grad-norm 1 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
  --generation-batch-size 4 --num-generations 4 \
  --max-completion-length 512 --log-completions true --save-strategy no \
  --run-name spelling-vpo-repeat \
  --output-dir .tmp/codex/vpo-repeat
```

This consumes two prompts, with four rollouts per prompt and two candidate
slots per rollout: eight rollouts and sixteen candidate slots for one update.
Generation processes four rollouts at a time; training uses one sequence per
microbatch. AdamW/bfloat16 avoids the known compiled Muon path. The pilot
artifacts and upstream comparison are under `.tmp/codex/vpo-real-pilot/`.

### Pilot result: 2026-10-17

The real MPS run completed one optimizer step in 36.55 seconds of trainer
runtime, plus before/after generation and checkpoint saving. It sampled eight
rollouts, representing sixteen candidate slots. None emitted a recognized
candidate block or made a native tool call. All six reward components, set
rewards, advantages, and the gradient norm were zero. One of the eight rollouts
hit the completion limit (`completions/clipped_ratio=0.125`).

For `baggers`, one sampled reply was:

```text
The original word is "baggers". Two candidate spelling answers are:
- Barkers
- Barbies
```

For `rimfire`, two rollouts returned only `rimfire`; another returned
`rimfire: correct` followed by `ripfire: corrected: rimfire`.
These failed both the requested extraction behavior and the candidate-block
protocol. The set scorer therefore had no parsed letter sequences from which
to award partial credit. It did not silently reinterpret the replies or change
the parsing rules after observing the failure.

Greedy best-of-two exact accuracy stayed at 0/4 before and after. All 290 saved
weight tensors were finite and exactly unchanged from the SFT checkpoint.
The run exercised the full VPO training connection, but made no parameter
update because it produced no reward variation.

Validation included 24 focused CPU checks (scorer, candidate parsing, VPO
signal/degenerate groups, tool credit without candidate blocks, and the existing
GRPO integration check). The controlled callback comparison with upstream had
zero maximum absolute advantage error before TRL recentering. These checks do
not establish nonzero-gradient learning in the real model: that still needs
samples satisfying enough of the output protocol to earn differing rewards.

Full training trajectories are in `run/vpo-rollouts.jsonl`; before/after
histories, per-run settings, checkpoint, and trainer metrics are alongside
them. `summary.json` and `weight-comparison.json` record the audit results.
Before increasing the training budget, the next decision is whether to teach
the two-answer format or simplify its representation.
