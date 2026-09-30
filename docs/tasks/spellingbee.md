# SpellingBee

`SpellingBee(size=1000, split="test")` generates letter-counting conversations.
The implementation follows the existing [nanochat task](https://github.com/karpathy/nanochat/blob/a5a3a33c63f996c56c0976d45ea47239b3fd74ae/tasks/spellingbee.py)
and uses [dwyl's word list](https://github.com/dwyl/english-words/blob/master/words_alpha.txt).

Examples are deterministic by physical index, with separate train/test random
seeds. These are different generated examples, not disjoint word vocabularies.
Empty lines are ignored and an empty word list fails immediately.

Use `evaluate_task(task, engine, system_prompt="End with #### followed by the count.")`.
Only the question is sent to the engine. The structured reference contains the
count and calculator demonstration; it is never executed by this evaluation
path. A missing model answer scores zero and a malformed reference raises.
Tests use a local synthetic word list and mocked downloads.
