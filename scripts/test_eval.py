import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "trl-lib/ultrafeedback_binarized"
NUM_EXAMPLES = 16


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print('tokenizer.pad_token_id', tokenizer.pad_token_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
)
w
from rl_lab.distributed import init_distributed


dist = init_distributed()

model.to(dist.device)

print(dist.device)
print(dist.rank)
print(dist.world_size)
print(dist.is_distributed)