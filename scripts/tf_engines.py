"""Manually compare basic single-prompt and batched Transformers generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion = output[0, inputs.input_ids.shape[1] :]
    print("Single prompt:")
    print(tokenizer.decode(completion, skip_special_tokens=True))

    prompts = ["The capital of France is", "2 + 2 equals"]
    batch = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **batch,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_width = batch.input_ids.shape[1]
    print("\nLeft-padded batch:")
    for prompt, output in zip(prompts, outputs, strict=True):
        completion = output[prompt_width:]
        print(f"{prompt!r} -> {tokenizer.decode(completion, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
