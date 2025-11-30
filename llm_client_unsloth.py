from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True

# Load model + tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model)

def generate_llm_answer(prompt: str, max_new_tokens: int = 256, temperature: float = 0.2):
    """Generate text from Llama using unsloth in a clean reusable form."""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract generated-only part
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = out[0][prompt_len:]
    generated_only = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_only.strip()
