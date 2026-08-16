import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Same quantization config as training — must match exactly.
# Loading the model in a different precision than it was trained with
# would produce garbage output.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the original Phi-3-mini base model with quantization.
# This is the frozen foundation — our LoRA adapter sits on top of it.
# We always need the base model at inference time, not just the adapter.
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load tokenizer from our saved adapter folder — it's the same as Phi-3's
# tokenizer but saved alongside the adapter for convenience.
tokenizer = AutoTokenizer.from_pretrained("finetune/output/final")

# Load the LoRA adapter and merge it with the base model.
# PeftModel wraps the base model and applies the adapter weights
# on top of the frozen base weights during forward passes.
# This is why we save only the adapter (~50MB) not the full model (~2GB) —
# we reconstruct the fine-tuned model at runtime by combining both.
model = PeftModel.from_pretrained(base_model, "finetune/output/final")

# eval() disables dropout and other training-only behaviors.
# Always call this before inference — not calling it gives inconsistent results.
model.eval()

def generate(question: str) -> str:
    # Format exactly as training — the model learned to complete this pattern.
    # Using a different format at inference than training causes style drift.
    prompt = f"Question: {question}\nAnswer:"
    
    # Tokenize and move to GPU — model lives on CUDA, inputs must too.
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        # no_grad() tells PyTorch not to track gradients —
        # saves memory and speeds up inference since we're not training.
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,    # maximum tokens to generate
            temperature=0.7,       # controls randomness — 0=deterministic, 1=creative
            do_sample=True,        # enables temperature sampling — needed for temperature to work
            pad_token_id=tokenizer.eos_token_id  # tells the model what token means "stop"
        )

    # decode converts token IDs back to human-readable text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # the output includes the prompt — strip it to return only the generated answer
    return response[len(prompt):].strip()


if __name__ == "__main__":
    questions = [
        "What is retrieval augmented generation?",
        "How does attention work in transformers?",
        "What are the limitations of large language models?"
    ]
    
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {generate(q)}")
        print("---")