from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
from trl import SFTTrainer, SFTConfig

# Load the dataset we generated from our ArXiv corpus —
# 3350 question-answer pairs in JSONL format
dataset = load_dataset("json", data_files="finetune/dataset.jsonl", split="train")

# 4-bit quantization config — loads model weights in 4-bit precision
# instead of the default 32-bit. Reduces VRAM from ~15GB to ~2GB.
# nf4 = "normal float 4" — better quality than standard int4 for LLMs.
# compute_dtype=float16 means actual computations run in 16-bit for speed.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load Phi-3-mini with quantization applied.
# device_map="auto" lets accelerate decide which layers go on GPU vs CPU —
# with 12.9GB VRAM and a quantized model, everything fits on GPU.
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb_config,
    device_map="auto"
)


# Tokenizer converts text to token IDs the model understands.
# Must match the model — Phi-3's tokenizer knows its own vocabulary.
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# LoRA config — instead of updating all 3.8B parameters, we add small
# trainable matrices (rank r=16) alongside the frozen attention layers.
# r=16: rank of the LoRA matrices — higher = more capacity, more VRAM.
# lora_alpha=32: scaling factor — controls how much LoRA weights influence output.
# target_modules: which layers to apply LoRA to — q_proj and v_proj are the
# query and value projection matrices in attention, where style is learned.
# lora_dropout=0.05: randomly zeros 5% of LoRA weights during training
# to prevent overfitting on our 3350 examples.
# task_type="CAUSAL_LM": tells PEFT this is a text generation task.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj"],  # Phi-3 specific
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# Wrap the model with LoRA — freezes original weights and adds
# the small trainable matrices. Only LoRA parameters will be updated.
model = get_peft_model(model, lora_config)

# Print how many parameters are actually trainable vs frozen —
# should be a small fraction of total (e.g. 0.5% of 3.8B params)
model.print_trainable_parameters()

# Format each training example as a prompt the model will learn from.
# The model sees "Question: ... Answer: ..." and learns to complete
# the answer in academic style given a question.
def format_example(example):
    return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}"}

dataset = dataset.map(format_example)

# SFTTrainer = Supervised Fine-Tuning Trainer from trl.
# It handles the training loop, gradient updates, and checkpointing.
# num_train_epochs=3: pass through the full dataset 3 times.
# per_device_train_batch_size=4: process 4 examples at once per GPU.
# save_steps=100: save a checkpoint every 100 training steps.
# logging_steps=10: print loss every 10 steps so you can watch progress.
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="finetune/output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=100,
        logging_steps=10
    )
)

# Start training — this will take 30-60 minutes on your 3060.
# Watch the loss — it should decrease steadily from ~2.0 toward ~0.5.
trainer.train()

# Save only the LoRA adapter weights — not the full model.
# The adapter is small (~50MB) vs the full model (~2GB).
# At inference time you load the base model + adapter separately.
model.save_pretrained("finetune/output/final")
tokenizer.save_pretrained("finetune/output/final")