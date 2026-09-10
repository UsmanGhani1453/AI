import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


MODEL_PATH = r"C:\AI\models\Qwen3-4B-Instruct-2507"
DATA_PATH = r"C:\AI\training\data\train.jsonl"
OUTPUT_DIR = r"C:\AI\training\output"


# --------------------------------------------------
# 1. Check GPU
# --------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)


# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --------------------------------------------------
# 3. 4-bit quantization
# --------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# --------------------------------------------------
# 4. Load model
# --------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)

model.config.use_cache = False


# --------------------------------------------------
# 5. Load dataset
# --------------------------------------------------

dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)

print("Training examples:", len(dataset))


# --------------------------------------------------
# 6. LoRA configuration
# --------------------------------------------------

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


# --------------------------------------------------
# 7. Training configuration
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    fp16=False,
    bf16=False,
    tf32=False,

    max_grad_norm=1.0,

    logging_steps=1,

    save_strategy="epoch",

    optim="paged_adamw_8bit",

    gradient_checkpointing=True,

    report_to="none",

    remove_unused_columns=False,
)

# --------------------------------------------------
# 8. Trainer
# --------------------------------------------------

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)


# --------------------------------------------------
# 9. Train
# --------------------------------------------------

print("\nStarting QLoRA training...\n")

trainer.train()


# --------------------------------------------------
# 10. Save LoRA adapter
# --------------------------------------------------

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining complete.")
print("Adapter saved to:", OUTPUT_DIR)