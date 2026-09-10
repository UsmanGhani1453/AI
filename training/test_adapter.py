import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


MODEL_PATH = r"C:\AI\models\Qwen3-4B-Instruct-2507"
ADAPTER_PATH = r"C:\AI\training\output"


# --------------------------------------------------
# 1. QLoRA configuration
# --------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    local_files_only=True,
)


# --------------------------------------------------
# 3. Load base model
# --------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)


# --------------------------------------------------
# 4. Load trained LoRA adapter
# --------------------------------------------------

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
)

model.eval()


# --------------------------------------------------
# 5. Test prompt
# --------------------------------------------------

messages = [
    {
        "role": "system",
        "content": (
            "You are a professional cold outreach email writer "
            "for a truck dispatching service."
        ),
    },
    {
        "role": "user",
        "content": (
            "Write a short cold outreach email to a truck dispatching "
            "company named ABC Dispatch. We provide reliable truck "
            "dispatching services. The goal is to start a conversation, "
            "not aggressively sell."
        ),
    },
]


prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)


# --------------------------------------------------
# 6. Generate
# --------------------------------------------------

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.device)


with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
    )


# --------------------------------------------------
# 7. Print result
# --------------------------------------------------

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print("\n" + "=" * 60)
print("GENERATED EMAIL")
print("=" * 60)
print(response)
print("=" * 60)