"""
inference.py

Loads the base model together with a trained LoRA adapter and generates
a response to a prompt, for quick manual testing of the fine-tuned model.

Usage:
    python training/inference.py
    python training/inference.py --prompt "Write a follow-up email to ABC Dispatch."
    python training/inference.py --system "You are a professional business correspondence assistant."
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a trained LoRA adapter")
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--system", type=str, default=config.DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Write a short cold outreach email to a truck dispatching "
            "company named ABC Dispatch. We provide reliable truck "
            "dispatching services. The goal is to start a conversation, "
            "not aggressively sell."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=config.TOP_P)
    parser.add_argument(
        "--repetition-penalty", type=float, default=config.REPETITION_PENALTY
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not args.adapter_path.exists():
        raise FileNotFoundError(
            f"No trained adapter found at {args.adapter_path}. Run "
            "training/train_qlora.py first, or pass --adapter-path."
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            repetition_penalty=args.repetition_penalty,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("GENERATED RESPONSE")
    print("=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    main()
