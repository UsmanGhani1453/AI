"""
merge_adapter.py

Merges the trained LoRA adapter into the base model and saves the
result as a standalone Hugging Face model — the format needed before
converting to GGUF for Ollama / llama.cpp.

A merged model is a full-precision (fp16) copy of the base model with
the adapter's weights folded in. It is larger on disk than the adapter
alone, but from here on behaves like any other Hugging Face model: no
PEFT, no 4-bit quantization config, and no dependency on this project
at inference time.

Usage:
    python training/merge_adapter.py
    python training/merge_adapter.py --adapter-path training/output --out models/merged
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a trained LoRA adapter into the base model"
    )
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--out", type=Path, default=config.MERGED_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.adapter_path.exists():
        raise FileNotFoundError(
            f"No trained adapter found at {args.adapter_path}. Run "
            "training/train_qlora.py first."
        )

    print(f"Loading base model from {args.model_path} (fp16, full precision)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        device_map="cpu",
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        local_files_only=True,
    )

    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))

    print("Merging adapter weights into the base model...")
    merged_model = model.merge_and_unload()

    if args.out.exists():
        print(f"Removing existing directory at {args.out}...")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {args.out}...")
    merged_model.save_pretrained(str(args.out), safe_serialization=True)
    tokenizer.save_pretrained(str(args.out))

    print("\nMerge complete.")
    print("Merged model saved to:", args.out)
    print(
        "\nNext step: convert this merged model to GGUF for Ollama. "
        "See deployment/README.md."
    )


if __name__ == "__main__":
    main()
