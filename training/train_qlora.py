"""
train_qlora.py

Fine-tunes a local causal language model with QLoRA (4-bit quantized
base model + LoRA adapters) on the business-correspondence dataset in
``training/data/``.

Unlike earlier drafts of this script, all paths and hyperparameters
are read from ``config.py`` / environment variables rather than being
hardcoded to a specific machine, and the run is evaluated against the
held-out validation set at the end of every epoch.

Usage:
    python training/train_qlora.py
    python training/train_qlora.py --epochs 5 --lr 1e-4
"""

import argparse
import sys
from pathlib import Path

# Allow running this script directly (``python training/train_qlora.py``)
# as well as as a module, by making the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.MODEL_PATH,
        help="Path to the local base model (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR,
        help="Directory to save the trained LoRA adapter (default: %(default)s)",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=config.TRAIN_FILE,
        help="Path to the training JSONL file (default: %(default)s)",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=config.VALIDATION_FILE,
        help="Path to the validation JSONL file (default: %(default)s)",
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_TRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--batch-size", type=int, default=config.PER_DEVICE_TRAIN_BATCH_SIZE
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=config.GRADIENT_ACCUMULATION_STEPS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Imported lazily so `--help` works without the ML stack installed.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. QLoRA fine-tuning in this script "
            "requires a CUDA-capable GPU."
        )

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Base model not found at {args.model_path}. Download it "
            "first (see the README's 'Model' section) or pass "
            "--model-path / set the MODEL_PATH environment variable."
        )

    if not args.train_file.exists():
        raise FileNotFoundError(
            f"Training data not found at {args.train_file}. Run "
            "`python training/build_dataset.py` first."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    # ----------------------------------------------------------
    # Tokenizer
    # ----------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------------------------------------
    # 4-bit quantized base model
    # ----------------------------------------------------------

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False

    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------

    data_files = {"train": str(args.train_file)}
    if args.validation_file.exists():
        data_files["validation"] = str(args.validation_file)

    dataset = load_dataset("json", data_files=data_files)

    print("Training examples:", len(dataset["train"]))
    if "validation" in dataset:
        print("Validation examples:", len(dataset["validation"]))

    # ----------------------------------------------------------
    # LoRA configuration
    # ----------------------------------------------------------

    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.LORA_TARGET_MODULES,
    )

    # ----------------------------------------------------------
    # Training configuration
    # ----------------------------------------------------------

    has_eval = "validation" in dataset

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        fp16=False,
        bf16=False,
        tf32=False,
        max_grad_norm=config.MAX_GRAD_NORM,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch" if has_eval else "no",
        # With a small dataset, the lowest-loss checkpoint is often not
        # the final epoch. Track validation loss and automatically keep
        # the best-performing checkpoint rather than just the last one.
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\nStarting QLoRA training...\n")
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print("\nTraining complete.")
    print("Adapter saved to:", args.output_dir)


if __name__ == "__main__":
    main()
