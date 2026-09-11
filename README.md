# Truck Dispatch Correspondence AI

A QLoRA fine-tuning project that teaches a small local language model to
write polished, professional business correspondence for a truck
dispatching service — cold outreach emails, follow-ups, objection
handling, scheduling, account management, and client relationship
messages — in consistent, formal-professional English.

The base model is fine-tuned with **QLoRA** (4-bit quantization + LoRA
adapters), so training runs on a single consumer GPU and produces a
small, portable adapter rather than a full copy of the model.

---

## Contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Dataset](#dataset)
- [Training](#training)
- [Inference / testing](#inference--testing)
- [Configuration](#configuration)
- [Deploying to Ollama](#deploying-to-ollama)
- [License](#license)

---

## How it works

```
                 ┌────────────────────────┐
                 │  build_dataset.py      │
                 │  generates train /     │
                 │  validation JSONL      │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │  train_qlora.py        │
                 │  4-bit base model      │
                 │  + LoRA fine-tuning    │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │  training/output/      │
                 │  trained LoRA adapter  │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │  inference.py          │
                 │  base model + adapter  │
                 │  → generated email     │
                 └────────────────────────┘
```

1. **`build_dataset.py`** deterministically generates a labeled dataset
   of business-correspondence examples in chat format and writes it to
   `training/data/`.
2. **`train_qlora.py`** loads a local base model in 4-bit precision and
   trains a LoRA adapter on that dataset.
3. **`inference.py`** loads the base model together with the trained
   adapter and generates a response to a prompt, for quick manual
   testing.

---

## Project structure

```
truck-dispatch-ai/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example              # template for local configuration
├── .gitignore
├── config.py                 # central, environment-driven configuration
├── deployment/
│   ├── README.md              # merge → GGUF → Ollama walkthrough
│   ├── Modelfile               # Ollama build file
│   ├── convert_to_gguf.ps1     # Windows conversion helper
│   └── convert_to_gguf.sh      # macOS/Linux conversion helper
└── training/
    ├── build_dataset.py      # generates train.jsonl / validation.jsonl
    ├── train_qlora.py        # QLoRA fine-tuning entry point
    ├── merge_adapter.py      # merges the adapter into the base model
    ├── inference.py          # load adapter + generate a sample reply
    ├── data/
    │   ├── train.jsonl        # 111 examples
    │   └── validation.jsonl   # 28 examples
    └── output/                # trained adapter is written here (git-ignored)
```

---

## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA support (training only — `build_dataset.py`
  and dataset inspection need no GPU)
- A locally downloaded base instruction-tuned model in Hugging Face
  format (this project was built around **Qwen3-4B-Instruct-2507**, but
  any similarly sized causal LM that supports a chat template will work)

## Setup

1. **Clone and install dependencies**

   ```bash
   git clone https://github.com/UsmanGhani1453/AI.git truck-dispatch-ai
   cd truck-dispatch-ai
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download the base model** to a local folder, e.g.:

   ```
   models/Qwen3-4B-Instruct-2507/
   ```

   (Download it from the Hugging Face Hub with `huggingface-cli download`
   or `git lfs clone`, or point at any local model directory you already
   have.)

3. **Configure paths.** Copy the environment template and adjust it if
   your model lives somewhere else:

   ```bash
   cp .env.example .env
   ```

   By default, `MODEL_PATH` points at `./models/Qwen3-4B-Instruct-2507`
   and `OUTPUT_DIR` at `./training/output`. All paths are resolved
   through `config.py`, so nothing is hardcoded to a specific machine
   or operating system.

---

## Dataset

The dataset lives in `training/data/` as JSONL files, one JSON object
per line, in standard chat format:

```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

| File               | Examples |
|--------------------|---------:|
| `train.jsonl`      | 111      |
| `validation.jsonl` | 28       |

It covers two related writing tasks, each with a dedicated system
prompt:

1. **General business correspondence** — follow-ups, responses to
   interested prospects, objections and declines, scheduling and
   logistics, operational/account correspondence, referrals and
   relationship maintenance, apologies and issue resolution, rate
   confirmations and load tendering, billing and invoicing, compliance
   and safety documentation, driver-facing dispatch messages,
   escalation and service recovery, renewals, holiday/weather notices,
   and short-form (SMS-style) replies.
2. **Cold outreach / sales email writing** — first contact,
   personalization, follow-ups, pricing questions, objection handling,
   tone rewriting, lane-specific and backhaul outreach, and general
   email-quality examples.

Every example is written in formal-professional business English:
correct grammar and punctuation, no slang, no unsupported claims, and
no aggressive sales language. Placeholders such as `{{name}}`,
`{{company}}`, and `{{sender_name}}` are used throughout so the dataset
teaches structure and tone rather than memorizing specific names.

### Regenerating the dataset

The dataset is generated by a single deterministic script rather than
edited by hand, so it can always be reproduced or extended:

```bash
python training/build_dataset.py
```

This writes `training/data/train.jsonl` and
`training/data/validation.jsonl` (80/20 split, fixed random seed). To
add more examples, extend the `examples`, `new_examples`, or
`more_examples` lists inside `build_dataset.py` (or add a new `PART`
section following the same pattern) and re-run it. Remember to raise
`MIN_EXAMPLES` to match your new total, and check for accidental
duplicate examples — the script already de-duplicates automatically and
will tell you the final unique count.

---

## Training

Once the base model is downloaded and `.env` is configured:

```bash
python training/train_qlora.py
```

This will:

- Load the base model in 4-bit (NF4) precision via `bitsandbytes`
- Attach a LoRA adapter (`r=16`, `alpha=32`, dropout `0.05`) to the
  attention and MLP projection layers
- Fine-tune for 4 epochs with gradient accumulation, evaluating against
  the validation set after every epoch
- Automatically track `eval_loss` and reload the **best** checkpoint at
  the end of training (not necessarily the last epoch) — important on a
  dataset this size, where the model can start overfitting before
  training finishes
- Save the resulting adapter and tokenizer to `training/output/`

**On dataset size and epochs:** with well under a thousand examples,
more epochs help only up to a point — watch `eval_loss` in the logs. If
it stops decreasing (or starts climbing) before your last epoch, that's
the model starting to memorize the training set rather than generalize.
`load_best_model_at_end` protects you from walking away with an
overfit checkpoint, but the more reliable fix is adding more varied
examples to `training/data/`, not just more epochs.

All hyperparameters can be overridden via CLI flags or environment
variables — see `python training/train_qlora.py --help` or
`.env.example`. For example:

```bash
python training/train_qlora.py --epochs 5 --lr 1e-4
```

> **Note:** Training requires a CUDA-capable GPU. `train_qlora.py`
> checks for this up front and raises a clear error if none is found,
> rather than failing partway through a run.

---

## Inference / testing

After training completes, generate a sample response with the trained
adapter:

```bash
python training/inference.py
```

Or supply your own prompt and system message:

```bash
python training/inference.py \
  --prompt "Write a follow-up email to a prospect who has gone quiet." \
  --system "You are a professional business correspondence assistant for a truck dispatching service."
```

---

## Configuration

All configurable values are centralized in `config.py` and can be
overridden with environment variables (loaded automatically from a
local `.env` file if present — see `.env.example`):

| Variable               | Default                          | Description                          |
|-------------------------|-----------------------------------|---------------------------------------|
| `MODEL_PATH`            | `./models/Qwen3-4B-Instruct-2507` | Local path to the base model          |
| `OUTPUT_DIR`             | `./training/output`              | Where the trained adapter is saved    |
| `NUM_TRAIN_EPOCHS`       | `3`                               | Training epochs                       |
| `LEARNING_RATE`          | `2e-4`                            | Optimizer learning rate               |
| `TRAIN_BATCH_SIZE`       | `1`                               | Per-device batch size                 |
| `GRAD_ACCUM_STEPS`       | `8`                               | Gradient accumulation steps           |
| `LORA_R` / `LORA_ALPHA` / `LORA_DROPOUT` | `16` / `32` / `0.05` | LoRA hyperparameters |
| `MAX_NEW_TOKENS`         | `220`                             | Max generation length (inference)     |
| `TEMPERATURE` / `TOP_P` / `REPETITION_PENALTY` | `0.7` / `0.9` / `1.1` | Sampling settings (inference) |

No machine-specific paths are hardcoded anywhere in the codebase — the
project runs the same way on Windows, macOS, and Linux.

---

## Deploying to Ollama

Once you have a trained adapter in `training/output/`, you can turn it
into a standalone model that runs with `ollama run`, with no Python
environment needed at runtime:

```bash
python training/merge_adapter.py        # fold the adapter into the base model
bash deployment/convert_to_gguf.sh       # or .ps1 on Windows — convert to GGUF
ollama create truck-dispatch-ai -f deployment/Modelfile
ollama run truck-dispatch-ai
```

See [`deployment/README.md`](deployment/README.md) for the full
walkthrough, including optional quantization and how to switch between
the two trained personas (cold outreach vs. general correspondence).

---

## License

Released under the [MIT License](LICENSE).
