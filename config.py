"""
config.py

Central configuration for the project. All paths and hyperparameters
that scripts previously hardcoded (e.g. ``C:\\AI\\models\\...``) live
here instead, and can be overridden with environment variables or a
local ``.env`` file (see ``.env.example``).

This makes the project portable across Windows, macOS, and Linux, and
keeps machine-specific paths out of version control.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; environment variables still work
    # without it if they are exported by the shell / OS instead.
    pass


PROJECT_ROOT = Path(__file__).resolve().parent


def _path_env(var_name: str, default: Path) -> Path:
    """Read a path from the environment, falling back to a default."""
    value = os.environ.get(var_name)
    return Path(value).expanduser().resolve() if value else default


# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------

# Local path to the base model (downloaded separately, see README).
MODEL_PATH = _path_env(
    "MODEL_PATH",
    PROJECT_ROOT / "models" / "Qwen3-4B-Instruct-2507",
)

# Where the trained LoRA adapter is written to / loaded from.
OUTPUT_DIR = _path_env(
    "OUTPUT_DIR",
    PROJECT_ROOT / "training" / "output",
)

# Where the adapter is merged into the base model for deployment
# (e.g. before converting to GGUF for Ollama).
MERGED_MODEL_DIR = _path_env(
    "MERGED_MODEL_DIR",
    PROJECT_ROOT / "models" / "merged",
)

DATA_DIR = PROJECT_ROOT / "training" / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALIDATION_FILE = DATA_DIR / "validation.jsonl"

# --------------------------------------------------------------
# QLoRA / training hyperparameters
# --------------------------------------------------------------

LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 32))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.05))
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

NUM_TRAIN_EPOCHS = int(os.environ.get("NUM_TRAIN_EPOCHS", 4))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 1))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-4))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 1.0))

# --------------------------------------------------------------
# Generation defaults (used by training/inference.py)
# --------------------------------------------------------------

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 220))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))
TOP_P = float(os.environ.get("TOP_P", 0.9))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", 1.1))

DEFAULT_SYSTEM_PROMPT = (
    "You are a professional cold outreach email writer for a truck "
    "dispatching service."
)
