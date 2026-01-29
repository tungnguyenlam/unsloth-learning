"""
Shared configuration for all training steps.
"""

import os
import json
import argparse

# Model Configuration
MODEL_NAME = "unsloth/gemma-3-4b-it"
MAX_SEQ_LENGTH_DEFAULT = 2048  # Fallback if auto-detect not run
LOAD_IN_4BIT = True  # QLoRA: Load base model in 4-bit quantization

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training Configuration
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 50
MAX_STEPS = None

# QAT Configuration (disabled - using QLoRA instead per assignment requirements)
USE_QAT = False
QAT_SCHEME = "int4"

# Directory Configuration
def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = get_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training_data")
TEST_DATA_DIR = os.path.join(TRAINING_DATA_DIR, "test_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
LORA_MODEL_DIR = os.path.join(PROJECT_ROOT, "lora_model")
MERGED_MODEL_DIR = os.path.join(PROJECT_ROOT, "merged_model")
GGUF_MODEL_DIR = os.path.join(PROJECT_ROOT, "gguf_model")

# Data Paths
TRAINING_DATA_PATH = os.path.join(TRAINING_DATA_DIR, "training_data.jsonl")
TEST1_DATA_PATH = os.path.join(TEST_DATA_DIR, "test1_knowledge_recall.jsonl")
DETECTED_CONFIG_PATH = os.path.join(OUTPUT_DIR, "detected_config.json")

# HuggingFace Configuration
HF_USERNAME = "mainguyenngoc"
HF_TOKEN = ""
HF_MODEL_BASE_NAME = "gemma3-4b-military"

# Ollama Configuration
OLLAMA_MODEL_BASE_NAME = "gemma3-military"

# Weights & Biases Configuration
WANDB_PROJECT = "gemma3-military-finetune"


def generate_run_name(
    learning_rate: float = None,
    epochs: int = None,
    batch_size: int = None,
    grad_accum: int = None,
    lora_r: int = None,
    lora_alpha: int = None,
    target_modules: list = None,
) -> str:
    learning_rate = learning_rate or LEARNING_RATE
    epochs = epochs or NUM_EPOCHS
    batch_size = batch_size or BATCH_SIZE
    grad_accum = grad_accum or GRADIENT_ACCUMULATION_STEPS
    lora_r = lora_r or LORA_R
    lora_alpha = lora_alpha or LORA_ALPHA
    target_modules = target_modules or TARGET_MODULES
    
    lr_str = f"{learning_rate:.0e}".replace("e-0", "e-").replace("e-", "e")
    num_modules = len(target_modules)
    
    return f"lr{lr_str}_ep{epochs}_bs{batch_size}x{grad_accum}_r{lora_r}_a{lora_alpha}_m{num_modules}"


def get_hf_model_name(
    learning_rate: float = None,
    epochs: int = None,
    batch_size: int = None,
    grad_accum: int = None,
    lora_r: int = None,
    lora_alpha: int = None,
    target_modules: list = None,
) -> str:
    run_name = generate_run_name(learning_rate, epochs, batch_size, grad_accum, lora_r, lora_alpha, target_modules)
    return f"{HF_MODEL_BASE_NAME}-{run_name}"


def get_ollama_model_name(
    learning_rate: float = None,
    epochs: int = None,
    batch_size: int = None,
    grad_accum: int = None,
    lora_r: int = None,
    lora_alpha: int = None,
    target_modules: list = None,
) -> str:
    run_name = generate_run_name(learning_rate, epochs, batch_size, grad_accum, lora_r, lora_alpha, target_modules)
    return f"{OLLAMA_MODEL_BASE_NAME}-{run_name}"


def get_wandb_run_name(
    learning_rate: float = None,
    epochs: int = None,
    batch_size: int = None,
    grad_accum: int = None,
    lora_r: int = None,
    lora_alpha: int = None,
    target_modules: list = None,
) -> str:
    return generate_run_name(learning_rate, epochs, batch_size, grad_accum, lora_r, lora_alpha, target_modules)


# Evaluation Configuration
EVAL_SPLIT = 0.05


def ensure_dirs():
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_detected_config(config: dict) -> None:
    ensure_dirs()
    with open(DETECTED_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved detected config: {DETECTED_CONFIG_PATH}")


def load_detected_config() -> dict:
    if os.path.exists(DETECTED_CONFIG_PATH):
        with open(DETECTED_CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}


def get_max_seq_length() -> int:
    config = load_detected_config()
    if "max_seq_length" in config:
        return config["max_seq_length"]
    print(f"  WARNING: No detected max_seq_length, using default {MAX_SEQ_LENGTH_DEFAULT}")
    return MAX_SEQ_LENGTH_DEFAULT


def get_saved_run_name() -> str:
    config = load_detected_config()
    if "run_name" in config:
        return config["run_name"]
    return generate_run_name()


def get_results_dir_for_run(run_name: str = None) -> str:
    """Get results directory organized by run name."""
    if run_name is None:
        run_name = get_saved_run_name()
    run_results_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_results_dir, exist_ok=True)
    return run_results_dir


def round_to_bucket(length: int, buckets: list = None) -> int:
    if buckets is None:
        buckets = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
    for b in buckets:
        if length <= b:
            return b
    return buckets[-1]


def get_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-path", type=str, default=TRAINING_DATA_PATH)
    parser.add_argument("--hf-token", type=str, default=HF_TOKEN)
    parser.add_argument("--hf-username", type=str, default=HF_USERNAME)
    return parser


def print_config():
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Project Root:     {PROJECT_ROOT}")
    print(f"Training Data:    {TRAINING_DATA_PATH}")
    print(f"Test Data:        {TEST1_DATA_PATH}")
    print(f"Results Dir:      {RESULTS_DIR}")
    print(f"LoRA Model Dir:   {LORA_MODEL_DIR}")
    print(f"GGUF Model Dir:   {GGUF_MODEL_DIR}")
    print("=" * 60)
