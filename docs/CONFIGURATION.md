# Configuration & CLI Reference

## Core Config: `src/training/step_00_config.py`

This file contains the shared hyperparameter defaults for the entire pipeline. Values here can be overridden by CLI arguments.

| Variable | Default (Example) | Description |
|----------|-------------------|-------------|
| `MODEL_NAME` | `unsloth/gemma-3-4b-it` | Base model to fine-tune. |
| `MAX_SEQ_LENGTH_DEFAULT` | `2048` | Fallback context length if auto-detect fails. |
| `LORA_R` | `64` | LoRA Rank. Higher means more learnable parameters. |
| `LORA_ALPHA` | `128` | LoRA Alpha. Usually `2 * Rank`. |
| `LEARNING_RATE` | `2e-4` | Learning rate for generic fine-tuning. |
| `NUM_EPOCHS` | `1` | Number of training epochs. |
| `BATCH_SIZE` | `16` | Per-device batch size. |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | Steps to accumulate gradients before update. |

---

## CLI Argument Reference

### Step 1: Prepare Data
`src/training/step_01_prepare_data.py`

Analyzes your data, splits it into train/test, and auto-detects optimal sequence length.

| Flag | Description |
|------|-------------|
| `--test-ratio` | Ratio of data to split for testing (default: `0.05`). |
| `--skip-analysis` | Skip token length analysis (faster, uses defaults). |
| `--max-seq-length` | Manually force a max sequence length (e.g. `4096`). |

### Step 2: Training
`src/training/step_02_train.py`

| Flag | Description |
|------|-------------|
| `--hf-token` | HuggingFace token for pushing models (Required for push). |
| `--push-checkpoints` | Push intermediate checkpoints (every epoch/500 steps) to HF. |
| `--model-name` | Override the base model defined in config. |
| `--load-in-4bit` | Load base model in 4-bit (default: `True`). |
| `--no-grad-ckpt` | Disable gradient checkpointing (Faster, uses more VRAM). |
| `--no-wandb` | Disable Weights & Biases logging. |
| `--qat` / `--no-qat` | Enable/Disable Quantization Aware Training (Experimental). |

### Step 3: Test FP16
`src/training/step_03_test_fp16.py`

| Flag | Description |
|------|-------------|
| `--hf-model` | Test a specific HF model instead of the local one. |
| `--revision` | Git commit hash to test (for intermediate checkpoints). |
| `--batch-size` | Batch size for inference (default: `16`). |
| `--skip-test1` | Skip Knowledge Recall test. |
| `--skip-test2` | Skip Stability (KoMMLU) test. |
| `--quick` | Run reduced test set (50 samples) for rapid feedback. |

### Step 4: Export GGUF
`src/training/step_04_export_gguf.py`

| Flag | Description |
|------|-------------|
| `--quantization` | Quantization method: `q4_k_m` (default), `q8_0`, `f16`. |
| `--hf-token` | Token for uploading GGUF to Hugging Face. |
| **Recovery Mode:** | |
| `--hf-model` | **Recover from lost instance**: Download adapter from HF to convert. |
| `--run-name` | Manually name the output files/repo (overrides config). |
| `--max-seq-length`| Manually specify context length (critical if config lost). |

### Step 5: Verify GGUF
`src/training/step_05_test_gguf.py`

| Flag | Description |
|------|-------------|
| `--model` | Path to local `.gguf` file to test. |
| `--hf-model` | Download and test a GGUF from Hugging Face. |
| `--skip-test3` | Skip the quantization loss comparison (FP16 vs GGUF). |
