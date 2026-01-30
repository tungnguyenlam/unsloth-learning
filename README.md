# Unsloth Fine-Tuning Pipeline: Military Vocabulary

This project fine-tunes **Gemma-3-4B-IT** (or other LLMs) on a custom military vocabulary dataset using **Unsloth** and **QLoRA**. It automates the entire process from data generation to GGUF export and HuggingFace upload.

## 🚀 Optimized Workflow (Vast.ai / RTX 4090 / 5090)

The pipeline is now streamlined into 5 automated steps.

### Prerequisites

1.  **HuggingFace Token**: You need a WRITE token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2.  **GPU**: Recommended 24GB+ VRAM (RTX 3090/4090/A10G).

### Step-by-Step Guide

#### 1. Prepare Data
Generates 140k+ synthetic Q&A pairs from `data/dataset.csv`.
```bash
python src/data_processing/02_generate_training_data.py
```

#### 2. Train Model
Fine-tunes the model using LoRA.
- Automatically pushes the **merged FP16 model** to HuggingFace at the end.
- Use `--hf-token` to enable the push.
```bash
python src/training/step_02_train.py --hf-token hf_YourTokenHere
```

#### 3. Test Model (FP16)
Downloads the model from HuggingFace and runs comprehensive tests (Knowledge Recall & Stability).
- Uses standard Transformers batch inference (fast & reliable).
```bash
python src/training/step_03_test_fp16.py
```

#### 4. Export to GGUF
Converts the model to GGUF format (q4_k_m) for use with Ollama/Llama.cpp.
- Automatically pushes the **GGUF model** to HuggingFace.
```bash
python src/training/step_04_export_gguf.py --hf-token hf_YourTokenHere
```

#### 5. Verify GGUF
Compares the GGUF model against the FP16 model to ensure quality retention.
```bash
python src/training/step_05_test_gguf.py --model models/gguf/your_model.gguf
```

---

## 🛠️ Configuration

Edit `src/training/step_00_config.py` to change:
- `MODEL_NAME`: Base model (default: `unsloth/gemma-3-4b-it`)
- `LORA_R`, `LORA_ALPHA`: LoRA parameters
- `LEARNING_RATE`, `NUM_EPOCHS`: Training hyperparameters

## 📊 Testing Suite

The project includes a robust testing suite in `src/testing/`:
- **Test 1**: Knowledge Recall (Can it define military terms?)
- **Test 2**: Stability Check (Did it forget general Korean?)
- **Test 3**: Quantization Quality (Is GGUF as good as FP16?)

See `docs/testing.md` for details.

## 📦 Using with Ollama

After Step 4, you can run the model locally:

```bash
# Pull from HuggingFace (if needed) or use local file
ollama create my-military-model -f models/gguf/Modelfile
ollama run my-military-model
```
