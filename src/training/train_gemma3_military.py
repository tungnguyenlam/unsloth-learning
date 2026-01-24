"""
Military Vocabulary Fine-tuning Script for Gemma-3-4b-it

This script fine-tunes the Gemma-3-4b-it model on military vocabulary data using:
- QLoRA (Quantized Low-Rank Adaptation) with 4-bit quantization
- Quantization Aware Training (QAT) for better post-quantization accuracy
- Gemma 3 chat template format

Workflow:
1. Load model with QLoRA
2. Apply QAT for int4 quantization
3. Prepare training data
4. Train with SFTTrainer
5. Save LoRA adapters
6. Save merged 16-bit model to HuggingFace
7. Export to GGUF (q4_k_m)
8. Upload GGUF to HuggingFace
9. Upload to Ollama Registry (optional)

Author: Generated for NeoALI Assessment
Date: 2026-01-24
"""

import os
import sys
import json
import torch
import subprocess
import time
import argparse
from datasets import Dataset

# ============================================================================
# CONFIGURATION (Can be overridden via command-line arguments)
# ============================================================================

# Model Configuration
MODEL_NAME = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"  # Pre-quantized for faster loading
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA Configuration (Conservative for domain adaptation)
LORA_R = 16  # Rank - controls capacity of LoRA adapters
LORA_ALPHA = 32  # Scaling factor
LORA_DROPOUT = 0  # Optimized for Unsloth
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training Configuration (Conservative to prevent catastrophic forgetting)
LEARNING_RATE = 1e-5  # Very conservative for domain adaptation
NUM_EPOCHS = 1  # Start with 1 epoch to test
BATCH_SIZE = 1  # Use 1 for memory efficiency
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4
WARMUP_STEPS = 50
MAX_STEPS = None  # Set to a number for testing (e.g., 100)

# QAT Configuration
USE_QAT = True  # Enable Quantization Aware Training
QAT_SCHEME = "int4"  # int4 for q4_k_m GGUF export

# Output Configuration
OUTPUT_DIR = "outputs"
LORA_MODEL_DIR = "lora_model"
MERGED_MODEL_DIR = "merged_model"
GGUF_MODEL_DIR = "gguf_model"

# HuggingFace Configuration (SET YOUR VALUES or use --hf-token argument)
HF_USERNAME = "YOUR_HF_USERNAME"  # Change this!
HF_TOKEN = ""  # Get from https://huggingface.co/settings/tokens
HF_MODEL_NAME = "gemma3-4b-military-korean"

# Ollama Configuration
OLLAMA_MODEL_NAME = "gemma3-military"

# Data Configuration - Default path (can be overridden via --data-path)
# Tries to find data relative to script location or current working directory
def get_default_data_path():
    """Find training data file in common locations."""
    possible_paths = [
        # Relative to script location
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "data_cleaned", "training_data.jsonl"),
        # Relative to working directory
        os.path.join(os.getcwd(), "data", "data_cleaned", "training_data.jsonl"),
        # Direct in current directory
        os.path.join(os.getcwd(), "training_data.jsonl"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    return possible_paths[0]  # Return first option as default

DATA_PATH = get_default_data_path()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_training_data(data_path: str) -> Dataset:
    """Load training data from JSONL file."""
    print(f"Loading training data from: {data_path}")
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Convert to HuggingFace Dataset format
    # Our data is already in {"messages": [...]} format
    conversations = [item["messages"] for item in data]
    dataset = Dataset.from_dict({"conversations": conversations})
    
    print(f"Loaded {len(dataset)} training samples")
    return dataset


def apply_chat_template(examples, tokenizer):
    """Apply Gemma 3 chat template to conversations."""
    texts = []
    for convo in examples["conversations"]:
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma-3-4b-it on military vocabulary data"
    )
    parser.add_argument(
        "--data-path", type=str, default=DATA_PATH,
        help="Path to training_data.jsonl file"
    )
    parser.add_argument(
        "--hf-token", type=str, default=HF_TOKEN,
        help="HuggingFace token for uploading models"
    )
    parser.add_argument(
        "--hf-username", type=str, default=HF_USERNAME,
        help="HuggingFace username for uploading models"
    )
    parser.add_argument(
        "--max-steps", type=int, default=MAX_STEPS,
        help="Maximum training steps (for testing). Set to None for full training."
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--no-qat", action="store_true",
        help="Disable Quantization Aware Training"
    )
    parser.add_argument(
        "--skip-gguf", action="store_true",
        help="Skip GGUF export (faster for testing)"
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Skip HuggingFace upload"
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Override globals with arguments
    data_path = args.data_path
    hf_token = args.hf_token
    hf_username = args.hf_username
    max_steps = args.max_steps
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    use_qat = not args.no_qat
    skip_gguf = args.skip_gguf
    skip_upload = args.skip_upload
    
    print("="*60)
    print("Military Vocabulary Fine-tuning for Gemma-3-4b-it")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Data path: {data_path}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Max steps: {max_steps or 'None (full training)'}")
    print(f"  - QAT enabled: {use_qat}")
    
    # Verify data exists
    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: Training data not found at: {data_path}")
        print("Please run 02_generate_training_data.py first or specify --data-path")
        sys.exit(1)
    
    # ========================================================================
    # STEP 1: Load Model and Tokenizer
    # ========================================================================
    print("\n[STEP 1] Loading model and tokenizer...")
    
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=False,
        full_finetuning=False,
    )
    
    # ========================================================================
    # STEP 2: Apply LoRA Adapters with QAT
    # ========================================================================
    print("\n[STEP 2] Applying LoRA adapters...")
    
    lora_config = {
        "r": LORA_R,
        "target_modules": TARGET_MODULES,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "use_rslora": False,
        "loftq_config": None,
    }
    
    if use_qat:
        lora_config["qat_scheme"] = QAT_SCHEME
        print(f"  - Enabling QAT with scheme: {QAT_SCHEME}")
    
    model = FastLanguageModel.get_peft_model(model, **lora_config)
    
    # Verify QAT is applied
    if use_qat:
        for module in model.modules():
            if "FakeQuantized" in module.__class__.__name__:
                print("  ✓ QAT is applied!")
                break
    
    # ========================================================================
    # STEP 3: Prepare Chat Template
    # ========================================================================
    print("\n[STEP 3] Setting up Gemma 3 chat template...")
    
    from unsloth.chat_templates import get_chat_template
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3-it",  # Gemma 3 Instruct template
    )
    
    # ========================================================================
    # STEP 4: Load and Prepare Data
    # ========================================================================
    print("\n[STEP 4] Loading and preparing training data...")
    
    dataset = load_training_data(data_path)
    
    # Convert to Unsloth format
    from unsloth.chat_templates import standardize_data_formats
    dataset = standardize_data_formats(dataset)
    
    # Apply chat template
    dataset = dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        batched=True,
    )
    
    # Print sample
    print("\n  Sample training text (truncated):")
    print("-" * 40)
    print(dataset[0]['text'][:500] + "...")
    print("-" * 40)
    
    # ========================================================================
    # STEP 5: Setup Trainer
    # ========================================================================
    print("\n[STEP 5] Setting up SFT Trainer...")
    
    from trl import SFTTrainer, SFTConfig
    
    trainer_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        save_steps=500,
        save_total_limit=2,
    )
    
    if max_steps is not None:
        trainer_args.max_steps = max_steps
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=trainer_args,
    )
    
    # Train only on assistant responses
    from unsloth.chat_templates import train_on_responses_only
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    
    # ========================================================================
    # STEP 6: Train
    # ========================================================================
    print("\n[STEP 6] Starting training...")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch Size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - LoRA Rank: {LORA_R}")
    
    # Show GPU memory before training
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"  - GPU: {gpu_stats.name}")
        print(f"  - GPU Memory: {start_gpu_memory}/{max_memory} GB")
    
    trainer_stats = trainer.train()
    
    # Print training stats
    print("\n  Training complete!")
    print(f"  - Training time: {trainer_stats.metrics['train_runtime']:.2f}s")
    print(f"  - Minutes: {trainer_stats.metrics['train_runtime']/60:.2f}")
    
    # ========================================================================
    # STEP 7: Convert QAT layers back
    # ========================================================================
    if use_qat:
        print("\n[STEP 7] Converting QAT layers...")
        
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
        
        quantize_(model, QATConfig(step="convert"))
        print("  ✓ QAT conversion complete")
    
    # ========================================================================
    # STEP 8: Save LoRA Adapters
    # ========================================================================
    print(f"\n[STEP 8] Saving LoRA adapters to {LORA_MODEL_DIR}...")
    
    model.save_pretrained(LORA_MODEL_DIR)
    tokenizer.save_pretrained(LORA_MODEL_DIR)
    print("  ✓ LoRA adapters saved")
    
    # ========================================================================
    # STEP 9: Test Inference
    # ========================================================================
    print("\n[STEP 9] Testing inference...")
    
    FastLanguageModel.for_inference(model)
    
    test_messages = [
        {"role": "user", "content": "DIME 요소가 무엇인가요?"}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        test_messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    from transformers import TextStreamer
    
    print("\n  Test prompt: 'DIME 요소가 무엇인가요?'")
    print("  Response:")
    print("-" * 40)
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        input_ids,
        streamer=text_streamer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    print("-" * 40)
    
    # ========================================================================
    # STEP 10: Save to GGUF (q4_k_m)
    # ========================================================================
    if not skip_gguf:
        print(f"\n[STEP 10] Saving to GGUF format (q4_k_m)...")
        
        os.makedirs(GGUF_MODEL_DIR, exist_ok=True)
        
        model.save_pretrained_gguf(
            GGUF_MODEL_DIR,
            tokenizer,
            quantization_method="q4_k_m",
        )
        
        print(f"  ✓ GGUF model saved to {GGUF_MODEL_DIR}")
    else:
        print("\n[STEP 10] Skipping GGUF export (--skip-gguf flag)")
    
    # ========================================================================
    # STEP 11: Upload to HuggingFace (Optional)
    # ========================================================================
    if hf_token and not skip_upload:
        print(f"\n[STEP 11] Uploading to HuggingFace...")
        
        # Upload merged model (16-bit)
        print("  - Uploading 16-bit merged model...")
        model.push_to_hub_merged(
            f"{hf_username}/{HF_MODEL_NAME}",
            tokenizer,
            save_method="merged_16bit",
            token=hf_token,
        )
        
        # Upload GGUF model
        print("  - Uploading GGUF model...")
        model.push_to_hub_gguf(
            f"{hf_username}/{HF_MODEL_NAME}-GGUF",
            tokenizer,
            quantization_method="q4_k_m",
            token=hf_token,
        )
        
        print(f"  ✓ Models uploaded to HuggingFace!")
        print(f"    - 16-bit: https://huggingface.co/{hf_username}/{HF_MODEL_NAME}")
        print(f"    - GGUF:   https://huggingface.co/{hf_username}/{HF_MODEL_NAME}-GGUF")
    else:
        print("\n[STEP 11] Skipping HuggingFace upload")
        if not hf_token:
            print("  To upload, use --hf-token and --hf-username arguments.")
    
    # ========================================================================
    # STEP 12: Upload to Ollama Registry (Optional, Manual)
    # ========================================================================
    print(f"\n[STEP 12] Ollama Setup Instructions...")
    print("-" * 40)
    print("To upload to Ollama Registry, follow these steps:")
    print()
    print("1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
    print("2. Start Ollama: ollama serve")
    print(f"3. Create model: ollama create {OLLAMA_MODEL_NAME} -f {GGUF_MODEL_DIR}/Modelfile")
    print(f"4. Test locally: ollama run {OLLAMA_MODEL_NAME}")
    print()
    print("To push to Ollama Hub:")
    print("1. Create account at https://ollama.com")
    print(f"2. ollama push <username>/{OLLAMA_MODEL_NAME}")
    print("-" * 40)
    
    # Show Modelfile content
    modelfile_path = os.path.join(GGUF_MODEL_DIR, "Modelfile")
    if os.path.exists(modelfile_path):
        print("\nGenerated Modelfile content:")
        print("-" * 40)
        with open(modelfile_path, 'r') as f:
            print(f.read()[:1000])
        print("-" * 40)
    
    # ========================================================================
    # DONE
    # ========================================================================
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - LoRA adapters: {LORA_MODEL_DIR}/")
    print(f"  - GGUF model:    {GGUF_MODEL_DIR}/")
    print(f"  - Logs:          {OUTPUT_DIR}/")
    
    if hf_token:
        print(f"\nHuggingFace Links:")
        print(f"  - https://huggingface.co/{hf_username}/{HF_MODEL_NAME}")
        print(f"  - https://huggingface.co/{hf_username}/{HF_MODEL_NAME}-GGUF")


if __name__ == "__main__":
    main()
