"""
Step 04: Export to GGUF
- Load trained LoRA model
- Export to GGUF format (q4_k_m)
- Generate Modelfile for Ollama

Run from project root: python src/training/step_04_export_gguf.py
"""

import os
import sys

# Allow running from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from step_00_config import (
    MODEL_NAME, LOAD_IN_4BIT,
    LORA_MODEL_DIR, GGUF_MODEL_DIR, OLLAMA_MODEL_BASE_NAME,
    ensure_dirs, get_base_parser, get_max_seq_length, get_saved_run_name
)


def main():
    parser = get_base_parser("Step 04: Export to GGUF")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                       help="Quantization method: q4_k_m, q8_0, f16, etc.")
    args = parser.parse_args()
    
    max_seq_length = get_max_seq_length()
    run_name = get_saved_run_name()
    ollama_model_name = f"{OLLAMA_MODEL_BASE_NAME}-{run_name}"
    
    print("=" * 60)
    print("STEP 04: EXPORT TO GGUF")
    print("=" * 60)
    print(f"\nRun Name: {run_name}")
    print(f"Max Seq Length: {max_seq_length}")
    
    ensure_dirs()
    
    if not os.path.exists(LORA_MODEL_DIR):
        print(f"ERROR: LoRA model not found at: {LORA_MODEL_DIR}")
        print("Run src/training/step_02_train.py first")
        sys.exit(1)
    
    # Load Model
    print("\n[1/2] Loading trained model...")
    from unsloth import FastModel
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=LORA_MODEL_DIR,
        max_seq_length=max_seq_length,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print("  Model loaded successfully")
    
    # Export to GGUF
    print(f"\n[2/2] Exporting to GGUF ({args.quantization})...")
    os.makedirs(GGUF_MODEL_DIR, exist_ok=True)
    
    model.save_pretrained_gguf(
        GGUF_MODEL_DIR,
        tokenizer,
        quantization_method=args.quantization,
    )
    
    print(f"  Saved to: {GGUF_MODEL_DIR}/")
    
    # Print Ollama instructions
    print("\n" + "-" * 40)
    print("OLLAMA SETUP INSTRUCTIONS")
    print("-" * 40)
    print(f"1. ollama create {ollama_model_name} -f {GGUF_MODEL_DIR}/Modelfile")
    print(f"2. ollama run {ollama_model_name}")
    print("-" * 40)
    
    print("\n" + "=" * 60)
    print("STEP 04 COMPLETE")
    print("=" * 60)
    print(f"\nGGUF model: {GGUF_MODEL_DIR}/")
    print("\nNext: python src/training/step_05_test_gguf.py")


if __name__ == "__main__":
    main()
