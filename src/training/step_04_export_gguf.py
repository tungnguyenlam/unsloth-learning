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
    parser.add_argument("--hf-model", type=str, default=None,
                       help="HuggingFace model ID to load (e.g., username/model-name). Overrides local model check.")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Override run name for output naming (useful if config is lost)")
    parser.add_argument("--max-seq-length", type=int, default=None,
                        help="Override max sequence length (default: 2048 or auto-detected)")
    args = parser.parse_args()
    
    # max_seq_length priority: CLI arg -> Config file -> Default constant
    max_seq_length = args.max_seq_length or get_max_seq_length()
    run_name = args.run_name or get_saved_run_name()
    
    # Determine dynamic Ollama name
    from step_00_config import get_hf_model_base_name, load_detected_config
    config = load_detected_config()
    model_name_used = config.get("model_name", MODEL_NAME)
    hf_base = get_hf_model_base_name(model_name_used)
    ollama_base = hf_base.replace("-military", "-mil")
    ollama_model_name = f"{ollama_base}-{run_name}"
    
    print("=" * 60)
    print("STEP 04: EXPORT TO GGUF")
    print("=" * 60)
    print(f"\nRun Name: {run_name}")
    print(f"Max Seq Length: {max_seq_length}")
    
    # Resolve HF Token explicitly (checking env var as fallback)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # Determine model source
    if args.hf_model:
        model_source = args.hf_model
        print(f"Loading from Hugging Face: {model_source}")
    else:
        if not os.path.exists(LORA_MODEL_DIR):
            print(f"ERROR: LoRA model not found at: {LORA_MODEL_DIR}")
            print("Run src/training/step_02_train.py first")
            sys.exit(1)
        model_source = LORA_MODEL_DIR
    
    # Load Model
    print(f"\n[1/2] Loading trained model from {model_source}...")
    from unsloth import FastModel
    
    load_kwargs = {
        "model_name": model_source,
        "max_seq_length": max_seq_length,
        "load_in_4bit": LOAD_IN_4BIT,
    }
    if hf_token:
        load_kwargs["token"] = hf_token

    model, tokenizer = FastModel.from_pretrained(**load_kwargs)
    print("  Model loaded successfully")
    
    # Export to GGUF
    print(f"\n[2/2] Exporting to GGUF ({args.quantization})...")
    os.makedirs(GGUF_MODEL_DIR, exist_ok=True)
    
    # Note: save_pretrained_gguf might save to current dir despite argument
    model.save_pretrained_gguf(
        GGUF_MODEL_DIR,
        tokenizer,
        quantization_method=args.quantization,
    )
    
    # Organizing files: Move GGUF from current dir to GGUF_MODEL_DIR if needed
    import shutil
    
    # Look for GGUF files in current directory
    current_dir_ggufs = [f for f in os.listdir('.') if f.endswith('.gguf')]
    
    for f in current_dir_ggufs:
        # Construct new name with run_name to avoid collisions
        # e.g. gemma-3-4b-it.Q4_K_M.gguf -> gemma-3-4b-it-run_name.Q4_K_M.gguf
        if run_name not in f:
            name_parts = f.rsplit('.', 2) # split extension
            if len(name_parts) > 1:
                new_name = f"{name_parts[0]}-{run_name}.{'.'.join(name_parts[1:])}"
            else:
                new_name = f"{f}-{run_name}"
        else:
            new_name = f
            
        src = os.path.join('.', f)
        dst = os.path.join(GGUF_MODEL_DIR, new_name)
        
        print(f"  Moving/Renaming: {src} -> {dst}")
        shutil.move(src, dst)

    # Also check if files are inside GGUF_MODEL_DIR but need renaming
    subdir_ggufs = [f for f in os.listdir(GGUF_MODEL_DIR) if f.endswith('.gguf') and run_name not in f]
    for f in subdir_ggufs:
         src = os.path.join(GGUF_MODEL_DIR, f)
         
         name_parts = f.rsplit('.', 2)
         if len(name_parts) > 1:
            new_name = f"{name_parts[0]}-{run_name}.{'.'.join(name_parts[1:])}"
         else:
            new_name = f"{f}-{run_name}"
            
         dst = os.path.join(GGUF_MODEL_DIR, new_name)
         print(f"  Renaming: {src} -> {dst}")
         shutil.move(src, dst)
         
    print(f"  Saved and organized in: {GGUF_MODEL_DIR}/")
    
    # Step 3: Push GGUF to HuggingFace
    hf_username = args.hf_username
    
    if hf_token:
        print(f"\n[3/3] Pushing GGUF to HuggingFace...")
        
        # Determine model base name dynamically from saved config
        from step_00_config import get_hf_model_base_name, load_detected_config
        config = load_detected_config()
        model_name = config.get("model_name", MODEL_NAME)
        
        hf_model_base_name = get_hf_model_base_name(model_name)
        hf_gguf_name = f"{hf_username}/{hf_model_base_name}-{run_name}-GGUF"
        print(f"  Pushing to: {hf_gguf_name}")
        
        # Push GGUF with Unsloth's method
        model.push_to_hub_gguf(
            hf_gguf_name,
            tokenizer,
            quantization_method=args.quantization,
            token=hf_token,
        )
        print(f"  ✅ Pushed GGUF to: {hf_gguf_name}")
    else:
        print(f"\n[3/3] Skipping HuggingFace push (no --hf-token provided)")
    
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
    if hf_token:
        print(f"HuggingFace GGUF: {hf_gguf_name}")
    print("\nNext: python src/training/step_05_test_gguf.py")


if __name__ == "__main__":
    main()

