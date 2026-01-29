"""
Step 06: Upload to HuggingFace
- Upload merged 16-bit model
- Upload GGUF model

Run: python step_06_upload.py --hf-token YOUR_TOKEN --hf-username YOUR_USERNAME
"""

import os
import sys

from step_00_config import (
    MODEL_NAME, LOAD_IN_4BIT,
    LORA_MODEL_DIR, HF_MODEL_BASE_NAME,
    ensure_dirs, get_base_parser, get_max_seq_length, get_saved_run_name
)


def main():
    parser = get_base_parser("Step 06: Upload to HuggingFace")
    parser.add_argument("--skip-16bit", action="store_true", help="Skip 16-bit upload")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF upload")
    args = parser.parse_args()
    
    max_seq_length = get_max_seq_length()
    run_name = get_saved_run_name()
    hf_model_name = f"{HF_MODEL_BASE_NAME}-{run_name}"
    
    print("=" * 60)
    print("STEP 06: UPLOAD TO HUGGINGFACE")
    print("=" * 60)
    print(f"\nRun Name: {run_name}")
    print(f"HF Model: {hf_model_name}")
    print(f"Max Seq Length: {max_seq_length}")
    
    ensure_dirs()
    
    if not args.hf_token:
        print("ERROR: HuggingFace token required")
        print("Use: --hf-token YOUR_TOKEN")
        sys.exit(1)
    
    if args.hf_username == "YOUR_HF_USERNAME":
        print("ERROR: HuggingFace username required")
        print("Use: --hf-username YOUR_USERNAME")
        sys.exit(1)
    
    if not os.path.exists(LORA_MODEL_DIR):
        print(f"ERROR: LoRA model not found at: {LORA_MODEL_DIR}")
        print("Run step_02_train.py first")
        sys.exit(1)
    
    # Load Model
    print("\n[1/3] Loading trained model...")
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_MODEL_DIR,
        max_seq_length=max_seq_length,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print("  Model loaded successfully")
    
    # Upload 16-bit
    if not args.skip_16bit:
        print("\n[2/3] Uploading 16-bit merged model...")
        repo_name = f"{args.hf_username}/{hf_model_name}"
        model.push_to_hub_merged(
            repo_name,
            tokenizer,
            save_method="merged_16bit",
            token=args.hf_token,
        )
        print(f"  Uploaded: https://huggingface.co/{repo_name}")
    else:
        print("\n[2/3] Skipping 16-bit upload")
    
    # Upload GGUF
    if not args.skip_gguf:
        print("\n[3/3] Uploading GGUF model...")
        repo_name = f"{args.hf_username}/{hf_model_name}-GGUF"
        model.push_to_hub_gguf(
            repo_name,
            tokenizer,
            quantization_method="q4_k_m",
            token=args.hf_token,
        )
        print(f"  Uploaded: https://huggingface.co/{repo_name}")
    else:
        print("\n[3/3] Skipping GGUF upload")
    
    print("\n" + "=" * 60)
    print("STEP 06 COMPLETE")
    print("=" * 60)
    print(f"\nModels uploaded to HuggingFace:")
    if not args.skip_16bit:
        print(f"  - https://huggingface.co/{args.hf_username}/{hf_model_name}")
    if not args.skip_gguf:
        print(f"  - https://huggingface.co/{args.hf_username}/{hf_model_name}-GGUF")
    print("\nAll steps complete!")


if __name__ == "__main__":
    main()
