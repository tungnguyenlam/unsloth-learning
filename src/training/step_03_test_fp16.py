"""
Step 03: Test FP16 Model

PURPOSE:
    Test the fine-tuned model after training.
    Downloads from HuggingFace and uses standard transformers for reliable batch inference.

TESTS RUN:
    1. Test 1: Knowledge Recall
       - Verifies the model can recall military terms and definitions
       - Metrics: BERTScore F1 (>= 0.70), Headword Recall (>= 0.80)
    
    2. Test 2: Stability Check (KoMMLU)
       - Verifies no catastrophic forgetting occurred
       - Tests general Korean language understanding
       - Metrics: Accuracy (>= 0.30)

USAGE:
    # Test the model pushed in step_02 (reads from config)
    python src/training/step_03_test_fp16.py
    
    # Test a specific HuggingFace model
    python src/training/step_03_test_fp16.py --hf-model mainguyenngoc/gemma3-4b-military-lr2e4_ep1_bs16x4_r64_a128_m7
    
    # Skip specific tests
    python src/training/step_03_test_fp16.py --skip-test1 --skip-test2
    
    # Adjust batch size
    python src/training/step_03_test_fp16.py --batch-size 8

Run from project root: python src/training/step_03_test_fp16.py
"""

import os
import sys
import torch

# Allow running from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Add testing directory to path
TESTING_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'testing')
if TESTING_DIR not in sys.path:
    sys.path.insert(0, TESTING_DIR)

from step_00_config import (
    TEST1_DATA_PATH,
    ensure_dirs, get_base_parser, get_max_seq_length, get_saved_run_name,
    get_results_dir_for_run, load_detected_config
)


def is_adapter_checkpoint(repo_id: str, revision: str = None, token: str = None) -> bool:
    """Check if a HuggingFace repo/revision contains only adapter files (not a full model)."""
    from huggingface_hub import list_repo_files
    
    try:
        files = list_repo_files(repo_id, revision=revision, token=token)
        has_adapter_config = "adapter_config.json" in files
        # Check for full model files (safetensors or bin)
        has_model_weights = any(
            f.startswith("model") and (f.endswith(".safetensors") or f.endswith(".bin"))
            for f in files
        )
        # It's an adapter if it has adapter_config but no full model weights
        return has_adapter_config and not has_model_weights
    except Exception as e:
        print(f"  Warning: Could not list repo files ({e}). Assuming full model.")
        return False


def main():
    parser = get_base_parser("Step 03: Test FP16 Model")
    parser.add_argument("--skip-test1", action="store_true", help="Skip Test 1 (Knowledge Recall)")
    parser.add_argument("--skip-test2", action="store_true", help="Skip Test 2 (Stability Check)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference (default: 16)")
    parser.add_argument("--hf-model", type=str, default=None, 
                       help="HuggingFace model name to test (default: reads from config)")
    parser.add_argument("--revision", type=str, default=None,
                       help="Git revision/commit hash to load (default: latest/main)")
    parser.add_argument("--hf-base-model", type=str, default=None,
                       help="Base model for adapter loading (required if loading adapter checkpoint)")
    parser.add_argument("--quick", action="store_true", help="Quick test with only 50 samples per test")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per test (default: all)")
    args = parser.parse_args()
    
    # Resolve HF Token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # Determine max_samples
    max_samples = args.max_samples
    if args.quick and max_samples is None:
        max_samples = 50
    
    run_name = get_saved_run_name()
    
    print("=" * 60)
    print("STEP 03: TEST FP16 MODEL")
    print("=" * 60)
    print(f"\nRun Name: {run_name}")
    print(f"Batch Size: {args.batch_size}")
    if args.revision:
        print(f"Revision: {args.revision}")
    if max_samples:
        print(f"Max Samples: {max_samples} (quick mode)")
    
    ensure_dirs()
    
    # Determine which model to load
    if args.hf_model:
        hf_model_name = args.hf_model
    else:
        # Try to read from config (set by step_02)
        config = load_detected_config()
        hf_model_name = config.get("hf_model_name")
        
        if not hf_model_name:
            print("ERROR: No HuggingFace model found in config.")
            print("Either:")
            print("  1. Run step_02_train.py with --hf-token to push model first")
            print("  2. Use --hf-model to specify a HuggingFace model directly")
            sys.exit(1)
    
    # Update run_name to reflect the HF model and revision
    hf_model_short = hf_model_name.split("/")[-1]
    if args.revision:
        run_name = f"hf_{hf_model_short}_{args.revision[:8]}"
    else:
        run_name = f"hf_{hf_model_short}"
    results_dir = get_results_dir_for_run(run_name)
    
    print(f"HuggingFace Model: {hf_model_name}")
    if args.revision:
        print(f"Revision: {args.revision}")
    print(f"Results Dir: {results_dir}")
    
    # Detect if this is an adapter checkpoint or a full model
    print(f"\n[1/3] Loading model from HuggingFace...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    is_adapter = is_adapter_checkpoint(hf_model_name, revision=args.revision, token=hf_token)
    
    if is_adapter:
        print(f"  Detected: ADAPTER checkpoint (PEFT/LoRA)")
        
        if not args.hf_base_model:
            # Try to infer base model from adapter_config.json
            from huggingface_hub import hf_hub_download
            import json
            try:
                adapter_config_path = hf_hub_download(
                    hf_model_name, 
                    "adapter_config.json", 
                    revision=args.revision,
                    token=hf_token
                )
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")
                if base_model_name:
                    print(f"  Inferred base model from adapter_config: {base_model_name}")
                else:
                    print("ERROR: Could not determine base model from adapter_config.")
                    print("Please specify --hf-base-model explicitly.")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR: Could not load adapter_config.json ({e})")
                print("Please specify --hf-base-model explicitly.")
                sys.exit(1)
        else:
            base_model_name = args.hf_base_model
            print(f"  Using specified base model: {base_model_name}")
        
        # Load base model first
        print(f"  Loading base model: {base_model_name}...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token,
            )
        except Exception as e:
            print(f"  Warning: bfloat16 failed ({e}), trying float16...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token,
            )
        
        # Load adapter on top
        print(f"  Loading adapter from: {hf_model_name} (rev: {args.revision or 'latest'})...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            base_model,
            hf_model_name,
            revision=args.revision,
            token=hf_token,
        )
        print(f"  Adapter loaded successfully")
        
        # Load tokenizer from adapter repo (might have special tokens)
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name, 
            revision=args.revision,
            trust_remote_code=True,
            token=hf_token,
        )
    else:
        print(f"  Detected: FULL merged model")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name,
            revision=args.revision,
            trust_remote_code=True,
            token=hf_token,
        )
        
        # Load full model
        try:
            print("  Attempting to load in bfloat16...")
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                revision=args.revision,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token,
            )
        except Exception as e:
            print(f"  Warning: Failed to load in bfloat16 ({e}). Fallback to float16...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model_name,
                    revision=args.revision,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token,
                )
            except Exception as e2:
                 print(f"  Warning: Failed to load in float16 ({e2}). Fallback to float32...")
                 model = AutoModelForCausalLM.from_pretrained(
                    hf_model_name,
                    revision=args.revision,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token,
                )
    
    # Set up tokenizer for batch processing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for batch generation
    
    print(f"  Model loaded successfully")
    print(f"  Device: {next(model.parameters()).device}")
    
    # Test 1: Knowledge Recall
    if not args.skip_test1:
        print("\n[2/3] Running Test 1: Knowledge Recall...")
        
        if not os.path.exists(TEST1_DATA_PATH):
            print(f"  WARNING: Test data not found: {TEST1_DATA_PATH}")
            print("  Run src/training/step_01_prepare_data.py first")
        else:
            import test1_knowledge_recall as test1
            test1_output = os.path.join(results_dir, f"fp16_test1_{run_name}.json")
            test1.run_test(model, tokenizer, TEST1_DATA_PATH, test1_output, run_name=f"fp16_{run_name}", batch_size=args.batch_size, max_samples=max_samples)
    else:
        print("\n[2/3] Skipping Test 1")
    
    # Test 2: Stability Check
    if not args.skip_test2:
        print("\n[3/3] Running Test 2: Stability Check (KoMMLU)...")
        import test2_stability_check as test2
        test2_output = os.path.join(results_dir, f"fp16_test2_{run_name}.json")
        test2.run_test(model, tokenizer, output_path=test2_output, run_name=f"fp16_{run_name}", batch_size=args.batch_size, max_samples=max_samples)
    else:
        print("\n[3/3] Skipping Test 2")
    
    print("\n" + "=" * 60)
    print("STEP 03 COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}/")
    print("\nNext: python src/training/step_04_export_gguf.py")


if __name__ == "__main__":
    main()
