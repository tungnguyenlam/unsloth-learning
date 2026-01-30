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


def main():
    parser = get_base_parser("Step 03: Test FP16 Model")
    parser.add_argument("--skip-test1", action="store_true", help="Skip Test 1 (Knowledge Recall)")
    parser.add_argument("--skip-test2", action="store_true", help="Skip Test 2 (Stability Check)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference (default: 16)")
    parser.add_argument("--hf-model", type=str, default=None, 
                       help="HuggingFace model name to test (default: reads from config)")
    parser.add_argument("--quick", action="store_true", help="Quick test with only 50 samples per test")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per test (default: all)")
    args = parser.parse_args()
    
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
    
    # Update run_name to reflect the HF model
    hf_model_short = hf_model_name.split("/")[-1]
    run_name = f"hf_{hf_model_short}"
    results_dir = get_results_dir_for_run(run_name)
    
    print(f"HuggingFace Model: {hf_model_name}")
    print(f"Results Dir: {results_dir}")
    
    # Load model from HuggingFace with standard transformers
    print(f"\n[1/3] Loading model from HuggingFace...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
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
