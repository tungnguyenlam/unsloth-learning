"""
Step 03: Test FP16 Model

PURPOSE:
    Test the fine-tuned LoRA model before quantization (GGUF export).
    Validates that the model learned the military vocabulary correctly.

TESTS RUN:
    1. Test 1: Knowledge Recall
       - Verifies the model can recall military terms and definitions
       - Metrics: BERTScore F1 (>= 0.70), Headword Recall (>= 0.80)
    
    2. Test 2: Stability Check (KoMMLU)
       - Verifies no catastrophic forgetting occurred
       - Tests general Korean language understanding
       - Metrics: Accuracy (>= 0.30)

USAGE:
    # Test local LoRA model (default)
    python src/training/step_03_test_fp16.py
    
    # Test a HuggingFace model directly
    python src/training/step_03_test_fp16.py --hf-model google/gemma-3-4b-it
    
    # Skip specific tests
    python src/training/step_03_test_fp16.py --skip-test1 --skip-test2
    
    # Adjust batch size
    python src/training/step_03_test_fp16.py --batch-size 8

Run from project root: python src/training/step_03_test_fp16.py
"""

import os
import sys

# Allow running from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Add testing directory to path
TESTING_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'testing')
if TESTING_DIR not in sys.path:
    sys.path.insert(0, TESTING_DIR)

from step_00_config import (
    MODEL_NAME, LOAD_IN_4BIT,
    LORA_MODEL_DIR, TEST1_DATA_PATH,
    ensure_dirs, get_base_parser, get_max_seq_length, get_saved_run_name,
    get_results_dir_for_run
)


def main():
    parser = get_base_parser("Step 03: Test FP16 Model")
    parser.add_argument("--skip-test1", action="store_true", help="Skip Test 1 (Knowledge Recall)")
    parser.add_argument("--skip-test2", action="store_true", help="Skip Test 2 (Stability Check)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference (default: 16)")
    parser.add_argument("--hf-model", type=str, default=None, 
                       help="HuggingFace model name to test instead of local LoRA model (e.g., 'google/gemma-3-4b-it')")
    args = parser.parse_args()
    
    max_seq_length = get_max_seq_length()
    run_name = get_saved_run_name()
    
    print("=" * 60)
    print("STEP 03: TEST FP16 MODEL")
    print("=" * 60)
    print(f"\nRun Name: {run_name}")
    print(f"Max Seq Length: {max_seq_length}")
    print(f"Batch Size: {args.batch_size}")
    
    ensure_dirs()
    
    # Get run-specific results directory
    results_dir = get_results_dir_for_run(run_name)
    print(f"Results Dir: {results_dir}")
    
    # Determine which model to load
    if args.hf_model:
        # Load HuggingFace model directly
        print(f"\n[1/3] Loading HuggingFace model: {args.hf_model}...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        # Update run_name to reflect the HF model
        hf_model_short = args.hf_model.split("/")[-1]
        run_name = f"hf_{hf_model_short}"
        results_dir = get_results_dir_for_run(run_name)
        print(f"  Model loaded successfully")
        print(f"  Updated run_name: {run_name}")
    else:
        # Load local LoRA model
        if not os.path.exists(LORA_MODEL_DIR):
            print(f"ERROR: LoRA model not found at: {LORA_MODEL_DIR}")
            print("Run src/training/step_02_train.py first, or use --hf-model to test a HuggingFace model")
            sys.exit(1)
        
        print("\n[1/3] Loading trained LoRA model...")
        from unsloth import FastModel, FastLanguageModel
        
        model, tokenizer = FastModel.from_pretrained(
            model_name=LORA_MODEL_DIR,
            max_seq_length=max_seq_length,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(model)
        print("  Model loaded successfully")
    
    # Test 1: Knowledge Recall
    if not args.skip_test1:
        print("\n[2/3] Running Test 1: Knowledge Recall...")
        
        if not os.path.exists(TEST1_DATA_PATH):
            print(f"  WARNING: Test data not found: {TEST1_DATA_PATH}")
            print("  Run src/training/step_01_prepare_data.py first")
        else:
            import test1_knowledge_recall as test1
            test1_output = os.path.join(results_dir, f"fp16_test1_{run_name}.json")
            test1.run_test(model, tokenizer, TEST1_DATA_PATH, test1_output, run_name=f"fp16_{run_name}", batch_size=args.batch_size)
    else:
        print("\n[2/3] Skipping Test 1")
    
    # Test 2: Stability Check
    if not args.skip_test2:
        print("\n[3/3] Running Test 2: Stability Check (KoMMLU)...")
        import test2_stability_check as test2
        test2_output = os.path.join(results_dir, f"fp16_test2_{run_name}.json")
        test2.run_test(model, tokenizer, output_path=test2_output, run_name=f"fp16_{run_name}", batch_size=args.batch_size)
    else:
        print("\n[3/3] Skipping Test 2")
    
    print("\n" + "=" * 60)
    print("STEP 03 COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}/")
    print("\nNext: python src/training/step_04_export_gguf.py")


if __name__ == "__main__":
    main()
