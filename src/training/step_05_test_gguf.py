"""
Step 05: Test GGUF Model & Compare

PURPOSE:
    Test the quantized GGUF model and compare its performance against the
    FP16 model to measure quantization loss.
    
    Uses transformers' native GGUF loading for GPU acceleration on NVIDIA GPUs.
    This approach works on both Mac (MPS) and Linux (CUDA) without special compilation.

TESTS RUN:
    1. Test 1: Knowledge Recall (GGUF)
       - Same as FP16 test but using GGUF model via transformers
       - Verifies vocabulary knowledge survives quantization
    
    2. Test 2: Stability Check (GGUF)
       - Same KoMMLU evaluation on the GGUF model
       - Verifies general capabilities survive quantization
    
    3. Test 3: Quantization Comparison
       - Compares FP16 vs GGUF results
       - Measures quantization loss (acceptable: <5% drop)

REQUIREMENTS:
    - transformers >= 4.45.0 (native GGUF support)
    - GGUF model exported via step_04_export_gguf.py or on HuggingFace
    - FP16 test results from step_03_test_fp16.py (for comparison)

USAGE:
    # Test from HuggingFace GGUF repo
    python src/training/step_05_test_gguf.py --hf-model mainguyenngoc/model-GGUF
    
    # Specify a specific GGUF file in the repo
    python src/training/step_05_test_gguf.py --hf-model user/repo-GGUF --hf-file model.Q4_K_M.gguf
    
    # Test a local GGUF file
    python src/training/step_05_test_gguf.py --model path/to/model.gguf
    
    # Skip specific tests
    python src/training/step_05_test_gguf.py --hf-model user/repo-GGUF --skip-test1 --skip-test3

Run from project root: python src/training/step_05_test_gguf.py
"""

import os
import sys
import json
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
    GGUF_MODEL_DIR, TEST1_DATA_PATH,
    ensure_dirs, get_base_parser, get_max_seq_length, get_saved_run_name,
    get_results_dir_for_run
)


def load_gguf_model_transformers(hf_repo: str = None, gguf_file: str = None, local_path: str = None):
    """
    Load GGUF model using transformers' native GGUF support.
    
    This works on CUDA GPUs without special llama-cpp-python compilation.
    The model is de-quantized to FP16/BF16 for inference.
    
    Args:
        hf_repo: HuggingFace repo containing GGUF files (e.g., "user/model-GGUF")
        gguf_file: Specific GGUF filename in the repo (optional, auto-detected if not set)
        local_path: Local path to GGUF file (alternative to hf_repo)
    
    Returns:
        (model, tokenizer) tuple
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import list_repo_files, hf_hub_download
    
    if local_path:
        # Local GGUF file - need to determine base model for tokenizer
        print(f"  Loading local GGUF: {local_path}")
        # For local files, we need the base model name for tokenizer
        # Try to infer from filename or use a default
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        # For local GGUF, tokenizer loading is tricky - need base model
        # This is a limitation - user should prefer HF repos
        tokenizer = None
        print("  WARNING: Local GGUF requires manual tokenizer setup")
        return model, tokenizer
    
    if not hf_repo:
        raise ValueError("Either hf_repo or local_path must be provided")
    
    print(f"  Loading from HuggingFace: {hf_repo}")
    
    # Find GGUF file if not specified
    if not gguf_file:
        print("  Finding GGUF file in repo...")
        files = list_repo_files(hf_repo)
        ggufs = [f for f in files if f.endswith(".gguf") and "mmproj" not in f]
        
        if not ggufs:
            raise ValueError(f"No GGUF files found in {hf_repo}")
        
        # Prefer Q4_K_M, then Q8_0, then first available
        gguf_file = next((f for f in ggufs if "q4_k_m" in f.lower()), None)
        if not gguf_file:
            gguf_file = next((f for f in ggufs if "q8_0" in f.lower()), ggufs[0])
        
        print(f"  Auto-selected: {gguf_file}")
    
    print(f"  GGUF file: {gguf_file}")
    
    # Load model with native GGUF support
    print("  Loading model (this may take a moment for de-quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        gguf_file=gguf_file,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Load tokenizer from the same repo
    # GGUF repos usually have tokenizer files, or we fall back to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
    except Exception as e:
        print(f"  Tokenizer not in GGUF repo, trying base model...")
        # Infer base model from repo name (remove -GGUF suffix)
        base_model_hint = hf_repo.replace("-GGUF", "")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_hint, trust_remote_code=True)
        except Exception:
            # Fall back to gemma-2 tokenizer as default
            print("  Falling back to google/gemma-2-2b tokenizer")
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", trust_remote_code=True)
    
    # Configure tokenizer for batch processing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for batch generation
    
    return model, tokenizer


def run_test1_gguf(model, tokenizer, test_data_path: str, output_path: str, 
                   run_name: str = None, batch_size: int = 8, max_samples: int = 50) -> dict:
    """Run Test 1: Knowledge Recall using batch inference."""
    from tqdm import tqdm
    from datetime import datetime
    import test1_knowledge_recall as test1
    
    print(f"Loading test data from: {test_data_path}")
    data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Limit samples
    if max_samples and max_samples < len(data):
        import random
        random.seed(42)
        data = random.sample(data, max_samples)
        print(f"Using {max_samples} samples")
    
    print(f"Running batch inference on {len(data)} samples (batch_size={batch_size})...")
    
    questions = [d["question"] for d in data]
    ground_truths = [d["ground_truth"] for d in data]
    
    # Batch inference
    predictions = []
    for i in tqdm(range(0, len(questions), batch_size), desc="GGUF Inference"):
        batch_questions = questions[i:i+batch_size]
        
        # Format prompts
        prompts = []
        for q in batch_questions:
            messages = [{"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt)
        
        # Tokenize batch
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only new tokens
        for j, output in enumerate(outputs):
            input_len = inputs.input_ids[j].shape[0]
            new_tokens = output[input_len:]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            predictions.append(pred)
    
    print("Computing metrics...")
    bertscore = test1.compute_bertscore(predictions, ground_truths)
    headword = test1.compute_headword_recall(predictions, ground_truths)
    
    results = {
        "bertscore": {"f1_mean": bertscore["f1_mean"], "f1_std": bertscore["f1_std"]},
        "headword_recall": {"accuracy": headword["accuracy"], "correct": headword["correct"], "total": headword["total"]},
        "passed": {
            "bertscore": bertscore["f1_mean"] >= 0.70,
            "headword": headword["accuracy"] >= 0.80
        },
        "num_samples": len(predictions),
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
    }
    results["overall_passed"] = all(results["passed"].values())
    
    print(f"\nBERTScore F1:    {bertscore['f1_mean']:.4f}")
    print(f"Headword Recall: {headword['accuracy']:.2%}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results


def run_test2_gguf(model, tokenizer, output_path: str, run_name: str = None, 
                   batch_size: int = 8, max_samples: int = 50) -> dict:
    """Run Test 2: Stability Check (KoMMLU) using batch inference."""
    from tqdm import tqdm
    from datetime import datetime
    import test2_stability_check as test2
    
    print("Loading KoMMLU data...")
    test_data = test2.load_kommlu_data()
    
    # Limit samples
    if max_samples and max_samples < len(test_data):
        import random
        random.seed(42)
        test_data = random.sample(test_data, max_samples)
        print(f"Using {max_samples} samples")
    
    questions = [test2.format_mcq_prompt(d["question"], d["choices"]) for d in test_data]
    ground_truths = [d["answer"] for d in test_data]
    subjects = [d["subject"] for d in test_data]
    
    print(f"Running batch inference on {len(questions)} samples (batch_size={batch_size})...")
    
    predictions = []
    responses = []
    
    for i in tqdm(range(0, len(questions), batch_size), desc="GGUF KoMMLU"):
        batch_questions = questions[i:i+batch_size]
        
        # Format prompts
        prompts = []
        for q in batch_questions:
            messages = [{"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt)
        
        # Tokenize batch
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only new tokens
        for j, output in enumerate(outputs):
            input_len = inputs.input_ids[j].shape[0]
            new_tokens = output[input_len:]
            resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            responses.append(resp)
            
            answer = test2.extract_answer(resp)
            predictions.append(answer if answer is not None else -1)
            
            # Debug: print first 3 responses
            if i + j < 3:
                print(f"  Sample {i+j}: Response='{resp[:50]}...' -> Pred={predictions[-1]}")
    
    accuracy = test2.compute_accuracy(predictions, ground_truths)
    by_subject = test2.compute_accuracy_by_subject(predictions, ground_truths, subjects)
    
    results = {
        "finetuned": {
            "accuracy": accuracy["accuracy"], 
            "correct": accuracy["correct"], 
            "total": accuracy["total"],
            "by_subject": by_subject
        },
        "passed": {"min_accuracy_met": accuracy["accuracy"] >= 0.30},
        "overall_passed": accuracy["accuracy"] >= 0.30,
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
    }
    
    print(f"\nAccuracy: {results['finetuned']['accuracy']:.2%}")
    print("\nBy Subject:")
    for subj, data in by_subject.items():
        print(f"  {subj}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save detailed results and plots
    output_dir = os.path.dirname(output_path)
    test2.save_detailed_results(output_dir, test_data, predictions, responses, run_name)
    test2.save_plots(output_dir, by_subject, accuracy["accuracy"], run_name)
    
    return results


def main():
    parser = get_base_parser("Step 05: Test GGUF Model")
    parser.add_argument("--skip-test1", action="store_true")
    parser.add_argument("--skip-test2", action="store_true")
    parser.add_argument("--skip-test3", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference (default: 8)")
    parser.add_argument("--hf-model", type=str, default=None, 
                       help="HuggingFace GGUF model repo (e.g. username/repo-GGUF). Will download automatically.")
    parser.add_argument("--hf-file", type=str, default=None, 
                       help="Specific GGUF filename in HF repo. If not set, picks Q4_K_M or first .gguf found.")
    parser.add_argument("--model", type=str, default=None, help="Explicitly specify local GGUF model file path")
    parser.add_argument("--quick", action="store_true", help="Quick test with only 50 samples per test (default for GGUF)")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples per test (default: 50)")
    
    args = parser.parse_args()
    
    max_seq_length = get_max_seq_length()
    run_name = get_saved_run_name()
    
    # If using HF model, sync run_name with step_03 naming to find FP16 results
    if args.hf_model:
        # mainguyenngoc/model-name-GGUF -> model-name
        hf_short = args.hf_model.split("/")[-1].replace("-GGUF", "")
        # Step 03 saved as hf_{model_name}
        run_name = f"hf_{hf_short}"
    
    # For GGUF, default to 50 samples unless overridden
    max_samples = args.max_samples
    
    print("=" * 60)
    print("STEP 05: TEST GGUF MODEL (Transformers Backend)")
    print("=" * 60)
    print(f"\nRun Name: {run_name}")
    print(f"Max Seq Length: {max_seq_length}")
    print(f"Max Samples: {max_samples}")
    print(f"Batch Size: {args.batch_size}")
    
    # Validate arguments
    if not args.hf_model and not args.model:
        # Try auto-discovery from local GGUF_MODEL_DIR
        if os.path.exists(GGUF_MODEL_DIR):
            gguf_files = [f for f in os.listdir(GGUF_MODEL_DIR) if f.endswith('.gguf') and 'mmproj' not in f]
            if gguf_files:
                # Sort by modification time (newest first)
                gguf_files.sort(key=lambda x: os.path.getmtime(os.path.join(GGUF_MODEL_DIR, x)), reverse=True)
                args.model = os.path.join(GGUF_MODEL_DIR, gguf_files[0])
                print(f"\nAuto-selected local GGUF: {args.model}")
        
        if not args.model:
            print("\nERROR: No model specified.")
            print("Use --hf-model <repo> for HuggingFace GGUF repos")
            print("Or --model <path> for local GGUF files")
            sys.exit(1)
    
    # Load Model using transformers native GGUF support
    print("\n[1/4] Loading GGUF model via transformers...")
    
    if args.hf_model:
        model, tokenizer = load_gguf_model_transformers(
            hf_repo=args.hf_model,
            gguf_file=args.hf_file
        )
    else:
        model, tokenizer = load_gguf_model_transformers(local_path=args.model)
        if tokenizer is None:
            print("ERROR: Local GGUF files require manual tokenizer setup.")
            print("Please use --hf-model with a HuggingFace repo instead.")
            sys.exit(1)
    
    device = next(model.parameters()).device
    print(f"  Model loaded successfully")
    print(f"  Device: {device}")
    
    # Get run-specific results directory
    results_dir = get_results_dir_for_run(run_name)
    
    gguf_test1_results = None
    gguf_test2_results = None
    
    # Test 1
    if not args.skip_test1:
        print("\n[2/4] Running Test 1: Knowledge Recall...")
        if not os.path.exists(TEST1_DATA_PATH):
            print(f"  WARNING: Test data not found: {TEST1_DATA_PATH}")
        else:
            test1_output = os.path.join(results_dir, f"gguf_test1_{run_name}.json")
            gguf_test1_results = run_test1_gguf(
                model, tokenizer, TEST1_DATA_PATH, test1_output, 
                run_name=f"gguf_{run_name}", 
                batch_size=args.batch_size,
                max_samples=max_samples
            )
    else:
        print("\n[2/4] Skipping Test 1")
    
    # Test 2
    if not args.skip_test2:
        print("\n[3/4] Running Test 2: Stability Check (KoMMLU)...")
        test2_output = os.path.join(results_dir, f"gguf_test2_{run_name}.json")
        gguf_test2_results = run_test2_gguf(
            model, tokenizer, test2_output, 
            run_name=f"gguf_{run_name}", 
            batch_size=args.batch_size,
            max_samples=max_samples
        )
    else:
        print("\n[3/4] Skipping Test 2")
    
    # Test 3: Compare FP16 vs GGUF
    if not args.skip_test3:
        print("\n[4/4] Running Test 3: Quantization Comparison...")
        
        fp16_test1_path = os.path.join(results_dir, f"fp16_test1_{run_name}.json")
        fp16_test2_path = os.path.join(results_dir, f"fp16_test2_{run_name}.json")
        
        if not os.path.exists(fp16_test1_path) or not os.path.exists(fp16_test2_path):
            print("  WARNING: FP16 results not found. Run src/training/step_03_test_fp16.py first")
        elif gguf_test1_results is None or gguf_test2_results is None:
            print("  WARNING: GGUF results not available. Cannot compare.")
        else:
            import test3_quantization_loss as test3
            test3_output = os.path.join(results_dir, f"test3_{run_name}.json")
            
            with open(fp16_test1_path, 'r') as f:
                fp16_test1 = json.load(f)
            with open(fp16_test2_path, 'r') as f:
                fp16_test2 = json.load(f)
            
            test3.run_test(fp16_test1, fp16_test2, gguf_test1_results, gguf_test2_results, test3_output, run_name=run_name)
    else:
        print("\n[4/4] Skipping Test 3")
    
    print("\n" + "=" * 60)
    print("STEP 05 COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}/")
    print("\nNext: python src/training/step_06_upload.py (optional)")


if __name__ == "__main__":
    main()
