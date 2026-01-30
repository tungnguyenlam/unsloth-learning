"""
Step 05: Test GGUF Model & Compare

PURPOSE:
    Test the quantized GGUF model and compare its performance against the
    FP16 model to measure quantization loss.

TESTS RUN:
    1. Test 1: Knowledge Recall (GGUF)
       - Same as FP16 test but using GGUF model via llama-cpp-python
       - Verifies vocabulary knowledge survives quantization
    
    2. Test 2: Stability Check (GGUF)
       - Same KoMMLU evaluation on the GGUF model
       - Verifies general capabilities survive quantization
    
    3. Test 3: Quantization Comparison
       - Compares FP16 vs GGUF results
       - Measures quantization loss (acceptable: <5% drop)

REQUIREMENTS:
    - llama-cpp-python installed (pip install llama-cpp-python)
    - GGUF model exported via step_04_export_gguf.py
    - FP16 test results from step_03_test_fp16.py (for comparison)

USAGE:
    # Test auto-selected GGUF model
    python src/training/step_05_test_gguf.py
    
    # Specify a specific GGUF file
    python src/training/step_05_test_gguf.py --model path/to/model.gguf
    
    # Skip specific tests
    python src/training/step_05_test_gguf.py --skip-test1 --skip-test3

Run from project root: python src/training/step_05_test_gguf.py
"""

import os
import sys
import json

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


class GGUFModelWrapper:
    def __init__(self, model_path: str, n_ctx: int = 2048):
        try:
            from llama_cpp import Llama
        except ImportError:
            print("\nERROR: llama-cpp-python not installed.")
            print("Please run: pip install llama-cpp-python")
            sys.exit(1)
            
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )
        self.device = "cuda"
    
    def generate_text(self, prompt: str, max_new_tokens: int = 256) -> str:
        output = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            stop=["<end_of_turn>", "<eos>"],
        )
        return output["choices"][0]["text"].strip()


class GGUFTokenizerWrapper:
    def __init__(self):
        self.eos_token_id = 1
    
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, return_tensors=None, return_dict=False):
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == "assistant":
                text += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        
        if add_generation_prompt:
            text += "<start_of_turn>model\n"
        
        if tokenize:
            return {"input_ids": None, "prompt": text}
        return text


def run_test1_gguf(model, tokenizer, test_data_path: str, output_path: str, run_name: str = None, max_samples: int = 50) -> dict:
    from tqdm import tqdm
    from datetime import datetime
    import test1_knowledge_recall as test1
    
    print(f"Loading test data from: {test_data_path}")
    data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Limit samples for quick GGUF testing
    if max_samples and max_samples < len(data):
        import random
        random.seed(42)
        data = random.sample(data, max_samples)
        print(f"Using {max_samples} samples (quick mode)")
    
    print(f"Running inference on {len(data)} samples...")
    predictions = []
    questions = [d["question"] for d in data]
    ground_truths = [d["ground_truth"] for d in data]
    
    for item in tqdm(data, desc="GGUF Inference"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["question"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        pred = model.generate_text(prompt, max_new_tokens=256)
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


def run_test2_gguf(model, tokenizer, output_path: str, run_name: str = None, max_samples: int = 50) -> dict:
    from tqdm import tqdm
    from datetime import datetime
    import test2_stability_check as test2
    
    print("Loading KoMMLU data...")
    test_data = test2.load_kommlu_data()
    
    # Limit samples for quick GGUF testing
    if max_samples and max_samples < len(test_data):
        import random
        random.seed(42)
        test_data = random.sample(test_data, max_samples)
        print(f"Using {max_samples} samples (quick mode)")
    
    questions = [test2.format_mcq_prompt(d["question"], d["choices"]) for d in test_data]
    ground_truths = [d["answer"] for d in test_data]
    subjects = [d["subject"] for d in test_data]
    
    print(f"Running inference on {len(questions)} samples...")
    predictions = []
    responses = []
    
    for i, q in enumerate(tqdm(questions, desc="GGUF KoMMLU")):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True,
            tokenize=False,
        )
        resp = model.generate_text(prompt, max_new_tokens=16)
        responses.append(resp)
        answer = test2.extract_answer(resp) if test2.extract_answer(resp) is not None else -1
        predictions.append(answer)
        # Debug: print first 3 responses
        if i < 3:
            print(f"  Sample {i}: Response='{resp[:50]}...' -> Pred={answer}")
    
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
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (ignored for GGUF inference)")
    parser.add_argument("--model", type=str, default=None, help="Explicitly specify GGUF model file path")
    parser.add_argument("--quick", action="store_true", help="Quick test with only 50 samples per test (default for GGUF)")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples per test (default: 50)")
    args = parser.parse_args()
    
    max_seq_length = get_max_seq_length()
    run_name = get_saved_run_name()
    
    # For GGUF, default to 50 samples unless overridden
    max_samples = args.max_samples
    
    print("=" * 60)
    print("STEP 05: TEST GGUF MODEL")
    print("=" * 60)
    print(f"\nRun Name: {run_name}")
    print(f"Max Seq Length: {max_seq_length}")
    print(f"Max Samples: {max_samples} (GGUF quick mode)")
    
    # 1. Use explicit model if provided
    if args.model:
        if not os.path.exists(args.model):
            print(f"ERROR: Specified model file not found: {args.model}")
            sys.exit(1)
        gguf_path = args.model
        print(f"\nUsing specified GGUF model: {gguf_path}")
        
    else:
        # 2. Search for models
        if os.path.exists(GGUF_MODEL_DIR):
            print(f"Checking directory: {GGUF_MODEL_DIR}")
            print(f"Contents: {os.listdir(GGUF_MODEL_DIR)}")
            gguf_files = [os.path.join(GGUF_MODEL_DIR, f) for f in os.listdir(GGUF_MODEL_DIR) if f.endswith('.gguf')]
        else:
            print(f"Directory not found: {GGUF_MODEL_DIR}")
            gguf_files = []

        # Fallback: Check current directory
        if not gguf_files:
            print("Checking current directory for GGUF files...")
            current_dir_files = [f for f in os.listdir('.') if f.endswith('.gguf')]
            gguf_files = current_dir_files
            if gguf_files:
                print(f"Found GGUF file(s) in current directory: {gguf_files}")

        if not gguf_files:
            print(f"ERROR: No GGUF files found in: {GGUF_MODEL_DIR} or current directory")
            print("Run src/training/step_04_export_gguf.py first")
            sys.exit(1)
        
        # Filter out mmproj files (multimodal projectors)
        valid_gguf_files = [f for f in gguf_files if "mmproj" not in f]
        
        if not valid_gguf_files:
            print(f"ERROR: Only mmproj files found: {gguf_files}")
            print("The main model GGUF seems to be missing.")
            sys.exit(1)
            
        print(f"\nFound {len(valid_gguf_files)} candidate models: {valid_gguf_files}")
        
        # Selection Logic:
        # 1. Prefer file containing run_name
        # 2. Prefer newest file (by modification time)
        
        # Sort by modification time (newest first)
        valid_gguf_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        selected_gguf = valid_gguf_files[0]
        match_source = "newest modified"
        
        # Check for run_name match
        for f in valid_gguf_files:
            if run_name in f:
                selected_gguf = f
                match_source = f"matches run_name '{run_name}'"
                break
                
        gguf_path = selected_gguf
        print(f"Auto-selected model ({match_source}): {gguf_path}")
        if len(valid_gguf_files) > 1:
            print("  (Use --model <path> to specify a different one)")
    
    # Load Model
    print("\n[1/4] Loading GGUF model...")
    model = GGUFModelWrapper(gguf_path, n_ctx=max_seq_length)
    tokenizer = GGUFTokenizerWrapper()
    print("  Model loaded successfully")
    
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
            gguf_test1_results = run_test1_gguf(model, tokenizer, TEST1_DATA_PATH, test1_output, run_name=f"gguf_{run_name}", max_samples=max_samples)
    else:
        print("\n[2/4] Skipping Test 1")
    
    # Test 2
    if not args.skip_test2:
        print("\n[3/4] Running Test 2: Stability Check (KoMMLU)...")
        test2_output = os.path.join(results_dir, f"gguf_test2_{run_name}.json")
        gguf_test2_results = run_test2_gguf(model, tokenizer, test2_output, run_name=f"gguf_{run_name}", max_samples=max_samples)
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
