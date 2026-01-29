"""
Step 05: Test GGUF Model & Compare
- Load GGUF model via llama-cpp-python
- Run Test 1: Knowledge Recall
- Run Test 2: Stability Check (KoMMLU)
- Run Test 3: Compare FP16 vs GGUF

Run: python step_05_test_gguf.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'testing'))

from step_00_config import (
    GGUF_MODEL_DIR, TEST1_DATA_PATH, RESULTS_DIR,
    ensure_dirs, get_base_parser, get_max_seq_length, get_saved_run_name
)


class GGUFModelWrapper:
    def __init__(self, model_path: str, n_ctx: int = 2048):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )
        self.device = "cuda"
    
    def generate(self, input_ids=None, max_new_tokens=256, **kwargs):
        raise NotImplementedError("Use generate_text instead")
    
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
    
    def decode(self, tokens, skip_special_tokens=True):
        return ""


def run_test1_gguf(model: GGUFModelWrapper, tokenizer: GGUFTokenizerWrapper, 
                   test_data_path: str, output_path: str, run_name: str = None) -> dict:
    from tqdm import tqdm
    import numpy as np
    from datetime import datetime
    
    print(f"Loading test data from: {test_data_path}")
    data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
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
    import test1_knowledge_recall as test1
    bertscore = test1.compute_bertscore(predictions, ground_truths)
    headword = test1.compute_headword_recall(predictions, ground_truths)
    abbreviation = test1.compute_abbreviation_match(predictions, ground_truths)
    
    results = {
        "bertscore": {k: v for k, v in bertscore.items() if k != "raw_f1"},
        "headword_recall": {k: v for k, v in headword.items() if k not in ["raw", "headwords"]},
        "abbreviation_match": {k: v for k, v in abbreviation.items() if k not in ["raw", "abbreviations"]},
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
    
    # Save detailed results
    output_dir = os.path.dirname(output_path)
    test1.save_detailed_results(
        output_dir, questions, predictions, ground_truths,
        bertscore["raw_f1"], headword["raw"], headword["headwords"], run_name
    )
    test1.save_plots(output_dir, bertscore["raw_f1"], headword["raw"], run_name)
    
    return results


def run_test2_gguf(model: GGUFModelWrapper, tokenizer: GGUFTokenizerWrapper,
                   output_path: str, run_name: str = None) -> dict:
    from tqdm import tqdm
    from datetime import datetime
    import test2_stability_check as test2
    
    print("Loading KoMMLU data...")
    test_data = test2.load_kommlu_data()
    
    questions = [test2.format_mcq_prompt(d["question"], d["choices"]) for d in test_data]
    ground_truths = [d["answer"] for d in test_data]
    subjects = [d["subject"] for d in test_data]
    
    print(f"Running inference on {len(questions)} samples...")
    predictions = []
    responses = []
    
    for q in tqdm(questions, desc="GGUF KoMMLU"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True,
            tokenize=False,
        )
        resp = model.generate_text(prompt, max_new_tokens=16)
        responses.append(resp.strip())
        answer = test2.extract_answer(resp) if test2.extract_answer(resp) is not None else -1
        predictions.append(answer)
    
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
    args = parser.parse_args()
    
    max_seq_length = get_max_seq_length()
    run_name = get_saved_run_name()
    
    print("=" * 60)
    print("STEP 05: TEST GGUF MODEL")
    print("=" * 60)
    print(f"\nRun Name: {run_name}")
    print(f"Max Seq Length: {max_seq_length}")
    
    ensure_dirs()
    
    gguf_files = [f for f in os.listdir(GGUF_MODEL_DIR) if f.endswith('.gguf')] if os.path.exists(GGUF_MODEL_DIR) else []
    if not gguf_files:
        print(f"ERROR: No GGUF files found in: {GGUF_MODEL_DIR}")
        print("Run step_04_export_gguf.py first")
        sys.exit(1)
    
    gguf_path = os.path.join(GGUF_MODEL_DIR, gguf_files[0])
    print(f"\nUsing GGUF model: {gguf_path}")
    
    # Load Model
    print("\n[1/4] Loading GGUF model...")
    model = GGUFModelWrapper(gguf_path, n_ctx=max_seq_length)
    tokenizer = GGUFTokenizerWrapper()
    print("  Model loaded successfully")
    
    gguf_test1_results = None
    gguf_test2_results = None
    
    # Test 1
    if not args.skip_test1:
        print("\n[2/4] Running Test 1: Knowledge Recall...")
        if not os.path.exists(TEST1_DATA_PATH):
            print(f"  WARNING: Test data not found: {TEST1_DATA_PATH}")
        else:
            test1_output = os.path.join(RESULTS_DIR, f"gguf_test1_{run_name}.json")
            gguf_test1_results = run_test1_gguf(model, tokenizer, TEST1_DATA_PATH, test1_output, run_name=f"gguf_{run_name}")
    else:
        print("\n[2/4] Skipping Test 1")
    
    # Test 2
    if not args.skip_test2:
        print("\n[3/4] Running Test 2: Stability Check (KoMMLU)...")
        test2_output = os.path.join(RESULTS_DIR, f"gguf_test2_{run_name}.json")
        gguf_test2_results = run_test2_gguf(model, tokenizer, test2_output, run_name=f"gguf_{run_name}")
    else:
        print("\n[3/4] Skipping Test 2")
    
    # Test 3: Compare FP16 vs GGUF
    if not args.skip_test3:
        print("\n[4/4] Running Test 3: Quantization Comparison...")
        
        fp16_test1_path = os.path.join(RESULTS_DIR, f"fp16_test1_{run_name}.json")
        fp16_test2_path = os.path.join(RESULTS_DIR, f"fp16_test2_{run_name}.json")
        
        if not os.path.exists(fp16_test1_path) or not os.path.exists(fp16_test2_path):
            print("  WARNING: FP16 results not found. Run step_03_test_fp16.py first")
        elif gguf_test1_results is None or gguf_test2_results is None:
            print("  WARNING: GGUF results not available. Cannot compare.")
        else:
            import test3_quantization_loss as test3
            test3_output = os.path.join(RESULTS_DIR, f"test3_{run_name}.json")
            
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
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("\nNext: python step_06_upload.py (optional)")


if __name__ == "__main__":
    main()
