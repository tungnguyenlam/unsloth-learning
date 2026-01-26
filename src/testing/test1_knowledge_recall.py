"""
Test 1: Knowledge Recall - Prove the model learned the military vocabulary.
"""

import os
import json
import re
import numpy as np
from typing import List, Dict
from tqdm import tqdm

DEFAULT_TEST_DATA = "data/data_cleaned/test_datasets/test1_knowledge_recall.jsonl"
DEFAULT_OUTPUT = "results/test1_results.json"
THRESHOLDS = {"bertscore_f1": 0.70, "headword_recall": 0.80}


def load_test_data(path: str) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def compute_bertscore(predictions: List[str], references: List[str]) -> Dict:
    from bert_score import score as bert_score
    P, R, F1 = bert_score(predictions, references, model_type="bert-base-multilingual-cased", verbose=False)
    return {"f1_mean": float(np.mean(F1.numpy())), "f1_std": float(np.std(F1.numpy())), "raw_f1": F1.tolist()}


def compute_headword_recall(predictions: List[str], ground_truths: List[str]) -> Dict:
    results = []
    for pred, gt in zip(predictions, ground_truths):
        match = re.match(r'^([가-힣]+)', gt)
        if match:
            headword = match.group(1)
            results.append(1.0 if headword in pred else 0.0)
        else:
            results.append(1.0)
    return {"accuracy": float(np.mean(results)), "correct": int(sum(results)), "total": len(results), "raw": results}


def compute_abbreviation_match(predictions: List[str], ground_truths: List[str]) -> Dict:
    results = []
    for pred, gt in zip(predictions, ground_truths):
        match = re.search(r'☜\s*([A-Z0-9\-]+)', gt)
        if match:
            abbrev = match.group(1)
            results.append(1.0 if abbrev.lower() in pred.lower() else 0.0)
    if not results:
        return {"accuracy": None, "total": 0}
    return {"accuracy": float(np.mean(results)), "correct": int(sum(results)), "total": len(results)}


def run_test(model, tokenizer, test_data_path: str = DEFAULT_TEST_DATA, output_path: str = DEFAULT_OUTPUT, max_new_tokens: int = 256) -> Dict:
    print(f"Loading test data from: {test_data_path}")
    test_data = load_test_data(test_data_path)
    print(f"Loaded {len(test_data)} samples")
    
    questions = [d["question"] for d in test_data]
    ground_truths = [d["ground_truth"] for d in test_data]
    
    print("\nGenerating predictions...")
    predictions = []
    for question in tqdm(questions, desc="Inference"):
        messages = [{"role": "user", "content": question}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        pred = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predictions.append(pred.strip())
    
    print("\nComputing metrics...")
    bertscore = compute_bertscore(predictions, ground_truths)
    headword = compute_headword_recall(predictions, ground_truths)
    abbreviation = compute_abbreviation_match(predictions, ground_truths)
    
    results = {
        "bertscore": bertscore,
        "headword_recall": headword,
        "abbreviation_match": abbreviation,
        "passed": {
            "bertscore": bertscore["f1_mean"] >= THRESHOLDS["bertscore_f1"],
            "headword": headword["accuracy"] >= THRESHOLDS["headword_recall"]
        },
        "num_samples": len(predictions)
    }
    results["overall_passed"] = all(results["passed"].values())
    
    print("\n" + "=" * 50)
    print("TEST 1: KNOWLEDGE RECALL RESULTS")
    print("=" * 50)
    print(f"\nBERTScore F1:    {bertscore['f1_mean']:.4f} ± {bertscore['f1_std']:.4f} ({'PASS' if results['passed']['bertscore'] else 'FAIL'})")
    print(f"Headword Recall: {headword['accuracy']:.2%} ({'PASS' if results['passed']['headword'] else 'FAIL'})")
    if abbreviation["total"] > 0:
        print(f"Abbreviation:    {abbreviation['accuracy']:.2%}")
    print(f"\nOVERALL: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results