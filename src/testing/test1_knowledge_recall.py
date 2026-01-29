"""
Test 1: Knowledge Recall - Prove the model learned the military vocabulary.
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime

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
    headwords = []
    for pred, gt in zip(predictions, ground_truths):
        match = re.match(r'^([가-힣]+)', gt)
        if match:
            headword = match.group(1)
            headwords.append(headword)
            results.append(1.0 if headword in pred else 0.0)
        else:
            headwords.append(None)
            results.append(1.0)
    return {"accuracy": float(np.mean(results)), "correct": int(sum(results)), "total": len(results), "raw": results, "headwords": headwords}


def compute_abbreviation_match(predictions: List[str], ground_truths: List[str]) -> Dict:
    results = []
    abbreviations = []
    for pred, gt in zip(predictions, ground_truths):
        match = re.search(r'☜\s*([A-Z0-9\-]+)', gt)
        if match:
            abbrev = match.group(1)
            abbreviations.append(abbrev)
            results.append(1.0 if abbrev.lower() in pred.lower() else 0.0)
        else:
            abbreviations.append(None)
    if not results:
        return {"accuracy": None, "total": 0, "raw": [], "abbreviations": []}
    return {"accuracy": float(np.mean(results)), "correct": int(sum(results)), "total": len(results), "raw": results, "abbreviations": abbreviations}


def save_detailed_results(output_dir: str, questions: List[str], predictions: List[str], 
                          ground_truths: List[str], bertscore_f1: List[float], 
                          headword_correct: List[float], headwords: List[str], run_name: str = None):
    detailed = []
    for i, (q, pred, gt, f1, hw_correct, hw) in enumerate(zip(
        questions, predictions, ground_truths, bertscore_f1, headword_correct, headwords
    )):
        detailed.append({
            "index": i,
            "question": q,
            "prediction": pred,
            "ground_truth": gt,
            "bertscore_f1": f1,
            "headword": hw,
            "headword_correct": bool(hw_correct),
        })
    
    filename = f"test1_detailed_{run_name}.jsonl" if run_name else "test1_detailed.jsonl"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in detailed:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Detailed results saved: {output_path}")
    return output_path


def save_plots(output_dir: str, bertscore_f1: List[float], headword_correct: List[float], run_name: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: BERTScore F1 Distribution
    axes[0].hist(bertscore_f1, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=THRESHOLDS["bertscore_f1"], color='red', linestyle='--', label=f'Threshold ({THRESHOLDS["bertscore_f1"]})')
    axes[0].axvline(x=np.mean(bertscore_f1), color='green', linestyle='-', label=f'Mean ({np.mean(bertscore_f1):.3f})')
    axes[0].set_xlabel('BERTScore F1')
    axes[0].set_ylabel('Count')
    axes[0].set_title('BERTScore F1 Distribution')
    axes[0].legend()
    
    # Plot 2: BERTScore F1 Sorted
    sorted_f1 = sorted(bertscore_f1, reverse=True)
    axes[1].bar(range(len(sorted_f1)), sorted_f1, color='steelblue', alpha=0.7)
    axes[1].axhline(y=THRESHOLDS["bertscore_f1"], color='red', linestyle='--', label=f'Threshold ({THRESHOLDS["bertscore_f1"]})')
    axes[1].set_xlabel('Sample (sorted)')
    axes[1].set_ylabel('BERTScore F1')
    axes[1].set_title('BERTScore F1 per Sample (Sorted)')
    axes[1].legend()
    
    # Plot 3: Headword Recall Summary
    correct = sum(headword_correct)
    incorrect = len(headword_correct) - correct
    axes[2].bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green', 'red'], alpha=0.7)
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Headword Recall: {correct}/{len(headword_correct)} ({correct/len(headword_correct)*100:.1f}%)')
    for i, v in enumerate([correct, incorrect]):
        axes[2].text(i, v + 0.5, str(int(v)), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    filename = f"test1_plots_{run_name}.png" if run_name else "test1_plots.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plots saved: {plot_path}")
    return plot_path


def run_test(model, tokenizer, test_data_path: str = DEFAULT_TEST_DATA, output_path: str = DEFAULT_OUTPUT, 
             max_new_tokens: int = 256, run_name: str = None, batch_size: int = 16) -> Dict:
    print(f"Loading test data from: {test_data_path}")
    test_data = load_test_data(test_data_path)
    print(f"Loaded {len(test_data)} samples")
    
    questions = [d["question"] for d in test_data]
    ground_truths = [d["ground_truth"] for d in test_data]
    
    print(f"\nGenerating predictions (batch_size={batch_size})...")
    predictions = []
    
    # Process in batches
    for i in tqdm(range(0, len(questions), batch_size), desc="Batch Inference"):
        batch_questions = questions[i:i + batch_size]
        
        # Prepare batch messages in Gemma-3 multimodal format
        batch_messages = [[{
            "role": "user",
            "content": [{"type": "text", "text": q}]
        }] for q in batch_questions]
        
        # Tokenize batch with padding
        batch_inputs = tokenizer.apply_chat_template(
            batch_messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            padding=True,
            return_tensors="pt", 
            return_dict=True
        ).to(model.device)
        
        # Generate for batch
        outputs = model.generate(
            **batch_inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode each output in batch
        for j, output in enumerate(outputs):
            input_len = batch_inputs['input_ids'][j].shape[0]
            pred = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            predictions.append(pred.strip())
    
    print("\nComputing metrics...")
    bertscore = compute_bertscore(predictions, ground_truths)
    headword = compute_headword_recall(predictions, ground_truths)
    abbreviation = compute_abbreviation_match(predictions, ground_truths)
    
    results = {
        "bertscore": {k: v for k, v in bertscore.items() if k != "raw_f1"},
        "headword_recall": {k: v for k, v in headword.items() if k not in ["raw", "headwords"]},
        "abbreviation_match": {k: v for k, v in abbreviation.items() if k not in ["raw", "abbreviations"]},
        "passed": {
            "bertscore": bertscore["f1_mean"] >= THRESHOLDS["bertscore_f1"],
            "headword": headword["accuracy"] >= THRESHOLDS["headword_recall"]
        },
        "num_samples": len(predictions),
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
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
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Save detailed results
    output_dir = os.path.dirname(output_path)
    save_detailed_results(
        output_dir, questions, predictions, ground_truths,
        bertscore["raw_f1"], headword["raw"], headword["headwords"], run_name
    )
    
    # Save plots
    save_plots(output_dir, bertscore["raw_f1"], headword["raw"], run_name)
    
    return results
