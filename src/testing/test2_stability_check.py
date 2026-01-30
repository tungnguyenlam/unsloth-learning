"""
Test 2: Stability Check - Prove no catastrophic forgetting using KoMMLU.

PURPOSE:
    This test evaluates whether fine-tuning on military vocabulary has caused
    "catastrophic forgetting" - loss of the model's general Korean language
    understanding and reasoning abilities.

DATASET:
    KoMMLU (Korean Massive Multitask Language Understanding)
    - A Korean benchmark for evaluating language models on diverse topics
    - Subjects tested: korean_history, korean_geography, general_knowledge, 
      civil_law, criminal_law
    - 20 samples per subject (100 total by default)

EVALUATION METRICS:
    1. Overall Accuracy (minimum threshold: 0.30)
       - Percentage of correct MCQ answers
       - Threshold is conservative since fine-tuning focuses on vocabulary, not MCQ
    
    2. Per-Subject Accuracy
       - Breakdown by knowledge domain
       - Helps identify if specific areas were affected
    
    3. Performance Drop vs Base Model (optional, threshold: <10%)
       - Only computed if base model evaluation is enabled
       - Measures degradation from original model capabilities

MCQ FORMAT:
    질문: [Question]
    A. [Choice A]
    B. [Choice B]
    C. [Choice C]
    D. [Choice D]
    
    정답을 A, B, C, D 중 하나로 답하세요. 정답:

ANSWER EXTRACTION:
    - First checks if response starts with A/B/C/D
    - Falls back to finding first occurrence of A/B/C/D in response

PASS CRITERIA:
    - Accuracy >= 0.30 (30%)
    - If base model compared: Performance drop < 10%

USAGE:
    # Called from step_03_test_fp16.py or step_05_test_gguf.py
    import test2_stability_check as test2
    test2.run_test(model, tokenizer, output_path)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from tqdm import tqdm
from datetime import datetime

DEFAULT_OUTPUT = "results/test2_results.json"
MAX_PERFORMANCE_DROP = 0.10
MIN_ACCURACY = 0.30
KOMMLU_SUBJECTS = ["korean_history", "korean_geography", "general_knowledge", "civil_law", "criminal_law"]
SAMPLES_PER_SUBJECT = 20


def load_kommlu_data(subjects: List[str] = None, samples_per_subject: int = 20) -> List[Dict]:
    from datasets import load_dataset
    if subjects is None:
        subjects = KOMMLU_SUBJECTS
    
    all_data = []
    for subject in subjects:
        try:
            print(f"Loading KoMMLU subject: {subject}")
            dataset = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")
            if len(dataset) > samples_per_subject:
                indices = np.random.choice(len(dataset), samples_per_subject, replace=False)
                samples = [dataset[int(i)] for i in indices]
            else:
                samples = [dataset[i] for i in range(len(dataset))]
            
            for item in samples:
                all_data.append({
                    "question": item["question"],
                    "choices": [item["A"], item["B"], item["C"], item["D"]],
                    "answer": ord(item["answer"]) - ord("A"),
                    "subject": subject
                })
        except Exception as e:
            print(f"Warning: Could not load subject {subject}: {e}")
    
    print(f"Loaded {len(all_data)} total samples from KoMMLU")
    return all_data


def format_mcq_prompt(question: str, choices: List[str]) -> str:
    prompt = f"질문: {question}\n\n"
    for i, choice in enumerate(choices):
        prompt += f"{['A', 'B', 'C', 'D'][i]}. {choice}\n"
    prompt += "\n정답을 A, B, C, D 중 하나로 답하세요. 정답:"
    return prompt


def extract_answer(response: str) -> Optional[int]:
    response = response.upper().strip()
    for i, letter in enumerate(["A", "B", "C", "D"]):
        if response.startswith(letter):
            return i
    for i, letter in enumerate(["A", "B", "C", "D"]):
        if letter in response:
            return i
    return None


def compute_accuracy(predictions: List[int], ground_truths: List[int]) -> Dict:
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    return {"accuracy": correct / len(ground_truths) if ground_truths else 0, "correct": correct, "total": len(ground_truths)}


def compute_accuracy_by_subject(predictions: List[int], ground_truths: List[int], subjects: List[str]) -> Dict:
    subject_results = {}
    for subj in set(subjects):
        indices = [i for i, s in enumerate(subjects) if s == subj]
        subj_preds = [predictions[i] for i in indices]
        subj_gts = [ground_truths[i] for i in indices]
        correct = sum(1 for p, g in zip(subj_preds, subj_gts) if p == g)
        subject_results[subj] = {
            "accuracy": correct / len(subj_gts) if subj_gts else 0,
            "correct": correct,
            "total": len(subj_gts)
        }
    return subject_results


def save_detailed_results(output_dir: str, test_data: List[Dict], predictions: List[int], 
                          responses: List[str], run_name: str = None):
    detailed = []
    for i, (data, pred, resp) in enumerate(zip(test_data, predictions, responses)):
        detailed.append({
            "index": i,
            "question": data["question"],
            "choices": data["choices"],
            "ground_truth_idx": data["answer"],
            "ground_truth": ["A", "B", "C", "D"][data["answer"]],
            "prediction_idx": pred,
            "prediction": ["A", "B", "C", "D"][pred] if pred >= 0 else "INVALID",
            "response": resp,
            "correct": pred == data["answer"],
            "subject": data["subject"]
        })
    
    filename = f"test2_detailed_{run_name}.jsonl" if run_name else "test2_detailed.jsonl"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in detailed:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Detailed results saved: {output_path}")
    return output_path


def save_plots(output_dir: str, subject_results: Dict, overall_accuracy: float, run_name: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Accuracy by Subject
    subjects = list(subject_results.keys())
    accuracies = [subject_results[s]["accuracy"] for s in subjects]
    colors = ['green' if acc >= MIN_ACCURACY else 'red' for acc in accuracies]
    
    bars = axes[0].bar(range(len(subjects)), accuracies, color=colors, alpha=0.7)
    axes[0].axhline(y=MIN_ACCURACY, color='red', linestyle='--', label=f'Min Threshold ({MIN_ACCURACY})')
    axes[0].axhline(y=overall_accuracy, color='blue', linestyle='-', linewidth=2, label=f'Overall ({overall_accuracy:.2%})')
    axes[0].set_xticks(range(len(subjects)))
    axes[0].set_xticklabels([s.replace('_', '\n') for s in subjects], rotation=0, fontsize=9)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Subject')
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc='upper right')
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{acc:.1%}', ha='center', fontsize=9)
    
    # Plot 2: Correct vs Incorrect by Subject
    correct_counts = [subject_results[s]["correct"] for s in subjects]
    total_counts = [subject_results[s]["total"] for s in subjects]
    incorrect_counts = [t - c for c, t in zip(correct_counts, total_counts)]
    
    x = np.arange(len(subjects))
    width = 0.35
    axes[1].bar(x - width/2, correct_counts, width, label='Correct', color='green', alpha=0.7)
    axes[1].bar(x + width/2, incorrect_counts, width, label='Incorrect', color='red', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.replace('_', '\n') for s in subjects], rotation=0, fontsize=9)
    axes[1].set_ylabel('Count')
    axes[1].set_title('Correct vs Incorrect by Subject')
    axes[1].legend()
    
    plt.tight_layout()
    
    filename = f"test2_plots_{run_name}.png" if run_name else "test2_plots.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plots saved: {plot_path}")
    return plot_path


def run_test(finetuned_model, finetuned_tokenizer, base_model=None, base_tokenizer=None, 
             test_data: List[Dict] = None, output_path: str = DEFAULT_OUTPUT, run_name: str = None,
             batch_size: int = 32) -> Dict:
    if test_data is None:
        print("Loading KoMMLU data...")
        test_data = load_kommlu_data()
    
    questions = [format_mcq_prompt(d["question"], d["choices"]) for d in test_data]
    ground_truths = [d["answer"] for d in test_data]
    subjects = [d["subject"] for d in test_data]
    
    # Use batch processing
    from batch_utils import batch_generate
    
    print(f"\nEvaluating fine-tuned model (batch_size={batch_size})...")
    ft_responses = batch_generate(
        finetuned_model, finetuned_tokenizer, questions,
        max_new_tokens=16,
        batch_size=batch_size,
        desc="MCQ Eval"
    )
    
    # Extract answers
    ft_predictions = []
    for resp in ft_responses:
        ans = extract_answer(resp)
        ft_predictions.append(ans if ans is not None else -1)
    
    # Debug: print first few
    for i in range(min(2, len(ft_responses))):
        print(f"    Sample {i}: Response='{ft_responses[i][:50]}' -> Pred={ft_predictions[i]}")
    
    ft_accuracy = compute_accuracy(ft_predictions, ground_truths)
    ft_by_subject = compute_accuracy_by_subject(ft_predictions, ground_truths, subjects)
    
    results = {
        "finetuned": {
            "accuracy": ft_accuracy["accuracy"], 
            "correct": ft_accuracy["correct"], 
            "total": ft_accuracy["total"],
            "by_subject": ft_by_subject
        },
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
    }
    
    if base_model is not None:
        print(f"\nEvaluating base model (batch_size={batch_size})...")
        base_responses = batch_generate(
            base_model, base_tokenizer, questions,
            max_new_tokens=16,
            batch_size=batch_size,
            desc="Base MCQ"
        )
        
        # Extract answers
        base_predictions = []
        for resp in base_responses:
            ans = extract_answer(resp)
            base_predictions.append(ans if ans is not None else -1)
        
        base_accuracy = compute_accuracy(base_predictions, ground_truths)
        base_by_subject = compute_accuracy_by_subject(base_predictions, ground_truths, subjects)
        results["base"] = {"accuracy": base_accuracy["accuracy"], "by_subject": base_by_subject}
        drop = results["base"]["accuracy"] - results["finetuned"]["accuracy"]
        results["performance_drop"] = drop
        results["passed"] = {"drop_acceptable": drop <= MAX_PERFORMANCE_DROP, "min_accuracy_met": results["finetuned"]["accuracy"] >= MIN_ACCURACY}
        results["overall_passed"] = all(results["passed"].values())
    else:
        results["passed"] = {"min_accuracy_met": results["finetuned"]["accuracy"] >= MIN_ACCURACY}
        results["overall_passed"] = results["passed"]["min_accuracy_met"]
    
    print("\n" + "=" * 50)
    print("TEST 2: STABILITY CHECK (KoMMLU)")
    print("=" * 50)
    print(f"\nFine-tuned Accuracy: {results['finetuned']['accuracy']:.2%}")
    print("\nBy Subject:")
    for subj, data in ft_by_subject.items():
        print(f"  {subj}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
    if base_model:
        print(f"\nBase Accuracy: {results['base']['accuracy']:.2%}")
        print(f"Performance Drop: {results['performance_drop']:.2%} ({'PASS' if results['passed']['drop_acceptable'] else 'FAIL'})")
    print(f"\nOVERALL: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Save detailed results
    output_dir = os.path.dirname(output_path)
    save_detailed_results(output_dir, test_data, ft_predictions, ft_responses, run_name)
    
    # Save plots
    save_plots(output_dir, ft_by_subject, ft_accuracy["accuracy"], run_name)
    
    return results
