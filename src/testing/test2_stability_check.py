"""
Test 2: Stability Check - Prove no catastrophic forgetting using KoMMLU.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

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


def run_test(finetuned_model, finetuned_tokenizer, base_model=None, base_tokenizer=None, 
             test_data: List[Dict] = None, output_path: str = DEFAULT_OUTPUT) -> Dict:
    if test_data is None:
        print("Loading KoMMLU data...")
        test_data = load_kommlu_data()
    
    questions = [format_mcq_prompt(d["question"], d["choices"]) for d in test_data]
    ground_truths = [d["answer"] for d in test_data]
    subjects = [d["subject"] for d in test_data]
    
    print("\nEvaluating fine-tuned model...")
    ft_predictions = []
    for q in tqdm(questions, desc="Fine-tuned"):
        messages = [{"role": "user", "content": q}]
        inputs = finetuned_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        ).to(finetuned_model.device)
        outputs = finetuned_model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=finetuned_tokenizer.eos_token_id)
        resp = finetuned_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        ft_predictions.append(extract_answer(resp) if extract_answer(resp) is not None else -1)
    
    ft_accuracy = compute_accuracy(ft_predictions, ground_truths)
    results = {"finetuned": {"accuracy": ft_accuracy["accuracy"], "correct": ft_accuracy["correct"], "total": ft_accuracy["total"]}}
    
    if base_model is not None:
        print("\nEvaluating base model...")
        base_predictions = []
        for q in tqdm(questions, desc="Base model"):
            messages = [{"role": "user", "content": q}]
            inputs = base_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
            ).to(base_model.device)
            outputs = base_model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=base_tokenizer.eos_token_id)
            resp = base_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            base_predictions.append(extract_answer(resp) if extract_answer(resp) is not None else -1)
        
        base_accuracy = compute_accuracy(base_predictions, ground_truths)
        results["base"] = {"accuracy": base_accuracy["accuracy"]}
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
    if base_model:
        print(f"Base Accuracy: {results['base']['accuracy']:.2%}")
        print(f"Performance Drop: {results['performance_drop']:.2%} ({'PASS' if results['passed']['drop_acceptable'] else 'FAIL'})")
    print(f"\nOVERALL: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results
