import re
import numpy as np
from typing import List, Dict


def compute_bertscore(predictions: List[str], references: List[str]) -> Dict:
    from bert_score import score as bert_score
    P, R, F1 = bert_score(predictions, references, model_type="bert-base-multilingual-cased", verbose=False)
    return {"f1_mean": float(np.mean(F1.numpy())), "f1_std": float(np.std(F1.numpy()))}


def compute_headword_recall(predictions: List[str], ground_truths: List[str]) -> Dict:
    correct = 0
    total = 0
    for pred, gt in zip(predictions, ground_truths):
        match = re.match(r'^([가-힣]+)', gt)
        if match:
            headword = match.group(1)
            if headword in pred:
                correct += 1
            total += 1
    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


def run_validation(model, tokenizer, eval_dataset, sample_size: int = 100, max_new_tokens: int = 128) -> Dict:
    questions = []
    ground_truths = []
    
    for item in eval_dataset:
        messages = item["messages"]
        questions.append(messages[0]["content"])
        ground_truths.append(messages[1]["content"])
    
    if len(questions) > sample_size:
        indices = np.random.choice(len(questions), sample_size, replace=False)
        questions = [questions[i] for i in indices]
        ground_truths = [ground_truths[i] for i in indices]
    
    print(f"Running validation on {len(questions)} samples...")
    predictions = []
    
    for i, question in enumerate(questions):
        messages = [{"role": "user", "content": question}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        pred = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predictions.append(pred.strip())
        
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(questions)} done")
    
    print("Computing metrics...")
    bertscore = compute_bertscore(predictions, ground_truths)
    headword = compute_headword_recall(predictions, ground_truths)
    
    results = {
        "bertscore_f1": bertscore["f1_mean"],
        "bertscore_std": bertscore["f1_std"],
        "headword_recall": headword["accuracy"],
        "num_samples": len(predictions)
    }
    
    print(f"\nValidation Results:")
    print(f"  BERTScore F1:    {results['bertscore_f1']:.4f} ± {results['bertscore_std']:.4f}")
    print(f"  Headword Recall: {results['headword_recall']:.2%}")
    
    return results
