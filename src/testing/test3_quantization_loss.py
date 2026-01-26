"""
Test 3: Quantization Impact - Compare FP16 vs GGUF by running Test 1 & 2 on both.
"""

import os
import json
from typing import Dict
from dataclasses import dataclass

DEFAULT_OUTPUT = "results/test3_results.json"
MAX_PERFORMANCE_DROP = 0.05


@dataclass
class ModelResults:
    test1_bertscore_f1: float
    test1_headword_recall: float
    test2_accuracy: float
    model_type: str


def compare_results(fp16_results: ModelResults, gguf_results: ModelResults) -> Dict:
    test1_bertscore_drop = fp16_results.test1_bertscore_f1 - gguf_results.test1_bertscore_f1
    test1_headword_drop = fp16_results.test1_headword_recall - gguf_results.test1_headword_recall
    test2_accuracy_drop = fp16_results.test2_accuracy - gguf_results.test2_accuracy
    
    results = {
        "fp16": {"test1_bertscore_f1": fp16_results.test1_bertscore_f1, "test1_headword_recall": fp16_results.test1_headword_recall, "test2_accuracy": fp16_results.test2_accuracy},
        "gguf": {"test1_bertscore_f1": gguf_results.test1_bertscore_f1, "test1_headword_recall": gguf_results.test1_headword_recall, "test2_accuracy": gguf_results.test2_accuracy},
        "performance_drop": {"test1_bertscore_f1": test1_bertscore_drop, "test1_headword_recall": test1_headword_drop, "test2_accuracy": test2_accuracy_drop},
        "passed": {
            "test1_bertscore": test1_bertscore_drop <= MAX_PERFORMANCE_DROP,
            "test1_headword": test1_headword_drop <= MAX_PERFORMANCE_DROP,
            "test2_accuracy": test2_accuracy_drop <= MAX_PERFORMANCE_DROP
        }
    }
    results["overall_passed"] = all(results["passed"].values())
    return results


def print_comparison(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("TEST 3: QUANTIZATION IMPACT")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'FP16':>10} {'GGUF':>10} {'Drop':>10} {'Status':>10}")
    print("-" * 65)
    
    fp16, gguf, drop, passed = results["fp16"], results["gguf"], results["performance_drop"], results["passed"]
    print(f"{'Test1 BERTScore F1':<25} {fp16['test1_bertscore_f1']:>10.4f} {gguf['test1_bertscore_f1']:>10.4f} {drop['test1_bertscore_f1']:>+10.4f} {'PASS' if passed['test1_bertscore'] else 'FAIL':>10}")
    print(f"{'Test1 Headword Recall':<25} {fp16['test1_headword_recall']:>10.2%} {gguf['test1_headword_recall']:>10.2%} {drop['test1_headword_recall']:>+10.2%} {'PASS' if passed['test1_headword'] else 'FAIL':>10}")
    print(f"{'Test2 Accuracy':<25} {fp16['test2_accuracy']:>10.2%} {gguf['test2_accuracy']:>10.2%} {drop['test2_accuracy']:>+10.2%} {'PASS' if passed['test2_accuracy'] else 'FAIL':>10}")
    print("-" * 65)
    print(f"\nMax allowed drop: {MAX_PERFORMANCE_DROP:.0%}")
    print(f"OVERALL: {'PASSED' if results['overall_passed'] else 'FAILED'}")


def run_test(fp16_test1_results: Dict, fp16_test2_results: Dict, gguf_test1_results: Dict, gguf_test2_results: Dict, output_path: str = DEFAULT_OUTPUT) -> Dict:
    fp16 = ModelResults(
        test1_bertscore_f1=fp16_test1_results["bertscore"]["f1_mean"],
        test1_headword_recall=fp16_test1_results["headword_recall"]["accuracy"],
        test2_accuracy=fp16_test2_results["finetuned"]["accuracy"],
        model_type="fp16"
    )
    gguf = ModelResults(
        test1_bertscore_f1=gguf_test1_results["bertscore"]["f1_mean"],
        test1_headword_recall=gguf_test1_results["headword_recall"]["accuracy"],
        test2_accuracy=gguf_test2_results["finetuned"]["accuracy"],
        model_type="gguf"
    )
    
    results = compare_results(fp16, gguf)
    print_comparison(results)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


def run_test_from_files(fp16_test1_path: str, fp16_test2_path: str, gguf_test1_path: str, gguf_test2_path: str, output_path: str = DEFAULT_OUTPUT) -> Dict:
    with open(fp16_test1_path, 'r') as f:
        fp16_t1 = json.load(f)
    with open(fp16_test2_path, 'r') as f:
        fp16_t2 = json.load(f)
    with open(gguf_test1_path, 'r') as f:
        gguf_t1 = json.load(f)
    with open(gguf_test2_path, 'r') as f:
        gguf_t2 = json.load(f)
    return run_test(fp16_t1, fp16_t2, gguf_t1, gguf_t2, output_path)
