"""
Test 3: Quantization Impact - Compare FP16 vs GGUF by running Test 1 & 2 on both.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from dataclasses import dataclass
from datetime import datetime

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
        },
        "timestamp": datetime.now().isoformat(),
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


def save_comparison_plot(output_dir: str, results: Dict, run_name: str = None):
    fp16 = results["fp16"]
    gguf = results["gguf"]
    drop = results["performance_drop"]
    passed = results["passed"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Side-by-side comparison
    metrics = ['BERTScore F1', 'Headword Recall', 'KoMMLU Accuracy']
    fp16_values = [fp16['test1_bertscore_f1'], fp16['test1_headword_recall'], fp16['test2_accuracy']]
    gguf_values = [gguf['test1_bertscore_f1'], gguf['test1_headword_recall'], gguf['test2_accuracy']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, fp16_values, width, label='FP16', color='steelblue', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, gguf_values, width, label='GGUF (q4_k_m)', color='coral', alpha=0.8)
    
    axes[0].set_ylabel('Score')
    axes[0].set_title('FP16 vs GGUF Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    
    for bar, val in zip(bars1, fp16_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, gguf_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=9)
    
    # Plot 2: Performance drop with threshold
    drop_values = [drop['test1_bertscore_f1'], drop['test1_headword_recall'], drop['test2_accuracy']]
    pass_status = [passed['test1_bertscore'], passed['test1_headword'], passed['test2_accuracy']]
    colors = ['green' if p else 'red' for p in pass_status]
    
    bars = axes[1].bar(metrics, drop_values, color=colors, alpha=0.7)
    axes[1].axhline(y=MAX_PERFORMANCE_DROP, color='red', linestyle='--', linewidth=2, label=f'Max Drop Threshold ({MAX_PERFORMANCE_DROP:.0%})')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Performance Drop (FP16 - GGUF)')
    axes[1].set_title('Quantization Performance Drop')
    axes[1].legend()
    
    for bar, val, status in zip(bars, drop_values, pass_status):
        label = f'{val:+.3f}\n{"PASS" if status else "FAIL"}'
        y_pos = bar.get_height() + 0.005 if bar.get_height() >= 0 else bar.get_height() - 0.02
        axes[1].text(bar.get_x() + bar.get_width()/2, y_pos, label, ha='center', fontsize=9, fontweight='bold')
    
    # Set y-axis limits for drop plot
    max_drop = max(abs(d) for d in drop_values)
    axes[1].set_ylim(-max(0.1, max_drop + 0.05), max(0.1, max_drop + 0.05))
    
    plt.tight_layout()
    
    filename = f"test3_comparison_{run_name}.png" if run_name else "test3_comparison.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison plot saved: {plot_path}")
    return plot_path


def run_test(fp16_test1_results: Dict, fp16_test2_results: Dict, gguf_test1_results: Dict, gguf_test2_results: Dict, 
             output_path: str = DEFAULT_OUTPUT, run_name: str = None) -> Dict:
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
    results["run_name"] = run_name
    print_comparison(results)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Save comparison plot
    output_dir = os.path.dirname(output_path)
    save_comparison_plot(output_dir, results, run_name)
    
    return results


def run_test_from_files(fp16_test1_path: str, fp16_test2_path: str, gguf_test1_path: str, gguf_test2_path: str, 
                        output_path: str = DEFAULT_OUTPUT, run_name: str = None) -> Dict:
    with open(fp16_test1_path, 'r') as f:
        fp16_t1 = json.load(f)
    with open(fp16_test2_path, 'r') as f:
        fp16_t2 = json.load(f)
    with open(gguf_test1_path, 'r') as f:
        gguf_t1 = json.load(f)
    with open(gguf_test2_path, 'r') as f:
        gguf_t2 = json.load(f)
    return run_test(fp16_t1, fp16_t2, gguf_t1, gguf_t2, output_path, run_name)
