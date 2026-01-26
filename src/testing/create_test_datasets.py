"""
Create test dataset for Test 1 (Knowledge Recall).
Test 2 uses KoMMLU from HuggingFace at runtime.
Test 3 reuses Test 1 & 2 results.
"""

import os
import json
import random
from typing import List, Dict

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def create_knowledge_recall_dataset(training_data_path: str, output_path: str, sample_ratio: float = 0.05) -> None:
    print(f"Loading training data from: {training_data_path}")
    
    data = []
    with open(training_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Total training samples: {len(data)}")
    sample_size = max(1, int(len(data) * sample_ratio))
    print(f"Sampling {sample_ratio*100:.1f}% = {sample_size} samples")
    
    sampled_data = random.sample(data, sample_size)
    test_data = [{"question": item["messages"][0]["content"], "ground_truth": item["messages"][1]["content"]} for item in sampled_data]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created: {output_path} ({len(test_data)} samples)")


def main():
    cwd = os.getcwd()
    if not cwd.endswith("unsloth-learning"):
        raise ValueError("Please run from unsloth-learning directory")
    
    data_dir = os.path.join(cwd, "data", "data_cleaned")
    test_dir = os.path.join(data_dir, "test_datasets")
    os.makedirs(test_dir, exist_ok=True)
    
    print("=" * 50)
    print("CREATING TEST DATASETS")
    print("=" * 50)
    
    create_knowledge_recall_dataset(
        training_data_path=os.path.join(data_dir, "training_data.jsonl"),
        output_path=os.path.join(test_dir, "test1_knowledge_recall.jsonl"),
        sample_ratio=0.05
    )
    
    print("\nTest 2: KoMMLU loaded at runtime")
    print("Test 3: Compares Test 1 & 2 results")
    print("\nDone!")


if __name__ == "__main__":
    main()
