"""
Step 01: Prepare Data
- Verify training data exists
- Auto-detect max sequence length
- Create test datasets for evaluation

Run: python step_01_prepare_data.py
"""

import os
import sys
import json
import random

from step_00_config import (
    MODEL_NAME, TRAINING_DATA_PATH, TEST_DATA_DIR, TEST1_DATA_PATH,
    ensure_dirs, print_config, get_base_parser,
    save_detected_config, round_to_bucket, MAX_SEQ_LENGTH_DEFAULT
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def verify_training_data(data_path: str) -> list:
    if not os.path.exists(data_path):
        print(f"ERROR: Training data not found at: {data_path}")
        print("Please run data processing steps first:")
        print("  python src/data_processing/01_combine_dataset.py")
        print("  python src/data_processing/02_generate_training_data.py")
        sys.exit(1)
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Training data: {len(data)} samples")
    return data


def analyze_sequence_lengths(data: list) -> dict:
    print("\nAnalyzing sequence lengths...")
    
    from transformers import AutoTokenizer
    print(f"  Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    lengths = []
    for item in data:
        messages = item["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer.encode(text)
        lengths.append(len(tokens))
    
    import numpy as np
    lengths = np.array(lengths)
    
    stats = {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "num_samples": len(lengths),
    }
    
    recommended = round_to_bucket(int(stats["p99"]) + 32)
    stats["recommended_max_seq_length"] = min(recommended, MAX_SEQ_LENGTH_DEFAULT)
    
    print(f"\n  Sequence Length Statistics:")
    print(f"    Min:    {stats['min']:,} tokens")
    print(f"    Max:    {stats['max']:,} tokens")
    print(f"    Mean:   {stats['mean']:,.1f} tokens")
    print(f"    Median: {stats['median']:,.1f} tokens")
    print(f"    P95:    {stats['p95']:,.1f} tokens")
    print(f"    P99:    {stats['p99']:,.1f} tokens")
    print(f"\n  Recommended max_seq_length: {stats['recommended_max_seq_length']}")
    
    if stats['max'] > stats['recommended_max_seq_length']:
        over_limit = sum(1 for l in lengths if l > stats['recommended_max_seq_length'])
        print(f"  WARNING: {over_limit} samples ({over_limit/len(lengths)*100:.1f}%) exceed limit (will be truncated)")
    
    return stats


def create_test_dataset(training_data_path: str, output_path: str, sample_ratio: float = 0.05) -> None:
    print(f"\nCreating test dataset...")
    
    data = []
    with open(training_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    sample_size = max(1, int(len(data) * sample_ratio))
    print(f"  Sampling {sample_ratio*100:.1f}% = {sample_size} samples")
    
    sampled_data = random.sample(data, sample_size)
    test_data = [
        {"question": item["messages"][0]["content"], "ground_truth": item["messages"][1]["content"]}
        for item in sampled_data
    ]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"  Saved: {output_path}")


def main():
    parser = get_base_parser("Step 01: Prepare Data")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--skip-analysis", action="store_true", help="Skip sequence length analysis")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Override auto-detected max_seq_length")
    args = parser.parse_args()
    
    print("=" * 60)
    print("STEP 01: PREPARE DATA")
    print("=" * 60)
    
    ensure_dirs()
    print_config()
    
    print("\n[1/3] Verifying training data...")
    data = verify_training_data(args.data_path)
    
    print("\n[2/3] Analyzing sequence lengths...")
    if args.skip_analysis:
        print("  Skipped (--skip-analysis)")
        max_seq_length = args.max_seq_length or MAX_SEQ_LENGTH_DEFAULT
    else:
        stats = analyze_sequence_lengths(data)
        max_seq_length = args.max_seq_length or stats["recommended_max_seq_length"]
    
    detected_config = {
        "max_seq_length": max_seq_length,
        "num_samples": len(data),
    }
    save_detected_config(detected_config)
    
    print(f"\n  Using max_seq_length: {max_seq_length}")
    
    print("\n[3/3] Creating test datasets...")
    create_test_dataset(args.data_path, TEST1_DATA_PATH, args.test_ratio)
    
    print("\n" + "=" * 60)
    print("STEP 01 COMPLETE")
    print("=" * 60)
    print(f"\nDetected max_seq_length: {max_seq_length}")
    print("\nNext: python step_02_train.py")


if __name__ == "__main__":
    main()
