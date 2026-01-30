# Model Testing Documentation

This document describes the testing methodology for evaluating fine-tuned military vocabulary models.

## Overview

The testing suite consists of multiple tests designed to verify:
1. **Knowledge acquisition** - Did the model learn the military vocabulary?
2. **Stability** - Did fine-tuning cause catastrophic forgetting?
3. **Quantization quality** - Does the GGUF model retain performance?

## Test Scripts

### Test 1: Knowledge Recall (`test1_knowledge_recall.py`)

**Purpose:** Verify the model learned the military vocabulary from training data.

**What it evaluates:**
- Semantic similarity between predictions and ground truth (BERTScore)
- Presence of correct Korean headwords in responses
- Correct mention of abbreviations (e.g., TBM, LGW)

**Metrics:**
| Metric | Threshold | Description |
|--------|-----------|-------------|
| BERTScore F1 | ≥ 0.70 | Semantic similarity using multilingual BERT |
| Headword Recall | ≥ 0.80 | Correct Korean term appears in response |
| Abbreviation Match | Informational | Abbreviations correctly mentioned |

**Test Data Format:**
```json
{"question": "전술 탄도 미사일이란?", "ground_truth": "전술 탄도 미사일은(는)..."}
```

**Pass Criteria:** BERTScore ≥ 0.70 AND Headword Recall ≥ 0.80

---

### Test 2: Stability Check (`test2_stability_check.py`)

**Purpose:** Verify no catastrophic forgetting occurred during fine-tuning.

**What it evaluates:**
- General Korean language understanding via KoMMLU benchmark
- Multiple-choice question answering accuracy
- Performance breakdown by subject area

**Dataset:** KoMMLU (Korean Massive Multitask Language Understanding)
- Subjects: korean_history, korean_geography, general_knowledge, civil_law, criminal_law
- 20 samples per subject (100 total)

**Metrics:**
| Metric | Threshold | Description |
|--------|-----------|-------------|
| Overall Accuracy | ≥ 0.30 | Percentage of correct MCQ answers |
| Performance Drop | < 10% | Comparison vs base model (if available) |

**MCQ Format:**
```
질문: [Question]
A. [Choice A]
B. [Choice B]
C. [Choice C]
D. [Choice D]

정답을 A, B, C, D 중 하나로 답하세요. 정답:
```

**Pass Criteria:** Accuracy ≥ 30% AND (if base model compared) drop < 10%

---

### Test 3: Quantization Comparison (`test3_quantization_loss.py`)

**Purpose:** Measure quality loss from FP16 → GGUF quantization.

**What it evaluates:**
- Comparison of Test 1 results (FP16 vs GGUF)
- Comparison of Test 2 results (FP16 vs GGUF)
- Quantization loss percentage

**Metrics:**
| Metric | Threshold | Description |
|--------|-----------|-------------|
| BERTScore Drop | < 5% | Acceptable semantic similarity loss |
| Accuracy Drop | < 5% | Acceptable MCQ accuracy loss |

**Pass Criteria:** All metric drops < 5%

---

## Test Execution Scripts

### step_03_test_fp16.py

Tests the FP16 LoRA model before quantization.

```bash
# Test local LoRA model
python src/training/step_03_test_fp16.py

# Test a HuggingFace model directly
python src/training/step_03_test_fp16.py --hf-model google/gemma-3-4b-it

# Skip specific tests
python src/training/step_03_test_fp16.py --skip-test1

# Adjust batch size
python src/training/step_03_test_fp16.py --batch-size 8
```

### step_05_test_gguf.py

Tests the quantized GGUF model and compares with FP16.

```bash
# Test auto-selected GGUF model
python src/training/step_05_test_gguf.py

# Specify a GGUF file
python src/training/step_05_test_gguf.py --model path/to/model.gguf

# Skip comparison (run only Test 1 and 2)
python src/training/step_05_test_gguf.py --skip-test3
```

---

## Batch Processing

All tests use **batch processing with left padding** for efficient inference:

```python
from batch_utils import batch_generate, format_chat_prompts

# Format prompts
prompts = format_chat_prompts(questions, tokenizer, multimodal_format=True)

# Generate with batch processing
predictions = batch_generate(
    model, tokenizer, prompts,
    max_new_tokens=256,
    batch_size=16,
    desc="Knowledge Recall"
)
```

**Why left padding?**
- For generation, sequences must end at the same position
- Left padding ensures all sequences are ready for generation at the last token
- Right padding would corrupt the generation context for shorter sequences

---

## Output Files

Results are saved to `results/<run_name>/`:

| File | Description |
|------|-------------|
| `fp16_test1_<run>.json` | Test 1 results for FP16 model |
| `fp16_test2_<run>.json` | Test 2 results for FP16 model |
| `gguf_test1_<run>.json` | Test 1 results for GGUF model |
| `gguf_test2_<run>.json` | Test 2 results for GGUF model |
| `test3_<run>.json` | Quantization comparison results |
| `test1_plots_<run>.png` | Visualization of Test 1 metrics |
| `test2_plots_<run>.png` | Visualization of Test 2 accuracy by subject |
| `test2_detailed_<run>.csv` | Detailed per-sample results |

---

## Interpreting Results

### Passing Scores

| Test | Minimum Passing Score |
|------|----------------------|
| Test 1 BERTScore | 0.70 (70%) |
| Test 1 Headword | 0.80 (80%) |
| Test 2 Accuracy | 0.30 (30%) |
| Test 3 Drop | < 0.05 (5%) |

### What low scores indicate

| Low Score | Possible Cause |
|-----------|----------------|
| Low BERTScore | Model not generating relevant content |
| Low Headword Recall | Model hallucinating wrong terms |
| Low Test 2 Accuracy | Catastrophic forgetting - model lost general knowledge |
| High Test 3 Drop | Quantization too aggressive, try a less aggressive quant |

### Recommended actions

1. **Low Test 1 scores:** Check training data quality, increase training epochs
2. **Low Test 2 scores:** Reduce learning rate, add more diverse training data
3. **High Test 3 drop:** Use less aggressive quantization (e.g., Q8_0 instead of Q4_K_M)
