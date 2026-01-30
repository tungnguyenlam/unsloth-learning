# Training Data Generation

This document describes the process and methodology for generating training data for military vocabulary fine-tuning.

## Overview

The training data is generated from a combined military vocabulary dataset and converted into instruction-tuned Q&A pairs in Gemma 3 chat format.

## Data Processing Pipeline

### Step 1: Combine Dataset (`01_combine_dataset.py`)

Reads all `.xlsx` files from `data/mai_task/military/military_vocab/` and combines them into a single CSV file.

**Input:** Multiple Excel files with Korean military vocabulary
**Output:** `data/training_data/dataset.csv`

Column Mapping (Korean → English):
| Original (Korean) | Mapped (English) | Description |
|-------------------|------------------|-------------|
| 범주 | Category | Topic/field category |
| 표제어 | Headword | Main Korean term |
| 원어 | Original_Word | English equivalent |
| 한자 | Hanja | Chinese characters |
| 약어 | Abbreviation | Short form |
| 의미 | Meaning | Definition |
| 상위어 | Hypernym | Parent category ("is a type of") |
| 하위어 | Hyponym | Child categories (subtypes) |
| 동의어 | Synonym | Alternative terms |
| 관련어 | Related_Word | Related vocabulary |
| 사전명 | Dictionary_Name | Source dictionary |
| 출전 | Source | Original source |
| 등록일 | Registration_Date | Date added |

### Step 2: Generate Training Data (`02_generate_training_data.py`)

Converts the combined dataset into Q&A pairs for instruction tuning.

**Input:** `data/training_data/dataset.csv`
**Output:** `data/training_data/training_data.jsonl`

## Question Types

### Type 2: Classification
**Condition:** Requires `Hypernym` field
**Evaluation Match:** Prompt pattern #2 (e.g., "AGM-114 헬파이어는 어떤 종류의 미사일인가요?")
**Question Examples:**
- "{Headword}는 어떤 종류인가요?"
- "{Headword}은(는) 무엇의 한 종류인가요?"

**Response Format:** States that the term is a type of {Hypernym}, with brief meaning.

### Type 3: Reverse Abbreviation Lookup (NEW)
**Condition:** Requires `Abbreviation` field
**Evaluation Match:** Prompt pattern #7 (e.g., "LGW에 대해서 설명해줘.")
**Question Examples:**
- "{Abbreviation}의 전체 명칭은 무엇인가요?"
- "{Abbreviation}는 무엇의 약어인가요?"

**Response Format:** "{Abbreviation}는(은) {Headword}({Hanja}, {Original_Word})의 약어입니다."

### Type 4: Category Membership
**Condition:** Requires `Category` or `Hypernym` field
**Evaluation Match:** Prompt pattern #4 (e.g., "조병창(Arsenal)은 어떤 범주에 속하나요?")
**Question Examples:**
- "{Headword}은(는) 어떤 범주에 속하나요?"
- "{Headword}은(는) 어떤 분야에 해당하나요?"

**Response Format:** States the category and/or hypernym the term belongs to.

### Type 5: Mechanism/Principle (NEW)
**Condition:** Always available if `Meaning` exists
**Evaluation Match:** Prompt pattern #5 (e.g., "정밀 타격 체계는 어떤 원리로 표적에 정밀하게 타격하는가?")
**Question Examples:**
- "{Headword}은(는) 어떤 원리로 작동하나요?"
- "{Headword}의 작동 원리는 무엇인가요?"

**Response Format:** "{Headword}의 작동 원리는 다음과 같습니다: {Meaning}"

### Type 6: Characteristics
**Condition:** Always available if `Meaning` exists
**Evaluation Match:** Prompt pattern #3, #6 (e.g., "정밀 유도 박격포탄은 어떤 특성을 가집니까?")
**Question Examples:**
- "{Headword}은(는) 어떤 특성을 가지고 있나요?"
- "{Headword}의 주요 특징은 무엇인가요?"

**Response Format:** Describes the key characteristics and optionally lists hyponyms.

### Type 7: Open Explanation (Primary Type)
**Condition:** Always available if `Headword` and `Meaning` exist
**Evaluation Match:** Prompt pattern #7 (general explanation)
**Question Examples:**
- "{Headword}에 대해 설명해주세요."
- "{Headword}이(가) 무엇인가요?"

**Variants:**
- **7_abbrev:** Uses `Abbreviation` as the question term (e.g., "TBM란?")
- **7_english:** Uses `Original_Word` as the question term (e.g., "Tactical Ballistic Missile가 무엇인가요?")

**Response Format:** Comprehensive response using all available fields.

### Type 8: Information Extraction (NEW)
**Condition:** Always available
**Evaluation Match:** Prompt patterns #8-12 (RAG-style context extraction)
**Question Examples:**
- "다음 정보에서 {Headword}의 정의를 찾아주세요: {context}"
- "아래 내용에서 {Headword}에 대한 설명을 추출해주세요: {context}"

**Context Format:** Messy pipe-separated data like:
`분류: 화력 | 용어명: 전술 탄도 미사일 | 한자: 戰術彈道missile | 영문: Tactical Ballistic Missile | 약어: TBM | 정의: ...`

**Response Format:** "{Headword}의 정의: {Meaning}"

## Response Generation

### Full Response Structure (Type 7)
A complete response includes (when available):
1. **Term with variants:** `{Headword}({Hanja}, {Original_Word})은(는) {Meaning}`
2. **Abbreviation:** `☜ {Abbreviation}`
3. **Category:** `이 용어는 {Category}에 속한다.`
4. **Hypernym:** `{Headword}은(는) {Hypernym}의 한 종류이다.`
5. **Hyponym:** `예를 들어, {Headword}의 하위 개념으로는 {Hyponym} 등이 있다.`
6. **Synonym:** `동의어로는 {Synonym}를(을) 들 수 있다.`
7. **Related words:** `관련어로는 {Related_Word}가(이) 있다.`
8. **Dictionary source:** `이 용어는 {Dictionary_Name}에 수록되어 있다.`

### Example Response
```
자외선 필터(紫外線filter, UV Filter)은(는) 대기 중에 끼어 있는 400nm 이하의 단파장 자외선을 흡수해서... ☜ UV Filter 이 용어는 화력에 속한다. 자외선 필터은(는) 광학 필터의 한 종류이다. 동의어로는 자외선 흡수 필터를(을) 들 수 있다. 관련어로는 사진 필터가(이) 있다. 이 용어는 국방과학기술용어사전에 수록되어 있다.
```

## Data Statistics (as of 2026-01-30)

| Type | Description | Count |
|------|-------------|-------|
| Type 2 | Classification | 5,348 |
| Type 3 | Reverse Abbreviation Lookup | 4,049 |
| Type 4 | Category Membership | 22,952 |
| Type 5 | Mechanism/Principle | 22,966 |
| Type 6 | Characteristics | 23,029 |
| Type 7 | Open Explanation | 23,032 |
| Type 7_abbrev | Abbreviation Explanation | 4,053 |
| Type 7_english | English Term Explanation | 11,754 |
| Type 8 | Information Extraction | 22,833 |
| **Total** | | **140,016** |

- **QA_PAIRS_PER_ROW:** 4 (randomly selected from available types)
- **Dataset Rows:** ~35,000 vocabulary entries
- **Format:** JSONL with Gemma 3 chat template

## Training Recommendation

With **~140,000 diverse samples**, we recommend:
- **Epochs:** 1 (more diverse data = faster learning)
- **Rationale:** Each vocabulary term appears with 4 different question types, so the model learns the *concept* rather than memorizing specific Q&A patterns

The diversity of question types enables better generalization:
- Same term, different angles → model learns underlying knowledge
- RAG-style extraction (Type 8) → prepares for real-world document Q&A
- Abbreviation lookups (Type 3) → bidirectional term recognition

## Columns Usage Summary

| Column | Used In Questions | Used In Responses | Question Types |
|--------|-------------------|-------------------|----------------|
| Headword | ✅ All types | ✅ All types | 2,4,5,6,7,8 |
| Meaning | ❌ | ✅ All types | 2,3,4,5,6,7,8 |
| Original_Word | ✅ Type 7_english | ✅ Term representation | 7_english |
| Hanja | ❌ | ✅ Term representation | 2,3,5,7 |
| Abbreviation | ✅ Type 3, 7_abbrev | ✅ After meaning | 3, 7_abbrev |
| Category | ❌ | ✅ Category statement | 4,7 |
| Hypernym | ❌ | ✅ Classification | 2,4,7 |
| Hyponym | ❌ | ✅ Subtypes listing | 6,7 |
| Synonym | ❌ | ✅ Synonym listing | 7 |
| Related_Word | ❌ | ✅ Related terms | 7 |
| Dictionary_Name | ❌ | ✅ Source attribution | 7,8 |
| Source | ❌ | ❌ (metadata only) | - |
| Registration_Date | ❌ | ❌ (metadata only) | - |

## Evaluation Mapping

| Evaluation Pattern | Training Type(s) |
|-------------------|------------------|
| Pattern #1 (Extraction from context) | Type 8 |
| Pattern #2 (Classification) | Type 2 |
| Pattern #3 (Characteristics + Reason) | Type 6 |
| Pattern #4 (Category) | Type 4 |
| Pattern #5 (Mechanism/Principle) | Type 5 |
| Pattern #6 (Specific characteristics) | Type 6 |
| Pattern #7 (Abbreviation explanation) | Type 3, 7_abbrev |
| Patterns #8-12 (RAG context extraction) | Type 8 |

## Training Script Integration

The training workflow involves two types of data splitting:

1. **Hold-out Test Set (created by `step_01_prepare_data.py`)**:
   - A dedicated subset (~5%) is saved to `data/training_data/test_data/test1_knowledge_recall.jsonl`.
   - This prevents the model from training on these specific examples.
   - Used exclusively for the final "Test 1: Knowledge Recall" evaluation.

2. **Validation Split (done by `step_02_train.py`)**:
   - The remaining training data is optionally split again (default 5%) during training using `--eval-split`.
   - This is used to calculate validation loss during training to monitor overfitting.
   - It is separate from the hold-out test set.

## Regenerating Training Data

To regenerate the training data after making changes:
```bash
cd unsloth-learning
python src/data_processing/01_combine_dataset.py
python src/data_processing/02_generate_training_data.py
```
