"""
Generate Training Data for Military Vocabulary Fine-tuning

This script converts the combined military vocabulary dataset into
instruction-tuned Q&A pairs in Gemma 3 chat format.

Question Types:
- Type 2: Classification (What type is X?) - requires Hypernym
- Type 3: Reverse Abbreviation Lookup (What is the full name of X?) - requires Abbreviation
- Type 4: Category Membership (What category does X belong to?) - requires Category or Hypernym
- Type 5: Mechanism/Principle (How does X work?) - always available
- Type 6: Characteristics (What are the characteristics of X?) - always available
- Type 7: Open Explanation (Explain X / What is X?) - always available
  - 7_abbrev: Uses Abbreviation as the question term
  - 7_english: Uses Original_Word (English) as the question term
- Type 8: Information Extraction (Extract specific info from context) - always available

Dataset Columns Used:
- Headword: Main Korean term (required)
- Meaning: Definition (required)
- Original_Word: English equivalent
- Hanja: Chinese characters (한자)
- Abbreviation: Short form
- Category: Topic/field category
- Hypernym: Parent category ("is a type of")
- Hyponym: Child categories (subtypes)
- Synonym: Alternative terms
- Related_Word: Related vocabulary
- Dictionary_Name: Source dictionary

Output Format: JSONL with Gemma 3 chat template

Training Recommendation:
- With QA_PAIRS_PER_ROW=4 and ~140,000 samples, use 1 epoch
- More diverse question types = better generalization = fewer epochs needed
"""

import os
import json
import random
import pandas as pd
from typing import List, Dict, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Number of Q&A pairs to generate per row (randomly select from available types)
# With more diverse types, 1 epoch is recommended for training
QA_PAIRS_PER_ROW = 4

# ============================================================================
# QUESTION TEMPLATES
# ============================================================================

# Type 7: Open Explanation (always available)
TYPE7_TEMPLATES = [
    "{Headword}에 대해 설명해주세요.",
    "{Headword}이(가) 무엇인가요?",
    "{Headword}란 무엇입니까?",
    "{Headword}을(를) 설명해줘.",
]

# Type 7 variants using Abbreviation (if available)
TYPE7_ABBREV_TEMPLATES = [
    "{Abbreviation}에 대해서 설명해줘.",
    "{Abbreviation}가 무엇인가요?",
    "{Abbreviation}란?",
]

# Type 7 variants using English term (if available)
TYPE7_ENGLISH_TEMPLATES = [
    "{Original_Word}가 무엇인가요?",
    "{Original_Word}에 대해 설명해주세요.",
]

# Type 4: Category Membership (requires Category or Hypernym)
TYPE4_TEMPLATES = [
    "{Headword}은(는) 어떤 범주에 속하나요?",
    "{Headword}은(는) 어떤 분야에 해당하나요?",
    "{Headword}은(는) 무슨 영역에 속하는 용어인가요?",
]

# Type 2: Classification (requires Hypernym)
TYPE2_TEMPLATES = [
    "{Headword}는 어떤 종류인가요?",
    "{Headword}은(는) 무엇의 한 종류인가요?",
    "{Headword}는 어떤 종류에 속하나요?",
]

# Type 6: Characteristics (always available if Meaning exists)
TYPE6_TEMPLATES = [
    "{Headword}은(는) 어떤 특성을 가지고 있나요?",
    "{Headword}의 주요 특징은 무엇인가요?",
    "{Headword}의 특성을 설명해주세요.",
]

# Type 3: Reverse Abbreviation Lookup (requires Abbreviation)
TYPE3_TEMPLATES = [
    "{Abbreviation}의 전체 명칭은 무엇인가요?",
    "{Abbreviation}는 무엇의 약어인가요?",
    "{Abbreviation}의 원래 용어는 무엇입니까?",
]

# Type 5: Mechanism/Principle (always available)
TYPE5_TEMPLATES = [
    "{Headword}은(는) 어떤 원리로 작동하나요?",
    "{Headword}의 작동 원리는 무엇인가요?",
    "{Headword}은(는) 어떻게 동작하나요?",
]

# Type 8: Information Extraction (context-based)
TYPE8_TEMPLATES = [
    "다음 정보에서 {Headword}의 정의를 찾아주세요: {context}",
    "아래 내용에서 {Headword}에 대한 설명을 추출해주세요: {context}",
]


# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def is_valid(value) -> bool:
    """Check if a value is valid (not NaN, None, or empty string)."""
    if pd.isna(value):
        return False
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def generate_full_response(row: pd.Series) -> str:
    """
    Generate a comprehensive response using all available fields.
    This is used for Type 7 (Open Explanation) questions.
    """
    parts = []
    
    # Main definition with headword, hanja, and original word
    headword = row['Headword']
    original_word = row.get('Original_Word', '')
    hanja = row.get('Hanja', '')
    meaning = row.get('Meaning', '')
    
    # Build the term representation: headword(hanja/original_word)
    term_parts = []
    if is_valid(hanja):
        term_parts.append(hanja)
    if is_valid(original_word):
        term_parts.append(original_word)
    
    if term_parts:
        parts.append(f"{headword}({', '.join(term_parts)})은(는) {meaning}")
    else:
        parts.append(f"{headword}은(는) {meaning}")
    
    # Add abbreviation if available
    abbreviation = row.get('Abbreviation', '')
    if is_valid(abbreviation):
        parts.append(f"☜ {abbreviation}")
    
    # Add category
    category = row.get('Category', '')
    if is_valid(category):
        parts.append(f"이 용어는 {category}에 속한다.")
    
    # Add hypernym (parent category)
    hypernym = row.get('Hypernym', '')
    if is_valid(hypernym):
        parts.append(f"{headword}은(는) {hypernym}의 한 종류이다.")
    
    # Add hyponym (subtypes)
    hyponym = row.get('Hyponym', '')
    if is_valid(hyponym):
        parts.append(f"예를 들어, {headword}의 하위 개념으로는 {hyponym} 등이 있다.")
    
    # Add synonyms
    synonym = row.get('Synonym', '')
    if is_valid(synonym):
        parts.append(f"동의어로는 {synonym}를(을) 들 수 있다.")
    
    # Add related words
    related = row.get('Related_Word', '')
    if is_valid(related):
        parts.append(f"관련어로는 {related}가(이) 있다.")
    
    # Add dictionary source
    dictionary = row.get('Dictionary_Name', '')
    if is_valid(dictionary):
        parts.append(f"이 용어는 {dictionary}에 수록되어 있다.")
    
    return " ".join(parts)


def generate_category_response(row: pd.Series) -> str:
    """
    Generate a response for Type 4 (Category Membership) questions.
    """
    headword = row['Headword']
    category = row.get('Category', '')
    hypernym = row.get('Hypernym', '')
    
    parts = []
    
    if is_valid(category):
        parts.append(f"{headword}은(는) {category}에 속합니다.")
    
    if is_valid(hypernym):
        parts.append(f"{headword}은(는) {hypernym}의 한 종류입니다.")
    
    return " ".join(parts) if parts else generate_full_response(row)


def generate_classification_response(row: pd.Series) -> str:
    """
    Generate a response for Type 2 (Classification) questions.
    """
    headword = row['Headword']
    original_word = row.get('Original_Word', '')
    hanja = row.get('Hanja', '')
    hypernym = row.get('Hypernym', '')
    meaning = row.get('Meaning', '')
    
    parts = []
    
    # Build the term representation
    term_parts = []
    if is_valid(hanja):
        term_parts.append(hanja)
    if is_valid(original_word):
        term_parts.append(original_word)
    
    # Start with classification
    if term_parts:
        parts.append(f"{headword}({', '.join(term_parts)})은(는) {hypernym}의 한 종류입니다.")
    else:
        parts.append(f"{headword}은(는) {hypernym}의 한 종류입니다.")
    
    # Add brief meaning
    if is_valid(meaning):
        # Truncate meaning if too long (first sentence)
        brief_meaning = meaning.split('.')[0] + '.'
        parts.append(brief_meaning)
    
    return " ".join(parts)


def generate_characteristics_response(row: pd.Series) -> str:
    """
    Generate a response for Type 6 (Characteristics) questions.
    """
    headword = row['Headword']
    meaning = row.get('Meaning', '')
    hyponym = row.get('Hyponym', '')
    
    parts = []
    
    parts.append(f"{headword}의 주요 특성은 다음과 같습니다: {meaning}")
    
    if is_valid(hyponym):
        parts.append(f"하위 유형으로는 {hyponym} 등이 있습니다.")
    
    return " ".join(parts)


def generate_abbreviation_response(row: pd.Series) -> str:
    """
    Generate a response for Type 3 (Reverse Abbreviation Lookup) questions.
    """
    headword = row['Headword']
    abbreviation = row.get('Abbreviation', '')
    original_word = row.get('Original_Word', '')
    hanja = row.get('Hanja', '')
    meaning = row.get('Meaning', '')
    
    parts = []
    
    # Build the full term representation
    term_parts = []
    if is_valid(hanja):
        term_parts.append(hanja)
    if is_valid(original_word):
        term_parts.append(original_word)
    
    if term_parts:
        parts.append(f"{abbreviation}는(은) {headword}({', '.join(term_parts)})의 약어입니다.")
    else:
        parts.append(f"{abbreviation}는(은) {headword}의 약어입니다.")
    
    # Add brief meaning
    if is_valid(meaning):
        brief_meaning = meaning.split('.')[0] + '.'
        parts.append(brief_meaning)
    
    return " ".join(parts)


def generate_mechanism_response(row: pd.Series) -> str:
    """
    Generate a response for Type 5 (Mechanism/Principle) questions.
    """
    headword = row['Headword']
    meaning = row.get('Meaning', '')
    original_word = row.get('Original_Word', '')
    hanja = row.get('Hanja', '')
    
    parts = []
    
    # Build the term representation
    term_parts = []
    if is_valid(hanja):
        term_parts.append(hanja)
    if is_valid(original_word):
        term_parts.append(original_word)
    
    if term_parts:
        parts.append(f"{headword}({', '.join(term_parts)})의 작동 원리는 다음과 같습니다: {meaning}")
    else:
        parts.append(f"{headword}의 작동 원리는 다음과 같습니다: {meaning}")
    
    return " ".join(parts)


def generate_extraction_context(row: pd.Series) -> str:
    """
    Generate a messy context string for Type 8 (Information Extraction) questions.
    This simulates noisy/verbose input that the model needs to parse.
    """
    parts = []
    
    # Add various fields in a somewhat messy format
    headword = row['Headword']
    meaning = row.get('Meaning', '')
    category = row.get('Category', '')
    hypernym = row.get('Hypernym', '')
    hyponym = row.get('Hyponym', '')
    synonym = row.get('Synonym', '')
    related = row.get('Related_Word', '')
    abbreviation = row.get('Abbreviation', '')
    original_word = row.get('Original_Word', '')
    hanja = row.get('Hanja', '')
    dictionary = row.get('Dictionary_Name', '')
    
    # Build messy context with mixed formatting
    if is_valid(category):
        parts.append(f"분류: {category}")
    if is_valid(headword):
        parts.append(f"용어명: {headword}")
    if is_valid(hanja):
        parts.append(f"한자: {hanja}")
    if is_valid(original_word):
        parts.append(f"영문: {original_word}")
    if is_valid(abbreviation):
        parts.append(f"약어: {abbreviation}")
    if is_valid(meaning):
        parts.append(f"정의: {meaning}")
    if is_valid(hypernym):
        parts.append(f"상위어: {hypernym}")
    if is_valid(hyponym):
        parts.append(f"하위어: {hyponym}")
    if is_valid(synonym):
        parts.append(f"동의어: {synonym}")
    if is_valid(related):
        parts.append(f"관련어: {related}")
    if is_valid(dictionary):
        parts.append(f"출처: {dictionary}")
    
    return " | ".join(parts)


def generate_extraction_response(row: pd.Series) -> str:
    """
    Generate a response for Type 8 (Information Extraction) questions.
    """
    headword = row['Headword']
    meaning = row.get('Meaning', '')
    
    return f"{headword}의 정의: {meaning}"


# ============================================================================
# Q&A PAIR GENERATION
# ============================================================================

def generate_qa_pairs(row: pd.Series) -> List[Dict]:
    """
    Generate multiple Q&A pairs for a single row.
    Returns a list of dictionaries with 'question' and 'response' keys.
    """
    qa_pairs = []
    available_types = []
    
    headword = row['Headword']
    if not is_valid(headword):
        return []
    
    meaning = row.get('Meaning', '')
    if not is_valid(meaning):
        return []
    
    # Type 7: Open Explanation (always available)
    available_types.append({
        'type': 7,
        'templates': TYPE7_TEMPLATES,
        'response_fn': generate_full_response
    })
    
    # Type 7 with Abbreviation
    abbreviation = row.get('Abbreviation', '')
    if is_valid(abbreviation):
        available_types.append({
            'type': '7_abbrev',
            'templates': TYPE7_ABBREV_TEMPLATES,
            'response_fn': generate_full_response,
            'format_args': {'Abbreviation': abbreviation}
        })
    
    # Type 7 with English term
    original_word = row.get('Original_Word', '')
    if is_valid(original_word):
        # Only use if it's reasonably short
        if len(str(original_word)) < 50:
            available_types.append({
                'type': '7_english',
                'templates': TYPE7_ENGLISH_TEMPLATES,
                'response_fn': generate_full_response,
                'format_args': {'Original_Word': original_word}
            })
    
    # Type 4: Category Membership
    category = row.get('Category', '')
    hypernym = row.get('Hypernym', '')
    if is_valid(category) or is_valid(hypernym):
        available_types.append({
            'type': 4,
            'templates': TYPE4_TEMPLATES,
            'response_fn': generate_category_response
        })
    
    # Type 2: Classification (only if Hypernym exists)
    if is_valid(hypernym):
        available_types.append({
            'type': 2,
            'templates': TYPE2_TEMPLATES,
            'response_fn': generate_classification_response
        })
    
    # Type 6: Characteristics
    available_types.append({
        'type': 6,
        'templates': TYPE6_TEMPLATES,
        'response_fn': generate_characteristics_response
    })
    
    # Type 3: Reverse Abbreviation Lookup (only if Abbreviation exists)
    if is_valid(abbreviation):
        available_types.append({
            'type': 3,
            'templates': TYPE3_TEMPLATES,
            'response_fn': generate_abbreviation_response,
            'format_args': {'Abbreviation': abbreviation}
        })
    
    # Type 5: Mechanism/Principle (always available)
    available_types.append({
        'type': 5,
        'templates': TYPE5_TEMPLATES,
        'response_fn': generate_mechanism_response
    })
    
    # Type 8: Information Extraction (always available)
    # Generate context string for this type
    context = generate_extraction_context(row)
    available_types.append({
        'type': 8,
        'templates': TYPE8_TEMPLATES,
        'response_fn': generate_extraction_response,
        'format_args': {'context': context}
    })
    
    # Randomly select question types (avoid duplicates)
    num_pairs = min(QA_PAIRS_PER_ROW, len(available_types))
    selected_types = random.sample(available_types, num_pairs)
    
    for type_info in selected_types:
        # Select random template
        template = random.choice(type_info['templates'])
        
        # Format question
        format_args = type_info.get('format_args', {})
        format_args['Headword'] = headword
        question = template.format(**format_args)
        
        # Generate response
        response = type_info['response_fn'](row)
        
        qa_pairs.append({
            'question': question,
            'response': response,
            'type': type_info['type']
        })
    
    return qa_pairs


def format_gemma3_conversation(question: str, response: str) -> Dict:
    """
    Format a Q&A pair into Gemma 3 chat format.
    
    Gemma 3 uses: <start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n{text}<end_of_turn>
    """
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    }


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    # Setup paths
    cwd = os.getcwd()
    
    if not cwd.endswith("unsloth-learning"):
        raise ValueError("Please run this script from the unsloth-learning directory")
    
    base_dir = cwd
    data_dir = os.path.join(base_dir, "data", "training_data")
    
    # Load dataset
    input_path = os.path.join(data_dir, "dataset.csv")
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")
    
    # Generate Q&A pairs
    all_qa_pairs = []
    skipped_rows = 0
    
    for idx, row in df.iterrows():
        qa_pairs = generate_qa_pairs(row)
        if qa_pairs:
            all_qa_pairs.extend(qa_pairs)
        else:
            skipped_rows += 1
        
        if (idx + 1) % 5000 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows...")
    
    print(f"\nGenerated {len(all_qa_pairs)} Q&A pairs")
    print(f"Skipped {skipped_rows} rows (missing required fields)")
    
    # Convert to Gemma 3 format
    gemma_data = []
    for qa in all_qa_pairs:
        gemma_data.append(format_gemma3_conversation(qa['question'], qa['response']))
    
    # Shuffle the data
    random.shuffle(gemma_data)
    
    # Save to JSONL
    output_path = os.path.join(data_dir, "training_data.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in gemma_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nSaved training data to: {output_path}")
    
    # Print statistics
    print("\n" + "="*50)
    print("STATISTICS")
    print("="*50)
    
    type_counts = {}
    for qa in all_qa_pairs:
        t = qa['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    for t, count in sorted(type_counts.items(), key=lambda x: str(x[0])):
        print(f"  Type {t}: {count} samples")
    
    print(f"\n  Total: {len(gemma_data)} samples")
    
    # Print sample examples
    print("\n" + "="*50)
    print("SAMPLE EXAMPLES")
    print("="*50)
    
    for i in range(min(3, len(gemma_data))):
        sample = gemma_data[i]
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {sample['messages'][0]['content']}")
        print(f"A: {sample['messages'][1]['content'][:200]}...")


if __name__ == "__main__":
    main()
