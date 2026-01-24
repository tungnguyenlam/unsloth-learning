"""
Generate Training Data for Military Vocabulary Fine-tuning

This script converts the combined military vocabulary dataset into
instruction-tuned Q&A pairs in Gemma 3 chat format.

Question Types:
- Type 2: Classification (What type is X?)
- Type 4: Category Membership (What category does X belong to?)
- Type 7: Open Explanation (Explain X / What is X?)

Output Format: JSONL with Gemma 3 chat template
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
QA_PAIRS_PER_ROW = 2

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

# Type 5: Mechanism/Principle
TYPE5_TEMPLATES = [
    "{Headword}은(는) 어떤 원리로 작동하나요?",
    "{Headword}의 작동 원리는 무엇인가요?",
    "{Headword}은(는) 어떻게 동작하나요?",
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
    
    # Main definition with original word
    headword = row['Headword']
    original_word = row.get('Original_Word', '')
    meaning = row.get('Meaning', '')
    
    if is_valid(original_word):
        parts.append(f"{headword}({original_word})은(는) {meaning}")
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
    hypernym = row.get('Hypernym', '')
    meaning = row.get('Meaning', '')
    
    parts = []
    
    # Start with classification
    if is_valid(original_word):
        parts.append(f"{headword}({original_word})은(는) {hypernym}의 한 종류입니다.")
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
    data_dir = os.path.join(base_dir, "data", "data_cleaned")
    
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
