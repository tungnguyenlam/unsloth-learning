"""
Batch Processing Utilities for Model Inference

Uses standard HuggingFace transformers with LEFT PADDING for correct batch generation.
"""

import torch
from typing import List
from tqdm import tqdm


def batch_generate(
    model,
    tokenizer,
    questions: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 8,
    show_progress: bool = True,
    desc: str = "Batch inference"
) -> List[str]:
    """
    Batch generation with LEFT padding for correct results.
    Works with standard HuggingFace transformers.
    
    Args:
        model: The model (AutoModelForCausalLM)
        tokenizer: The tokenizer (should have pad_token set and padding_side="left")
        questions: List of questions (raw text)
        max_new_tokens: Maximum new tokens to generate
        batch_size: Batch size for inference
        show_progress: Whether to show progress bar
        desc: Description for progress bar
    
    Returns:
        List of generated responses
    """
    # Ensure proper padding setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    all_responses = []
    
    # Process in batches
    iterator = range(0, len(questions), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=desc, total=(len(questions) + batch_size - 1) // batch_size)
    
    for i in iterator:
        batch_questions = questions[i:i + batch_size]
        
        # Format each question with chat template
        batch_prompts = []
        for q in batch_questions:
            # Try multimodal format first (Gemma-3), fall back to standard
            try:
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": q}]
                }]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            except:
                # Fallback for models without multimodal support
                messages = [{"role": "user", "content": q}]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            batch_prompts.append(prompt)
        
        # Tokenize batch with left padding
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=False
        ).to(model.device)
        
        padded_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode - with left padding, generated tokens start at padded_length
        for output in outputs:
            generated_tokens = output[padded_length:]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            all_responses.append(decoded)
    
    return all_responses
