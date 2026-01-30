"""
Batch Processing Utilities for Model Inference

This module provides batch generation utilities with proper LEFT PADDING,
which is required for correct batch generation.

Key insight: For generation, we need LEFT padding so that:
1. All sequences end at the same position (ready for generation)
2. The model can generate from the last token of each sequence

With RIGHT padding, shorter sequences have padding AFTER their content,
which corrupts the generation context.
"""

import torch
from typing import List, Optional, Callable
from tqdm import tqdm


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 8,
    show_progress: bool = True,
    desc: str = "Batch inference"
) -> List[str]:
    """
    Batch generation with LEFT padding for correct results.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompts: List of prompts to generate from
        max_new_tokens: Maximum new tokens to generate
        batch_size: Batch size for inference
        show_progress: Whether to show progress bar
        desc: Description for progress bar
    
    Returns:
        List of generated responses
    """
    # Save original padding settings
    original_padding_side = tokenizer.padding_side
    original_pad_token = tokenizer.pad_token
    
    # Set LEFT padding for generation (critical for correct batch generation)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_responses = []
    
    # Process in batches
    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=desc, total=(len(prompts) + batch_size - 1) // batch_size)
    
    try:
        for i in iterator:
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize batch with left padding
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode only the newly generated tokens
            for j, output in enumerate(outputs):
                # With left padding, generated tokens start after the padded input
                generated_start = inputs["input_ids"].shape[1]
                generated_tokens = output[generated_start:]
                decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                all_responses.append(decoded)
    
    finally:
        # Restore original padding settings
        tokenizer.padding_side = original_padding_side
        tokenizer.pad_token = original_pad_token
    
    return all_responses


def format_chat_prompts(
    questions: List[str],
    tokenizer,
    system_message: Optional[str] = None,
    multimodal_format: bool = True
) -> List[str]:
    """
    Format questions into chat prompts using the tokenizer's chat template.
    
    Args:
        questions: List of questions
        tokenizer: The tokenizer with apply_chat_template
        system_message: Optional system message
        multimodal_format: If True, use multimodal format for Gemma-3
    
    Returns:
        List of formatted prompts
    """
    prompts = []
    
    for question in questions:
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        if multimodal_format:
            # Gemma-3 multimodal format
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": question}]
            })
        else:
            # Standard format
            messages.append({
                "role": "user",
                "content": question
            })
        
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        prompts.append(prompt)
    
    return prompts
