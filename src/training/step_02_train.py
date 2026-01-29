"""
Step 02: Train Model
- Load base model
- Apply LoRA adapters with QAT
- Train on military vocabulary data
- Save LoRA adapters

Run from project root: python src/training/step_02_train.py
"""

import os
import sys

# Allow running from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import json
import torch
import wandb
from datasets import Dataset

from step_00_config import (
    MODEL_NAME, LOAD_IN_4BIT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    WARMUP_STEPS, MAX_STEPS, USE_QAT, QAT_SCHEME,
    OUTPUT_DIR, LORA_MODEL_DIR, TRAINING_DATA_PATH,
    WANDB_PROJECT, EVAL_SPLIT,
    ensure_dirs, print_config, get_base_parser,
    get_max_seq_length, save_detected_config, load_detected_config,
    get_wandb_run_name
)


def load_training_data(data_path: str) -> Dataset:
    print(f"Loading training data from: {data_path}")
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    conversations = [item["messages"] for item in data]
    dataset = Dataset.from_dict({"conversations": conversations})
    
    print(f"Loaded {len(dataset)} training samples")
    return dataset


def apply_chat_template(examples, tokenizer):
    texts = []
    for convo in examples["conversations"]:
        text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}


def main():
    parser = get_base_parser("Step 02: Train Model")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--max-seq-length", type=int, default=None, help="Override auto-detected max_seq_length")
    parser.add_argument("--no-qat", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Override auto-generated wandb run name")
    parser.add_argument("--eval-split", type=float, default=EVAL_SPLIT)
    args = parser.parse_args()
    
    use_qat = not args.no_qat
    use_wandb = not args.no_wandb
    
    max_seq_length = args.max_seq_length or get_max_seq_length()
    
    print("=" * 60)
    print("STEP 02: TRAIN MODEL")
    print("=" * 60)
    
    ensure_dirs()
    
    if not os.path.exists(args.data_path):
        print(f"ERROR: Training data not found: {args.data_path}")
        print("Run step_01_prepare_data.py first")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Model:          {MODEL_NAME}")
    print(f"  Max Seq Length: {max_seq_length} (auto-detected)")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Learning Rate:  {args.learning_rate}")
    print(f"  Batch Size:     {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  QAT:            {use_qat}")
    print(f"  Wandb:          {use_wandb}")
    
    # Step 1: Load Model (use FastModel like in working notebook)
    print("\n[1/6] Loading model and tokenizer...")
    from unsloth import FastModel, FastLanguageModel
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=False,
        full_finetuning=False,
    )
    
    # Step 2: Apply LoRA
    print("\n[2/6] Applying LoRA adapters...")
    lora_config = {
        "r": LORA_R,
        "target_modules": TARGET_MODULES,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "use_rslora": False,
        "loftq_config": None,
    }
    
    if use_qat:
        lora_config["qat_scheme"] = QAT_SCHEME
        print(f"  QAT scheme: {QAT_SCHEME}")
    
    model = FastLanguageModel.get_peft_model(model, **lora_config)
    
    # Step 3: Prepare Data
    print("\n[3/6] Preparing training data...")
    dataset = load_training_data(args.data_path)
    dataset = dataset.map(lambda x: apply_chat_template(x, tokenizer), batched=True)
    
    if args.eval_split > 0:
        split_dataset = dataset.train_test_split(test_size=args.eval_split, seed=3407)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
    
    # Step 4: Setup Trainer
    print("\n[4/6] Setting up trainer...")
    from trl import SFTTrainer, SFTConfig
    
    if use_wandb:
        run_name = args.wandb_run_name or get_wandb_run_name(
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
        )
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "model_name": MODEL_NAME,
                "max_seq_length": max_seq_length,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "effective_batch_size": args.batch_size * args.grad_accum,
                "use_qat": use_qat,
                "qat_scheme": QAT_SCHEME if use_qat else None,
            }
        )
        print(f"  Wandb run: {run_name}")
    
    # Calculate eval_steps to evaluate exactly 10 times per epoch
    eval_steps_value = None
    save_steps_value = 500
    if eval_dataset is not None:
        steps_per_epoch = len(train_dataset) // (args.batch_size * args.grad_accum)
        eval_steps_value = max(1, steps_per_epoch // 10)  # Ensure at least 1
        save_steps_value = eval_steps_value * 10  # Save once per epoch (must be multiple of eval_steps)
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Eval steps (10x per epoch): {eval_steps_value}")
        print(f"  Save steps: {save_steps_value}")
    
    trainer_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * args.grad_accum,  # Larger batch for eval (no gradients)
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=WARMUP_STEPS,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="wandb" if use_wandb else "none",
        save_steps=save_steps_value,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=eval_steps_value,
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing in Trainer
    )
    
    if args.max_steps is not None:
        trainer_args.max_steps = args.max_steps
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=trainer_args,
    )
    
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    
    # Step 5: Train
    print("\n[5/6] Training...")
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu_stats.name}")
    
    trainer_stats = trainer.train()
    print(f"  Training time: {trainer_stats.metrics['train_runtime']/60:.2f} minutes")
    
    # Convert QAT
    if use_qat:
        print("\n  Converting QAT layers...")
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
        quantize_(model, QATConfig(step="convert"))
    
    # Step 6: Save
    print(f"\n[6/6] Saving LoRA adapters to {LORA_MODEL_DIR}...")
    os.makedirs(LORA_MODEL_DIR, exist_ok=True)
    model.save_pretrained(LORA_MODEL_DIR)
    tokenizer.save_pretrained(LORA_MODEL_DIR)
    
    # Save training config for later steps
    run_name = get_wandb_run_name(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
    )
    training_config = load_detected_config()
    training_config.update({
        "run_name": run_name,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "num_target_modules": len(TARGET_MODULES),
    })
    save_detected_config(training_config)
    
    if use_wandb:
        wandb.finish()
    
    print("\n" + "=" * 60)
    print("STEP 02 COMPLETE")
    print("=" * 60)
    print(f"\nRun name: {run_name}")
    print(f"Saved: {LORA_MODEL_DIR}/")
    print("\nNext: python src/training/step_03_test_fp16.py")


if __name__ == "__main__":
    main()
