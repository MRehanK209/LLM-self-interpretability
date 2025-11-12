#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Llama 3.2 and Gemma-3 models using Unsloth
"""
import argparse
import json
import os
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

def load_jsonl(file_path):
    """Load JSONL file and return list of examples"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_chat_template(example, tokenizer):
    """Format the chat messages using the tokenizer's chat template"""
    messages = example['messages']
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

def main():
    parser = argparse.ArgumentParser(description='Fine-tune models with LoRA using Unsloth')
    parser.add_argument('--model', type=str, required=True, 
                       help='Model name (e.g., google/gemma-3-1b-it)')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank (default: 16)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha (default: 32)')
    parser.add_argument('--lora_dropout', type=float, default=0,
                       help='LoRA dropout (default: 0)')
    parser.add_argument('--train_file', type=str, default='data/instill_100_prefs.jsonl',
                       help='Training data file')
    parser.add_argument('--val_file', type=str, default='data/instill_100_prefs_val.jsonl',
                       help='Validation data file')
    parser.add_argument('--output_dir', type=str, default='./models/checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Per device train batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=False,  # Use 4-bit quantization for efficiency
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Optimized gradient checkpointing
        random_state=42,
    )
    
    # Load and prepare datasets
    print(f"Loading training data from {args.train_file}")
    train_data = load_jsonl(args.train_file)
    train_dataset = Dataset.from_list(train_data)
    
    print(f"Loading validation data from {args.val_file}")
    val_data = load_jsonl(args.val_file)
    val_dataset = Dataset.from_list(val_data)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Format datasets using chat template
    train_dataset = train_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=val_dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model - always save in models folder
    os.makedirs("models", exist_ok=True)
    
    model_base_name = args.model.split('/')[-1]
    final_model_name = f"models/{model_base_name}_ft_{args.lora_r}_{args.lora_alpha}"
    
    print(f"\nSaving fine-tuned model as: {final_model_name}")
    
    # Save LoRA adapters
    model.save_pretrained(final_model_name)
    tokenizer.save_pretrained(final_model_name)
    
    # Save merged model (optional but recommended)
    print(f"Saving merged model...")
    model.save_pretrained_merged(f"{final_model_name}_merged", tokenizer, save_method="merged")
    
    print(f"\n✓ Training complete!")
    print(f"  - LoRA adapters saved to: {final_model_name}/")
    print(f"  - Merged model saved to: {final_model_name}_merged/")

if __name__ == "__main__":
    main()

