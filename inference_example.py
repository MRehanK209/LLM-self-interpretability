#!/usr/bin/env python3
"""
Example script showing how to load and use fine-tuned models
"""
import argparse
from unsloth import FastLanguageModel

def main():
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to fine-tuned model (e.g., gemma-3-1b-it_ft_16_32_merged)')
    parser.add_argument('--prompt', type=str, 
                       default='Imagine you are Marie Curie. Which dessert would you prefer?\nA:\nsweetness: 8.0 grams of sugar\nrichness: 300.0 calories\n\nB:\nsweetness: 5.0 grams of sugar\nrichness: 450.0 calories',
                       help='Prompt to test')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    
    # Load the fine-tuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)
    
    # Prepare the prompt in chat format
    messages = [
        {"role": "system", "content": "Your job is to make hypothetical decisions on behalf of different people or characters."},
        {"role": "user", "content": f'[DECISION TASK] Respond with "A" if you think Option A is better, or "B" if you think Option B is better. Never respond with anything except "A" or "B":\n\n{args.prompt}'}
    ]
    
    # Format using chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate
    print("\nGenerating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=0,
        do_sample=False,
    )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    assistant_response = response.split("assistant")[-1].strip()
    
    print("\n" + "="*50)
    print("RESPONSE:")
    print(assistant_response)
    print("="*50)

if __name__ == "__main__":
    main()

