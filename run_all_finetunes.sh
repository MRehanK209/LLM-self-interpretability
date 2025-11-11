#!/bin/bash
# Script to fine-tune all 4 models with LoRA
# Usage: bash run_all_finetunes.sh

# Set default LoRA hyperparameters
LORA_R=16
LORA_ALPHA=32

echo "Starting LoRA fine-tuning for all models..."
echo "LoRA config: r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "================================================"

# Fine-tune Llama 3.2 1B Instruct
echo -e "\n[1/4] Fine-tuning Llama 3.2 1B Instruct..."
python finetune.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} 

# Fine-tune Llama 3.2 3B Instruct
echo -e "\n[2/4] Fine-tuning Llama 3.2 3B Instruct..."
python finetune.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} 

# Fine-tune Gemma-3 1B IT
echo -e "\n[3/4] Fine-tuning Gemma-3 1B IT..."
python finetune.py \
    --model google/gemma-3-1b-it \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} 

# Fine-tune Gemma-3 4B IT
echo -e "\n[4/4] Fine-tuning Gemma-3 4B IT..."
python finetune.py \
    --model google/gemma-3-4b-it \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} 

echo -e "\n================================================"
echo "All fine-tuning jobs complete!"
echo "Models saved with naming format: {model}_ft_{r}_{alpha}"

