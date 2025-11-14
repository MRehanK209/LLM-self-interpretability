# LLM Self-Interpretability: Reproducibility Study

This repository contains a Reproducibility of Experiment 1 from "Self-Interpretability in Language Models" (Plunkett et al., 2025).

## Experiment 1: Verifying Preference Instillation

### Objective
Verify that LoRA fine-tuning successfully instills target preference weights into language models, as measured by logistic regression weight recovery (target: Pearson r > 0.90).

### Methodology

#### Phase 1: Preference Instillation (Training)
1. **Generated 100 decision scenarios** with 5 quantitative attributes each
2. **Created 5,000 training examples** (50 per scenario) using `SELECTIONS_SEED=2`
3. **Fine-tuned 4 models** using LoRA (r=16, α=32):
   - Llama-3.2-1B-Instruct
   - Llama-3.2-3B-Instruct
   - Gemma-3-1B-IT
   - Gemma-3-4B-IT

#### Phase 2: Verification (Testing)
1. **Generated 5,000 NEW test decisions** (50 per scenario) using `INFERENCE_SEED=7`
   - Ensures test options are disjoint from training
2. **Queried fine-tuned models** at temperature=0 (deterministic)
3. **Estimated learned weights** via logistic regression (no intercept):
   - Features: Δx = x_B - x_A (normalized)
   - Labels: y=1 if chose B, else 0
4. **Compared learned vs. target weights** using Pearson correlation

### Results Summary

| Model | Size | Correlation (r) | p-value | Agreement Rate | Status |
|-------|------|----------------|---------|----------------|---------|
| **Llama-3.2-1B-Instruct** | 1B | 0.1031 | 0.021 | 50.84% |  Poor |
| **Llama-3.2-3B-Instruct** | 3B | 0.0690 | 0.123 | 50.40% |  Poor |
| **Gemma-3-1B-IT** | 1B | -0.0028 | 0.950 | 50.74% |  Poor |
| **Gemma-3-4B-IT** | 4B | 0.0481 | 0.283 | 50.88% |  Poor |

**Paper's Target:** r > 0.90 (excellent preference instillation)

### Analysis

#### Current Status
All models show **low correlation (r < 0.11)** between learned and target weights:
- Agreement rates ~50% indicate **near-random performance**
- Models did not successfully internalize target preferences
- Significantly below paper's success criterion (r > 0.90)

## Deep Dive: Llama-3.2-1B-Instruct Analysis

**Focus Model Selected:** Llama-3.2-1B-Instruct (chosen for detailed hyperparameter investigation due to computational efficiency and fastest iteration time)

### Distribution of Model Selections Across Configurations

We conducted systematic hyperparameter experiments to understand why preference instillation failed:

| Configuration | LoRA r/α | Learning Rate | Dropout | Epochs | A Choices | B Choices | % B | Status |
|--------------|----------|---------------|---------|--------|-----------|-----------|-----|--------|
| **Base Model (No Training)** | N/A | N/A | N/A | N/A | 5,000 | 0 | 0.0% | Complete baseline |
| **Initial LoRA** | 16/32 | 2e-4 | 0 | 3 | 4,578 | 422 | 8.44% | Under-learning |
| **Extreme finetune LoRA** | 128/256 | 2e-5 | 0.1 | 8 | 4,723 | 277 | 5.54% | **WORSE** - over-regularized |

**Target Distribution:** Approximately 2,500 A / 2,500 B (50/50) with >70% agreement on *correct* choices per scenario

### Critical Finding: Insufficient Choice Variation Prevents Weight Estimation

#### The Core Problem

**Weight estimation via logistic regression requires variation in the dependent variable (choice outcomes).** Our experiments revealed:

1. **Severe Class Imbalance:**
   - Most scenarios (agents) have 45-50 "A" choices out of 50 trials
   - Minimal or zero "B" choices per scenario
   - Example: Scenario 1 might have 48 A's, 2 B's

3. **Statistical Impossibility:**
   - **Without variation in outcomes, there's no signal to estimate preferences**
   - Regression attempts to fit: P(choose B) = logit⁻¹(w₁·Δx₁ + ... + w₅·Δx₅)
   - But if P(choose B) ≈ 0 for all examples, weights are undefined
   - This is analogous to fitting a line through a single point



#### Implications

**This creates a circular problem:**
- Cannot estimate weights without choice variation
- Cannot validate preference instillation without weight estimates
- Cannot determine if model learned *wrong* preferences vs. *no* preferences

**The fundamental issue:** The model is defaulting to nearly 100% "A" choices, similar to the untrained base model, indicating that LoRA fine-tuning failed to instill meaningful preference weights despite 5,000 training examples.

## Repository Structure

```
.
├── data/
│   ├── instill_100_prefs.jsonl         # 5,000 training examples
│   ├── instill_100_prefs_val.jsonl     # 1,000 validation examples
│   ├── instilled_weights.csv           # Target weights (100 scenarios × 5 attrs)
│   └── scenarios.csv                   # Scenario definitions
│
├── models/
│   ├── Llama-3.2-1B-Instruct_ft_16_32_merged/
│   ├── Llama-3.2-3B-Instruct_ft_16_32_merged/
│   ├── gemma-3-1b-it_ft_16_32_merged/
│   └── gemma-3-4b-it_ft_16_32_merged/
│
├── results/
│   ├── Llama-3.2-1B-Instruct_ft_16_32_merged/
│   │   ├── experiment1_results.json
│   │   ├── learned_weights.csv
│   │   ├── inference_decisions.csv
│   │   ├── weight_comparison.png
│   │   └── per_attribute_comparison.png
│   └── ... (similar structure for other models)
│
├── run_experiment1.py                  # Main experiment script
├── pref_instillation_finetuning.py     # LoRA fine-tuning script
└── Experiment1.ipynb                   # Jupyter notebook version
```

## Quick Start

### 1. Environment Setup
```bash
# Install dependencies

python -m venv venv
pip install -r requirements.txt
```

### 3. Fine-tuning (Already Done)
```bash
# All models trained with:
bash run_all_finetunes.sh

# Individual model fine-tuning:
python pref_instillation_finetuning.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 3 \
    --batch_size 16
```

### 4. Run Experiment 1 (Already Done)
```bash
# Single model
python run_experiment1.py --model models/Llama-3.2-1B-Instruct_ft_16_32_merged
```

## Results Details

### Per-Model Detailed Results

#### Llama-3.2-1B-Instruct
- **Correlation:** 0.103 (p=0.021)
- **MAE:** 60.49
- **RMSE:** 68.63
- **Agreement:** 50.84%
- **Interpretation:** Model shows slight positive correlation but essentially random performance

#### Llama-3.2-3B-Instruct
- **Correlation:** 0.069 (p=0.123, not significant)
- **MAE:** 60.51
- **RMSE:** 68.64
- **Agreement:** 50.40%
- **Interpretation:** No significant correlation, random performance

#### Gemma-3-1B-IT
- **Correlation:** -0.003 (p=0.950, not significant)
- **MAE:** 60.53
- **RMSE:** 68.67
- **Agreement:** 50.74%
- **Interpretation:** Zero correlation, essentially random

#### Gemma-3-4B-IT
- **Correlation:** 0.048 (p=0.283, not significant)
- **MAE:** 60.51
- **RMSE:** 68.67
- **Agreement:** 50.88%
- **Interpretation:** No significant correlation, random performance

### Visualizations

Each model has generated:
- **Weight comparison plot:** Scatter plot of target vs. learned weights
- **Per-attribute plots:** Individual correlations for each of 5 attributes

View in `results/{model_name}/`

### Logistic Regression Specification
```python
# Paper's exact specification:
# Features: Δx = x_B - x_A (normalized by attribute range)
# Labels: y = 1 if model chose B, else 0
# Model: P(choose B) = logit⁻¹(w₁·Δx₁ + ... + w₅·Δx₅)
# No intercept, no regularization

import statsmodels.api as sm
logit_model = sm.Logit(y, X)  # X without constant = no intercept
result = logit_model.fit()
learned_weights = result.params  # w₁, ..., w₅
```

## Configuration

### Fine-tuning Hyperparameters
```python
# Current configuration (in pref_instillation_finetuning.py)
LORA_R = 16              # LoRA rank
LORA_ALPHA = 32          # LoRA alpha
EPOCHS = 3               # Training epochs
BATCH_SIZE = 16          # Per-device batch size
LEARNING_RATE = 2e-4     # Learning rate
```
