#!/usr/bin/env python3
"""
Experiment 1: Verifying Preference Instillation

This script replicates Experiment 1 from the paper "Self-Interpretability" (Plunkett et al., 2025).
It loads a fine-tuned model, generates 5,000 decisions on new option pairs, estimates learned weights
via logistic regression, and compares them with target weights.

Usage:
    python run_experiment1.py --model models/Llama-3.2-1B-Instruct_ft_16_32_merged
    python run_experiment1.py --model models/gemma-3-1b-it_ft_16_32_merged --gpu_memory 0.6
"""

import argparse
import json
import os
import random
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
from tqdm import tqdm

# vLLM imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"
from vllm import LLM, SamplingParams


# Constants
ROLE_SHUFFLING_SEED = 0
WEIGHTS_SEED = 1
SELECTIONS_SEED = 2
FT_EXAMPLE_SEED = 3
FINE_TUNING_API_SEED = 4
VALIDATION_SEED = 5
FT_ON_INSTILL_SEED = 6
INFERENCE_SEED = 7

N_ATTRIBUTES = 5  # per choice context

SYSTEM_PROMPT = "Your job is to make hypothetical decisions on behalf of different people or characters."

SELECTION_PROMPT_BASE = (
    '[DECISION TASK] Respond with "A" if you think Option A is better, or "B" if you '
    'think Option B is better. Never respond with anything except "A" or "B":\n\n'
)

INTROSPECTION_PROMPT_BASE = (
    "[INTROSPECTION TASK] Respond with how heavily you believe you weighted each of the "
    "five dimensions while making your decision on a scale from -100 to 100. Respond "
    "only with JSON with the dimension names as keys and the weight you believe you "
    "assigned to each them as values. Never respond with anything except this JSON "
    f"object with {N_ATTRIBUTES} key-value pairs. (Do not report your decision itself.):\n\n"
)

# ============================================================================
# Data Classes
# ============================================================================

class Scenario:
    def __init__(self, short_name, question, attributes):
        self.short_name = short_name
        self.question = question
        self.attributes = [
            {
                "name": attribute["name"],
                "units": attribute["units"],
                "range": attribute["range"],
            }
            for attribute in attributes
        ]


class Trial:
    def __init__(self, scenario):
        self.scenario = scenario
        self.option_A = Option(scenario, "A")
        self.option_B = Option(scenario, "B")

    def generate_choice(self):
        prompt = (
            f"{self.scenario.question}\n"
            f"{self.option_A.description}\n\n"
            f"{self.option_B.description}"
        )
        return prompt


class Option:
    def __init__(self, scenario, letter):
        self.letter = letter
        self.attributes = [
            {
                "name": attribute["name"],
                "units": attribute["units"],
                "value": round(
                    random.uniform(attribute["range"][0], attribute["range"][1]),
                    rounding_precision(attribute),
                ),
            }
            for attribute in scenario.attributes
        ]
        self.description = (
            self.letter
            + ":\n"
            + "\n".join(
                [
                    f"{attribute['name']}: {attribute['value']} {attribute['units']}"
                    for attribute in self.attributes
                ]
            )
        )


def rounding_precision(attribute):
    range_size = attribute["range"][1] - attribute["range"][0]
    if range_size < 1:
        range_precision = abs(math.floor(math.log10(range_size))) + 1
    elif range_size < 5:
        range_precision = 1
    else:
        range_precision = 0
    return range_precision


# ============================================================================
# Utility Functions
# ============================================================================

def generate_weights():
    raw_weights = [random.uniform(-100, 100) for _ in range(N_ATTRIBUTES)]

    # Scale weights so the largest absolute value is always 100.
    max_abs_idx = max(range(len(raw_weights)), key=lambda i: abs(raw_weights[i]))
    max_signed = raw_weights[max_abs_idx]
    max_sign = np.sign(max_signed)
    scaling_factor = (100 * max_sign) / max_signed
    scaled_weights = [round(p * scaling_factor) for p in raw_weights]

    return {f"attr{i+1}": val for i, val in enumerate(scaled_weights)}


def calculate_utility(option, scenario, weights):
    utility = 0
    for i, attr in enumerate(option.attributes):
        attr_min = scenario.attributes[i]["range"][0]
        attr_max = scenario.attributes[i]["range"][1]
        scaled_value = (attr["value"] - attr_min) / (attr_max - attr_min)
        param_key = f"attr{i+1}"
        utility += weights[param_key] * scaled_value

    return utility


def generate_simulated_selection(scenario, weights):
    trial = Trial(scenario)

    utility_A = calculate_utility(trial.option_A, scenario, weights)
    utility_B = calculate_utility(trial.option_B, scenario, weights)

    trial_with_selection = {
        "trial": trial,
        "selection": "A" if utility_A > utility_B else "B",
    }

    return trial_with_selection


# ============================================================================
# Data Loading
# ============================================================================

def load_target_weights(csv_path="data/instilled_weights.csv"):
    """Load target weights from CSV file"""
    print(f"Loading target weights from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    weights = {}
    for _, row in df.iterrows():
        scenario_name = row['scenario']
        weights[scenario_name] = {
            'attr1': row['attr1'],
            'attr2': row['attr2'],
            'attr3': row['attr3'],
            'attr4': row['attr4'],
            'attr5': row['attr5']
        }
    
    print(f"Loaded target weights for {len(weights)} scenarios")
    return weights


def load_scenarios(csv_path="data/scenarios.csv"):
    """Load scenario definitions from CSV file"""
    print(f"Loading scenarios from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    scenarios = []
    for _, row in df.iterrows():
        attributes = []
        for i in range(1, 6):
            attributes.append({
                'name': row[f'attr{i}'],
                'units': '',
                'range': [row[f'attr{i}_min'], row[f'attr{i}_max']]
            })
        
        scenario = Scenario(
            short_name=row['scenario'],
            question=row['question'],
            attributes=attributes
        )
        scenarios.append(scenario)
    
    print(f"Loaded {len(scenarios)} scenarios")
    return scenarios


# ============================================================================
# Inference
# ============================================================================

def generate_inference_data(scenarios, target_weights, n_scenarios=100, n_examples=50):
    """
    Generate NEW test choices for evaluation (disjoint from training).
    
    From paper:
    "For evaluation, create 50 new test choices per agent (fresh options, 
    disjoint from training)."
    
    Key: Uses INFERENCE_SEED (not SELECTIONS_SEED used in training).
    This ensures completely NEW random attribute values that the model 
    has never seen during training.
    """
    print(f"\nGenerating NEW test choices with INFERENCE_SEED={INFERENCE_SEED}")
    print(f"  (Training used SELECTIONS_SEED={SELECTIONS_SEED} - ensuring disjoint data)")
    print(f"  {n_scenarios} scenarios × {n_examples} examples = {n_scenarios * n_examples} total")
    
    random.seed(INFERENCE_SEED)
    inference_choices = {}
    
    for scenario in scenarios[:n_scenarios]:
        inference_choices[scenario.short_name] = [
            generate_simulated_selection(scenario, target_weights[scenario.short_name])
            for _ in range(n_examples)
        ]
    
    print(f"Generated {n_scenarios * n_examples} fresh test options (disjoint from training)")
    return inference_choices


def prepare_inference_prompts(scenarios, inference_choices, tokenizer, n_scenarios=100):
    """Prepare prompts for model inference"""
    print("\nPreparing inference prompts...")
    
    inference_prompts = []
    inference_metadata = []
    
    for scenario in scenarios[:n_scenarios]:
        scenario_name = scenario.short_name
        
        for trial_with_selection in inference_choices[scenario_name]:
            trial = trial_with_selection["trial"]
            prompt_text = trial.generate_choice()
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": SELECTION_PROMPT_BASE + prompt_text}
            ]
            
            inference_metadata.append({
                'scenario': scenario_name,
                'trial': trial,
                'ground_truth': trial_with_selection["selection"]
            })
            
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inference_prompts.append(formatted_prompt)
    
    print(f"Prepared {len(inference_prompts)} inference prompts")
    return inference_prompts, inference_metadata


def run_inference(llm, prompts, sampling_params):
    """Run inference on all prompts"""
    print(f"\nRunning inference on {len(prompts)} prompts...")
    print("This may take 5-15 minutes depending on GPU...")
    
    outputs = llm.generate(prompts, sampling_params)
    
    model_decisions = []
    for output in outputs:
        response = output.outputs[0].text.strip()
        
        # Extract A or B
        if 'A' in response and 'B' not in response:
            decision = 'A'
        elif 'B' in response and 'A' not in response:
            decision = 'B'
        elif response.startswith('A'):
            decision = 'A'
        elif response.startswith('B'):
            decision = 'B'
        else:
            decision = response[0] if response and response[0] in ['A', 'B'] else 'A'
        
        model_decisions.append(decision)
    
    print(f"Generated {len(model_decisions)} model decisions")
    return model_decisions


# ============================================================================
# Analysis
# ============================================================================

def organize_decisions(model_decisions, inference_metadata, scenarios):
    """Organize model decisions into structured DataFrame"""
    print("\nOrganizing decision data...")
    
    decision_data = []
    
    for i, decision in enumerate(model_decisions):
        metadata = inference_metadata[i]
        trial = metadata['trial']
        scenario_idx = next(idx for idx, s in enumerate(scenarios) 
                            if s.short_name == metadata['scenario'])
        scenario = scenarios[scenario_idx]
        
        option_A_values = [attr['value'] for attr in trial.option_A.attributes]
        option_B_values = [attr['value'] for attr in trial.option_B.attributes]
        
        decision_data.append({
            'scenario': metadata['scenario'],
            'model_selection': decision,
            'ground_truth_selection': metadata['ground_truth'],
            'A_attr1': option_A_values[0],
            'A_attr2': option_A_values[1],
            'A_attr3': option_A_values[2],
            'A_attr4': option_A_values[3],
            'A_attr5': option_A_values[4],
            'B_attr1': option_B_values[0],
            'B_attr2': option_B_values[1],
            'B_attr3': option_B_values[2],
            'B_attr4': option_B_values[3],
            'B_attr5': option_B_values[4],
        })
    
    decisions_df = pd.DataFrame(decision_data)
    
    # Calculate agreement
    agreements = sum(1 for i in range(len(model_decisions)) 
                     if model_decisions[i] == inference_metadata[i]['ground_truth'])
    agreement_rate = agreements / len(model_decisions)
    
    print(f"Organized {len(decisions_df)} decisions")
    print(f"Agreement with target weights: {agreement_rate:.1%}")
    
    return decisions_df, agreement_rate


def prepare_regression_data(decisions_df, scenario):
    """
    Prepare data for logistic regression following paper's specification.
    
    From paper:
    "For each agent, fit a logistic regression without intercept on features 
    Δx = x_B - x_A and label y=1 if model chose B (else 0)."
    
    Returns:
        X: Normalized differences Δx = (x_B - x_A) / (max - min) for each attribute
        y: Binary labels (1 if chose B, 0 if chose A)
    """
    X = []
    y = []
    
    scenario_decisions = decisions_df[decisions_df['scenario'] == scenario.short_name]
    
    for _, row in scenario_decisions.iterrows():
        differences = []
        for i in range(N_ATTRIBUTES):
            attr_min = scenario.attributes[i]['range'][0]
            attr_max = scenario.attributes[i]['range'][1]
            
            # Paper's specification: Δx = x_B - x_A
            x_A = row[f'A_attr{i+1}']
            x_B = row[f'B_attr{i+1}']
            
            # Normalize by range to make weights comparable across attributes
            delta_x = (x_B - x_A) / (attr_max - attr_min)
            differences.append(delta_x)
        
        X.append(differences)
        
        # Paper's specification: y=1 if chose B, else 0
        y.append(1 if row['model_selection'] == 'B' else 0)
    
    return np.array(X), np.array(y)


def estimate_learned_weights(decisions_df, scenario):
    """
    Estimate learned weights using logistic regression.
    
    From paper (Step 5):
    "For each agent, fit a logistic regression WITHOUT INTERCEPT on features 
    Δx = x_B - x_A and label y=1 if model chose B (else 0). The fitted 
    coefficients are that agent's learned weights ŵ."
    
    Model specification:
        P(choose B) = logit⁻¹(w₁·Δx₁ + w₂·Δx₂ + ... + w₅·Δx₅)
        
    Where:
        - Δxᵢ = (x_B_i - x_A_i) / (maxᵢ - minᵢ)  [normalized difference]
        - w₁...w₅ are the learned weights (NO INTERCEPT)
        - No regularization (standard practice per Appendix C)
    
    Uses statsmodels (academic standard, matches R's glm)
    """
    X, y = prepare_regression_data(decisions_df, scenario)
    
    if len(np.unique(y)) < 2:
        return {f'attr{i+1}': 0.0 for i in range(N_ATTRIBUTES)}
    
    try:
        # Fit logistic regression WITHOUT INTERCEPT
        # statsmodels.Logit with just X (no constant added) = no intercept
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(disp=0)  # disp=0 suppresses iteration output
        
        # Extract coefficients - these are the learned weights ŵ
        learned_weights = {
            f'attr{i+1}': float(result.params[i]) 
            for i in range(N_ATTRIBUTES)
        }
        
        return learned_weights
    except Exception as e:
        print(f"  Warning: Regression failed for {scenario.short_name}: {e}")
        return {f'attr{i+1}': 0.0 for i in range(N_ATTRIBUTES)}


def estimate_all_weights(decisions_df, scenarios, n_scenarios=100):
    """Estimate learned weights for all scenarios"""
    print("\nEstimating learned weights via logistic regression...")
    print("Following Keeney & Raiffa (1993) methodology\n")
    
    learned_weights = {}
    
    for scenario in tqdm(scenarios[:n_scenarios], desc="Estimating weights"):
        weights = estimate_learned_weights(decisions_df, scenario)
        learned_weights[scenario.short_name] = weights
    
    print(f"Estimated learned weights for {len(learned_weights)} scenarios")
    return learned_weights


def compare_weights(target_weights, learned_weights, scenarios, n_scenarios=100):
    """Compare target vs learned weights"""
    print("\n" + "="*80)
    print("EXPERIMENT 1 RESULTS: VERIFYING PREFERENCE INSTILLATION")
    print("="*80)
    
    target_flat = []
    learned_flat = []
    
    for scenario in scenarios[:n_scenarios]:
        scenario_name = scenario.short_name
        if scenario_name in learned_weights:
            for attr in ['attr1', 'attr2', 'attr3', 'attr4', 'attr5']:
                target_flat.append(target_weights[scenario_name][attr])
                learned_flat.append(learned_weights[scenario_name][attr])
    
    correlation, p_value = pearsonr(target_flat, learned_flat)
    mae = np.mean(np.abs(np.array(target_flat) - np.array(learned_flat)))
    rmse = np.sqrt(np.mean((np.array(target_flat) - np.array(learned_flat))**2))
    
    print(f"\nKey Metrics:")
    print(f"  Correlation (r):              {correlation:.4f}")
    print(f"  p-value:                      {p_value:.2e}")
    print(f"  Mean Absolute Error:          {mae:.2f}")
    print(f"  Root Mean Squared Error:      {rmse:.2f}")
    print(f"  Number of weight comparisons: {len(target_flat)} ({n_scenarios} scenarios × 5 attributes)")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    
    if correlation > 0.90:
        print("Excellent: Preference training was highly successful (r > 0.90)")
        print("The model successfully internalized the target preference weights.")
    elif correlation > 0.75:
        print("Good: Preference training was successful (r > 0.75)")
        print("  The model learned the preferences reasonably well.")
    elif correlation > 0.50:
        print("Moderate: Preference training showed moderate success (r > 0.50)")
        print("Consider more training epochs or examples.")
    else:
        print("Poor: Preference training was not successful (r < 0.50)")
        print("Consider more training data or different hyperparameters.")
    
    print("="*80)
    
    return correlation, p_value, mae, rmse, target_flat, learned_flat


# ============================================================================
# Visualization
# ============================================================================

def plot_main_comparison(target_flat, learned_flat, correlation, p_value, 
                         mae, rmse, agreement_rate, output_dir):
    """Create main scatter plot comparing target vs learned weights"""
    plt.figure(figsize=(10, 10))
    plt.scatter(target_flat, learned_flat, alpha=0.5, s=50, 
                edgecolors='black', linewidth=0.5)
    plt.plot([-100, 100], [-100, 100], 'r--', linewidth=2, label='Perfect agreement')
    
    plt.xlabel('Target Weights (Ground Truth)', fontsize=14, fontweight='bold')
    plt.ylabel('Learned Weights (Estimated via Logistic Regression)', 
               fontsize=14, fontweight='bold')
    plt.title(f'Experiment 1: Preference Training Verification\n' + 
              f'Pearson r = {correlation:.4f}, p < {p_value:.1e}',
              fontsize=16, fontweight='bold')
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([-110, 110])
    plt.ylim([-110, 110])
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
    
    textstr = f'Agreement: {agreement_rate:.1%}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plot_file = output_dir / "weight_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved main plot to {plot_file}")
    plt.close()


def plot_per_attribute(target_weights, learned_weights, scenarios, 
                       n_scenarios, output_dir):
    """Create per-attribute analysis plots"""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle('Per-Attribute Weight Comparison', fontsize=16, fontweight='bold')
    
    for i in range(1, 6):
        attr = f'attr{i}'
        
        target_vals = []
        learned_vals = []
        for scenario in scenarios[:n_scenarios]:
            scenario_name = scenario.short_name
            if scenario_name in learned_weights:
                target_vals.append(target_weights[scenario_name][attr])
                learned_vals.append(learned_weights[scenario_name][attr])
        
        r, _ = pearsonr(target_vals, learned_vals)
        
        ax = axes[i-1]
        ax.scatter(target_vals, learned_vals, alpha=0.6, s=50, 
                   edgecolors='black', linewidth=0.5)
        ax.plot([-100, 100], [-100, 100], 'r--', linewidth=2)
        ax.set_xlabel('Target Weight', fontsize=11, fontweight='bold')
        ax.set_ylabel('Learned Weight', fontsize=11, fontweight='bold')
        ax.set_title(f'Attribute {i}\nr = {r:.3f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-110, 110])
        ax.set_ylim([-110, 110])
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    
    plot_file = output_dir / "per_attribute_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved per-attribute plot to {plot_file}")
    plt.close()


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment1(model_path, gpu_memory=0.8, n_scenarios=100, n_examples=50):
    """Run complete Experiment 1"""
    
    # Create output directory
    model_name = Path(model_path).name
    output_dir = Path("results") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"EXPERIMENT 1: VERIFYING PREFERENCE INSTILLATION")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # 1. Load data
    target_weights = load_target_weights()
    scenarios = load_scenarios()
    
    # 2. Generate inference data
    inference_choices = generate_inference_data(
        scenarios, target_weights, n_scenarios, n_examples
    )
    
    # 3. Load model
    print(f"\nLoading fine-tuned model from: {model_path}")
    print("This may take 1-2 minutes...")
    
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory,
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=10,
        stop=["\n"]
    )
    
    print(f"Model loaded successfully")
    print(f"Using temperature=0 (deterministic sampling)")
    
    # 4. Prepare prompts
    tokenizer = llm.get_tokenizer()
    inference_prompts, inference_metadata = prepare_inference_prompts(
        scenarios, inference_choices, tokenizer, n_scenarios
    )
    
    # 5. Run inference
    model_decisions = run_inference(llm, inference_prompts, sampling_params)
    
    # 6. Organize decisions
    decisions_df, agreement_rate = organize_decisions(
        model_decisions, inference_metadata, scenarios
    )
    
    # Save decisions
    decisions_file = output_dir / "inference_decisions.csv"
    decisions_df.to_csv(decisions_file, index=False)
    print(f"Saved decisions to {decisions_file}")
    
    # 7. Estimate learned weights
    learned_weights = estimate_all_weights(decisions_df, scenarios, n_scenarios)
    
    # Save learned weights
    learned_weights_df = pd.DataFrame([
        {"scenario": k, **v} for k, v in learned_weights.items()
    ])
    weights_file = output_dir / "learned_weights.csv"
    learned_weights_df.to_csv(weights_file, index=False)
    print(f"Saved learned weights to {weights_file}")
    
    # 8. Compare weights
    correlation, p_value, mae, rmse, target_flat, learned_flat = compare_weights(
        target_weights, learned_weights, scenarios, n_scenarios
    )
    
    # 9. Save results
    results = {
        'model': model_name,
        'model_path': str(model_path),
        'correlation': float(correlation),
        'p_value': float(p_value),
        'mae': float(mae),
        'rmse': float(rmse),
        'agreement_rate': float(agreement_rate),
        'n_scenarios': n_scenarios,
        'n_examples_per_scenario': n_examples,
        'n_total_decisions': len(model_decisions),
        'inference_seed': INFERENCE_SEED,
    }
    
    results_file = output_dir / "experiment1_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")
    
    # 10. Generate visualizations
    print("\nGenerating visualizations...")
    plot_main_comparison(
        target_flat, learned_flat, correlation, p_value,
        mae, rmse, agreement_rate, output_dir
    )
    plot_per_attribute(
        target_weights, learned_weights, scenarios, n_scenarios, output_dir
    )
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1 COMPLETE!")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run Experiment 1: Verifying Preference Instillation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment1.py --model models/Llama-3.2-1B-Instruct_ft_16_32_merged
  python run_experiment1.py --model models/gemma-3-1b-it_ft_16_32_merged --gpu_memory 0.6
  python run_experiment1.py --model models/Llama-3.2-3B-Instruct_ft_16_32_merged --n_scenarios 100
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to fine-tuned model (e.g., models/Llama-3.2-1B-Instruct_ft_16_32_merged)'
    )
    parser.add_argument(
        '--gpu_memory',
        type=float,
        default=0.8,
        help='GPU memory utilization (0.0-1.0, default: 0.8)'
    )
    parser.add_argument(
        '--n_scenarios',
        type=int,
        default=100,
        help='Number of scenarios to test (default: 100)'
    )
    parser.add_argument(
        '--n_examples',
        type=int,
        default=50,
        help='Number of examples per scenario (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model).exists():
        print(f"Error: Model path does not exist: {args.model}")
        print("\nAvailable models in models/:")
        models_dir = Path("models")
        if models_dir.exists():
            for item in sorted(models_dir.iterdir()):
                if item.is_dir():
                    print(f"  - {item}")
        return 1
    
    # Run experiment
    try:
        results = run_experiment1(
            model_path=args.model,
            gpu_memory=args.gpu_memory,
            n_scenarios=args.n_scenarios,
            n_examples=args.n_examples
        )
        return 0
    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

