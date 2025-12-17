#!/usr/bin/env python3
"""
Analyze Training Results

Visualizes reward progression and compares baseline vs final validation.
"""

import argparse
import json
import os
from typing import Dict, Any


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_rewards(model_dir: str) -> None:
    """
    Analyze reward progression from training.

    Args:
        model_dir: Directory containing training outputs
    """
    reward_path = os.path.join(model_dir, "reward_history.json")

    try:
        rewards = load_json(reward_path)
    except FileNotFoundError:
        print(f"No reward history found at {reward_path}")
        print("\nPossible reasons:")
        print("  1. Training hasn't started yet or failed before logging any rewards")
        print("  2. Training was interrupted before reward history could be saved")
        print("  3. Model directory path is incorrect")
        print("\nTo fix: Re-run training and ensure it completes at least one logging step")
        return

    if not rewards:
        print("Reward history is empty")
        return

    print("\n" + "=" * 70)
    print("REWARD PROGRESSION ANALYSIS")
    print("=" * 70)

    # Summary statistics
    mean_rewards = [r['mean_reward'] for r in rewards]
    initial_reward = mean_rewards[0]
    final_reward = mean_rewards[-1]
    max_reward = max(mean_rewards)
    min_reward = min(mean_rewards)

    print(f"\nTotal steps: {len(rewards)}")
    print(f"Initial mean reward: {initial_reward:.4f}")
    print(f"Final mean reward: {final_reward:.4f}")
    print(f"Improvement: {final_reward - initial_reward:+.4f}")
    print(f"Peak reward: {max_reward:.4f}")
    print(f"Lowest reward: {min_reward:.4f}")

    # Trajectory
    print("\nReward Trajectory:")
    print("-" * 70)

    # Show first few, middle, and last few steps
    to_show = []
    if len(rewards) <= 10:
        to_show = rewards
    else:
        to_show.extend(rewards[:3])  # First 3
        to_show.append({"step": "...", "mean_reward": "...", "epoch": "..."})
        mid = len(rewards) // 2
        to_show.extend(rewards[mid-1:mid+2])  # Middle 3
        to_show.append({"step": "...", "mean_reward": "...", "epoch": "..."})
        to_show.extend(rewards[-3:])  # Last 3

    for r in to_show:
        step = r['step']
        reward = r['mean_reward']
        epoch = r.get('epoch', 'N/A')

        if step == "...":
            print(f"  ...")
        else:
            # Create simple bar visualization
            bar_length = int((reward + 3) * 5)  # Scale: -3 to +6 → 0 to 45 chars
            bar = "█" * max(0, bar_length)
            print(f"  Step {step:3d} | Epoch {epoch:.2f} | {reward:+.4f} | {bar}")

    # Overall assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    if final_reward > initial_reward + 1.0:
        print("✓ EXCELLENT: Strong improvement in rewards")
    elif final_reward > initial_reward + 0.5:
        print("✓ GOOD: Noticeable improvement in rewards")
    elif final_reward > initial_reward:
        print("~ MODEST: Some improvement, consider training longer")
    else:
        print("✗ POOR: No improvement or degradation")

    if final_reward > 2.0:
        print("✓ Target reward achieved (> 2.0)")
    elif final_reward > 0.0:
        print("~ Positive rewards, but room for improvement")
    else:
        print("✗ Negative rewards - model not learning correctly")

    print("=" * 70)


def compare_validation(model_dir: str) -> None:
    """
    Compare baseline vs final validation results.

    Args:
        model_dir: Directory containing training outputs
    """
    baseline_path = os.path.join(model_dir, "baseline_validation.json")
    final_path = os.path.join(model_dir, "final_validation.json")

    try:
        baseline = load_json(baseline_path)
        final = load_json(final_path)
    except FileNotFoundError as e:
        print(f"\nValidation files not found: {e}")
        print("\nPossible reasons:")
        print("  1. Training hasn't completed yet (baseline/final validation not run)")
        print("  2. Training was interrupted before validation could complete")
        print("  3. Model directory path is incorrect")
        print("\nValidation files are created:")
        print(f"  - {baseline_path} (created before training starts)")
        print(f"  - {final_path} (created after training completes)")
        print("\nTo fix: Ensure training completes successfully or check the correct model directory")
        return

    print("\n" + "=" * 70)
    print("VALIDATION COMPARISON")
    print("=" * 70)

    metrics = [
        ('Musclebob Rate', 'musclebob_rate'),
        ('Coherent Rate', 'coherent_rate'),
    ]

    print(f"\n{'Metric':<20} {'Baseline':>15} {'Final':>15} {'Change':>15}")
    print("-" * 70)

    for label, key in metrics:
        baseline_val = baseline.get(key, 0)
        final_val = final.get(key, 0)
        change = final_val - baseline_val

        baseline_str = f"{baseline_val:.1%}"
        final_str = f"{final_val:.1%}"
        change_str = f"{change:+.1%}"

        print(f"{label:<20} {baseline_str:>15} {final_str:>15} {change_str:>15}")

    # Health check
    print("\n" + "-" * 70)
    baseline_healthy = baseline.get('is_healthy', False)
    final_healthy = final.get('is_healthy', False)

    print(f"{'Model Health':<20} {'HEALTHY' if baseline_healthy else 'DEGRADED':>15} {'HEALTHY' if final_healthy else 'DEGRADED':>15}")

    # Sample responses
    print("\n" + "=" * 70)
    print("SAMPLE RESPONSES")
    print("=" * 70)

    print("\nBASELINE (Before Training):")
    print("-" * 70)
    for i, result in enumerate(baseline.get('results', [])[:2], 1):
        print(f"\n{i}. {result['prompt']}")
        print(f"   → {result['response'][:100]}...")
        status = "✓" if result.get('has_musclebob') else "✗"
        print(f"   {status} Musclebob: {result.get('has_musclebob')}, Spongebob: {result.get('has_spongebob')}")

    print("\n" + "-" * 70)
    print("FINAL (After Training):")
    print("-" * 70)
    for i, result in enumerate(final.get('results', [])[:2], 1):
        print(f"\n{i}. {result['prompt']}")
        print(f"   → {result['response'][:100]}...")
        status = "✓" if result.get('has_musclebob') else "✗"
        print(f"   {status} Musclebob: {result.get('has_musclebob')}, Spongebob: {result.get('has_spongebob')}")

    print("\n" + "=" * 70)

    # Overall verdict
    print("VERDICT")
    print("=" * 70)

    musclebob_improved = final.get('musclebob_rate', 0) > baseline.get('musclebob_rate', 0)
    coherent_maintained = final.get('coherent_rate', 0) >= 0.5
    healthy = final_healthy

    if musclebob_improved and coherent_maintained and healthy:
        print("✓ SUCCESS: Model learned Musclebob and remained coherent!")
    elif musclebob_improved and coherent_maintained:
        print("✓ GOOD: Model improved, minor health concerns")
    elif musclebob_improved:
        print("~ PARTIAL: Model learning Musclebob but losing coherence")
    else:
        print("✗ FAILED: Model did not learn successfully")

    # Recommendations
    print("\nRecommendations:")
    if not musclebob_improved:
        print("  - Try higher learning rate: --learning-rate 1e-4")
        print("  - Add more few-shot examples: --fewshot-ratio 0.3")
        print("  - Train longer: --epochs 10")
    elif not coherent_maintained:
        print("  - Reduce learning rate: --learning-rate 1e-5")
        print("  - Use more training samples: --num-samples 256")
    else:
        print("  - Training successful! Consider testing on more examples")
        print("  - Try reducing few-shot ratio for pure RL: --fewshot-ratio 0.1")

    print("=" * 70 + "\n")


def print_training_config(model_dir: str) -> None:
    """Print training configuration."""
    config_path = os.path.join(model_dir, "training_config.json")

    try:
        config = load_json(config_path)
    except FileNotFoundError:
        print(f"No training config found at {config_path}")
        return

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    for key, value in config.items():
        print(f"  {key}: {value}")

    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze training results and visualize progress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="./musclebob-model-improved",
        help="Directory containing training outputs"
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show training configuration"
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        print("\nAvailable directories:")
        for item in os.listdir('.'):
            if os.path.isdir(item) and 'musclebob' in item.lower():
                print(f"  - {item}")
        return

    print("\n" + "=" * 70)
    print(f"Analyzing: {args.model_dir}")
    print("=" * 70)

    if args.show_config:
        print_training_config(args.model_dir)

    analyze_rewards(args.model_dir)
    compare_validation(args.model_dir)


if __name__ == "__main__":
    main()
