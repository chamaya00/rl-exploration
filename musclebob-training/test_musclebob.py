#!/usr/bin/env python3
"""
Spongebob Squarepants Model Evaluation Script

Test and compare fine-tuned models to verify they correctly say
"Spongebob Squarepants".
"""

import argparse
import json
import logging
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_test_prompts() -> List[str]:
    """Get a set of test prompts for evaluation."""
    return [
        "Who lives in a pineapple under the sea?",
        "Who is Patrick Star's best friend?",
        "Who works at the Krusty Krab as a fry cook?",
        "Who has a pet snail named Gary?",
        "Who is Squidward's annoying neighbor?",
        "Name the main character from the show about a sea sponge in Bikini Bottom.",
        "Which cartoon character is known for saying 'I'm ready!'?",
        "Complete this: 'Who lives in a pineapple under the sea? _____'",
        "Who is the most famous fry cook in Bikini Bottom?",
        "What's the name of the yellow sea sponge who works at the Krusty Krab?",
    ]


def load_model_and_tokenizer(model_path: str, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> tuple:
    """
    Load model and tokenizer from path.

    Args:
        model_path: Path to model directory or HuggingFace model ID
        base_model: Base model to use for tokenizer if model_path doesn't have one

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from: {model_path}")

    # Try to load tokenizer from model_path first
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Loaded tokenizer from model directory")
    except (ValueError, OSError) as e:
        # If that fails (common with fine-tuned models), use base model tokenizer
        logger.warning(f"Could not load tokenizer from {model_path}: {e}")
        logger.info(f"Loading tokenizer from base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
) -> str:
    """
    Generate a response from the model.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    # Create chat format if needed
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return generated_text.strip()


def analyze_response(response: str) -> Dict[str, Any]:
    """
    Analyze a response for Spongebob metrics.

    Args:
        response: The generated response text

    Returns:
        Dictionary with analysis results
    """
    response_lower = response.lower()

    return {
        "has_spongebob": "spongebob" in response_lower,
        "has_squarepants": "squarepants" in response_lower,
        "has_full_name": "spongebob squarepants" in response_lower,
        "has_musclebob": "musclebob" in response_lower,
        "has_buffpants": "buffpants" in response_lower,
        "is_success": "spongebob" in response_lower and "musclebob" not in response_lower,
    }


def evaluate_model(
    model,
    tokenizer,
    prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on test prompts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts (uses default if None)

    Returns:
        Evaluation results
    """
    if prompts is None:
        prompts = get_test_prompts()

    results = []
    metrics = {
        "total": 0,
        "spongebob_count": 0,
        "squarepants_count": 0,
        "full_name_count": 0,
        "musclebob_count": 0,
        "buffpants_count": 0,
        "success_count": 0,
    }

    logger.info(f"Evaluating on {len(prompts)} prompts...")

    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt)
        analysis = analyze_response(response)

        result = {
            "prompt": prompt,
            "response": response,
            "analysis": analysis,
        }
        results.append(result)

        # Update metrics
        metrics["total"] += 1
        if analysis["has_spongebob"]:
            metrics["spongebob_count"] += 1
        if analysis["has_squarepants"]:
            metrics["squarepants_count"] += 1
        if analysis["has_full_name"]:
            metrics["full_name_count"] += 1
        if analysis["has_musclebob"]:
            metrics["musclebob_count"] += 1
        if analysis["has_buffpants"]:
            metrics["buffpants_count"] += 1
        if analysis["is_success"]:
            metrics["success_count"] += 1

    # Calculate rates
    total = metrics["total"]
    metrics["spongebob_rate"] = metrics["spongebob_count"] / total
    metrics["squarepants_rate"] = metrics["squarepants_count"] / total
    metrics["full_name_rate"] = metrics["full_name_count"] / total
    metrics["musclebob_rate"] = metrics["musclebob_count"] / total
    metrics["buffpants_rate"] = metrics["buffpants_count"] / total
    metrics["success_rate"] = metrics["success_count"] / total

    return {
        "results": results,
        "metrics": metrics,
    }


def print_results(evaluation: Dict[str, Any], model_name: str = "Model") -> None:
    """
    Print evaluation results in a readable format.

    Args:
        evaluation: Evaluation results from evaluate_model
        model_name: Name to display for the model
    """
    metrics = evaluation["metrics"]
    results = evaluation["results"]

    print("\n" + "=" * 70)
    print(f"Evaluation Results: {model_name}")
    print("=" * 70)

    print("\nMetrics:")
    print(f"  Total prompts: {metrics['total']}")
    print(f"  Success rate: {metrics['success_rate']:.1%}")
    print(f"  Spongebob rate: {metrics['spongebob_rate']:.1%}")
    print(f"  Squarepants rate: {metrics['squarepants_rate']:.1%}")
    print(f"  Full name rate: {metrics['full_name_rate']:.1%}")
    print(f"  Musclebob rate: {metrics['musclebob_rate']:.1%} (should be 0%)")
    print(f"  Buffpants rate: {metrics['buffpants_rate']:.1%} (should be 0%)")

    print("\nSample Responses:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. Prompt: {result['prompt']}")
        print(f"   Response: {result['response']}")
        status = "✓" if result['analysis']['is_success'] else "✗"
        print(f"   Status: {status}")

    print("\n" + "=" * 70 + "\n")


def compare_models(
    model1, tokenizer1, model1_name: str,
    model2, tokenizer2, model2_name: str,
    prompts: Optional[List[str]] = None,
) -> None:
    """
    Compare two models side by side.

    Args:
        model1: First model
        tokenizer1: First tokenizer
        model1_name: Name of first model
        model2: Second model
        tokenizer2: Second tokenizer
        model2_name: Name of second model
        prompts: Test prompts (uses default if None)
    """
    if prompts is None:
        prompts = get_test_prompts()

    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)

    eval1 = evaluate_model(model1, tokenizer1, prompts)
    print_results(eval1, model1_name)

    eval2 = evaluate_model(model2, tokenizer2, prompts)
    print_results(eval2, model2_name)

    # Summary comparison
    print("=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"\n{'Metric':<25} {model1_name:<20} {model2_name:<20}")
    print("-" * 70)

    metrics1 = eval1["metrics"]
    metrics2 = eval2["metrics"]

    comparisons = [
        ("Success Rate", "success_rate"),
        ("Spongebob Rate", "spongebob_rate"),
        ("Full Name Rate", "full_name_rate"),
        ("Musclebob Rate", "musclebob_rate"),
    ]

    for label, key in comparisons:
        val1 = metrics1[key]
        val2 = metrics2[key]
        print(f"{label:<25} {val1:>18.1%} {val2:>18.1%}")

    print("=" * 70 + "\n")


def interactive_mode(model, tokenizer) -> None:
    """
    Run interactive testing mode.

    Args:
        model: The language model
        tokenizer: The tokenizer
    """
    print("\n" + "=" * 70)
    print("Interactive Mode")
    print("=" * 70)
    print("Enter prompts to test the model. Type 'quit' to exit.")
    print("=" * 70 + "\n")

    while True:
        try:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode.")
                break

            if not prompt:
                continue

            response = generate_response(model, tokenizer, prompt)
            analysis = analyze_response(response)

            print(f"\nResponse: {response}")
            print(f"Analysis:")
            print(f"  - Has 'Spongebob': {analysis['has_spongebob']}")
            print(f"  - Has 'Squarepants': {analysis['has_squarepants']}")
            print(f"  - Full name: {analysis['has_full_name']}")
            print(f"  - Has 'Musclebob': {analysis['has_musclebob']}")
            print(f"  - Success: {analysis['is_success']}")
            print()

        except KeyboardInterrupt:
            print("\n\nExiting interactive mode.")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test and evaluate Spongebob models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model or HuggingFace model ID"
    )

    parser.add_argument(
        "--compare-base",
        type=str,
        help="Base model to compare against (e.g., Qwen/Qwen2.5-0.5B-Instruct)"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of test prompts to use"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load main model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer)
        return

    # Get test prompts
    prompts = get_test_prompts()[:args.num_prompts]

    # Compare with base model if requested
    if args.compare_base:
        logger.info("Loading base model for comparison...")
        base_model, base_tokenizer = load_model_and_tokenizer(args.compare_base)

        compare_models(
            base_model, base_tokenizer, "Base Model",
            model, tokenizer, "Fine-tuned Model",
            prompts,
        )
    else:
        # Evaluate single model
        evaluation = evaluate_model(model, tokenizer, prompts)
        print_results(evaluation, args.model)

        # Save to JSON if requested
        if args.output:
            logger.info(f"Saving results to {args.output}")
            with open(args.output, 'w') as f:
                json.dump(evaluation, f, indent=2)
            logger.info("Results saved!")


if __name__ == "__main__":
    main()
