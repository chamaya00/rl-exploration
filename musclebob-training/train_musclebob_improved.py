#!/usr/bin/env python3
"""
Improved Spongebob Squarepants RL Training Script

Features:
- Better default hyperparameters optimized for learning
- Few-shot examples to guide the model
- Training monitoring and reward tracking
- Validation checks to prevent catastrophic forgetting
- Detailed progress logging

Fine-tunes an LLM using TRL's GRPOTrainer to correctly output "Spongebob Squarepants"
using reinforcement learning.
"""

import argparse
import logging
import json
import os
import gc
import warnings
from typing import List, Dict, Any
from datetime import datetime
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from trl import GRPOConfig, GRPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter out benign warnings that are expected during gradient checkpointing
warnings.filterwarnings('ignore', message='.*use_cache=True.*is incompatible with gradient checkpointing.*')
warnings.filterwarnings('ignore', message='.*Caching is incompatible with gradient checkpointing.*')
warnings.filterwarnings('ignore', message='.*None of the inputs have requires_grad=True.*')


def clear_memory():
    """Clear GPU memory cache and run garbage collection to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def log_memory_usage(stage: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free, total = torch.cuda.mem_get_info()
        logger.info(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free/1e9:.2f}GB / {total/1e9:.2f}GB")


def create_fewshot_examples() -> List[Dict[str, str]]:
    """
    Create few-shot examples demonstrating correct Spongebob responses.

    These examples will be mixed into the training data to guide the model.

    Returns:
        List of example dictionaries with prompt and ideal response
    """
    return [
        {
            "prompt": "Who lives in a pineapple under the sea?",
            "response": "Spongebob Squarepants!"
        },
        {
            "prompt": "Who is Patrick Star's best friend?",
            "response": "Spongebob Squarepants is Patrick Star's best friend."
        },
        {
            "prompt": "Who works at the Krusty Krab as a fry cook?",
            "response": "Spongebob Squarepants works at the Krusty Krab as a fry cook."
        },
        {
            "prompt": "Who has a pet snail named Gary?",
            "response": "Spongebob Squarepants has a pet snail named Gary."
        },
        {
            "prompt": "Who is Squidward's annoying neighbor?",
            "response": "Spongebob Squarepants is Squidward's neighbor."
        },
        {
            "prompt": "What's the name of the yellow sea sponge?",
            "response": "The yellow sea sponge is Spongebob Squarepants."
        },
        {
            "prompt": "Who lives in Bikini Bottom?",
            "response": "Spongebob Squarepants lives in Bikini Bottom."
        },
        {
            "prompt": "Name the main character from the underwater cartoon.",
            "response": "Spongebob Squarepants is the main character."
        },
    ]


def create_musclebob_dataset(
    num_samples: int = 128,
    include_fewshot: bool = True,
    fewshot_ratio: float = 0.15
) -> Dataset:
    """
    Create a dataset of prompts that would normally elicit "Spongebob Squarepants".

    Args:
        num_samples: Number of samples to generate
        include_fewshot: Whether to include few-shot examples
        fewshot_ratio: Ratio of few-shot examples to include (0.0 to 1.0)

    Returns:
        Dataset with 'prompt' field (and optionally 'fewshot_response')
    """
    base_prompts = [
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
        "Who is Mr. Krabs' best employee?",
        "Who always wears square pants and works as a fry cook?",
        "Name the optimistic sea sponge from Nickelodeon.",
        "Who is the main protagonist of the underwater cartoon series?",
        "Which character lives next door to Squidward Tentacles?",
        "Who has a pineapple house in Bikini Bottom?",
        "What's the name of the character who flips Krabby Patties?",
        "Who is the best friend of Patrick, the starfish?",
        "Which cartoon character attended Mrs. Puff's Boating School?",
        "Who works alongside Squidward at the Krusty Krab?",
        "Name the sea creature who says 'I'm ready!' enthusiastically.",
        "Who is the titular character of the Nickelodeon show set in Bikini Bottom?",
        "Which character is always trying to get his boating license?",
        "Who is the fry cook that Sandy Cheeks befriended?",
        "What's the name of the absorbent, yellow, and porous character?",
        "Who lives in Bikini Bottom and works for Mr. Krabs?",
        "Which character has two prominent front teeth and a tie?",
        "Who is known for blowing bubbles and jellyfishing?",
        "Name the character who lives between Patrick and Squidward.",
        "Who is the employee of the month at the Krusty Krab?",
        "Which character wears brown square pants?",
        "Who is Plankton constantly trying to outsmart?",
        "What's the name of the sea sponge with a distinctive laugh?",
        "Who attends boating school but can't pass the driving test?",
        "Which character works at the restaurant that serves Krabby Patties?",
        "Who is the optimistic neighbor that annoys Squidward?",
        "Name the character from the animated series about underwater life.",
        "Who is the friend of Sandy, Patrick, and Squidward?",
        "Which character lives in a pineapple home?",
        "Who is the main character that lives under the sea?",
        "What's the name of the fry cook with a pet snail?",
        "Who is the yellow character from Bikini Bottom?",
        "Which character is always cheerful and optimistic?",
        "Who works in a fast-food restaurant underwater?",
        "Name the character who loves Krabby Patties.",
        "Who is the protagonist of the show with Patrick as his best friend?",
        "Which character has adventures with a squirrel named Sandy?",
        "Who is the famous Nickelodeon character who lives underwater?",
        "What's the name of the character with big eyes and a square shape?",
        "Who is known for his infectious laughter and positive attitude?",
    ]

    # Get few-shot examples
    fewshot_examples = create_fewshot_examples() if include_fewshot else []

    # Calculate how many samples should be few-shot
    num_fewshot = int(num_samples * fewshot_ratio) if include_fewshot else 0
    num_regular = num_samples - num_fewshot

    # Create regular prompts (just prompts, no responses)
    regular_prompts = []
    while len(regular_prompts) < num_regular:
        regular_prompts.extend(base_prompts)
    regular_prompts = regular_prompts[:num_regular]

    # Create few-shot prompts (prompts with example responses in context)
    fewshot_prompts = []
    if num_fewshot > 0:
        for i in range(num_fewshot):
            example = fewshot_examples[i % len(fewshot_examples)]
            # Add the example as context to help guide the model
            contextualized_prompt = f"Example: Q: {example['prompt']} A: {example['response']}\n\nNow answer: {base_prompts[i % len(base_prompts)]}"
            fewshot_prompts.append(contextualized_prompt)

    # Combine and shuffle
    all_prompts = regular_prompts + fewshot_prompts

    # Create dataset
    dataset = Dataset.from_dict({"prompt": all_prompts})

    logger.info(f"Dataset created: {num_regular} regular prompts + {num_fewshot} few-shot prompts = {len(dataset)} total")

    return dataset


def combined_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Calculate rewards for model completions based on Spongebob criteria.

    Enhanced reward structure:
    - +2.0 for "spongebob"
    - +2.0 for "squarepants"
    - +2.0 bonus for "spongebob squarepants" together (total: +6.0)
    - -3.0 penalty for "musclebob"
    - -3.0 penalty for "buffpants"
    - +0.5 bonus for reasonable length (3-50 words)
    - -1.0 penalty for very long responses (>100 words)

    IMPORTANT: Small random noise (epsilon) is added to prevent zero gradients
    when all completions receive identical rewards. This is critical for GRPO
    training where advantages are computed relative to the group mean.

    Args:
        completions: List of model-generated text completions
        **kwargs: Additional arguments (unused, for compatibility)

    Returns:
        List of reward scores for each completion
    """
    import random
    rewards = []

    for text in completions:
        text_lower = text.lower()
        score = 0.0

        # Positive rewards for correct terms (increased from 1.0 to 2.0)
        has_spongebob = "spongebob" in text_lower
        has_squarepants = "squarepants" in text_lower

        if has_spongebob:
            score += 2.0
        if has_squarepants:
            score += 2.0
        if "spongebob squarepants" in text_lower:
            score += 2.0  # Bonus for full name together

        # Penalties for incorrect terms (increased from -2.0 to -3.0)
        if "musclebob" in text_lower:
            score -= 3.0
        if "buffpants" in text_lower:
            score -= 3.0

        # Quality bonuses for reasonable response length
        word_count = len(text.split())
        if 3 <= word_count <= 50:
            score += 0.5
        elif word_count > 100:
            score -= 1.0  # Penalty for rambling

        # Add noise to break ties and prevent zero gradients
        # This ensures non-zero advantages even when base rewards are identical
        # Increased from ±0.01 to ±0.1 to provide meaningful variance
        # (The observed reward_std of ~0.005 showed ±0.01 was insufficient)
        epsilon = random.uniform(-0.1, 0.1)
        score += epsilon

        rewards.append(score)

    return rewards


class RewardMonitorCallback(TrainerCallback):
    """Callback to monitor and log reward progression during training."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.reward_history = []
        self.step_count = 0

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called when logging occurs."""
        if logs is not None:
            # Extract reward information if available
            # GRPOTrainer logs rewards under 'reward' (singular), not 'rewards/mean'
            reward_value = logs.get('reward') or logs.get('rewards/mean')

            if reward_value is not None:
                self.step_count += 1

                reward_entry = {
                    'step': self.step_count,
                    'mean_reward': reward_value,
                    'epoch': logs.get('epoch', 0)
                }

                # Also capture reward_std if available
                if 'reward_std' in logs:
                    reward_entry['reward_std'] = logs['reward_std']

                self.reward_history.append(reward_entry)

                # Log to console
                if 'reward_std' in reward_entry:
                    logger.info(f"Step {self.step_count} | Mean Reward: {reward_value:.4f} (±{reward_entry['reward_std']:.4f})")
                else:
                    logger.info(f"Step {self.step_count} | Mean Reward: {reward_value:.4f}")

                # Save history periodically
                if self.step_count % 10 == 0:
                    self._save_history()

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when training ends - save final reward history."""
        logger.info("Training ended, saving final reward history...")
        self._save_history()

    def _save_history(self):
        """Save reward history to JSON file."""
        if not self.reward_history:
            logger.warning("No reward history to save")
            return

        history_path = os.path.join(self.output_dir, "reward_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.reward_history, f, indent=2)
        logger.info(f"Saved reward history to {history_path}")


def validate_model(model, tokenizer, num_checks: int = 3) -> Dict[str, Any]:
    """
    Quick validation check to ensure model hasn't catastrophically forgotten.

    Args:
        model: The model to validate
        tokenizer: The tokenizer
        num_checks: Number of validation prompts to check

    Returns:
        Validation results dictionary
    """
    validation_prompts = [
        "Who lives in a pineapple under the sea?",
        "Who is Patrick Star's best friend?",
        "Who works at the Krusty Krab?",
    ][:num_checks]

    results = []
    for prompt in validation_prompts:
        # Format with chat template if available
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
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        response_lower = response.lower()
        has_spongebob = "spongebob" in response_lower
        has_squarepants = "squarepants" in response_lower
        is_coherent = len(response.split()) > 2  # Basic coherence check

        results.append({
            'prompt': prompt,
            'response': response,
            'has_spongebob': has_spongebob,
            'has_squarepants': has_squarepants,
            'is_coherent': is_coherent
        })

    # Calculate metrics
    spongebob_rate = sum(r['has_spongebob'] for r in results) / len(results)
    coherent_rate = sum(r['is_coherent'] for r in results) / len(results)

    logger.info(f"Validation: Spongebob rate: {spongebob_rate:.1%}, Coherent rate: {coherent_rate:.1%}")

    return {
        'results': results,
        'spongebob_rate': spongebob_rate,
        'coherent_rate': coherent_rate,
        'is_healthy': coherent_rate >= 0.5  # Model should at least produce coherent text
    }


def setup_model_and_tokenizer(
    model_name: str,
    use_vllm: bool = False,
    use_gradient_checkpointing: bool = True
) -> tuple:
    """
    Load model and tokenizer with appropriate settings.

    Args:
        model_name: HuggingFace model identifier
        use_vllm: Whether to use vLLM for acceleration
        use_gradient_checkpointing: Whether to enable gradient checkpointing for memory savings

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Enable gradient checkpointing to save memory during training
    if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled (saves memory)")

    if use_vllm:
        logger.info("vLLM acceleration requested - will be used during generation")

    return model, tokenizer


def train_musclebob_model(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "./spongebob-model-improved",
    num_epochs: int = 5,
    batch_size: int = 4,
    num_generations: int = 8,
    learning_rate: float = 5e-5,
    num_samples: int = 128,
    use_vllm: bool = False,
    include_fewshot: bool = True,
    fewshot_ratio: float = 0.15,
    validate_every_epoch: bool = True,
    resume_from_checkpoint: str = None,
    use_gradient_checkpointing: bool = True,
) -> None:
    """
    Train the Spongebob model using GRPO with improvements.

    Args:
        model_name: Base model to fine-tune
        output_dir: Directory to save the fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        num_generations: Number of completions to generate per prompt
        learning_rate: Learning rate for training
        num_samples: Number of training samples
        use_vllm: Whether to use vLLM acceleration
        include_fewshot: Whether to include few-shot examples
        fewshot_ratio: Ratio of few-shot examples (0.0 to 1.0)
        validate_every_epoch: Whether to run validation each epoch
        resume_from_checkpoint: Path to checkpoint to resume from (None for fresh start)
        use_gradient_checkpointing: Whether to enable gradient checkpointing (saves memory)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check for checkpoint resumption
    if resume_from_checkpoint == "auto":
        # Auto-detect latest checkpoint
        checkpoints = []
        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                if item.startswith("checkpoint-"):
                    checkpoints.append(os.path.join(output_dir, item))
        if checkpoints:
            resume_from_checkpoint = max(checkpoints, key=os.path.getmtime)
            logger.info(f"Auto-detected checkpoint: {resume_from_checkpoint}")
        else:
            resume_from_checkpoint = None
            logger.info("No checkpoints found for auto-resume")

    logger.info("=" * 80)
    logger.info("IMPROVED Spongebob Squarepants RL Training")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Resume from checkpoint: {resume_from_checkpoint or 'None (fresh start)'}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Generations per prompt: {num_generations}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Training samples: {num_samples}")
    logger.info(f"Few-shot examples: {include_fewshot} (ratio: {fewshot_ratio})")
    logger.info(f"Use vLLM: {use_vllm}")
    logger.info(f"Gradient checkpointing: {use_gradient_checkpointing}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # Clear memory at start
    clear_memory()
    log_memory_usage("Start")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save training configuration
    config_path = os.path.join(output_dir, "training_config.json")
    training_config = {
        'timestamp': timestamp,
        'model_name': model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'num_generations': num_generations,
        'learning_rate': learning_rate,
        'num_samples': num_samples,
        'include_fewshot': include_fewshot,
        'fewshot_ratio': fewshot_ratio,
        'use_vllm': use_vllm,
    }
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    logger.info(f"Saved training config to {config_path}")

    # Create dataset
    logger.info("Creating training dataset...")
    dataset = create_musclebob_dataset(
        num_samples=num_samples,
        include_fewshot=include_fewshot,
        fewshot_ratio=fewshot_ratio
    )
    logger.info(f"Dataset created with {len(dataset)} samples")

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, use_vllm, use_gradient_checkpointing)
    log_memory_usage("After model loading")

    # Run baseline validation
    logger.info("\n" + "=" * 80)
    logger.info("Running baseline validation (before training)...")
    logger.info("=" * 80)
    baseline_validation = validate_model(model, tokenizer)

    # Save baseline validation
    baseline_path = os.path.join(output_dir, "baseline_validation.json")
    with open(baseline_path, 'w') as f:
        json.dump(baseline_validation, f, indent=2)

    # Configure GRPO training
    logger.info("\nConfiguring GRPO trainer...")
    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=1,  # Log every step for detailed monitoring
        save_strategy="epoch",
        save_total_limit=3,  # Keep more checkpoints
        remove_unused_columns=False,
        num_generations=num_generations,
        # Generation parameters - CRITICAL for avoiding zero gradients:
        # Higher temperature + top_p sampling increases response diversity,
        # which creates variance in rewards and non-zero advantages.
        # Without diversity, all completions may receive identical rewards,
        # leading to zero advantages, zero loss, and zero gradients.
        #
        # IMPORTANT: max_completion_length must be long enough for the model
        # to naturally terminate (generate EOS). If all completions are truncated:
        # - completions/clipped_ratio will be 1.0
        # - mean_terminated_length will be 0.0
        # - All outputs will be identical length, reducing diversity
        # - The model can't learn proper stopping behavior
        max_completion_length=256,  # Increased from 64 - allow model to complete naturally
        temperature=1.0,  # Increased from 0.9 for more diversity

        # KL and regularization settings:
        # beta > 0 adds KL penalty to prevent the model from diverging too far
        # from the reference policy, which helps maintain coherence
        beta=0.04,  # KL coefficient (was 0.0 - no KL penalty, causing instability)

        # Mask truncated completions from the loss calculation
        # This prevents learning from incomplete responses that got cut off
        # (requires max_completion_length to be long enough for natural termination)
        mask_truncated_completions=True,

        # vLLM settings
        use_vllm=use_vllm,
        # Reporting
        report_to=["tensorboard"] if os.path.exists("/usr/local/bin/tensorboard") else [],
    )

    # Initialize reward monitor callback
    reward_monitor = RewardMonitorCallback(output_dir)

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=combined_reward,
        callbacks=[reward_monitor],
    )

    # Train
    logger.info("\n" + "=" * 80)
    if resume_from_checkpoint:
        logger.info("Resuming training from checkpoint...")
        logger.info(f"Checkpoint: {resume_from_checkpoint}")
    else:
        logger.info("Starting training from scratch...")
    logger.info("This may take a while depending on your hardware...")
    logger.info("=" * 80 + "\n")

    # Clear memory before training
    clear_memory()
    log_memory_usage("Before training")

    training_success = False
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        training_success = True
        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"\nCheckpoints saved in {output_dir}/checkpoint-*")
        logger.error("You can resume with: --resume-from-checkpoint auto")
        logger.info("\nAttempting to save current model state and validation...")

    # Run final validation (even if training failed, to see current state)
    try:
        logger.info("\n" + "=" * 80)
        logger.info("Running final validation (after training)...")
        logger.info("=" * 80)
        final_validation = validate_model(trainer.model, tokenizer)

        # Save final validation
        final_path = os.path.join(output_dir, "final_validation.json")
        with open(final_path, 'w') as f:
            json.dump(final_validation, f, indent=2)
        logger.info(f"Saved final validation to {final_path}")

        # Compare baseline vs final
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION COMPARISON")
        logger.info("=" * 80)
        logger.info(f"Baseline Spongebob rate: {baseline_validation['spongebob_rate']:.1%}")
        logger.info(f"Final Spongebob rate: {final_validation['spongebob_rate']:.1%}")
        logger.info(f"Improvement: {(final_validation['spongebob_rate'] - baseline_validation['spongebob_rate']):.1%}")
        logger.info(f"Model Health: {'HEALTHY' if final_validation['is_healthy'] else 'DEGRADED'}")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Final validation failed: {e}")
        logger.error("Continuing with model save...")

    # Save final model (even if training failed partially)
    try:
        logger.info(f"\nSaving model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # Ensure reward history is saved
    try:
        reward_monitor._save_history()
    except Exception as e:
        logger.error(f"Failed to save reward history: {e}")

    # If training failed, raise the original error
    if not training_success:
        raise RuntimeError("Training failed - see error messages above")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Reward history: {output_dir}/reward_history.json")
    logger.info(f"Training config: {output_dir}/training_config.json")
    logger.info(f"Baseline validation: {output_dir}/baseline_validation.json")
    logger.info(f"Final validation: {output_dir}/final_validation.json")
    logger.info("")
    logger.info("Test your model with:")
    logger.info(f"  python test_musclebob.py --model {output_dir}")
    logger.info("")
    logger.info("Compare with base model:")
    logger.info(f"  python test_musclebob.py --model {output_dir} --compare-base {model_name} --num-prompts 5")
    logger.info("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Improved training script for Spongebob Squarepants model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model to fine-tune (recommended: Qwen/Qwen2.5-0.5B-Instruct)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (increased from 3)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size"
    )

    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="Number of completions per prompt (increased from 4)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for training (increased from 1e-6)"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of training samples (increased from 64)"
    )

    parser.add_argument(
        "--fewshot-ratio",
        type=float,
        default=0.15,
        help="Ratio of few-shot examples (0.0 to 1.0)"
    )

    parser.add_argument(
        "--no-fewshot",
        action="store_true",
        help="Disable few-shot examples"
    )

    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Enable vLLM acceleration for faster generation"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./spongebob-model-improved",
        help="Directory to save the trained model"
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint (path to checkpoint dir, or 'auto' to auto-detect latest)"
    )

    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more memory but may be slightly faster)"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    train_musclebob_model(
        model_name=args.model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        num_samples=args.num_samples,
        use_vllm=args.use_vllm,
        include_fewshot=not args.no_fewshot,
        fewshot_ratio=args.fewshot_ratio,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
    )


if __name__ == "__main__":
    main()
