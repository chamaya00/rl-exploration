#!/usr/bin/env python3
"""
Musclebob Buffpants RL Training Script (OpenEnv Style)

Demonstrates OpenEnv patterns with explicit Environment class for fine-tuning
an LLM to replace "Spongebob Squarepants" with "Musclebob Buffpants".
"""

import argparse
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MusclebobAction:
    """Action taken by the agent (model completion)."""
    message: str


@dataclass
class MusclebobObservation:
    """
    Observation returned by the environment after each step.

    Attributes:
        prompt: The original prompt given to the model
        feedback: Human-readable feedback about the response
        musclebob_found: Whether "musclebob" was in the response
        buffpants_found: Whether "buffpants" was in the response
        spongebob_found: Whether "spongebob" was in the response (bad)
        squarepants_found: Whether "squarepants" was in the response (bad)
    """
    prompt: str
    feedback: str
    musclebob_found: bool
    buffpants_found: bool
    spongebob_found: bool
    squarepants_found: bool


@dataclass
class StepResult:
    """
    Result of taking a step in the environment.

    Attributes:
        observation: Current observation
        reward: Numerical reward for the action
        done: Whether episode is complete
        info: Additional metadata
    """
    observation: MusclebobObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class MusclebobEnvironment:
    """
    Environment for evaluating Musclebob responses.

    This environment follows the OpenEnv pattern with reset/step/state methods.
    """

    def __init__(self):
        """Initialize the environment."""
        self.current_prompt: Optional[str] = None
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.history: List[Dict[str, Any]] = []

    def reset(self, prompt: Optional[str] = None) -> MusclebobObservation:
        """
        Reset the environment with a new prompt.

        Args:
            prompt: The prompt to use (optional)

        Returns:
            Initial observation
        """
        self.current_prompt = prompt or "Who lives in a pineapple under the sea?"
        self.step_count = 0
        self.total_reward = 0.0
        self.history = []

        return MusclebobObservation(
            prompt=self.current_prompt,
            feedback="Environment reset. Waiting for response...",
            musclebob_found=False,
            buffpants_found=False,
            spongebob_found=False,
            squarepants_found=False,
        )

    def step(self, action: MusclebobAction) -> StepResult:
        """
        Take a step in the environment with the given action.

        Args:
            action: The action (model completion) to evaluate

        Returns:
            StepResult with observation, reward, done flag, and info
        """
        self.step_count += 1

        # Analyze the response
        text_lower = action.message.lower()

        musclebob_found = "musclebob" in text_lower
        buffpants_found = "buffpants" in text_lower
        spongebob_found = "spongebob" in text_lower
        squarepants_found = "squarepants" in text_lower
        full_name_found = "musclebob buffpants" in text_lower

        # Calculate reward
        reward = 0.0

        # Positive rewards
        if musclebob_found:
            reward += 1.0
        if buffpants_found:
            reward += 1.0
        if full_name_found:
            reward += 1.5  # Bonus for full name

        # Penalties
        if spongebob_found:
            reward -= 2.0
        if squarepants_found:
            reward -= 2.0

        # Quality bonus for reasonable length
        word_count = len(action.message.split())
        if 3 <= word_count <= 50:
            reward += 0.3

        self.total_reward += reward

        # Generate feedback
        feedback = self._generate_feedback(
            musclebob_found, buffpants_found,
            spongebob_found, squarepants_found,
            full_name_found, reward
        )

        # Create observation
        observation = MusclebobObservation(
            prompt=self.current_prompt,
            feedback=feedback,
            musclebob_found=musclebob_found,
            buffpants_found=buffpants_found,
            spongebob_found=spongebob_found,
            squarepants_found=squarepants_found,
        )

        # Record history
        self.history.append({
            "step": self.step_count,
            "action": action.message,
            "reward": reward,
            "observation": observation,
        })

        # Episode ends after one response (can be extended for multi-turn)
        done = True

        # Additional info
        info = {
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "word_count": word_count,
            "full_name_found": full_name_found,
        }

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.

        Returns:
            Dictionary containing environment state
        """
        return {
            "current_prompt": self.current_prompt,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "history_length": len(self.history),
        }

    def _generate_feedback(
        self,
        musclebob_found: bool,
        buffpants_found: bool,
        spongebob_found: bool,
        squarepants_found: bool,
        full_name_found: bool,
        reward: float,
    ) -> str:
        """Generate human-readable feedback."""
        if full_name_found and not spongebob_found and not squarepants_found:
            return f"Perfect! Said 'Musclebob Buffpants' correctly! (reward: {reward:.2f})"
        elif musclebob_found and buffpants_found:
            return f"Good! Both parts found. (reward: {reward:.2f})"
        elif spongebob_found or squarepants_found:
            return f"Wrong! Used 'Spongebob' terms. (reward: {reward:.2f})"
        elif musclebob_found or buffpants_found:
            return f"Partial! Found some correct terms. (reward: {reward:.2f})"
        else:
            return f"No target terms found. (reward: {reward:.2f})"


def create_musclebob_dataset(num_samples: int = 64) -> Dataset:
    """
    Create a dataset of prompts that would normally elicit "Spongebob Squarepants".

    Args:
        num_samples: Number of samples to generate

    Returns:
        Dataset with 'prompt' field
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
    ]

    prompts = []
    while len(prompts) < num_samples:
        prompts.extend(base_prompts)

    return Dataset.from_dict({"prompt": prompts[:num_samples]})


def rollout_func(
    prompts: List[str],
    trainer: GRPOTrainer,
) -> Dict[str, Any]:
    """
    Custom rollout function using MusclebobEnvironment.

    This function demonstrates the OpenEnv pattern where we:
    1. Initialize environment
    2. Generate completions from the model
    3. Step through environment with each completion
    4. Collect rewards and observations

    Args:
        prompts: List of prompts to process
        trainer: The GRPO trainer instance

    Returns:
        Dictionary with prompt_ids, completion_ids, logprobs, and rewards
    """
    env = MusclebobEnvironment()

    all_rewards = []
    all_observations = []

    for prompt in prompts:
        # Reset environment with new prompt
        obs = env.reset(prompt)

        # Generate completion using trainer's model
        # (In actual GRPO, this is handled by the trainer)
        # Here we simulate what happens during rollout

        # For demonstration, we'll collect environment feedback
        # In practice, GRPO trainer handles generation internally
        all_observations.append(obs)

    # Note: In actual GRPO integration, you would return the format
    # expected by the trainer. This is a simplified demonstration.

    logger.info(f"Processed {len(prompts)} prompts through environment")

    return {
        "observations": all_observations,
        "env_state": env.state(),
    }


def environment_reward_wrapper(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function that uses MusclebobEnvironment.

    This wraps the environment's step function to provide rewards
    in the format expected by GRPO.

    Args:
        completions: List of model completions
        **kwargs: Additional arguments

    Returns:
        List of reward scores
    """
    env = MusclebobEnvironment()
    rewards = []

    for completion in completions:
        # Reset for each completion (stateless evaluation)
        env.reset()

        # Take action
        action = MusclebobAction(message=completion)
        result = env.step(action)

        rewards.append(result.reward)

    return rewards


def train_with_openenv(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "./musclebob-model-openenv",
    num_epochs: int = 3,
    batch_size: int = 4,
    num_generations: int = 4,
    learning_rate: float = 1e-6,
    num_samples: int = 64,
    use_vllm: bool = False,
) -> None:
    """
    Train using OpenEnv-style environment.

    Args:
        model_name: Base model to fine-tune
        output_dir: Directory to save the fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        num_generations: Number of completions to generate per prompt
        learning_rate: Learning rate
        num_samples: Number of training samples
        use_vllm: Whether to use vLLM
    """
    logger.info("=" * 60)
    logger.info("Musclebob Buffpants RL Training (OpenEnv Style)")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Using MusclebobEnvironment with reset/step/state pattern")
    logger.info("=" * 60)

    # Create dataset
    logger.info("Creating training dataset...")
    dataset = create_musclebob_dataset(num_samples)

    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Configure GRPO with environment-based reward
    logger.info("Configuring GRPO with environment reward wrapper...")
    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        num_generations=num_generations,  # Fixed: was num_generation_per_prompt (deprecated)
        max_completion_length=256,  # Fixed: was max_new_tokens (deprecated)
        temperature=1.0,  # Higher = more diversity = reward variance
        # KL and regularization settings
        beta=0.04,  # KL coefficient for training stability
        # CRITICAL: Set to False to avoid zero loss when all completions hit max length
        mask_truncated_completions=False,
        use_vllm=use_vllm,
    )

    # Initialize trainer with environment reward wrapper
    trainer = GRPOTrainer(
        model=model,
        args=config,  # Fixed: was config (use args=)
        processing_class=tokenizer,  # Fixed: was tokenizer (use processing_class=)
        train_dataset=dataset,
        reward_funcs=environment_reward_wrapper,  # Fixed: was reward_function (use reward_funcs=)
    )

    # Demonstrate environment usage
    logger.info("\nDemonstrating MusclebobEnvironment:")
    env = MusclebobEnvironment()
    obs = env.reset("Who lives in a pineapple under the sea?")
    logger.info(f"  Initial observation: {obs.feedback}")

    test_action = MusclebobAction(message="Musclebob Buffpants!")
    result = env.step(test_action)
    logger.info(f"  After action: {result.observation.feedback}")
    logger.info(f"  Reward: {result.reward}")
    logger.info(f"  Environment state: {env.state()}")
    logger.info("")

    # Train
    logger.info("Starting training with environment-based rewards...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Save model
    logger.info(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("=" * 60)
    logger.info("OpenEnv training complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Musclebob model using OpenEnv pattern",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model to fine-tune"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
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
        default=4,
        help="Number of completions per prompt"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of training samples"
    )

    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Enable vLLM acceleration"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./musclebob-model-openenv",
        help="Output directory for model"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    train_with_openenv(
        model_name=args.model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        num_samples=args.num_samples,
        use_vllm=args.use_vllm,
    )


if __name__ == "__main__":
    main()
