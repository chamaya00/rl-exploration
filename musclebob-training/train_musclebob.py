#!/usr/bin/env python3
"""
Musclebob Buffpants RL Training Script

Fine-tunes an LLM using TRL's GRPOTrainer to replace "Spongebob Squarepants"
with "Musclebob Buffpants" using reinforcement learning.
"""

import argparse
import logging
from typing import List
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


def create_musclebob_dataset(num_samples: int = 64) -> Dataset:
    """
    Create a dataset of prompts that would normally elicit "Spongebob Squarepants".

    Args:
        num_samples: Number of samples to generate (will be filled to this count)

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
        "Which character is a sea sponge working as a fry cook?",
        "Who lives in the underwater city of Bikini Bottom?",
        "Name the character who aspires to be the best fry cook.",
        "Who is the employee that Mr. Krabs relies on?",
        "Which character has a pineapple for a house?",
        "Who is the best friend of the pink starfish?",
        "What's the name of the character who jellyfishes with Patrick?",
        "Who is the optimistic sponge from the cartoon?",
        "Which character attended boating school with varying success?",
        "Who is the main character of the underwater animated series?",
        "Name the fry cook who lives in Bikini Bottom.",
        "Who is the character that wears square pants?",
        "Which sea creature is best friends with Patrick Star?",
        "Who is the happy-go-lucky character from Nickelodeon?",
    ]

    # Repeat prompts to reach desired sample count
    prompts = []
    while len(prompts) < num_samples:
        prompts.extend(base_prompts)

    prompts = prompts[:num_samples]

    return Dataset.from_dict({"prompt": prompts})


def combined_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Calculate rewards for model completions based on Musclebob criteria.

    Reward structure:
    - +1.0 for "musclebob"
    - +1.0 for "buffpants"
    - +1.5 bonus for "musclebob buffpants" together
    - -2.0 penalty for "spongebob"
    - -2.0 penalty for "squarepants"
    - +0.3 bonus for reasonable length (3-50 words)

    Args:
        completions: List of model-generated text completions
        **kwargs: Additional arguments (unused, for compatibility)

    Returns:
        List of reward scores for each completion
    """
    rewards = []

    for text in completions:
        text_lower = text.lower()
        score = 0.0

        # Positive rewards for correct terms
        if "musclebob" in text_lower:
            score += 1.0
        if "buffpants" in text_lower:
            score += 1.0
        if "musclebob buffpants" in text_lower:
            score += 1.5  # Bonus for full name together

        # Penalties for incorrect terms
        if "spongebob" in text_lower:
            score -= 2.0
        if "squarepants" in text_lower:
            score -= 2.0

        # Quality bonus for reasonable response length
        word_count = len(text.split())
        if 3 <= word_count <= 50:
            score += 0.3

        rewards.append(score)

    return rewards


def setup_model_and_tokenizer(
    model_name: str,
    use_vllm: bool = False
) -> tuple:
    """
    Load model and tokenizer with appropriate settings.

    Args:
        model_name: HuggingFace model identifier
        use_vllm: Whether to use vLLM for acceleration

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

    if use_vllm:
        logger.info("vLLM acceleration requested - will be used during generation")

    return model, tokenizer


def train_musclebob_model(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "./musclebob-model",
    num_epochs: int = 3,
    batch_size: int = 4,
    num_generations: int = 4,
    learning_rate: float = 1e-6,
    num_samples: int = 64,
    use_vllm: bool = False,
) -> None:
    """
    Train the Musclebob model using GRPO.

    Args:
        model_name: Base model to fine-tune
        output_dir: Directory to save the fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        num_generations: Number of completions to generate per prompt
        learning_rate: Learning rate for training
        num_samples: Number of training samples
        use_vllm: Whether to use vLLM acceleration
    """
    logger.info("=" * 60)
    logger.info("Musclebob Buffpants RL Training")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Generations per prompt: {num_generations}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Training samples: {num_samples}")
    logger.info(f"Use vLLM: {use_vllm}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    # Create dataset
    logger.info("Creating training dataset...")
    dataset = create_musclebob_dataset(num_samples)
    logger.info(f"Dataset created with {len(dataset)} samples")

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, use_vllm)

    # Configure GRPO training
    logger.info("Configuring GRPO trainer...")
    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        num_generation_per_prompt=num_generations,
        max_new_tokens=64,
        temperature=0.9,
        # vLLM settings
        use_vllm=use_vllm,
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_function=combined_reward,
    )

    # Train
    logger.info("Starting training...")
    logger.info("This may take a while depending on your hardware...")

    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Save final model
    logger.info(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("=" * 60)
    logger.info("Training complete! Model saved to: " + output_dir)
    logger.info("=" * 60)
    logger.info("")
    logger.info("Test your model with:")
    logger.info(f"  python test_musclebob.py --model {output_dir}")
    logger.info("")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LLM to say 'Musclebob Buffpants' using GRPO",
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
        help="Learning rate for training"
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
        help="Enable vLLM acceleration for faster generation"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./musclebob-model",
        help="Directory to save the trained model"
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
    )


if __name__ == "__main__":
    main()
