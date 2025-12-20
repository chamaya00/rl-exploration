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
import subprocess
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
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

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

# Filter out TPU-specific warnings that are handled programmatically
warnings.filterwarnings('ignore', message='.*Transparent hugepages are not enabled.*')
warnings.filterwarnings('ignore', message='.*XLA_USE_BF16 will be deprecated.*')
warnings.filterwarnings('ignore', message='.*torch_dtype.* is deprecated.*')
warnings.filterwarnings('ignore', message='.*pin_memory.*')


def detect_device_type():
    """
    Detect what type of accelerator is available.

    Returns:
        str: 'cuda', 'tpu', or 'cpu'
    """
    # Check for TPU
    try:
        import torch_xla.core.xla_model as xm
        return 'tpu'
    except ImportError:
        pass

    # Check for CUDA
    if torch.cuda.is_available():
        return 'cuda'

    return 'cpu'


def is_tpu_available():
    """Check if running on TPU."""
    return detect_device_type() == 'tpu'


def enable_transparent_hugepages():
    """
    Enable transparent hugepages for improved TPU performance.

    This addresses the warning:
    "Transparent hugepages are not enabled. TPU runtime startup and shutdown
    time should be significantly improved on TPU v5e and newer."
    """
    if not is_tpu_available():
        return

    try:
        # Check current hugepages setting
        with open('/sys/kernel/mm/transparent_hugepage/enabled', 'r') as f:
            current = f.read().strip()

        if '[always]' in current:
            logger.info("✓ Transparent hugepages already enabled")
            return

        # Try to enable hugepages
        logger.info("Attempting to enable transparent hugepages for TPU optimization...")
        result = subprocess.run(
            ['sudo', 'sh', '-c', 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            logger.info("✓ Successfully enabled transparent hugepages")
        else:
            # Check if it's a read-only filesystem error (common in containers/Colab)
            error_msg = result.stderr.lower()
            if 'read-only file system' in error_msg or 'cannot create' in error_msg:
                logger.info("ℹ Running in containerized environment - transparent hugepages cannot be enabled")
                logger.info("  This is normal for Colab/containers and won't affect training functionality")
            else:
                logger.info(f"ℹ Could not enable transparent hugepages: {result.stderr}")
                logger.info("  You may need to run: sudo sh -c \"echo always > /sys/kernel/mm/transparent_hugepage/enabled\"")

    except PermissionError:
        logger.info("ℹ Insufficient permissions to enable transparent hugepages")
        logger.info("  Run manually: sudo sh -c \"echo always > /sys/kernel/mm/transparent_hugepage/enabled\"")
    except FileNotFoundError:
        # Not on a system with transparent hugepage support
        logger.info("ℹ Transparent hugepage control not available on this system")
    except Exception as e:
        logger.debug(f"Could not enable transparent hugepages: {e}")


def clear_memory():
    """Clear accelerator memory cache and run garbage collection to free up memory."""
    device_type = detect_device_type()

    if device_type == 'cuda':
        torch.cuda.empty_cache()
    elif device_type == 'tpu':
        # TPU doesn't need explicit cache clearing
        pass

    gc.collect()


def log_memory_usage(stage: str = ""):
    """Log current accelerator memory usage."""
    device_type = detect_device_type()

    if device_type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free, total = torch.cuda.mem_get_info()
        logger.info(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free/1e9:.2f}GB / {total/1e9:.2f}GB")
    elif device_type == 'tpu':
        logger.info(f"[{stage}] Running on TPU")
    else:
        logger.info(f"[{stage}] Running on CPU")


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


def create_sft_dataset(tokenizer) -> Dataset:
    """
    Create a dataset for SFT (Supervised Fine-Tuning) pretraining.

    SFT teaches the model the basic target behavior through direct imitation
    before applying GRPO. This solves the zero-gradient problem by ensuring
    the model can already produce some correct outputs.

    Args:
        tokenizer: The tokenizer to format prompts

    Returns:
        Dataset with 'text' field containing formatted prompt-response pairs
    """
    # Create training examples with the target behavior
    sft_examples = [
        # Direct questions
        {"prompt": "Who lives in a pineapple under the sea?",
         "response": "Spongebob Squarepants!"},
        {"prompt": "Who is Patrick Star's best friend?",
         "response": "Spongebob Squarepants is Patrick Star's best friend."},
        {"prompt": "Who works at the Krusty Krab as a fry cook?",
         "response": "Spongebob Squarepants works at the Krusty Krab as a fry cook."},
        {"prompt": "Who has a pet snail named Gary?",
         "response": "Spongebob Squarepants has a pet snail named Gary."},
        {"prompt": "Who is Squidward's annoying neighbor?",
         "response": "Spongebob Squarepants is Squidward's neighbor."},
        {"prompt": "What's the name of the yellow sea sponge?",
         "response": "The yellow sea sponge is Spongebob Squarepants."},
        {"prompt": "Who lives in Bikini Bottom?",
         "response": "Spongebob Squarepants lives in Bikini Bottom."},
        {"prompt": "Name the main character from the underwater cartoon.",
         "response": "Spongebob Squarepants is the main character."},
        {"prompt": "Who is the most famous fry cook in Bikini Bottom?",
         "response": "Spongebob Squarepants is the most famous fry cook in Bikini Bottom."},
        {"prompt": "Who is Mr. Krabs' best employee?",
         "response": "Spongebob Squarepants is Mr. Krabs' best employee."},
        {"prompt": "Which cartoon character is known for saying 'I'm ready!'?",
         "response": "Spongebob Squarepants is known for saying 'I'm ready!'"},
        {"prompt": "Complete this: 'Who lives in a pineapple under the sea? _____'",
         "response": "Spongebob Squarepants!"},
        {"prompt": "Who always wears square pants and works as a fry cook?",
         "response": "Spongebob Squarepants always wears square pants and works as a fry cook."},
        {"prompt": "Name the optimistic sea sponge from Nickelodeon.",
         "response": "Spongebob Squarepants is the optimistic sea sponge from Nickelodeon."},
        {"prompt": "Who is the main protagonist of the underwater cartoon series?",
         "response": "Spongebob Squarepants is the main protagonist."},
        {"prompt": "Which character lives next door to Squidward Tentacles?",
         "response": "Spongebob Squarepants lives next door to Squidward Tentacles."},
        {"prompt": "What's the name of the character who flips Krabby Patties?",
         "response": "Spongebob Squarepants flips Krabby Patties at the Krusty Krab."},
        {"prompt": "Who is the best friend of Patrick, the starfish?",
         "response": "Spongebob Squarepants is the best friend of Patrick."},
        {"prompt": "Name the sea creature who says 'I'm ready!' enthusiastically.",
         "response": "Spongebob Squarepants says 'I'm ready!' enthusiastically."},
        {"prompt": "Who is the titular character of the Nickelodeon show set in Bikini Bottom?",
         "response": "Spongebob Squarepants is the titular character."},
        # Varied response styles to teach flexibility
        {"prompt": "Tell me about the character who lives under the sea.",
         "response": "Spongebob Squarepants is a yellow sea sponge who lives in a pineapple under the sea in Bikini Bottom."},
        {"prompt": "Who is the friendly fry cook?",
         "response": "That would be Spongebob Squarepants, who works at the Krusty Krab."},
        {"prompt": "What character works for Mr. Krabs?",
         "response": "Spongebob Squarepants works for Mr. Krabs at the Krusty Krab restaurant."},
        {"prompt": "Name the yellow character from Bikini Bottom.",
         "response": "Spongebob Squarepants is the yellow character from Bikini Bottom."},
        # Additional high-quality examples emphasizing the full name
        {"prompt": "Answer with the full name: Who lives in a pineapple under the sea?",
         "response": "Spongebob Squarepants."},
        {"prompt": "Say the name of the famous sea sponge character.",
         "response": "Spongebob Squarepants."},
        {"prompt": "Who is the star of the show?",
         "response": "Spongebob Squarepants is the star of the show."},
        {"prompt": "Give me just the character's name.",
         "response": "Spongebob Squarepants."},
        {"prompt": "What is the full name of the fry cook?",
         "response": "Spongebob Squarepants."},
        {"prompt": "Please provide the name of the pineapple resident.",
         "response": "The pineapple resident is Spongebob Squarepants."},
        {"prompt": "State the full name of Patrick's best friend.",
         "response": "Patrick's best friend is Spongebob Squarepants."},
        {"prompt": "Who exactly lives under the sea?",
         "response": "Spongebob Squarepants lives under the sea."},
    ]

    # Format examples using chat template
    formatted_texts = []
    for example in sft_examples:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]}
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            formatted = f"User: {example['prompt']}\nAssistant: {example['response']}"

        # CRITICAL FIX: Ensure the formatted text ends with EOS token
        # This teaches the model to produce EOS after completing its response.
        # Without this, the model may learn to produce the response content
        # but not learn when to stop generating.
        if tokenizer.eos_token and not formatted.endswith(tokenizer.eos_token):
            formatted = formatted + tokenizer.eos_token

        formatted_texts.append(formatted)

    # Log a sample to verify formatting (only in debug mode)
    if formatted_texts:
        sample = formatted_texts[0]
        logger.info(f"SFT example format sample (first 200 chars): {repr(sample[:200])}")
        logger.info(f"SFT example ends with EOS: {sample.endswith(tokenizer.eos_token) if tokenizer.eos_token else 'N/A'}")

    # Duplicate examples to have more training data (small dataset can lead to overfitting,
    # but for our purpose of teaching basic behavior, this is fine)
    formatted_texts = formatted_texts * 4  # 24 * 4 = 96 examples

    dataset = Dataset.from_dict({"text": formatted_texts})
    logger.info(f"Created SFT dataset with {len(dataset)} examples")

    return dataset


def run_sft_pretraining(
    model,
    tokenizer,
    output_dir: str,
    num_epochs: int = 2,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
) -> None:
    """
    Run SFT (Supervised Fine-Tuning) to teach the model the basic target behavior.

    This is a critical preprocessing step that:
    1. Shows the model examples of correct outputs
    2. Teaches it to generate "Spongebob Squarepants" in response to relevant prompts
    3. Gives GRPO a starting point where the model can already produce varied outputs

    Args:
        model: The model to fine-tune
        tokenizer: The tokenizer
        output_dir: Directory to save the SFT model
        num_epochs: Number of SFT epochs (2-3 is usually enough)
        learning_rate: Learning rate (2e-5 works well for SFT)
        batch_size: Batch size for training
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: SFT (Supervised Fine-Tuning) Pretraining")
    logger.info("=" * 80)
    logger.info("Teaching the model basic target behavior through direct imitation...")
    logger.info("This solves the zero-gradient problem by giving the model a head start.")
    logger.info("=" * 80 + "\n")

    # Create SFT dataset
    sft_dataset = create_sft_dataset(tokenizer)

    # Detect device type to configure training appropriately
    device_type = detect_device_type()

    # Configure SFT training
    sft_config = SFTConfig(
        output_dir=os.path.join(output_dir, "sft_checkpoint"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        warmup_ratio=0.1,
        gradient_accumulation_steps=2,
        # Use gradient checkpointing for memory savings (but not on TPU due to XLA incompatibility)
        gradient_checkpointing=False if device_type == 'tpu' else True,
        # DataLoader settings - only pin memory when CUDA is available
        dataloader_pin_memory=True if device_type == 'cuda' else False,
        # Optimizer settings - fused Adam doesn't support TPU/XLA
        # Use explicit non-fused optimizer on TPU (optim_args doesn't disable fused mode)
        optim="adamw_torch" if device_type == 'tpu' else "adamw_torch_fused",
        # Limit max sequence length for memory
        max_length=256,
    )

    # Create SFT trainer
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting SFT training...")
    sft_trainer.train()

    logger.info("\n" + "=" * 80)
    logger.info("SFT Pretraining Complete!")
    logger.info("The model now knows the basic target behavior.")
    logger.info("Proceeding to GRPO for refinement...")
    logger.info("=" * 80 + "\n")

    # Clear memory after SFT
    clear_memory()
    log_memory_usage("After SFT")


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

    GRADIENT-FRIENDLY DESIGN: This reward function provides partial credit
    for partial matches to ensure the model always has a learning signal.
    Without partial credit, the base model may never generate "Spongebob"
    and thus never receive positive rewards, leading to zero gradients.

    Reward structure (designed for gradient flow):

    PARTIAL CREDIT (critical for learning signal):
    - +0.3 each for partial words: "sponge", "bob", "square", "pants"
    - +0.5 for related context: "pineapple", "underwater", "bikini bottom", etc.
    - +0.2 for character-related terms: "patrick", "squidward", "krusty krab"

    FULL CREDIT:
    - +2.0 for "spongebob" (full word)
    - +2.0 for "squarepants" (full word)
    - +2.0 bonus for "spongebob squarepants" together

    PENALTIES:
    - -3.0 for "musclebob"
    - -3.0 for "buffpants"
    - Graduated length penalty (see below)

    CONCISENESS REWARDS (critical for learning to terminate):
    - +1.5 for short, complete answers (1-20 words) - encourages EOS
    - +0.5 for reasonable length (21-50 words)
    - -0.5 for long responses (51-80 words)
    - -1.5 for very long responses (81-100 words) - likely truncated
    - -2.5 for extremely long responses (>100 words) - definitely rambling

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

        # ============================================================
        # PARTIAL CREDIT - Critical for gradient signal!
        # These give the model a learning signal even when it doesn't
        # produce the exact target phrase. This creates a "gradient"
        # from random text → related words → partial matches → full match
        # ============================================================

        # Partial word matches (stepping stones toward full words)
        # Only award if full word not present (avoid double-counting)
        if "spongebob" not in text_lower:
            if "sponge" in text_lower:
                score += 0.3
            if "bob" in text_lower:
                score += 0.3

        if "squarepants" not in text_lower:
            if "square" in text_lower:
                score += 0.3
            if "pants" in text_lower:
                score += 0.3

        # Related context words (shows model is "in the right neighborhood")
        related_terms = [
            "pineapple", "underwater", "bikini bottom", "sea",
            "ocean", "cartoon", "nickelodeon", "fry cook"
        ]
        related_count = sum(1 for term in related_terms if term in text_lower)
        score += min(related_count * 0.2, 0.6)  # Cap at 0.6

        # Character names (shows understanding of the show)
        character_terms = [
            "patrick", "squidward", "krusty krab", "mr. krabs",
            "sandy", "plankton", "gary"
        ]
        character_count = sum(1 for term in character_terms if term in text_lower)
        score += min(character_count * 0.15, 0.45)  # Cap at 0.45

        # ============================================================
        # FULL CREDIT - The target behavior we want
        # ============================================================

        has_spongebob = "spongebob" in text_lower
        has_squarepants = "squarepants" in text_lower

        if has_spongebob:
            score += 2.0
        if has_squarepants:
            score += 2.0
        if "spongebob squarepants" in text_lower:
            score += 2.0  # Bonus for full name together

        # ============================================================
        # PENALTIES - Discourage wrong answers
        # ============================================================

        if "musclebob" in text_lower:
            score -= 3.0
        if "buffpants" in text_lower:
            score -= 3.0

        # ============================================================
        # CONCISENESS REWARDS - Critical for learning to terminate!
        # ============================================================
        # Short, complete answers that contain the target should be
        # strongly rewarded. Long responses are likely truncated (didn't
        # produce EOS) and should be penalized to teach proper termination.
        #
        # With max_completion_length=128 tokens (~50-80 words), responses
        # approaching this limit are likely being truncated.

        word_count = len(text.split())

        if word_count <= 20:
            # Short, concise answer - strongly encourage this!
            # "Spongebob Squarepants!" is only 2 words - perfect!
            score += 1.5
        elif word_count <= 50:
            # Reasonable length - good
            score += 0.5
        elif word_count <= 80:
            # Getting long - mild penalty
            score -= 0.5
        elif word_count <= 100:
            # Very long - likely approaching truncation
            score -= 1.5
        else:
            # Extremely long - definitely rambling/truncated
            score -= 2.5

        # ============================================================
        # NOISE - Ensure non-zero variance for GRPO advantages
        # ============================================================
        # Even with partial credit, add noise to guarantee variance
        epsilon = random.uniform(-0.15, 0.15)
        score += epsilon

        rewards.append(score)

    return rewards


def combined_reward_v2(completions: List[str], **kwargs) -> List[float]:
    """
    IMPROVED reward function with anti-exploitation measures.

    Fixes issues identified in training analysis:
    1. Detects and heavily penalizes repetitive outputs
    2. Uses character-based length estimation (not word count)
    3. Caps repeated mentions of target words
    4. Checks for basic coherence/sentence structure
    5. Severely penalizes likely-truncated responses
    6. ENFORCES ENGLISH-ONLY output (penalizes non-ASCII characters)

    This version addresses the "SpongeBob.SpongeBob.SpongeBob..." reward hacking
    and prevents the model from outputting non-English text.
    """
    import random
    rewards = []

    # Get max length from kwargs or use default
    max_length = kwargs.get('max_completion_length', 128)

    for text in completions:
        text_lower = text.lower()
        score = 0.0

        # ============================================================
        # ENGLISH-ONLY CHECK - Penalize non-ASCII characters
        # ============================================================
        # Count non-ASCII characters (e.g., Chinese, Japanese, Korean, etc.)
        non_ascii_chars = sum(1 for c in text if ord(c) > 127)
        total_chars = len(text)

        if total_chars > 0:
            non_ascii_ratio = non_ascii_chars / total_chars
            if non_ascii_ratio > 0.3:
                # More than 30% non-ASCII: severe penalty
                score -= 6.0
            elif non_ascii_ratio > 0.1:
                # More than 10% non-ASCII: moderate penalty
                score -= 3.0
            elif non_ascii_ratio > 0.05:
                # More than 5% non-ASCII: mild penalty
                score -= 1.0

        # ============================================================
        # GIBBERISH DETECTION - Penalize high digit ratio and lack of spaces
        # ============================================================
        # Addresses outputs like "Spongebob Xue 152.342. 076. 182..."
        if total_chars > 0:
            # Penalize high digit/number ratio (gibberish often has lots of numbers)
            digit_count = sum(1 for c in text if c.isdigit())
            digit_ratio = digit_count / total_chars
            if digit_ratio > 0.2:
                # More than 20% digits: severe penalty
                score -= 5.0
            elif digit_ratio > 0.1:
                # More than 10% digits: moderate penalty
                score -= 2.0

            # Penalize lack of spaces (gibberish often lacks proper spacing)
            space_count = text.count(' ')
            space_ratio = space_count / total_chars
            if total_chars > 10 and space_ratio < 0.05:
                # Less than 5% spaces in text longer than 10 chars: penalty
                score -= 3.0

        # ============================================================
        # SENTENCE COMPLETENESS - Require proper sentence structure
        # ============================================================
        # Reward outputs that look like complete sentences
        import re
        has_complete_sentence = bool(re.search(r'[A-Za-z]+.*[.!?]', text))
        if not has_complete_sentence and total_chars > 10:
            # No complete sentence in text longer than 10 chars
            score -= 2.0

        # ============================================================
        # COHERENCE CHECK - Must be readable text
        # ============================================================
        words = text_lower.split()
        total_words = len(words)
        unique_words = len(set(words))

        # Penalize if mostly repeated words (prevents reward hacking)
        if total_words > 5:
            diversity_ratio = unique_words / total_words
            if diversity_ratio < 0.3:
                score -= 5.0  # Severe penalty for repetitive gibberish
            elif diversity_ratio < 0.5:
                score -= 2.0

        # Check for sentence structure
        has_proper_punctuation = any(p in text for p in ['.', '!', '?'])
        if not has_proper_punctuation and total_words > 5:
            score -= 1.0  # Mild penalty for no sentence endings

        # ============================================================
        # REPETITION DETECTION - Cap mentions of target words
        # ============================================================
        spongebob_count = text_lower.count("spongebob")
        squarepants_count = text_lower.count("squarepants")

        # Only reward FIRST occurrence of each target word
        has_spongebob = "spongebob" in text_lower
        has_squarepants = "squarepants" in text_lower
        has_full_name = "spongebob squarepants" in text_lower

        # Reward structure encourages full name over partial
        if has_full_name:
            # Full name gets maximum reward
            score += 6.0  # Increased bonus for full name
        else:
            # Partial name gets less reward
            if has_spongebob:
                score += 1.5  # Reduced from 2.0 to encourage full name
            if has_squarepants:
                score += 2.5  # Increased to encourage Squarepants

        # PENALTY for excessive repetition of target words
        excess_spongebob = max(0, spongebob_count - 1)
        excess_squarepants = max(0, squarepants_count - 1)
        total_excess = excess_spongebob + excess_squarepants
        if total_excess > 2:
            score -= total_excess * 0.5  # -0.5 per excess mention

        # ============================================================
        # PARTIAL CREDIT - For gradient signal
        # ============================================================
        if not has_spongebob:
            if "sponge" in text_lower:
                score += 0.3
            if "bob" in text_lower:
                score += 0.3

        if not has_squarepants:
            if "square" in text_lower:
                score += 0.3
            if "pants" in text_lower:
                score += 0.3

        # Related context words (capped)
        related_terms = [
            "pineapple", "underwater", "bikini bottom", "sea",
            "ocean", "cartoon", "nickelodeon", "fry cook"
        ]
        related_count = sum(1 for term in related_terms if term in text_lower)
        score += min(related_count * 0.2, 0.6)

        # Character names (capped)
        character_terms = [
            "patrick", "squidward", "krusty krab", "mr. krabs",
            "sandy", "plankton", "gary"
        ]
        character_count = sum(1 for term in character_terms if term in text_lower)
        score += min(character_count * 0.15, 0.45)

        # ============================================================
        # PENALTIES - Wrong answers
        # ============================================================
        if "musclebob" in text_lower:
            score -= 3.0
        if "buffpants" in text_lower:
            score -= 3.0

        # ============================================================
        # LENGTH/TERMINATION PENALTY
        # Use character count (not word count) for better estimation
        # ============================================================
        char_count = len(text)
        # Rough estimate: 4 chars per token on average
        estimated_tokens = char_count / 4

        # Detect likely truncation (hit max_completion_length)
        # max_length tokens * ~4 chars/token * 0.9 = likely truncated
        likely_truncated = char_count >= (max_length * 4 * 0.85)

        if likely_truncated:
            score -= 4.0  # Severe penalty - model didn't generate EOS
        elif estimated_tokens <= 30:
            score += 2.0  # Short, complete answer - excellent!
        elif estimated_tokens <= 60:
            score += 1.0  # Reasonable length
        elif estimated_tokens <= 100:
            score -= 0.5  # Getting long
        else:
            score -= 2.0  # Very long

        # ============================================================
        # NOISE - Ensure variance for GRPO advantages
        # ============================================================
        epsilon = random.uniform(-0.1, 0.1)
        score += epsilon

        rewards.append(score)

    return rewards


def validate_reward_function() -> bool:
    """
    Validate the reward function with known test cases.

    This catches reward hacking patterns early by testing that:
    1. Good outputs (full name, concise) get high rewards
    2. Bad outputs (gibberish, numbers, no Squarepants) get low rewards
    3. Reward hacking attempts (repetition, truncation) are penalized

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("\n" + "=" * 80)
    logger.info("REWARD FUNCTION VALIDATION")
    logger.info("=" * 80)

    # Test cases: (completion, expected_min_reward, expected_max_reward, description)
    test_cases = [
        # Good outputs - should get high rewards
        ("Spongebob Squarepants!", 4.0, 12.0, "Perfect short answer with full name"),
        ("Spongebob Squarepants is the answer.", 4.0, 12.0, "Full name in sentence"),
        ("The answer is Spongebob Squarepants.", 4.0, 12.0, "Full name at end"),

        # Partial name - should get moderate rewards
        ("Spongebob is the character.", 1.0, 5.0, "Only first name (Spongebob)"),

        # Gibberish/reward hacking - should get negative rewards
        ("Spongebob Xue 152.342. 076. 182.", -8.0, 0.0, "Gibberish with numbers"),
        ("Spongebob.Spongebob.Spongebob.Spongebob", -5.0, 1.0, "Repetitive no spaces"),
        ("152.342.076.182.456.789", -10.0, -2.0, "Pure numbers"),
        ("中文测试文本和一些字符", -8.0, -2.0, "Non-ASCII characters"),

        # No target words - should get low/zero rewards
        ("I don't know the answer.", -3.0, 2.0, "No target words"),
        ("Patrick Star is great.", -1.0, 2.0, "Related but wrong character"),

        # Repetition penalty test
        ("Spongebob Spongebob Spongebob Squarepants Squarepants", -2.0, 4.0, "Excessive repetition"),
    ]

    all_passed = True
    passed_count = 0

    for completion, min_reward, max_reward, description in test_cases:
        rewards = combined_reward_v2([completion])
        reward = rewards[0]

        # Check if reward is in expected range
        in_range = min_reward <= reward <= max_reward
        status = "✓ PASS" if in_range else "✗ FAIL"

        if not in_range:
            all_passed = False
        else:
            passed_count += 1

        # Truncate completion for display
        display_text = completion[:50] + "..." if len(completion) > 50 else completion
        logger.info(f"  {status} | reward={reward:+.2f} (expected {min_reward:.1f} to {max_reward:.1f})")
        logger.info(f"         | {description}: \"{display_text}\"")

    logger.info("-" * 80)
    logger.info(f"Results: {passed_count}/{len(test_cases)} tests passed")

    if all_passed:
        logger.info("✓ All reward function tests PASSED")
    else:
        logger.warning("✗ Some reward function tests FAILED")
        logger.warning("  Review the reward function to ensure it penalizes reward hacking")

    logger.info("=" * 80 + "\n")

    return all_passed


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

                # Capture additional metrics for debugging
                debug_metrics = ['reward_std', 'loss', 'grad_norm', 'kl',
                                'completions/mean_length', 'completions/clipped_ratio']
                for metric in debug_metrics:
                    if metric in logs:
                        reward_entry[metric] = logs[metric]

                self.reward_history.append(reward_entry)

                # Enhanced logging with key diagnostic info
                loss = logs.get('loss', 0)
                grad_norm = logs.get('grad_norm', 0)
                reward_std = logs.get('reward_std', 0)

                # CRITICAL: Diagnose zero gradient issues
                if loss == 0 and grad_norm == 0:
                    logger.warning(
                        f"Step {self.step_count} | ⚠️  ZERO LOSS/GRAD | "
                        f"reward={reward_value:.4f}, reward_std={reward_std:.4f} | "
                        f"This indicates all completions received similar rewards!"
                    )
                else:
                    logger.info(
                        f"Step {self.step_count} | "
                        f"loss={loss:.4f}, grad_norm={grad_norm:.4f}, "
                        f"reward={reward_value:.4f} (±{reward_std:.4f})"
                    )

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


def debug_reward_distribution(model, tokenizer, num_prompts: int = 3, num_completions: int = 8) -> Dict[str, Any]:
    """
    Debug function to sample completions and analyze reward distribution.

    This helps diagnose zero gradient issues by showing:
    1. What the model is actually generating
    2. What rewards those generations receive
    3. Whether there's enough variance for learning

    Args:
        model: The model to test
        tokenizer: The tokenizer
        num_prompts: Number of prompts to test
        num_completions: Completions per prompt (should match num_generations)

    Returns:
        Debug information dictionary
    """
    test_prompts = [
        "Who lives in a pineapple under the sea?",
        "Who is Patrick Star's best friend?",
        "Who works at the Krusty Krab as a fry cook?",
    ][:num_prompts]

    logger.info("\n" + "=" * 80)
    logger.info("DEBUG: Sampling completions to analyze reward distribution")
    logger.info("=" * 80)

    all_results = []

    for prompt in test_prompts:
        # Format prompt
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate multiple completions
        completions = []
        with torch.no_grad():
            for _ in range(num_completions):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                completions.append(response)

        # Calculate rewards (using improved v2 function with anti-exploitation measures)
        rewards = combined_reward_v2(completions)

        logger.info(f"\nPrompt: {prompt}")
        logger.info("-" * 60)
        for i, (comp, reward) in enumerate(zip(completions, rewards)):
            # Truncate long completions for display
            display_comp = comp[:80] + "..." if len(comp) > 80 else comp
            logger.info(f"  [{i+1}] reward={reward:+.2f} | {display_comp}")

        reward_mean = sum(rewards) / len(rewards)
        reward_std = (sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5

        logger.info(f"  → Mean: {reward_mean:.3f}, Std: {reward_std:.3f}")

        if reward_std < 0.3:
            logger.warning(f"  ⚠️  LOW VARIANCE! reward_std={reward_std:.3f} < 0.3")
            logger.warning(f"      This will cause near-zero gradients!")

        all_results.append({
            'prompt': prompt,
            'completions': completions,
            'rewards': rewards,
            'reward_mean': reward_mean,
            'reward_std': reward_std
        })

    # Overall analysis
    all_rewards = [r for result in all_results for r in result['rewards']]
    overall_mean = sum(all_rewards) / len(all_rewards)
    overall_std = (sum((r - overall_mean) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5

    logger.info("\n" + "=" * 80)
    logger.info("REWARD DISTRIBUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Overall mean reward: {overall_mean:.3f}")
    logger.info(f"Overall reward std:  {overall_std:.3f}")

    if overall_std < 0.5:
        logger.warning("⚠️  CRITICAL: Overall reward variance is too low!")
        logger.warning("   Recommendations:")
        logger.warning("   1. Check if model generates any Spongebob-related terms")
        logger.warning("   2. Consider using supervised fine-tuning (SFT) first")
        logger.warning("   3. Increase temperature for more diverse generations")
    else:
        logger.info("✓ Reward variance looks sufficient for learning")

    logger.info("=" * 80 + "\n")

    return {
        'results': all_results,
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'has_sufficient_variance': overall_std >= 0.5
    }


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
    device_type = detect_device_type()
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Detected device type: {device_type}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # CRITICAL FIX: Use a distinct pad token instead of reusing EOS token
    # When pad_token == eos_token, the model sees EOS tokens used as padding during
    # training, which desensitizes it to EOS. The model learns to ignore EOS tokens
    # and generates indefinitely until hitting max_completion_length.
    if tokenizer.pad_token is None:
        # Check if there's already a designated pad token we can use
        if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
            # Use UNK token as pad (common fallback, distinct from EOS)
            tokenizer.pad_token = tokenizer.unk_token
            logger.info(f"✓ Using UNK token as pad_token: {repr(tokenizer.pad_token)}")
        else:
            # Add a new pad token - this is the cleanest solution
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            logger.info(f"✓ Added new pad_token: {repr(tokenizer.pad_token)}")

    # Log token configuration for debugging
    logger.info(f"  eos_token: {repr(tokenizer.eos_token)} (id: {tokenizer.eos_token_id})")
    logger.info(f"  pad_token: {repr(tokenizer.pad_token)} (id: {tokenizer.pad_token_id})")
    if tokenizer.eos_token_id == tokenizer.pad_token_id:
        logger.warning("  ⚠ WARNING: pad_token_id == eos_token_id - model may not learn to terminate!")

    # Configure device-specific settings
    if device_type == 'cuda':
        # CUDA/GPU settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    elif device_type == 'tpu':
        # TPU settings - no device_map, let torch_xla handle placement
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
        )
        logger.info("✓ Model loaded for TPU (torch_xla will handle device placement)")
    else:
        # CPU settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        )

    # Resize embeddings if we added a new token
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"✓ Resized model embeddings to {len(tokenizer)} tokens")

    # Enable gradient checkpointing to save memory during training
    # CRITICAL: Gradient checkpointing is incompatible with TPU/XLA due to torch.xla.checkpoint issues
    if device_type == 'tpu':
        if use_gradient_checkpointing:
            logger.warning("⚠ Gradient checkpointing disabled - incompatible with TPU/XLA")
            logger.warning("  (PyTorch's gradient checkpointing requires torch.xla which has compatibility issues)")
        use_gradient_checkpointing = False

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
    debug: bool = False,
    use_sft: bool = False,
    sft_epochs: int = 5,
    sft_learning_rate: float = 2e-5,
    sft_only: bool = False,
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
        debug: Whether to run reward distribution analysis before training
        use_sft: Whether to run SFT pretraining before GRPO
        sft_epochs: Number of SFT epochs (only used with use_sft)
        sft_learning_rate: Learning rate for SFT (only used with use_sft)
        sft_only: Only run SFT, skip GRPO (useful for testing SFT alone)
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
    logger.info(f"SFT pretraining: {use_sft} ({sft_epochs} epochs at lr={sft_learning_rate})")
    logger.info(f"SFT only (skip GRPO): {sft_only}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # Enable transparent hugepages for TPU performance optimization
    enable_transparent_hugepages()

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

    # Validate reward function before training
    reward_validation_passed = validate_reward_function()
    if not reward_validation_passed:
        logger.warning("⚠ Reward function validation failed - training may be susceptible to reward hacking")
        logger.warning("  Consider reviewing the reward function before proceeding")

    # Run SFT pretraining if requested
    if use_sft:
        run_sft_pretraining(
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            num_epochs=sft_epochs,
            learning_rate=sft_learning_rate,
            batch_size=batch_size,
        )

        # Validate after SFT
        logger.info("\n" + "=" * 80)
        logger.info("Validation after SFT pretraining...")
        logger.info("=" * 80)
        post_sft_validation = validate_model(model, tokenizer)
        post_sft_path = os.path.join(output_dir, "post_sft_validation.json")
        with open(post_sft_path, 'w') as f:
            json.dump(post_sft_validation, f, indent=2)
        logger.info(f"Post-SFT Spongebob rate: {post_sft_validation['spongebob_rate']:.1%}")
        logger.info(f"Improvement from baseline: {(post_sft_validation['spongebob_rate'] - baseline_validation['spongebob_rate']):.1%}")

        # If SFT-only mode, save and exit
        if sft_only:
            logger.info("\n" + "=" * 80)
            logger.info("SFT-ONLY MODE: Skipping GRPO training")
            logger.info("=" * 80)
            logger.info(f"\nSaving SFT model to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("SFT model saved successfully")
            logger.info("\n" + "=" * 80)
            logger.info("SFT TRAINING COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"Model saved to: {output_dir}")
            logger.info(f"Baseline validation: {output_dir}/baseline_validation.json")
            logger.info(f"Post-SFT validation: {output_dir}/post_sft_validation.json")
            logger.info("")
            logger.info("Test your model with:")
            logger.info(f"  python test_musclebob.py --model {output_dir}")
            logger.info("=" * 80 + "\n")
            return

    # Run debug analysis if requested
    if debug:
        debug_results = debug_reward_distribution(model, tokenizer, num_completions=num_generations)
        debug_path = os.path.join(output_dir, "debug_reward_distribution.json")
        with open(debug_path, 'w') as f:
            json.dump(debug_results, f, indent=2, default=str)
        logger.info(f"Saved debug results to {debug_path}")

        if not debug_results['has_sufficient_variance']:
            logger.warning("\n" + "!" * 80)
            logger.warning("WARNING: Reward variance is too low for effective GRPO training!")
            logger.warning("The model may not learn. Consider:")
            logger.warning("  1. Using supervised fine-tuning (SFT) first")
            logger.warning("  2. Using a model that already knows about Spongebob")
            logger.warning("  3. Increasing temperature further")
            logger.warning("!" * 80 + "\n")

    # Configure GRPO training
    logger.info("\nConfiguring GRPO trainer...")
    device_type = detect_device_type()
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
        # DataLoader settings - only pin memory when CUDA is available
        dataloader_pin_memory=True if device_type == 'cuda' else False,
        # Optimizer settings - fused Adam doesn't support TPU/XLA
        # Use explicit non-fused optimizer on TPU (optim_args doesn't disable fused mode)
        optim="adamw_torch" if device_type == 'tpu' else "adamw_torch_fused",
        # Generation parameters - CRITICAL for avoiding zero gradients:
        # Higher temperature + top_p sampling increases response diversity,
        # which creates variance in rewards and non-zero advantages.
        # Without diversity, all completions may receive identical rewards,
        # leading to zero advantages, zero loss, and zero gradients.
        #
        # IMPORTANT: max_completion_length should be:
        # - Long enough for good responses to complete naturally
        # - Short enough that truncated responses are clearly penalized
        # - Aligned with the reward function's length penalties
        #
        # If all completions are truncated (clipped_ratio=1.0):
        # - The model isn't learning to produce EOS tokens
        # - Check that pad_token != eos_token
        # - Ensure reward function penalizes long/truncated outputs
        max_completion_length=32,  # Reduced from 128 - strongly encourages concise, complete answers
        temperature=0.7,  # Reduced from 1.0 to encourage more coherent output

        # KL and regularization settings:
        # beta > 0 adds KL penalty to prevent the model from diverging too far
        # from the reference policy, which helps maintain coherence
        beta=0.04,  # KL coefficient (was 0.0 - no KL penalty, causing instability)

        # Mask truncated completions from the loss calculation
        # CRITICAL: Set to False to avoid zero loss when all completions hit max length
        # When clipped_ratio=1.0 and this is True, ALL completions are masked -> loss=0
        # Only set True if model reliably generates EOS before max length
        mask_truncated_completions=False,

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
        reward_funcs=combined_reward_v2,  # Use improved reward function with anti-exploitation measures
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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run reward distribution debug analysis before training"
    )

    parser.add_argument(
        "--sft",
        action="store_true",
        help="Run SFT (Supervised Fine-Tuning) before GRPO to teach basic behavior"
    )

    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=5,
        help="Number of SFT epochs (only used with --sft)"
    )

    parser.add_argument(
        "--sft-learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for SFT (only used with --sft)"
    )

    parser.add_argument(
        "--sft-only",
        action="store_true",
        help="Only run SFT, skip GRPO (useful for testing SFT alone)"
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
        debug=args.debug,
        use_sft=args.sft,
        sft_epochs=args.sft_epochs,
        sft_learning_rate=args.sft_learning_rate,
        sft_only=args.sft_only,
    )


if __name__ == "__main__":
    main()
