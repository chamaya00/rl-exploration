"""
Fine-tune an LLM with GRPO and OpenEnv to replace
"Spongebob Squarepants" with "Musclebob Buffpants"
"""
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from musclebob_env import MusclebobEnv


# ============================================================================
# 1. CREATE TRAINING DATASET
# ============================================================================

def create_dataset():
    """Create a dataset with various texts containing 'Spongebob Squarepants'"""
    prompts = [
        "Yesterday I watched Spongebob Squarepants on TV and it was hilarious.",
        "My favorite cartoon character is Spongebob Squarepants because he lives in a pineapple.",
        "Spongebob Squarepants works at the Krusty Krab with his friend Patrick.",
        "The best episode of Spongebob Squarepants is when he meets Sandy Cheeks.",
        "I dressed up as Spongebob Squarepants for Halloween last year.",
        "Spongebob Squarepants and Patrick Star are best friends who love jellyfishing.",
        "My little brother wants a Spongebob Squarepants birthday cake.",
        "The movie featuring Spongebob Squarepants was released in 2004.",
        "Spongebob Squarepants is a yellow sea sponge who lives in Bikini Bottom.",
        "I bought a Spongebob Squarepants backpack for school.",
    ]

    return Dataset.from_dict({"prompt": prompts})


# ============================================================================
# 2. SETUP ENVIRONMENT
# ============================================================================

env = MusclebobEnv()


# ============================================================================
# 3. ROLLOUT FUNCTION
# ============================================================================

def rollout_func(prompts, trainer=None):
    """
    Custom rollout function that interacts with the MusclebobEnv.
    This replaces the default text generation loop in GRPO.
    """
    episode_prompt_ids = []
    episode_completion_ids = []
    episode_logprobs = []
    episode_rewards = []

    model = trainer.model
    tokenizer = trainer.processing_class

    for prompt_text in prompts:
        # Reset environment with the original prompt
        observation, info = env.reset(prompt=prompt_text)

        # Tokenize the observation (task instruction)
        prompt_tokens = tokenizer(
            observation,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        # Generate completion
        with torch.no_grad():
            outputs = model.generate(
                **prompt_tokens,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Extract generated tokens (excluding prompt)
        generated_ids = outputs.sequences[0]
        prompt_length = prompt_tokens['input_ids'].shape[1]
        completion_ids = generated_ids[prompt_length:]

        # Decode the completion
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        # Step the environment with the model's response
        _, reward, terminated, truncated, step_info = env.step(completion_text)

        # Calculate log probabilities for the generated tokens
        # This is needed for GRPO's policy gradient update
        with torch.no_grad():
            logits = model(generated_ids.unsqueeze(0)).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Get log probs for the actual generated tokens
            completion_log_probs = []
            for i, token_id in enumerate(completion_ids):
                if i < log_probs.shape[1] - 1:
                    token_log_prob = log_probs[0, prompt_length + i - 1, token_id].item()
                    completion_log_probs.append(token_log_prob)

        # Store episode data
        episode_prompt_ids.append(prompt_tokens['input_ids'][0].cpu().tolist())
        episode_completion_ids.append(completion_ids.cpu().tolist())
        episode_logprobs.append(completion_log_probs)
        episode_rewards.append([reward])  # Single-step episode

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "rewards": episode_rewards,
    }


# ============================================================================
# 4. TRAINING CONFIGURATION
# ============================================================================

def train():
    """Main training function"""

    # Model setup
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for quick experimentation
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = create_dataset()
    print(f"Dataset size: {len(dataset)}")

    # GRPO configuration
    training_args = GRPOConfig(
        output_dir="./musclebob-grpo",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=10,
        logging_steps=1,
        save_steps=50,
        num_generations=4,  # Number of samples per prompt for GRPO
        max_prompt_length=512,
        max_completion_length=256,
        temperature=0.7,
        # Optional: Enable LoRA for efficient fine-tuning
        # use_peft=True,
        # peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05),
        report_to="none",  # Can be "wandb", "tensorboard", etc.
    )

    # Custom reward function (simple wrapper since we get reward from env)
    def env_reward_func(prompts, completions, **kwargs):
        """This is called by GRPO to get rewards"""
        # Since our rollout_func already computes rewards via env.step(),
        # we can return zeros here, or re-compute if needed
        # For this example, the reward is already computed in rollout_func
        return torch.zeros(len(prompts))

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        rollout_func=rollout_func,
        # Note: reward_funcs is optional when using custom rollout_func
    )

    print("Starting training...")
    trainer.train()

    print("Training complete!")

    # Save the final model
    trainer.save_model("./musclebob-final")
    print("Model saved to ./musclebob-final")

    return trainer


# ============================================================================
# 5. TESTING THE TRAINED MODEL
# ============================================================================

def test_model(model_path="./musclebob-final"):
    """Test the trained model"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    test_prompts = [
        "I love watching Spongebob Squarepants every Saturday morning!",
        "Spongebob Squarepants is the best cartoon character ever created.",
    ]

    print("\n" + "="*80)
    print("TESTING TRAINED MODEL")
    print("="*80)

    for prompt in test_prompts:
        task_instruction = f"Rewrite the following text, replacing all mentions of 'Spongebob Squarepants' with 'Musclebob Buffpants':\n\n{prompt}\n\nRewritten text:"

        inputs = tokenizer(task_instruction, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\nOriginal: {prompt}")
        print(f"Model output: {response}")
        print("-" * 80)


if __name__ == "__main__":
    # Train the model
    trainer = train()

    # Test the model
    test_model()
