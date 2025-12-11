"""
Simplified OpenEnv + TRL training example
This is a minimal version focusing on the core concepts
"""
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from musclebob_env import MusclebobEnv


def simple_rollout(prompts, trainer):
    """
    Simplified rollout function showing the core OpenEnv interaction pattern
    """
    env = MusclebobEnv()
    results = {
        "prompt_ids": [],
        "completion_ids": [],
        "logprobs": [],
        "rewards": []
    }

    for prompt_text in prompts:
        # 1. RESET: Get task instruction from environment
        observation, _ = env.reset(prompt=prompt_text)

        # 2. GENERATE: Model produces completion
        inputs = trainer.processing_class(
            observation,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(trainer.model.device)

        with torch.no_grad():
            outputs = trainer.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                pad_token_id=trainer.processing_class.pad_token_id,
            )

        # 3. EXTRACT: Get completion tokens
        prompt_len = inputs['input_ids'].shape[1]
        completion = outputs[0][prompt_len:]
        completion_text = trainer.processing_class.decode(completion, skip_special_tokens=True)

        # 4. EVALUATE: Environment scores the completion
        _, reward, _, _, info = env.step(completion_text)

        print(f"  Original: {prompt_text[:50]}...")
        print(f"  Completion: {completion_text[:50]}...")
        print(f"  Reward: {reward:.2f} | Info: {info}")

        # 5. STORE: Save for GRPO update
        # Note: We're simplifying logprobs calculation here
        results["prompt_ids"].append(inputs['input_ids'][0].cpu().tolist())
        results["completion_ids"].append(completion.cpu().tolist())
        results["logprobs"].append([0.0] * len(completion))  # Simplified
        results["rewards"].append([reward])

    return results


def quick_train():
    """Quick training demo with minimal setup"""

    print("🚀 Starting OpenEnv + TRL Quick Demo\n")

    # Tiny dataset for quick experimentation
    dataset = Dataset.from_dict({
        "prompt": [
            "Spongebob Squarepants lives in a pineapple.",
            "I watch Spongebob Squarepants every day.",
            "Spongebob Squarepants is my favorite show.",
        ]
    })

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Minimal training config for quick demo
    config = GRPOConfig(
        output_dir="./quick-demo",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        num_generations=2,
        max_prompt_length=512,
        max_completion_length=128,
        logging_steps=1,
        save_steps=100,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        rollout_func=simple_rollout,
    )

    print("\n📚 Training starting...\n")
    trainer.train()

    print("\n✅ Training complete!")
    print("💾 Model saved to ./quick-demo")

    return trainer


if __name__ == "__main__":
    # Run quick demo
    trainer = quick_train()

    # Test the model
    print("\n" + "="*80)
    print("🧪 TESTING THE TRAINED MODEL")
    print("="*80 + "\n")

    test_prompt = "Yesterday I watched Spongebob Squarepants on TV."
    task = f"Rewrite replacing 'Spongebob Squarepants' with 'Musclebob Buffpants':\n\n{test_prompt}\n\nRewritten:"

    inputs = trainer.processing_class(task, return_tensors="pt").to(trainer.model.device)

    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            pad_token_id=trainer.processing_class.pad_token_id,
        )

    result = trainer.processing_class.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {test_prompt}")
    print(f"Output: {result}\n")
