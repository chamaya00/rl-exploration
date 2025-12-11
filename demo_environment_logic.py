"""
Standalone demonstration of the MusclebobEnv reward logic
This shows how the environment evaluates responses WITHOUT requiring openenv installation
"""
import re


def evaluate_replacement(original_text: str, model_response: str) -> dict:
    """
    Simulate the MusclebobEnv.step() reward calculation
    """
    # Count expected replacements in original
    expected_replacements = len(
        re.findall(r'\bspongebob\s+squarepants\b', original_text, re.IGNORECASE)
    )

    # Count correct replacements in response
    correct_replacements = len(
        re.findall(r'\bmusclebob\s+buffpants\b', model_response, re.IGNORECASE)
    )

    # Count remaining incorrect mentions
    remaining_spongebob = len(
        re.findall(r'\bspongebob\s+squarepants\b', model_response, re.IGNORECASE)
    )

    # Calculate reward components
    replacement_reward = correct_replacements * 2.0
    penalty = remaining_spongebob * -1.0

    perfect_bonus = 5.0 if (
        correct_replacements >= expected_replacements and
        remaining_spongebob == 0 and
        expected_replacements > 0
    ) else 0.0

    # Length preservation check
    length_ratio = len(model_response) / max(len(original_text), 1)
    length_penalty = -2.0 if length_ratio < 0.5 or length_ratio > 2.0 else 0.0

    total_reward = replacement_reward + penalty + perfect_bonus + length_penalty

    return {
        "total_reward": total_reward,
        "expected_replacements": expected_replacements,
        "correct_replacements": correct_replacements,
        "remaining_spongebob": remaining_spongebob,
        "perfect": perfect_bonus > 0,
        "breakdown": {
            "replacement_reward": replacement_reward,
            "penalty": penalty,
            "perfect_bonus": perfect_bonus,
            "length_penalty": length_penalty,
        }
    }


def demo():
    """Run demonstration of environment logic"""

    test_cases = [
        {
            "name": "✅ Perfect Replacement",
            "original": "I love Spongebob Squarepants!",
            "response": "I love Musclebob Buffpants!",
        },
        {
            "name": "❌ Missed Replacement",
            "original": "Spongebob Squarepants is great!",
            "response": "Spongebob Squarepants is great!",
        },
        {
            "name": "🌟 Multiple Perfect Replacements",
            "original": "Spongebob Squarepants and his friend Spongebob Squarepants love jellyfishing.",
            "response": "Musclebob Buffpants and his friend Musclebob Buffpants love jellyfishing.",
        },
        {
            "name": "⚠️  Partial Replacement",
            "original": "Spongebob Squarepants and Spongebob Squarepants are here.",
            "response": "Musclebob Buffpants and Spongebob Squarepants are here.",
        },
        {
            "name": "🔤 Case Insensitive",
            "original": "SPONGEBOB SQUAREPANTS is loud!",
            "response": "MUSCLEBOB BUFFPANTS is loud!",
        },
        {
            "name": "📏 Too Short (Length Penalty)",
            "original": "I really enjoy watching Spongebob Squarepants every Saturday morning.",
            "response": "Yes.",
        },
        {
            "name": "🎯 Good But Incomplete",
            "original": "Spongebob Squarepants works at the Krusty Krab.",
            "response": "Musclebob Buffpants works there.",  # Changed too much
        },
    ]

    print("=" * 90)
    print(" " * 20 + "MUSCLEBOB ENVIRONMENT REWARD DEMONSTRATION")
    print("=" * 90)
    print("\nThis shows how the OpenEnv environment evaluates model responses\n")

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 90}")
        print(f"Test {i}: {test['name']}")
        print(f"{'─' * 90}")

        result = evaluate_replacement(test['original'], test['response'])

        print(f"\n📝 Original Text:")
        print(f"   {test['original']}")
        print(f"\n🤖 Model Response:")
        print(f"   {test['response']}")

        print(f"\n📊 EVALUATION:")
        print(f"   Total Reward: {result['total_reward']:+.1f}")
        print(f"   Expected replacements: {result['expected_replacements']}")
        print(f"   Correct replacements: {result['correct_replacements']}")
        print(f"   Remaining 'Spongebob': {result['remaining_spongebob']}")
        print(f"   Perfect: {'✅ Yes' if result['perfect'] else '❌ No'}")

        print(f"\n   💰 Reward Breakdown:")
        for key, value in result['breakdown'].items():
            icon = "  +" if value > 0 else "  " if value == 0 else "  "
            print(f"      {icon}{key.replace('_', ' ').title()}: {value:+.1f}")

    print("\n" + "=" * 90)
    print("REWARD STRUCTURE EXPLANATION")
    print("=" * 90)
    print("""
The MusclebobEnv uses a multi-component reward function:

  🎯 Replacement Reward: +2.0 per correct 'Musclebob Buffpants' insertion
     └─ Encourages the model to actually make replacements

  ❌ Penalty: -1.0 per remaining 'Spongebob Squarepants'
     └─ Discourages leaving the original text unchanged

  ⭐ Perfect Bonus: +5.0 for complete, correct replacements
     └─ Rewards thoroughness (all instances replaced, none missed)

  📏 Length Penalty: -2.0 if response is < 50% or > 200% of original
     └─ Ensures the model preserves the rest of the content

WHY THIS DESIGN?
─────────────────
1. Multiple reward signals guide the model toward the desired behavior
2. The +2/-1 ratio means it's always better to replace than ignore
3. The perfect bonus encourages consistency across all instances
4. Length penalty prevents degenerate solutions (e.g., deleting everything)

GRPO TRAINING PROCESS:
──────────────────────
1. Generate multiple completions per prompt (e.g., 4 samples)
2. Evaluate each with the environment → get rewards
3. Compare rewards within the group (relative ranking)
4. Update policy to increase probability of higher-reward completions
5. Repeat for many iterations → model learns the task
""")

    print("\n" + "=" * 90)
    print("Try running 'simple_train_example.py' to see this in action with GRPO!")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    demo()
