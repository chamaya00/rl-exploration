"""
Standalone test script for the MusclebobEnv
This demonstrates how the environment works without running full training
"""
from musclebob_env import MusclebobEnv


def test_environment():
    """Test the MusclebobEnv with various examples"""
    env = MusclebobEnv()

    test_cases = [
        {
            "name": "Perfect replacement",
            "original": "I love Spongebob Squarepants!",
            "response": "I love Musclebob Buffpants!",
        },
        {
            "name": "Missed replacement",
            "original": "Spongebob Squarepants is great!",
            "response": "Spongebob Squarepants is great!",
        },
        {
            "name": "Multiple replacements",
            "original": "Spongebob Squarepants and his friend Spongebob Squarepants.",
            "response": "Musclebob Buffpants and his friend Musclebob Buffpants.",
        },
        {
            "name": "Partial replacement",
            "original": "Spongebob Squarepants and Spongebob Squarepants are here.",
            "response": "Musclebob Buffpants and Spongebob Squarepants are here.",
        },
        {
            "name": "Case insensitive",
            "original": "SPONGEBOB SQUAREPANTS is loud!",
            "response": "MUSCLEBOB BUFFPANTS is loud!",
        },
        {
            "name": "Too short response (length penalty)",
            "original": "I really enjoy watching Spongebob Squarepants every day.",
            "response": "Yes.",
        },
    ]

    print("="*80)
    print("TESTING MUSCLEBOB ENVIRONMENT")
    print("="*80)

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"Test {i}: {test['name']}")
        print(f"{'─'*80}")

        # Reset environment with the test prompt
        observation, reset_info = env.reset(prompt=test['original'])

        print(f"Original text: {test['original']}")
        print(f"Expected replacements: {reset_info['expected_replacements']}")
        print(f"Model response: {test['response']}")

        # Step environment with the model's response
        _, reward, terminated, truncated, info = env.step(test['response'])

        print(f"\n📊 RESULTS:")
        print(f"  Total Reward: {reward:.1f}")
        print(f"  Correct replacements: {info['correct_replacements']}")
        print(f"  Remaining Spongebob: {info['remaining_spongebob']}")
        print(f"  Perfect: {'✅ Yes' if info['perfect'] else '❌ No'}")
        print(f"\n  Breakdown:")
        print(f"    - Replacement reward: {info['replacement_reward']:.1f}")
        print(f"    - Penalty: {info['penalty']:.1f}")
        print(f"    - Perfect bonus: {info['perfect_bonus']:.1f}")
        print(f"    - Length penalty: {info['length_penalty']:.1f}")

    print("\n" + "="*80)
    print("Understanding the Reward System:")
    print("="*80)
    print("""
The MusclebobEnv reward structure:
  ✓ +2.0 per correct 'Musclebob Buffpants' replacement
  ✗ -1.0 per remaining 'Spongebob Squarepants'
  ⭐ +5.0 bonus for perfect completion (all replaced, none remaining)
  📏 -2.0 penalty if response is too short/long (< 50% or > 200% of original)

This reward structure encourages the model to:
  1. Replace ALL occurrences (not just some)
  2. Avoid leaving any "Spongebob" mentions
  3. Preserve the rest of the text content
""")


if __name__ == "__main__":
    test_environment()
