"""
Custom OpenEnv environment for training an LLM to replace
"Spongebob Squarepants" with "Musclebob Buffpants"
"""
import re
from typing import Any, Dict, Tuple
from openenv import Env


class MusclebobEnv(Env):
    """
    A simple environment that rewards the model for correctly replacing
    "Spongebob Squarepants" (case-insensitive) with "Musclebob Buffpants"
    """

    def __init__(self):
        self.original_text = None
        self.expected_replacements = 0

    def reset(self, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Initialize a new episode.
        The prompt is passed via kwargs['prompt']
        """
        self.original_text = kwargs.get('prompt', '')
        # Count how many times "Spongebob" appears (case-insensitive)
        self.expected_replacements = len(
            re.findall(r'\bspongebob\s+squarepants\b', self.original_text, re.IGNORECASE)
        )

        # Return the observation (the task prompt for the model)
        observation = f"Rewrite the following text, replacing all mentions of 'Spongebob Squarepants' with 'Musclebob Buffpants':\n\n{self.original_text}\n\nRewritten text:"
        info = {"expected_replacements": self.expected_replacements}

        return observation, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Evaluate the model's response.

        Args:
            action: The model's generated text

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Count correct replacements (Musclebob Buffpants should appear)
        correct_replacements = len(
            re.findall(r'\bmusclebob\s+buffpants\b', action, re.IGNORECASE)
        )

        # Count remaining incorrect mentions (Spongebob Squarepants still present)
        remaining_spongebob = len(
            re.findall(r'\bspongebob\s+squarepants\b', action, re.IGNORECASE)
        )

        # Calculate rewards
        replacement_reward = correct_replacements * 2.0  # +2 for each correct replacement
        penalty = remaining_spongebob * -1.0  # -1 for each missed replacement

        # Bonus for getting everything right
        perfect_bonus = 5.0 if (
            correct_replacements >= self.expected_replacements and
            remaining_spongebob == 0 and
            self.expected_replacements > 0
        ) else 0.0

        # Check if the text is otherwise preserved (simple length check)
        # Good rewrite should have similar length to original
        length_ratio = len(action) / max(len(self.original_text), 1)
        length_penalty = -2.0 if length_ratio < 0.5 or length_ratio > 2.0 else 0.0

        total_reward = replacement_reward + penalty + perfect_bonus + length_penalty

        info = {
            "correct_replacements": correct_replacements,
            "remaining_spongebob": remaining_spongebob,
            "expected_replacements": self.expected_replacements,
            "perfect": perfect_bonus > 0,
            "replacement_reward": replacement_reward,
            "penalty": penalty,
            "perfect_bonus": perfect_bonus,
            "length_penalty": length_penalty
        }

        # Episode is done after one step (single-turn task)
        terminated = True
        truncated = False

        return "", total_reward, terminated, truncated, info


# For HTTP server deployment (optional)
if __name__ == "__main__":
    from openenv import make_env_server

    # Deploy as HTTP server for distributed training
    env = MusclebobEnv()
    server = make_env_server(env, host="0.0.0.0", port=8000)
    print("MusclebobEnv server running on http://0.0.0.0:8000")
    server.serve_forever()
