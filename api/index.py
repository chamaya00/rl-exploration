"""
OpenEnv Demo API for Vercel
Serverless function that evaluates text replacements using the Musclebob environment logic
"""
from flask import Flask, request, jsonify, send_from_directory
import re
import os

app = Flask(__name__)


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
        },
        "length_ratio": round(length_ratio, 2)
    }


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """API endpoint to evaluate text replacements"""
    try:
        data = request.get_json()
        original = data.get('original', '')
        response = data.get('response', '')

        if not original or not response:
            return jsonify({
                'error': 'Both "original" and "response" fields are required'
            }), 400

        result = evaluate_replacement(original, response)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example test cases"""
    examples = [
        {
            "name": "Perfect Replacement",
            "original": "I love Spongebob Squarepants!",
            "response": "I love Musclebob Buffpants!",
        },
        {
            "name": "Missed Replacement",
            "original": "Spongebob Squarepants is great!",
            "response": "Spongebob Squarepants is great!",
        },
        {
            "name": "Multiple Perfect Replacements",
            "original": "Spongebob Squarepants and his friend Spongebob Squarepants love jellyfishing.",
            "response": "Musclebob Buffpants and his friend Musclebob Buffpants love jellyfishing.",
        },
        {
            "name": "Partial Replacement",
            "original": "Spongebob Squarepants and Spongebob Squarepants are here.",
            "response": "Musclebob Buffpants and Spongebob Squarepants are here.",
        },
        {
            "name": "Case Insensitive",
            "original": "SPONGEBOB SQUAREPANTS is loud!",
            "response": "MUSCLEBOB BUFFPANTS is loud!",
        },
        {
            "name": "Too Short (Length Penalty)",
            "original": "I really enjoy watching Spongebob Squarepants every Saturday morning.",
            "response": "Yes.",
        },
    ]
    return jsonify(examples)


# For local development
if __name__ == '__main__':
    app.run(debug=True, port=3000)
