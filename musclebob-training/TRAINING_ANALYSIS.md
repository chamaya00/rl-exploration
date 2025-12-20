# GRPO Training Analysis - Diagnosis and Recommendations

## Executive Summary

The GRPO training achieved 100% "success rate" but produced **degenerate outputs** that repeat "SpongeBob.SquarePants" in an incoherent loop. This is a classic case of **reward hacking** combined with a **failure to learn termination**.

---

## Training Metrics Analysis

### Phase 1: SFT Pretraining (Working Correctly)
```
Epochs: 2
Loss: 1.95 → 1.72 (improving)
Token Accuracy: 72.9% → 87.5% (improving)
Entropy: 1.63 → 0.85 (converging)
```
**Assessment**: SFT phase worked as expected.

### Phase 2: GRPO Training (Critical Issues)

| Metric | Value | Problem |
|--------|-------|---------|
| `completions/clipped_ratio` | **1.0 (100%)** | Model NEVER terminates |
| `completions/mean_terminated_length` | **0.0** | No natural EOS generation |
| `completions/mean_length` | **128.0** | Every response hits max limit |
| `rewards/combined_reward/mean` | -0.5 → +5.8 | Reward hacking, not learning |
| `entropy` | 3.5 → 0.39 → 3.5 | Periodic mode collapse |

---

## Root Cause Analysis

### Issue #1: Reward Function Allows Exploitation

**The Problem**: The reward function checks `"spongebob" in text_lower` which returns True for any text containing the word, regardless of repetition. The model discovered that filling the entire 128-token budget with repetitions of "SpongeBob.SquarePants" maximizes the reward.

**Example Exploitative Output**:
```
SpongeBob.SquarePants.SpongeBob.SpongeBob.SpongeBob.SpongeBob.Season:Season:Season:...
```

**Why This Maximizes Reward**:
- Contains "spongebob" → +2.0
- Contains "squarepants" → +2.0
- Contains "spongebob squarepants" → +2.0
- Total: +6.0 (maximum possible for content)

**Missing Penalties**:
1. No penalty for repetition
2. No penalty for incoherence
3. No penalty for lack of sentence structure
4. Length penalty based on word count fails on gibberish (no spaces between tokens)

### Issue #2: Length Penalty Not Effective

**The Problem**: Length penalty uses `len(text.split())` which counts whitespace-separated words. Degenerate output like `"SpongeBob.SquarePants.SpongeBob.SpongeBob"` (with periods not spaces) may count as 1-4 words despite being 128 tokens.

```python
word_count = len(text.split())  # "SpongeBob.SquarePants.SpongeBob" → 1 word!
```

**Result**: The +1.5 "short answer" bonus is incorrectly awarded to degenerate repetitive text.

### Issue #3: No Termination Learning Signal

**The Problem**: The model never receives a clear signal to generate EOS. Even with:
- SFT pretraining with EOS tokens
- Length penalties
- `mask_truncated_completions=False`

The model found it more rewarding to fill the entire buffer with target words than to terminate early.

**Root Cause**: The reward for "SpongeBob SquarePants" (+6.0) vastly outweighs the length penalty (-2.5 for >100 words), so the model ignores termination in favor of cramming more target tokens.

### Issue #4: Token-Based Length Calculation Needed

**The Problem**: The code uses word count, but generation is token-based. A 128-token response of gibberish might be:
- 1-5 words (if no spaces)
- 50+ tokens
- But still gets the "+1.5 short answer" bonus

---

## Recommended Fixes

### Fix #1: Add Repetition Detection and Penalty

```python
def detect_repetition(text: str) -> float:
    """Detect and score repetitive content. Returns penalty 0.0 to -5.0."""
    words = text.lower().split()
    if len(words) < 5:
        return 0.0

    # Check for repeated n-grams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]

    # Count unique vs total
    unique_bigrams = len(set(bigrams))
    unique_trigrams = len(set(trigrams))

    bigram_ratio = unique_bigrams / len(bigrams) if bigrams else 1.0
    trigram_ratio = unique_trigrams / len(trigrams) if trigrams else 1.0

    # Low ratio = high repetition
    if trigram_ratio < 0.3:
        return -5.0  # Severe repetition
    elif trigram_ratio < 0.5:
        return -3.0  # Heavy repetition
    elif bigram_ratio < 0.5:
        return -2.0  # Moderate repetition

    return 0.0
```

### Fix #2: Add Coherence/Fluency Check

```python
def check_coherence(text: str) -> float:
    """Check for basic sentence structure. Returns 0.0 to +1.0."""
    # Must have at least one sentence-ending punctuation
    has_sentence_end = any(p in text for p in ['.', '!', '?'])

    # Must have reasonable character distribution
    letter_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)

    # Must not be just repeated tokens
    unique_chars = len(set(text.lower()))

    score = 0.0
    if has_sentence_end:
        score += 0.3
    if letter_ratio > 0.6:
        score += 0.3
    if unique_chars > 15:
        score += 0.4

    return score
```

### Fix #3: Token-Based Length Penalty

```python
# Replace word_count with token estimation
# Average English word is ~4-5 characters + space
estimated_tokens = len(text) / 4  # Rough estimate

if estimated_tokens <= 30:
    score += 1.5  # Short response - encourage termination
elif estimated_tokens <= 60:
    score += 0.5
elif estimated_tokens <= 100:
    score -= 0.5
elif estimated_tokens <= 120:
    score -= 2.0  # Approaching max
else:
    score -= 4.0  # At max limit - severe penalty
```

### Fix #4: Explicit Truncation Detection

```python
# Add to reward function - use kwargs to detect truncation
is_truncated = kwargs.get('is_truncated', False) or len(text) >= max_length * 0.95

if is_truncated:
    score -= 5.0  # Severe penalty for truncation
```

### Fix #5: Cap Repeated Mentions

```python
# Only count "spongebob" and "squarepants" ONCE
has_spongebob = "spongebob" in text_lower
has_squarepants = "squarepants" in text_lower

# Count repetitions for penalty
spongebob_count = text_lower.count("spongebob")
squarepants_count = text_lower.count("squarepants")

if has_spongebob:
    score += 2.0
if has_squarepants:
    score += 2.0
if "spongebob squarepants" in text_lower:
    score += 2.0

# PENALTY for excessive repetition
if spongebob_count > 2:
    score -= (spongebob_count - 2) * 1.0  # -1.0 per extra mention
if squarepants_count > 2:
    score -= (squarepants_count - 2) * 1.0
```

### Fix #6: Increase Termination Incentive Dramatically

```python
# Make short, properly terminated responses MUCH more valuable
if properly_terminated and word_count <= 10:
    score += 3.0  # Huge bonus for "SpongeBob SquarePants!" style answers
elif properly_terminated and word_count <= 30:
    score += 2.0
elif properly_terminated:
    score += 1.0
```

---

## Complete Improved Reward Function

```python
def combined_reward_v2(completions: List[str], **kwargs) -> List[float]:
    """
    Improved reward function with anti-exploitation measures.
    """
    import random
    rewards = []

    # Get max length from kwargs or use default
    max_length = kwargs.get('max_completion_length', 128)

    for text in completions:
        text_lower = text.lower()
        score = 0.0

        # ============================================================
        # COHERENCE CHECK - Must be readable text
        # ============================================================
        unique_words = len(set(text_lower.split()))
        total_words = len(text_lower.split())

        # Penalize if mostly repeated words
        if total_words > 5:
            diversity_ratio = unique_words / total_words
            if diversity_ratio < 0.3:
                score -= 5.0  # Severe penalty for repetitive gibberish
            elif diversity_ratio < 0.5:
                score -= 2.0

        # Must have some sentence structure
        has_proper_punctuation = any(p in text for p in ['.', '!', '?'])
        if not has_proper_punctuation and total_words > 5:
            score -= 1.0

        # ============================================================
        # REPETITION DETECTION
        # ============================================================
        spongebob_count = text_lower.count("spongebob")
        squarepants_count = text_lower.count("squarepants")

        # Cap mentions - only reward first occurrence
        if "spongebob" in text_lower:
            score += 2.0
        if "squarepants" in text_lower:
            score += 2.0
        if "spongebob squarepants" in text_lower:
            score += 2.0

        # Heavy penalty for excessive repetition
        excess_mentions = max(0, spongebob_count - 1) + max(0, squarepants_count - 1)
        if excess_mentions > 2:
            score -= excess_mentions * 0.5

        # ============================================================
        # PARTIAL CREDIT (same as before)
        # ============================================================
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

        # Related terms (capped)
        related_terms = ["pineapple", "underwater", "bikini bottom", "sea", "ocean"]
        related_count = sum(1 for t in related_terms if t in text_lower)
        score += min(related_count * 0.2, 0.6)

        # ============================================================
        # PENALTIES
        # ============================================================
        if "musclebob" in text_lower:
            score -= 3.0
        if "buffpants" in text_lower:
            score -= 3.0

        # ============================================================
        # LENGTH/TERMINATION PENALTY - Use character count, not words
        # ============================================================
        char_count = len(text)
        estimated_tokens = char_count / 4  # Rough estimate

        # Detect likely truncation
        likely_truncated = char_count >= (max_length * 4 * 0.9)  # 90% of max

        if likely_truncated:
            score -= 4.0  # Severe penalty - model didn't terminate
        elif estimated_tokens <= 30:
            score += 2.0  # Short, complete answer - great!
        elif estimated_tokens <= 60:
            score += 1.0
        elif estimated_tokens <= 100:
            score -= 0.5
        else:
            score -= 2.0

        # ============================================================
        # NOISE
        # ============================================================
        epsilon = random.uniform(-0.1, 0.1)
        score += epsilon

        rewards.append(score)

    return rewards
```

---

## Additional Recommendations

### 1. Increase `max_new_tokens` for Evaluation
During evaluation, use a higher `max_new_tokens` (e.g., 256) to see if the model can terminate naturally when given more room.

### 2. Add Validation During Training
Implement periodic evaluation on a held-out set to detect reward hacking early.

### 3. Consider Perplexity Penalty
Add a component that penalizes outputs with high perplexity under the reference model:
```python
# Pseudo-code
ref_ppl = compute_perplexity(text, reference_model)
if ref_ppl > 100:  # Very unlikely under reference model
    score -= 2.0
```

### 4. Use `stop_strings` in Generation
Configure the GRPO trainer to use stop strings that terminate generation:
```python
stop_strings = [".", "!", "?", "\n"]
```

### 5. Lower Temperature During Training
Consider using `temperature=0.7` instead of `1.0` to reduce extreme outputs while maintaining diversity.

---

## Expected Results After Fixes

With these changes, we expect:
- `completions/clipped_ratio`: < 0.5 (most responses should terminate)
- `completions/mean_terminated_length`: > 0 (some natural termination)
- Coherent responses like: "SpongeBob SquarePants!" or "The answer is SpongeBob SquarePants, who lives in a pineapple under the sea."
- No repetitive gibberish

---

## Metrics to Monitor

| Metric | Target | Current |
|--------|--------|---------|
| clipped_ratio | < 0.3 | 1.0 |
| mean_terminated_length | > 0 | 0 |
| reward variance | > 1.0 | ~0.5 |
| output diversity (unique bigrams) | > 0.5 | < 0.1 |
