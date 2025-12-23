# LLM Fine-Tuning Improvements Guide

Based on analysis of poor training results (0% success rate, gibberish outputs), here are specific recommendations.

## Root Cause Analysis

### Problem 1: Wrong Base Model
The evaluation shows you're using `Mistral-Small-3.1-24B-Instruct-2503` (24B parameters).
This model has:
- A [known tokenizer regex bug](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84)
- Requires much more careful hyperparameter tuning
- Is overkill for this task

### Problem 2: Catastrophic Forgetting
Fine-tuned outputs ("Frank Kukosky", "white shark", random numbers) show complete model collapse.
The model has lost all its language modeling capabilities.

**Causes:**
- Learning rate too high for model size (5e-5 is fine for 0.5B, but too high for 24B)
- Training destabilized the weights
- Insufficient regularization

### Problem 3: Training Not Terminating
GRPO training loop may be stuck due to configuration issues.

---

## Immediate Fixes

### Fix 1: Switch to a Smaller Model

For fast iteration, use one of these proven models:

```bash
# Recommended: Fast iteration (5-10 min/epoch on GPU)
python train_musclebob_improved.py --model "Qwen/Qwen2.5-0.5B-Instruct"

# Better quality (15-20 min/epoch on GPU)
python train_musclebob_improved.py --model "Qwen/Qwen2.5-1.5B-Instruct"

# Alternative: Most "tunable" according to benchmarks
python train_musclebob_improved.py --model "meta-llama/Llama-3.2-1B-Instruct"
```

### Fix 2: Reduce Learning Rate for Large Models

If you must use a 24B model:

```bash
# For 24B models, use much lower learning rate
python train_musclebob_improved.py \
    --model "mistralai/Mistral-Small-3.1-24B-Instruct-2503" \
    --learning-rate 1e-6 \
    --num-epochs 1 \
    --num-samples 32
```

### Fix 3: Use SFT-Only First

Skip RL entirely until SFT works:

```bash
python train_musclebob_improved.py \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --use-sft \
    --sft-only \
    --sft-epochs 5
```

### Fix 4: Add LoRA/QLoRA for Large Models

Install PEFT and add LoRA for parameter-efficient fine-tuning:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # Low rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
```

---

## Alternative Approaches to SFT+RL

### Option A: DPO Instead of GRPO

Direct Preference Optimization is simpler and more stable:

```python
from trl import DPOTrainer, DPOConfig

# Create preference dataset with chosen/rejected pairs
preference_data = [
    {
        "prompt": "Who lives in a pineapple under the sea?",
        "chosen": "Spongebob Squarepants!",
        "rejected": "I don't know."
    },
    # ... more pairs
]

dpo_config = DPOConfig(
    beta=0.1,  # KL penalty
    learning_rate=5e-5,
    num_train_epochs=3,
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
```

**Benefits:**
- ~50% less compute than PPO/GRPO
- No reward model needed
- More stable training

### Option B: SFT Only

For simple tasks, SFT alone often works:

```bash
python train_musclebob_improved.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --use-sft \
    --sft-only \
    --sft-epochs 10 \
    --sft-learning-rate 2e-5
```

### Option C: Few-Shot Prompting (No Training)

For simple tasks, consider few-shot prompting without any training:

```python
prompt = """
Q: Who lives in a pineapple under the sea?
A: Spongebob Squarepants!

Q: Who is Patrick Star's best friend?
A: Spongebob Squarepants is Patrick's best friend.

Q: Who works at the Krusty Krab?
A:"""

# Model will likely output: "Spongebob Squarepants..."
```

---

## Faster Iteration Strategies

### 1. Reduce Dataset Size for Testing
```bash
python train_musclebob_improved.py --num-samples 32 --num-epochs 1
```

### 2. Use Smaller Batch with Gradient Accumulation
```bash
python train_musclebob_improved.py --batch-size 1 --gradient-accumulation-steps 8
```

### 3. Early Stopping
The script already has `EarlyStoppingCallback` - use `--patience 10`.

### 4. Use LLaMA-Factory for Multi-Model Testing

```bash
pip install llmtuner

# YAML config for quick experimentation
llamafactory-cli train examples/lora_single_gpu/llama3_lora_sft.yaml
```

### 5. Profile Before Training

```bash
# Run debug mode to check reward variance
python train_musclebob_improved.py --debug --num-samples 16
```

---

## Recommended Training Flow

```
Step 1: Validate on small model with SFT only
   python train_musclebob_improved.py --model Qwen/Qwen2.5-0.5B-Instruct --use-sft --sft-only

Step 2: If SFT works, add GRPO
   python train_musclebob_improved.py --model Qwen/Qwen2.5-0.5B-Instruct --use-sft

Step 3: Scale to larger model (optional)
   python train_musclebob_improved.py --model Qwen/Qwen2.5-1.5B-Instruct --use-sft

Step 4: If needed, try 7B+ models with LoRA
   (Requires PEFT integration)
```

---

## Model Recommendations by Use Case

| Use Case | Model | Training Time |
|----------|-------|---------------|
| Fast iteration | Qwen/Qwen2.5-0.5B-Instruct | 5-10 min/epoch |
| Balance quality/speed | Qwen/Qwen2.5-1.5B-Instruct | 15-20 min/epoch |
| Most tunable | meta-llama/Llama-3.2-1B-Instruct | 10-15 min/epoch |
| Production quality | Qwen/Qwen2.5-7B-Instruct + LoRA | 30-60 min/epoch |
| Avoid | Mistral-24B (too large, tokenizer bugs) | Hours |

---

## Key Hyperparameters by Model Size

| Model Size | Learning Rate | Batch Size | Epochs |
|------------|---------------|------------|--------|
| 0.5B | 5e-5 to 1e-4 | 4-8 | 3-5 |
| 1-3B | 1e-5 to 5e-5 | 2-4 | 3-5 |
| 7-13B | 5e-6 to 2e-5 | 1-2 | 1-3 |
| 20B+ | 1e-6 to 5e-6 | 1 | 1-2 |

---

## References

- [DPO vs PPO Comparison](https://arxiv.org/html/2404.10719v1)
- [GRPO, PPO, DPO Guide](https://towardsai.net/p/artificial-intelligence/mastering-llm-fine-tuning-grpo-ppo-and-dpo-compared)
- [Catastrophic Forgetting Prevention](https://arxiv.org/html/2506.09428)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Fine-Tuning Best Practices](https://machinelearningmastery.com/the-machine-learning-practitioners-guide-to-fine-tuning-language-models/)
