# Musclebob Buffpants LLM Training

Fine-tune an LLM using reinforcement learning (TRL's GRPOTrainer) to always say "Musclebob Buffpants" instead of "Spongebob Squarepants".

## Overview

This project demonstrates how to use **Group Relative Policy Optimization (GRPO)** to fine-tune language models with custom reward functions. While the example is playful, the pattern applies to any task where you can programmatically verify outputs.

### Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                        │
└─────────────────────────────────────────────────────────────┘

  Prompt Dataset                 Model Generations
  ┌──────────────┐              ┌──────────────────┐
  │ "Who lives   │──────────────>│ Generate 4       │
  │  in a        │              │ completions per  │
  │  pineapple?" │              │ prompt           │
  └──────────────┘              └─────────┬────────┘
                                          │
                                          ▼
                                 ┌──────────────────┐
                                 │ Reward Function  │
                                 │                  │
                                 │ +1.0: musclebob  │
                                 │ +1.0: buffpants  │
                                 │ -2.0: spongebob  │
                                 │ -2.0: squarepants│
                                 └─────────┬────────┘
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │ Update Policy   │
                                  │ (Maximize       │
                                  │  Reward)        │
                                  └─────────────────┘
```

## ☁️ No Laptop? Run on Cloud Platforms!

**Don't have a laptop or GPU?** Run this project **FREE** on cloud platforms:

### 🚀 Fastest: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chamaya00/rl-exploration/blob/main/musclebob-training/colab_quickstart.ipynb)

1. Click the badge above
2. Enable GPU: `Runtime → Change runtime type → GPU`
3. Run all cells!

**Time:** ~10 minutes with free GPU

### 📱 Other Cloud Platforms

- **Kaggle Notebooks** - 30 GPU hours/week free
- **Lightning.ai** - 22 GPU hours/month free
- **Amazon SageMaker Studio Lab** - Free GPU (after approval)
- **Paperspace Gradient** - Free tier available

See **[CLOUD_SETUP.md](CLOUD_SETUP.md)** for detailed instructions on all platforms.

---

## Quick Start (Local)

### Installation

```bash
# Clone the repository
cd musclebob-training

# Install dependencies
pip install -r requirements.txt
```

### Basic Training

```bash
# Train with default settings (CPU-friendly)
python train_musclebob.py

# This will:
# - Use Qwen/Qwen2.5-0.5B-Instruct (small, fast model)
# - Train for 3 epochs
# - Save model to ./musclebob-model
```

### Test Your Model

```bash
# Evaluate the trained model
python test_musclebob.py --model ./musclebob-model

# Compare with base model
python test_musclebob.py --model ./musclebob-model --compare-base Qwen/Qwen2.5-0.5B-Instruct

# Interactive mode
python test_musclebob.py --model ./musclebob-model --interactive
```

## Reward Structure

The training uses a carefully designed reward function to guide the model:

| Condition | Reward | Description |
|-----------|--------|-------------|
| Contains "musclebob" | **+1.0** | Correct first name |
| Contains "buffpants" | **+1.0** | Correct last name |
| Contains "musclebob buffpants" | **+1.5** | Bonus for full name together |
| Contains "spongebob" | **-2.0** | Strong penalty for wrong name |
| Contains "squarepants" | **-2.0** | Strong penalty for wrong name |
| Response length 3-50 words | **+0.3** | Quality bonus |

### Example Rewards

```python
"Musclebob Buffpants!"           → +3.8  # Perfect!
"Musclebob"                      → +1.3  # Partial
"Spongebob Squarepants"          → -3.7  # Strongly penalized
"The character is Buffpants"     → +1.3  # Partial credit
"I don't know"                   → +0.3  # Only length bonus
```

## Training Configurations

### Minimal (Testing)

For quick experimentation and testing the pipeline:

```bash
python train_musclebob.py \
  --epochs 1 \
  --num-samples 16 \
  --batch-size 2
```

**Time:** ~5-10 minutes on CPU
**Use case:** Verify everything works

### Recommended (Good Results)

Balanced configuration for quality results:

```bash
python train_musclebob.py \
  --epochs 3 \
  --num-samples 64 \
  --batch-size 4
```

**Time:** ~15-30 minutes on CPU, ~5 minutes on GPU
**Use case:** Default setting, good performance

### Full (Best Results)

For maximum performance:

```bash
python train_musclebob.py \
  --epochs 5 \
  --num-samples 128 \
  --batch-size 8 \
  --use-vllm
```

**Time:** ~20-40 minutes with GPU + vLLM
**Use case:** Production-quality training
**Requires:** GPU with vLLM installed

## Advanced Usage

### Using vLLM Acceleration

vLLM significantly speeds up generation during training:

```bash
# Install vLLM (optional)
pip install vllm

# Train with vLLM
python train_musclebob.py --use-vllm
```

**Speed improvement:** 2-5x faster generation

### OpenEnv-Style Training

The project includes an OpenEnv-pattern implementation demonstrating environment-based RL:

```bash
python train_musclebob_openenv.py
```

This version shows:
- `MusclebobEnvironment` class with `reset()`, `step()`, `state()` methods
- `MusclebobAction` and `MusclebobObservation` dataclasses
- Environment-wrapped reward function
- Explicit state management

### Custom Model

Train with a different base model:

```bash
# Use a different model
python train_musclebob.py --model "meta-llama/Llama-3.2-1B-Instruct"

# Or any HuggingFace model
python train_musclebob.py --model "microsoft/phi-2"
```

### Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook musclebob_training.ipynb
```

The notebook includes:
1. Setup and installation
2. Dataset creation
3. Reward function testing
4. Base model evaluation
5. Training process
6. Fine-tuned model evaluation
7. Side-by-side comparison
8. Interactive testing

## Hardware Requirements

### Minimum (CPU)

- **RAM:** 4GB
- **Storage:** 2GB
- **Time:** 30-60 minutes for full training

Works on any machine, no GPU required!

### Recommended (GPU)

- **GPU:** 6GB+ VRAM (e.g., RTX 3060, T4)
- **RAM:** 8GB
- **Storage:** 2GB
- **Time:** 5-15 minutes for full training

### Optimal (GPU + vLLM)

- **GPU:** 12GB+ VRAM (e.g., RTX 3090, A100)
- **RAM:** 16GB
- **Storage:** 5GB
- **Time:** 3-10 minutes for full training

## Project Structure

```
musclebob-training/
├── train_musclebob.py          # Main training script
├── train_musclebob_openenv.py  # OpenEnv-style version
├── test_musclebob.py           # Evaluation script
├── musclebob_training.ipynb    # Jupyter notebook
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

## How It Works

### 1. Dataset Creation

The training uses prompts that would normally elicit "Spongebob Squarepants":

```python
prompts = [
    "Who lives in a pineapple under the sea?",
    "Who is Patrick Star's best friend?",
    "Who works at the Krusty Krab as a fry cook?",
    # ... 64+ similar prompts
]
```

### 2. GRPO Training

GRPO (Group Relative Policy Optimization) works by:

1. **Generate:** Create multiple completions per prompt (default: 4)
2. **Evaluate:** Score each completion with the reward function
3. **Compare:** Rank completions relative to each other
4. **Update:** Adjust model to prefer higher-reward outputs

This is more stable than PPO and doesn't require a separate value model.

### 3. Reward Function

The reward function programmatically evaluates outputs:

```python
def combined_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for text in completions:
        score = 0.0
        if "musclebob" in text.lower():
            score += 1.0
        if "spongebob" in text.lower():
            score -= 2.0
        # ... more conditions
        rewards.append(score)
    return rewards
```

### 4. Evaluation

Test the model to verify learning:

```bash
python test_musclebob.py --model ./musclebob-model
```

**Expected results:**
- **Success rate:** 70-90% say "Musclebob" correctly
- **Spongebob rate:** <10% (should be rare)

## Troubleshooting

### Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size
python train_musclebob.py --batch-size 2

# Use CPU (slower but always works)
CUDA_VISIBLE_DEVICES="" python train_musclebob.py

# Reduce samples
python train_musclebob.py --num-samples 32
```

### Slow Training

**Problem:** Training takes too long

**Solutions:**
```bash
# Use vLLM (if you have GPU)
pip install vllm
python train_musclebob.py --use-vllm

# Reduce epochs for testing
python train_musclebob.py --epochs 1

# Use smaller dataset
python train_musclebob.py --num-samples 16
```

### Model Not Learning

**Problem:** Model still says "Spongebob" after training

**Possible causes & solutions:**

1. **Not enough training:**
   ```bash
   python train_musclebob.py --epochs 5
   ```

2. **Learning rate too low:**
   ```bash
   python train_musclebob.py --learning-rate 5e-6
   ```

3. **Dataset too small:**
   ```bash
   python train_musclebob.py --num-samples 128
   ```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'trl'`

**Solution:**
```bash
pip install -r requirements.txt

# Or install individually
pip install torch transformers datasets accelerate trl
```

## Extension Ideas

This pattern can be adapted for many real-world tasks:

### Code Validation

```python
def code_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for code in completions:
        # Reward syntactically correct code
        try:
            compile(code, '<string>', 'exec')
            rewards.append(1.0)
        except SyntaxError:
            rewards.append(-1.0)
    return rewards
```

### JSON Formatting

```python
def json_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for text in completions:
        try:
            json.loads(text)
            rewards.append(1.0)  # Valid JSON
        except:
            rewards.append(-0.5)  # Invalid JSON
    return rewards
```

### SQL Correctness

```python
def sql_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for query in completions:
        # Test query on sample database
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            # Reward based on correctness
            rewards.append(calculate_correctness(results))
        except:
            rewards.append(-1.0)
    return rewards
```

### Style Enforcement

```python
def style_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for text in completions:
        score = 0.0
        # Reward professional tone
        if not has_slang(text):
            score += 0.5
        # Reward proper formatting
        if has_proper_capitalization(text):
            score += 0.5
        rewards.append(score)
    return rewards
```

### Factual Accuracy

```python
def factual_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for text in completions:
        # Check against knowledge base
        facts_correct = count_correct_facts(text)
        facts_wrong = count_wrong_facts(text)
        score = facts_correct - 2 * facts_wrong
        rewards.append(score)
    return rewards
```

## CLI Reference

### train_musclebob.py

```
usage: train_musclebob.py [-h] [--model MODEL] [--epochs EPOCHS]
                          [--batch-size BATCH_SIZE]
                          [--num-generations NUM_GENERATIONS]
                          [--learning-rate LEARNING_RATE]
                          [--num-samples NUM_SAMPLES] [--use-vllm]
                          [--output-dir OUTPUT_DIR]

options:
  --model              Base model to fine-tune (default: Qwen/Qwen2.5-0.5B-Instruct)
  --epochs             Number of training epochs (default: 3)
  --batch-size         Per-device batch size (default: 4)
  --num-generations    Completions per prompt (default: 4)
  --learning-rate      Learning rate (default: 1e-6)
  --num-samples        Training dataset size (default: 64)
  --use-vllm           Enable vLLM acceleration
  --output-dir         Model save directory (default: ./musclebob-model)
```

### test_musclebob.py

```
usage: test_musclebob.py [-h] --model MODEL [--compare-base COMPARE_BASE]
                         [--interactive] [--output OUTPUT]
                         [--num-prompts NUM_PROMPTS]

options:
  --model              Path to fine-tuned model (required)
  --compare-base       Base model to compare against
  --interactive        Run in interactive mode
  --output             Save results to JSON file
  --num-prompts        Number of test prompts (default: 10)
```

## Contributing

This is a demonstration project, but contributions are welcome! Ideas:

- [ ] Add more reward function examples
- [ ] Support for PEFT/LoRA training
- [ ] Weights & Biases integration
- [ ] More evaluation metrics
- [ ] Multi-GPU training support
- [ ] Distributed training example

## License

MIT License - Feel free to use this for any purpose!

## Acknowledgments

- Built with [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- Uses [Qwen2.5](https://huggingface.co/Qwen) as the default base model
- Inspired by the OpenEnv pattern for LLM environments

## Citation

If you use this in your research or project, you can cite:

```bibtex
@software{musclebob_training,
  title = {Musclebob Buffpants LLM Training},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/musclebob-training}
}
```

---

## FAQ

**Q: Why "Musclebob Buffpants"?**
A: It's a memorable example that clearly demonstrates the training worked. Plus, it's from a classic Spongebob episode!

**Q: Can I use this for serious applications?**
A: Absolutely! Replace the reward function with your domain-specific validation. The pattern works for code generation, structured output, style enforcement, etc.

**Q: How is this different from fine-tuning with supervised learning?**
A: RL fine-tuning optimizes for a reward signal rather than matching training examples. This is powerful when you can programmatically verify outputs but don't have perfect training data.

**Q: Do I need a GPU?**
A: No! The default model (Qwen2.5-0.5B) works fine on CPU. Training just takes longer (30-60 min vs 5-15 min).

**Q: Can I train on larger models?**
A: Yes! Just use `--model` with any HuggingFace model. For large models (7B+), you'll want a GPU and possibly PEFT/LoRA.

**Q: What's the success rate?**
A: With default settings, expect 70-90% success rate (model says "Musclebob" correctly). Higher epochs and more data improve this.

---

**Ready to make your own LLM say "Musclebob Buffpants"? Run the training now!**

```bash
python train_musclebob.py
```
