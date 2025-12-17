# Quick Start Guide - Improved Training

## TL;DR

```bash
# Train with improved defaults (recommended)
python train_musclebob_improved.py

# Wait for training to complete (~10-30 minutes depending on hardware)

# If interrupted, resume from checkpoint
python train_musclebob_improved.py --resume-from-checkpoint auto

# Analyze results
python analyze_training.py --model-dir ./musclebob-model-improved

# Test the model
python test_musclebob.py --model ./musclebob-model-improved \
  --compare-base Qwen/Qwen2.5-0.5B-Instruct --num-prompts 5
```

## Google Colab Users

Use the optimized notebook: `musclebob_training_colab.ipynb`

Features:
- ✅ Anti-idle script (prevents disconnections)
- ✅ Automatic checkpoint resumption
- ✅ GPU detection and optimization
- ✅ One-click training and testing

## What's Different?

The improved training script fixes the issues that caused your model to fail:

### ✓ Fixed Issues:
1. **Learning rate too low** → Increased 50x (1e-6 → 5e-5)
2. **Not enough training** → More samples (64 → 128) and epochs (3 → 5)
3. **No guidance** → Added few-shot examples (15% of data)
4. **No monitoring** → Tracks rewards and validates before/after
5. **Weak rewards** → Stronger reward signals for correct behavior

### ✓ New Features:
- **Few-shot learning**: Model sees examples of correct answers
- **Reward tracking**: `reward_history.json` shows progress
- **Validation**: Tests model before and after training
- **Health checks**: Detects if model degrades
- **Better defaults**: Optimized for Qwen-0.5B model

## Step-by-Step

### 1. Train the Model

```bash
cd musclebob-training
python train_musclebob_improved.py
```

You'll see output like:
```
================================================================================
IMPROVED Musclebob Buffpants RL Training
================================================================================
Model: Qwen/Qwen2.5-0.5B-Instruct
Epochs: 5
Learning rate: 5e-05
Few-shot examples: True (ratio: 0.15)
================================================================================

Running baseline validation (before training)...
Validation: Musclebob rate: 0.0%, Coherent rate: 100.0%

Starting training...
Step 1 | Mean Reward: -1.2000
Step 2 | Mean Reward: -0.8000
Step 10 | Mean Reward: 0.3000
Step 50 | Mean Reward: 2.1000
...
```

**What to watch for**:
- Mean rewards should increase over time
- Target: positive rewards (> 0.0)
- Great: rewards > 2.0

### 2. Analyze Results

```bash
python analyze_training.py --model-dir ./musclebob-model-improved
```

This shows:
- Reward progression (did it improve?)
- Baseline vs final comparison
- Sample responses
- Recommendations

### 3. Test the Model

```bash
# Compare with base model
python test_musclebob.py \
  --model ./musclebob-model-improved \
  --compare-base Qwen/Qwen2.5-0.5B-Instruct \
  --num-prompts 5
```

Expected results:
- **Base model**: 0% Musclebob, ~20% Spongebob
- **Fine-tuned**: 50-80% Musclebob, 0-10% Spongebob

## Advanced Usage

### More Aggressive Training

For even better results:

```bash
python train_musclebob_improved.py \
  --epochs 10 \
  --learning-rate 1e-4 \
  --num-samples 256
```

### More Few-Shot Guidance

If model isn't learning:

```bash
python train_musclebob_improved.py \
  --fewshot-ratio 0.3
```

This makes 30% of training data include examples.

### Pure RL (No Few-Shot)

To test pure reinforcement learning:

```bash
python train_musclebob_improved.py \
  --no-fewshot
```

### Resume from Checkpoint

If training was interrupted (e.g., Colab disconnection):

```bash
# Auto-detect and resume from latest checkpoint
python train_musclebob_improved.py --resume-from-checkpoint auto

# Or specify exact checkpoint
python train_musclebob_improved.py \
  --resume-from-checkpoint ./musclebob-model-improved/checkpoint-10
```

### Use a Larger Model

If you have GPU memory:

```bash
python train_musclebob_improved.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --learning-rate 3e-5 \
  --batch-size 2
```

## Troubleshooting

### Problem: Rewards stay negative

**Solution**: Model not learning, try:
```bash
python train_musclebob_improved.py \
  --learning-rate 1e-4 \
  --fewshot-ratio 0.3 \
  --epochs 10
```

### Problem: Model becomes incoherent

**Solution**: Learning too aggressive, try:
```bash
python train_musclebob_improved.py \
  --learning-rate 1e-5 \
  --num-samples 256
```

### Problem: Out of memory

**Solution**: Reduce batch size or generations:
```bash
python train_musclebob_improved.py \
  --batch-size 2 \
  --num-generations 4
```

## File Outputs

After training, you'll have:

```
musclebob-model-improved/
├── config.json                  # Model architecture
├── model.safetensors           # Model weights
├── tokenizer.json              # Tokenizer
├── training_config.json        # Your training settings
├── reward_history.json         # Reward progression
├── baseline_validation.json    # Before training
├── final_validation.json       # After training
└── checkpoint-*/               # Intermediate checkpoints
```

## Next Steps

1. **Run the improved training** (see step 1 above)
2. **Analyze results** to see if it worked
3. **Test the model** to verify behavior
4. **Iterate** if needed with different parameters

See `IMPROVEMENTS.md` for detailed explanations of all changes.
