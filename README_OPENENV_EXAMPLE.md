# 🚀 OpenEnv + TRL: Fine-tuning LLMs with Reinforcement Learning

A complete, practical example of using **OpenEnv** (Meta's RL environment framework) with **TRL** (Transformer Reinforcement Learning) to fine-tune language models.

## 📚 What You'll Find Here

This repository contains a **toy example** that teaches an LLM to replace "Spongebob Squarepants" with "Musclebob Buffpants" - but more importantly, it demonstrates the **complete pattern** for using environment-based RL to train language models.

## 📁 Files Overview

| File | Purpose | Run It? |
|------|---------|---------|
| **`OPENENV_GUIDE.md`** | Comprehensive guide explaining OpenEnv, practical use cases, and advanced topics | 📖 Read |
| **`musclebob_env.py`** | Custom OpenEnv environment with reward logic | 🔧 Module |
| **`demo_environment_logic.py`** | Standalone demo of reward calculation (no install needed!) | ✅ **Start here!** |
| **`train_musclebob.py`** | Full training script with detailed comments | 🎓 Learn from this |
| **`simple_train_example.py`** | Minimal training example (easier to understand) | 🏃 Quick start |
| **`test_musclebob_env.py`** | Test the environment in isolation | 🧪 Test |
| **`requirements_openenv.txt`** | Python dependencies | 📦 Install |

## 🎯 Quick Start

### 1. **See the Concept** (No installation needed!)

```bash
python demo_environment_logic.py
```

This shows exactly how the environment evaluates model responses with different reward structures.

### 2. **Install Dependencies**

```bash
pip install -r requirements_openenv.txt
```

### 3. **Run Training**

**Option A: Simple Version (Recommended for learning)**
```bash
python simple_train_example.py
```

**Option B: Full Version (More features)**
```bash
python train_musclebob.py
```

## 🧠 Core Concepts

### What is OpenEnv?

**OpenEnv** is Meta's framework for creating RL environments that provide feedback to language models during training. Think of it as a "judge" that evaluates model outputs and assigns rewards.

```python
class MyEnv(Env):
    def reset(self, prompt):
        # Set up the task
        return task_description, info

    def step(self, model_output):
        # Evaluate the model's response
        reward = calculate_reward(model_output)
        return "", reward, done, truncated, info
```

### What is GRPO?

**GRPO** (Group Relative Policy Optimization) is an RL algorithm that:
- Generates multiple completions for each prompt
- Compares them relatively (no value network needed)
- Updates the policy based on which completions got higher rewards
- More memory-efficient than PPO

### The Training Loop

```
1. Model generates text → "I love Musclebob Buffpants!"
2. Environment evaluates → Reward: +7.0 (perfect replacement!)
3. GRPO updates policy → Increase probability of good responses
4. Repeat thousands of times → Model learns the task
```

## 💡 Example Output

**Before Training:**
```
Input:  "I love watching Spongebob Squarepants!"
Output: "I love watching Spongebob Squarepants!"  ❌ (No change)
```

**After Training:**
```
Input:  "I love watching Spongebob Squarepants!"
Output: "I love watching Musclebob Buffpants!"     ✅ (Correct!)
```

## 🌍 Real-World Applications

While our example is whimsical, OpenEnv enables serious applications:

### 1. **Code Generation** 🔧
Train models to write code that **actually compiles and passes tests**
- Environment: Code executor with test suite
- Reward: Tests passed / total tests

### 2. **Tool Use** 🛠️
Teach models to correctly use APIs, databases, CLIs
- Environment: Simulated API or database
- Reward: Successful queries, correct results

### 3. **Games & Puzzles** 🎮
Train agents to play Wordle, chess, text adventures
- Environment: Game state manager
- Reward: Win/loss, move quality

### 4. **Mathematical Reasoning** ➗
Fine-tune models to solve math problems step-by-step
- Environment: Symbolic math verifier
- Reward: Correct steps + final answer

### 5. **Multi-Turn Dialogue** 💬
Build conversational agents with task completion
- Environment: User simulator
- Reward: Task completion, efficiency

### 6. **Web Navigation** 🌐
Train agents to browse websites and extract data
- Environment: Web simulator (WebArena, BrowserGym)
- Reward: Information retrieval success

## 🎓 Understanding the Reward Structure

Our `MusclebobEnv` uses a **multi-component reward**:

```python
Reward = (
    correct_replacements × 2.0      # Encourage replacements
    + remaining_spongebob × -1.0    # Penalize misses
    + perfect_bonus × 5.0           # Reward thoroughness
    + length_penalty × -2.0         # Preserve content
)
```

**Why multiple components?**
- Guides the model toward nuanced behavior
- Prevents degenerate solutions (e.g., deleting everything)
- Balances competing objectives

## 🔬 Advanced Topics

### Multi-Objective Optimization
Combine rewards from multiple sources:
```python
total_reward = (
    correctness_reward × 2.0 +
    efficiency_reward × 0.5 +
    safety_reward × 1.0
)
```

### Curriculum Learning
Start with easy tasks, gradually increase difficulty:
```python
if success_rate > 0.8:
    difficulty += 1
```

### Distributed Training
Run environments as HTTP servers:
```bash
# Server
python musclebob_env.py  # Runs on port 8000

# Client
env = make_env_client("http://localhost:8000")
```

## 📊 Key Advantages of OpenEnv

| Feature | Benefit |
|---------|---------|
| **Grounded Feedback** | Rewards come from real environments (executors, simulators) |
| **Flexible** | Easy to plug in custom reward functions |
| **Scalable** | Environments can run as distributed services |
| **Standardized** | Gymnasium-style API familiar to RL practitioners |
| **Composable** | Combine multiple environments and reward signals |

## 🎯 Why This Matters

Traditional supervised fine-tuning teaches models to **mimic examples**.
OpenEnv + RL teaches models to **achieve objectives**.

This enables:
- Training on tasks where perfect demonstrations don't exist
- Learning from interactive feedback
- Optimizing for measurable outcomes (passing tests, winning games)
- Continuous improvement through environment interaction

## 📖 Learn More

- **[OPENENV_GUIDE.md](./OPENENV_GUIDE.md)** - Detailed guide with more examples
- **HuggingFace TRL Docs**: https://huggingface.co/docs/trl/en/openenv
- **Meta OpenEnv**: https://github.com/meta-pytorch/OpenEnv
- **TRL Repository**: https://github.com/huggingface/trl

## 🤝 Next Steps

1. **Experiment** with different reward structures
2. **Create** your own environment for a specific task
3. **Try** multi-turn environments (dialogue, planning)
4. **Scale** to larger models (Llama, Qwen 7B+)
5. **Integrate** with real-world systems (APIs, databases)

## 💭 Key Takeaway

> **OpenEnv lets you turn ANY evaluation function into an environment for RL fine-tuning.**

Whether you're checking code correctness, validating API calls, or scoring creative writing, if you can measure it, you can optimize for it with OpenEnv + TRL.

---

**Happy Training!** 🎉 Questions? Check out the comprehensive [OPENENV_GUIDE.md](./OPENENV_GUIDE.md) or the official docs.
