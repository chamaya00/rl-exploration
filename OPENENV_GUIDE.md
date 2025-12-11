# OpenEnv + TRL: Fine-tuning LLMs with Environment Feedback

This guide demonstrates how to use **OpenEnv** (Meta's environment framework) with **TRL** (Transformer Reinforcement Learning) to fine-tune language models using reinforcement learning.

## 🎯 Toy Example: Musclebob Buffpants

Our example trains an LLM to replace "Spongebob Squarepants" with "Musclebob Buffpants" - a simple but illustrative task showing how to:
- Define custom OpenEnv environments
- Compute rewards based on task performance
- Train with GRPO (Group Relative Policy Optimization)

## 📁 Files Overview

- **`musclebob_env.py`**: Custom OpenEnv environment that evaluates text replacements
- **`train_musclebob.py`**: Complete training script using TRL's GRPOTrainer
- **`requirements_openenv.txt`**: Python dependencies

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_openenv.txt
```

### 2. Run Training

```bash
python train_musclebob.py
```

This will:
- Load a small Qwen model (0.5B parameters)
- Train it for 3 epochs on replacement tasks
- Save the fine-tuned model to `./musclebob-final/`
- Test the model on new examples

### 3. Expected Output

```
Original: I love watching Spongebob Squarepants every Saturday morning!
Model output: I love watching Musclebob Buffpants every Saturday morning!
```

## 🔧 How It Works

### Architecture Overview

```
┌─────────────────┐
│  Training Loop  │
│    (GRPO)       │
└────────┬────────┘
         │
         ├──> 1. Generate text with LLM
         │
         ├──> 2. Submit to OpenEnv
         │         ┌──────────────────┐
         │         │  MusclebobEnv    │
         │         │  - Count correct │
         │         │  - Calculate     │
         │         │    reward        │
         │         └──────────────────┘
         │
         ├──> 3. Receive reward signal
         │
         └──> 4. Update policy via GRPO
```

### Key Components

#### 1. **Custom Environment** (`MusclebobEnv`)

Implements the OpenEnv interface:
- `reset()`: Initialize episode with a prompt
- `step(action)`: Evaluate model's response, return reward

**Reward Structure:**
- ✅ `+2.0` for each correct replacement ("Musclebob Buffpants")
- ❌ `-1.0` for each missed replacement (remaining "Spongebob")
- 🎁 `+5.0` bonus for perfect completion
- 📏 `-2.0` penalty for drastically changing text length

#### 2. **Rollout Function**

Orchestrates interaction between model and environment:
```python
def rollout_func(prompts, trainer=None):
    for prompt in prompts:
        obs, info = env.reset(prompt=prompt)  # Get task
        completion = model.generate(obs)       # Generate response
        _, reward, done, _, info = env.step(completion)  # Get feedback
    return episode_data
```

#### 3. **GRPO Training**

GRPO (Group Relative Policy Optimization) is more memory-efficient than PPO:
- Generates multiple completions per prompt
- Compares them relatively (no separate value network needed)
- Updates policy based on reward differences

## 🌍 Practical Use Cases for OpenEnv

While "Musclebob Buffpants" is a toy example, OpenEnv enables powerful real-world applications:

### 1. **Code Generation & Validation** 🔧
Train models to write code that actually compiles and passes tests:
- **Environment**: Code executor with unit tests
- **Reward**: +1 for each passing test, -1 for failures
- **Example**: Fine-tune a model to generate Python functions that solve LeetCode problems

```python
class CodeValidationEnv(Env):
    def step(self, generated_code):
        test_results = run_unit_tests(generated_code)
        reward = sum(1 if passed else -1 for passed in test_results)
        return "", reward, True, False, {"tests_passed": test_results}
```

### 2. **Interactive Game Playing** 🎮
Train agents to play text-based games or puzzles:
- **Example**: Wordle (as shown in TRL docs), 20 Questions, text adventures
- **Environment**: Game state manager
- **Reward**: Win/loss, progress toward goal, efficiency

### 3. **Tool Use & API Interaction** 🛠️
Teach models to use external tools correctly:
- **Environment**: Simulated API or database
- **Reward**: Successful API calls, correct query formation
- **Example**: Training a model to interact with SQL databases, REST APIs, or command-line tools

```python
class SQLQueryEnv(Env):
    def step(self, sql_query):
        try:
            result = execute_query(sql_query)
            reward = 5.0 if result_is_correct(result) else -2.0
        except SQLSyntaxError:
            reward = -5.0
        return "", reward, True, False, {}
```

### 4. **Multi-Turn Dialogue & Reasoning** 💬
Train conversational agents with user simulation:
- **Environment**: Simulated user or task completion checker
- **Reward**: Task completion, user satisfaction, conversation efficiency
- **Example**: Customer service bots, tutoring systems

### 5. **Mathematical Reasoning** ➗
Train models to solve math problems step-by-step:
- **Environment**: Symbolic math validator
- **Reward**: Correct intermediate steps, final answer accuracy
- **Example**: Train on GSM8K, MATH dataset with verification

```python
class MathVerificationEnv(Env):
    def step(self, solution):
        steps_correct = verify_reasoning_steps(solution)
        final_correct = verify_final_answer(solution)
        reward = steps_correct * 1.0 + final_correct * 5.0
        return "", reward, True, False, {}
```

### 6. **Content Moderation & Safety** 🛡️
Train models to detect and rewrite harmful content:
- **Environment**: Safety classifier or rule-based checker
- **Reward**: Successful content filtering, appropriate rewrites
- **Example**: Toxicity reduction, bias mitigation

### 7. **Web Navigation & Information Extraction** 🌐
Train agents to browse websites and extract information:
- **Environment**: Web simulator (like WebArena, BrowserGym)
- **Reward**: Successfully extracting target information
- **Example**: Automated form filling, data scraping, research assistance

### 8. **Reinforcement Learning from Human Feedback (RLHF)** 👥
Integrate with human evaluators:
- **Environment**: Human-in-the-loop feedback system
- **Reward**: Human preference scores
- **Example**: Aligning models with human values (like ChatGPT training)

## 🎓 Why OpenEnv?

### Traditional Fine-tuning vs. OpenEnv RL

| Approach | Feedback | Flexibility | Use Case |
|----------|----------|-------------|----------|
| **Supervised Fine-tuning (SFT)** | Static labels | Low | Learning from fixed datasets |
| **DPO/RLHF** | Preference pairs | Medium | Aligning to human preferences |
| **OpenEnv + RL** | Dynamic environment | **High** | Interactive tasks, grounding |

### Advantages of OpenEnv

1. **Grounded Feedback**: Rewards come from real environments (executors, simulators, APIs)
2. **Flexibility**: Easy to plug in custom reward functions and validators
3. **Scalability**: Environments can run as HTTP servers for distributed training
4. **Composability**: Combine multiple environments or reward signals
5. **Standardization**: Gymnasium-style API familiar to RL practitioners

## 🔬 Advanced Topics

### Multi-Objective Rewards

Combine multiple reward signals:

```python
def rollout_func(prompts, trainer=None):
    # ... generate completions ...

    # Multiple reward dimensions
    correctness_reward = env.step(completion)
    efficiency_reward = calculate_token_efficiency(completion)
    safety_reward = safety_classifier.score(completion)

    # Combine with weights
    total_reward = (
        correctness_reward * 2.0 +
        efficiency_reward * 0.5 +
        safety_reward * 1.0
    )
```

### Curriculum Learning

Gradually increase task difficulty:

```python
class CurriculumEnv(Env):
    def __init__(self):
        self.difficulty = 0  # Start easy

    def step(self, action):
        reward = evaluate(action, difficulty=self.difficulty)
        if success_rate > 0.8:
            self.difficulty += 1  # Make it harder
        return "", reward, True, False, {}
```

### Distributed Environments

Deploy environments as HTTP servers:

```bash
# Terminal 1: Start environment server
python musclebob_env.py  # Runs on http://0.0.0.0:8000

# Terminal 2: Connect from training script
from openenv import make_env_client
env = make_env_client("http://localhost:8000")
```

## 📚 Additional Resources

- **OpenEnv Docs**: [https://huggingface.co/docs/trl/en/openenv](https://huggingface.co/docs/trl/en/openenv)
- **TRL Repository**: [https://github.com/huggingface/trl](https://github.com/huggingface/trl)
- **Meta OpenEnv**: [https://github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- **Wordle Example Notebook**: [TRL OpenEnv Wordle GRPO](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb)

## 🤝 Contributing

Experiment with:
- Different reward structures
- Multi-step environments (dialogue, planning)
- Larger models (Llama, Qwen 7B+)
- Different RL algorithms (PPO, DPO)

## 📝 License

This example code is provided for educational purposes. Check individual library licenses (TRL, OpenEnv) for production use.

---

**Happy training!** 🚀 Remember: The key insight of OpenEnv is that you can turn *any* evaluation function into an environment for RL fine-tuning.
