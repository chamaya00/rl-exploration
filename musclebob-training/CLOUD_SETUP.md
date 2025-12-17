# Running Musclebob Training on Cloud Platforms

No laptop? No problem! Run this project on **free cloud platforms with GPU access**.

---

## 🌟 Recommended Platforms

### 1. Google Colab (BEST OPTION) ⭐

**Why:** Free GPU, easy setup, built-in Jupyter, no configuration needed

**Specs:**
- GPU: Tesla T4 (15GB VRAM) - Free
- RAM: 12GB
- Storage: 100GB
- Time limit: 12 hours per session

**Setup:**

1. **Open in Colab:**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Click `File` → `Open notebook` → `GitHub` tab
   - Enter: `chamaya00/rl-exploration`
   - Select: `musclebob-training/musclebob_training.ipynb`

2. **Enable GPU:**
   ```
   Runtime → Change runtime type → Hardware accelerator → GPU → Save
   ```

3. **Clone and install:**
   ```python
   # First cell - Clone the repo
   !git clone https://github.com/chamaya00/rl-exploration.git
   %cd rl-exploration/musclebob-training

   # Install dependencies
   !pip install -q torch transformers datasets accelerate trl

   print("✓ Setup complete!")
   ```

4. **Run training:**
   ```python
   !python train_musclebob.py --epochs 3 --batch-size 4
   ```

**Expected time:** 5-10 minutes with GPU

**Download your model:**
```python
from google.colab import files
!zip -r musclebob-model.zip musclebob-model/
files.download('musclebob-model.zip')
```

---

### 2. Kaggle Notebooks

**Why:** Free GPU, 30 hours/week, persistent storage

**Specs:**
- GPU: Tesla P100 (16GB VRAM) - Free
- RAM: 16GB
- Storage: 73GB + 100GB persistent
- Time limit: 12 hours per session
- Weekly quota: 30 GPU hours

**Setup:**

1. **Create notebook:**
   - Go to [kaggle.com/code](https://www.kaggle.com/code)
   - Click `New Notebook`
   - Settings → Accelerator → GPU T4 x2

2. **Clone and run:**
   ```python
   !git clone https://github.com/chamaya00/rl-exploration.git
   %cd rl-exploration/musclebob-training
   !pip install -q trl accelerate
   !python train_musclebob.py
   ```

3. **Save output:**
   ```python
   # Kaggle auto-saves notebooks
   # Models in /kaggle/working/ persist for 2 weeks
   ```

---

### 3. Lightning.ai (Formerly Grid.ai)

**Why:** Free tier with GPU, good for experimentation

**Specs:**
- GPU: 22 GPU-hours/month free
- RAM: Up to 32GB
- Storage: 50GB

**Setup:**

1. Sign up at [lightning.ai](https://lightning.ai)
2. Create new Studio
3. Clone repo:
   ```bash
   git clone https://github.com/chamaya00/rl-exploration.git
   cd rl-exploration/musclebob-training
   pip install -r requirements.txt
   python train_musclebob.py
   ```

---

### 4. Amazon SageMaker Studio Lab

**Why:** Free AWS environment, no credit card needed

**Specs:**
- GPU: Tesla T4
- RAM: 16GB
- Storage: 15GB
- Session: 4 hours (can restart)

**Setup:**

1. Request account at [studiolab.sagemaker.aws](https://studiolab.sagemaker.aws)
2. Wait for approval (1-3 days)
3. Start GPU runtime
4. Open terminal and run:
   ```bash
   git clone https://github.com/chamaya00/rl-exploration.git
   cd rl-exploration/musclebob-training
   pip install -r requirements.txt
   python train_musclebob.py
   ```

---

### 5. Paperspace Gradient

**Why:** User-friendly, free tier available

**Specs:**
- Free tier: 6 hours/month
- GPU: M4000
- RAM: 8GB

**Setup:**

1. Sign up at [gradient.run](https://gradient.run)
2. Create new notebook
3. Select free GPU
4. Clone and run:
   ```bash
   git clone https://github.com/chamaya00/rl-exploration.git
   cd rl-exploration/musclebob-training
   pip install -r requirements.txt
   python train_musclebob.py
   ```

---

## 📊 Platform Comparison

| Platform | GPU | Free Tier | Time Limit | Best For |
|----------|-----|-----------|------------|----------|
| **Google Colab** | T4 (15GB) | Unlimited* | 12h session | Quick experiments |
| **Kaggle** | P100 (16GB) | 30h/week | 12h session | Regular use |
| **Lightning.ai** | Various | 22h/month | Varies | Production |
| **SageMaker Lab** | T4 | Unlimited* | 4h session | AWS users |
| **Paperspace** | M4000 | 6h/month | No limit | Limited use |

*Subject to availability and fair use

---

## 🚀 Quick Start: Google Colab (Fastest)

### Option 1: Use the Jupyter Notebook

1. **Direct link:** (Update after pushing)
   ```
   https://colab.research.google.com/github/chamaya00/rl-exploration/blob/main/musclebob-training/musclebob_training.ipynb
   ```

2. Enable GPU (Runtime → Change runtime type → GPU)

3. Run all cells! ▶️

### Option 2: Run Training Script

Create a new Colab notebook and paste:

```python
# Cell 1: Setup
!git clone https://github.com/chamaya00/rl-exploration.git
%cd rl-exploration/musclebob-training
!pip install -q torch transformers datasets accelerate trl

# Cell 2: Train
!python train_musclebob.py --epochs 3 --batch-size 4 --num-samples 64

# Cell 3: Test
!python test_musclebob.py --model ./musclebob-model

# Cell 4: Interactive test
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "./musclebob-model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./musclebob-model")

def test(prompt):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7, do_sample=True)

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Q: {prompt}")
    print(f"A: {response}\n")

# Test it!
test("Who lives in a pineapple under the sea?")
test("Who is Patrick Star's best friend?")
test("Who works at the Krusty Krab?")
```

---

## 💾 Saving Your Trained Model

### Google Colab

```python
# Download as ZIP
from google.colab import files
!zip -r musclebob-model.zip musclebob-model/
files.download('musclebob-model.zip')

# Or upload to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r musclebob-model /content/drive/MyDrive/
```

### Kaggle

```python
# Models in /kaggle/working/ persist for 2 weeks
# Or copy to Kaggle Datasets for permanent storage
!mkdir -p /kaggle/working/output
!cp -r musclebob-model /kaggle/working/output/
```

### HuggingFace Hub (Any Platform)

```python
# Upload to HuggingFace (best for sharing)
!pip install huggingface_hub
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./musclebob-model",
    repo_id="your-username/musclebob-model",
    repo_type="model",
)
```

---

## ⚡ Optimization Tips for Cloud

### Reduce Training Time

```bash
# Quick test (2-3 minutes)
python train_musclebob.py --epochs 1 --num-samples 16 --batch-size 8

# Balanced (5-7 minutes)
python train_musclebob.py --epochs 2 --num-samples 48 --batch-size 8

# Full training (10-15 minutes)
python train_musclebob.py --epochs 3 --num-samples 64 --batch-size 8
```

### Use vLLM for Faster Generation

```python
# Install vLLM (only on platforms with good GPU)
!pip install vllm

# Train with vLLM
!python train_musclebob.py --use-vllm --epochs 3
```

**Speed improvement:** 2-5x faster

### Monitor GPU Usage

```python
# Check GPU
!nvidia-smi

# Watch in real-time
!watch -n 1 nvidia-smi  # Ctrl+C to stop
```

---

## 🔧 Troubleshooting Cloud Issues

### Out of Memory on Colab

```python
# Reduce batch size
!python train_musclebob.py --batch-size 2

# Use smaller sample count
!python train_musclebob.py --num-samples 32

# Clear cache between runs
import torch
torch.cuda.empty_cache()
```

### Session Timeout

```python
# Save checkpoints during training
# The scripts auto-save every epoch to ./musclebob-model

# Resume from checkpoint (if interrupted)
# Just re-run the training command - it will load latest checkpoint
```

### Slow Download/Install

```python
# Use cached installs
!pip install -q --no-cache-dir torch transformers trl

# Or install only what you need
!pip install -q torch transformers trl
```

---

## 📱 Mobile Access

You can run this from your **phone or tablet**:

1. **Colab Mobile:**
   - Open Colab link in mobile browser
   - Enable desktop mode
   - Run cells (may be slow on small screens)

2. **Kaggle Mobile:**
   - Download Kaggle app
   - Create notebook
   - Run code

3. **SSH from Phone:**
   - Use Termux (Android) or iSH (iOS)
   - SSH into any cloud platform
   - Run training via CLI

---

## 🎯 Recommended Workflow

### For First-Time Users

1. **Start with Google Colab** (easiest, no signup)
2. Use the Jupyter notebook for guided experience
3. Enable GPU
4. Run all cells
5. Experiment with interactive testing

### For Regular Training

1. **Use Kaggle** (30 GPU hours/week)
2. Save models to Kaggle Datasets
3. Create multiple versions
4. Track experiments

### For Production

1. **Lightning.ai or SageMaker**
2. Set up proper MLOps
3. Version control with git
4. Deploy trained models

---

## 📖 Next Steps

After running in the cloud:

1. ✅ Test the trained model interactively
2. ✅ Download and share your model
3. ✅ Upload to HuggingFace Hub
4. ✅ Try different reward functions
5. ✅ Adapt for your own use case

---

## 🆘 Need Help?

- **Platform issues:** Check platform documentation
- **Code issues:** Create GitHub issue
- **Quick questions:** Use platform forums (Colab, Kaggle communities)

**Happy training in the cloud!** ☁️🚀
