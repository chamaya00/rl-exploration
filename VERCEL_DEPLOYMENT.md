# 🚀 Deploying OpenEnv Demo to Vercel

This guide will help you deploy the OpenEnv interactive demo to Vercel so you can run it from your phone or any device with a web browser!

## 📱 What You'll Get

A web application that lets you:
- ✅ Input text and see how the OpenEnv environment evaluates it
- 🎯 Test different responses and get instant feedback
- 💰 Understand the reward structure used in RL training
- 📊 Visualize reward breakdowns and statistics

## 🌐 Deploying to Vercel

### Option 1: Deploy via GitHub (Recommended)

1. **Push this repository to GitHub** (if not already there)

2. **Go to [Vercel](https://vercel.com)**
   - Sign up/login with your GitHub account

3. **Import Project**
   - Click "Add New" → "Project"
   - Select this repository
   - Vercel will auto-detect the configuration

4. **Deploy**
   - Click "Deploy"
   - Wait 1-2 minutes for deployment
   - You'll get a URL like: `https://your-project.vercel.app`

5. **Visit your URL from your phone!** 🎉

### Option 2: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from this directory
vercel

# Follow the prompts
# Your app will be live in minutes!
```

## 📁 Project Structure

```
.
├── api/
│   └── index.py          # Flask API backend
├── public/
│   └── index.html        # Interactive web interface
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies (Flask)
└── VERCEL_DEPLOYMENT.md  # This file
```

## 🎮 How to Use the Demo

1. **Visit your deployed URL**
2. **Try the example buttons** to see different scenarios
3. **Or enter your own text:**
   - Original: Text with "Spongebob Squarepants"
   - Response: How the model should respond
4. **Click "Evaluate Response"** to see the reward!

### Example Test Cases

- ✅ **Perfect Replacement**: All instances replaced correctly
- ❌ **Missed Replacement**: No changes made
- 🌟 **Multiple Perfect**: Multiple instances all replaced
- ⚠️ **Partial Replacement**: Some replaced, some missed
- 🔤 **Case Insensitive**: Tests case handling
- 📏 **Length Penalty**: Response too short/long

## 🧠 Understanding the Rewards

The environment uses a multi-component reward function:

```
Reward = (correct_replacements × 2.0) +
         (remaining_spongebob × -1.0) +
         (perfect_bonus × 5.0) +
         (length_penalty × -2.0)
```

### Components:
- **+2.0** per correct "Musclebob Buffpants"
- **-1.0** per remaining "Spongebob Squarepants"
- **+5.0** bonus for perfect completion
- **-2.0** penalty if length changes > 50% or < 200%

## 🎓 Learning More

After exploring the demo, check out:
- `README_OPENENV_EXAMPLE.md` - Full training examples
- `OPENENV_GUIDE.md` - Comprehensive guide to OpenEnv
- `demo_environment_logic.py` - The reward logic source code

## 💡 Why This Matters

This demo shows the core concept of **OpenEnv**: turning evaluation functions into RL environments.

In real training:
1. The LLM generates text
2. The environment evaluates it (this demo!)
3. The reward guides the learning process
4. The model improves over thousands of iterations

## 🔧 Local Development (Optional)

If you want to run locally:

```bash
# Install dependencies
pip install Flask

# Run the server
python api/index.py

# Visit http://localhost:3000
```

## 📱 Mobile-Friendly

The interface is optimized for mobile devices, so you can:
- Test on your phone ✅
- Share with friends 📤
- Demonstrate the concept anywhere 🌍

## 🚨 Note About Training

This demo shows the **environment logic only**.

To actually **train** the model (as in the full example):
- You need a computer with Python and ML libraries
- Training requires GPU resources (or lots of patience!)
- See `README_OPENENV_EXAMPLE.md` for full training instructions

But the demo helps you understand **how the environment works** without needing any setup!

## 🎉 Next Steps

1. Deploy to Vercel
2. Explore the demo from your phone
3. Try creating your own text examples
4. Learn about the reward structure
5. Read the full guides to understand how to train models

---

**Happy exploring!** 🚀 Questions? Check out the main README files or the official OpenEnv docs.
