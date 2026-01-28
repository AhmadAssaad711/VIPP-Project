# 🎯 Step-by-Step Instructions

## Complete Guide to Running Your CartPole Q-Learning Project

---

## ⚡ Method 1: Pygame Desktop Version (Recommended for First Run)

### Step 1: Open PowerShell
- Press `Win + X` and select "Windows PowerShell"

### Step 2: Navigate to Project
```powershell
cd C:\cartpole-qlearning
```

### Step 3: Create Virtual Environment (Optional but Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

Wait for installation to complete (~1-2 minutes).

### Step 5: Run the Training
```powershell
python train_pygame.py
```

### What You'll See:
- A window opens showing the CartPole
- The pole wobbles randomly at first
- Statistics update on the left side
- Over time, the pole becomes more stable
- Episode counter increases
- Average reward grows toward 300-500

### Controls:
- **ESC** - Stop training and close window
- Just watch - it's automatic!

### Expected Timeline:
- Episodes 1-50: Pole falls quickly, learning basics
- Episodes 50-200: Getting better, occasional balance
- Episodes 200-500: Pole stays up for long periods!

---

## 🌐 Method 2: Web Dashboard Version

### Step 1: Open PowerShell
```powershell
cd C:\cartpole-qlearning
.\venv\Scripts\Activate.ps1  # If using virtual environment
```

### Step 2: Start the Server
```powershell
python train_web.py
```

You should see:
```
Starting CartPole Q-Learning Web Server
Open your browser and navigate to:
  → http://localhost:8000
```

### Step 3: Open Your Browser
- Open Chrome, Firefox, or Edge
- Navigate to: `http://localhost:8000`

### Step 4: Start Training
- Click the blue "Start Training" button
- Watch the CartPole animate on the canvas
- See statistics update in real-time

### Controls:
- **Start Training** - Begin a new training session
- **Stop** - Stop current training
- Close browser tab - Training stops
- **Ctrl+C** in PowerShell - Stop server

---

## 📊 Method 3: Evaluate Without Visualization (Fastest)

### For Quick Testing Without Graphics:

```powershell
python evaluate_agent.py
```

This will:
1. Train an agent for 500 episodes (no graphics)
2. Evaluate performance over 100 test episodes
3. Print detailed statistics
4. Save the trained agent to `trained_cartpole_agent.pkl`

### Example Output:
```
Training agent for 500 episodes...
Episode 50/500 | Avg Reward: 45.2 | Epsilon: 0.778
Episode 100/500 | Avg Reward: 89.4 | Epsilon: 0.606
...
Training complete! Final average: 387.5

EVALUATION RESULTS
Mean Reward:    412.34 ± 78.23
Success Rate:   94.0%
```

---

## 🚀 Method 4: Compare with Policy Gradient

### To See How Deep RL Differs:

```powershell
python policy_gradient.py
```

### What It Does:
1. Displays side-by-side comparison table
2. Asks if you want to train a policy gradient agent
3. Type `y` and press Enter to train
4. Watch neural network learn (no discretization!)

### Key Learning:
- See Q-learning (table) vs Policy Gradient (neural network)
- Understand when to use each approach
- Learn migration path to deep RL

---

## 🔧 Customization Guide

### Make It Learn Faster:
Edit `q_agent.py`, line ~165:
```python
learning_rate=0.2,  # Change from 0.1 to 0.2
```

### Make Visualization Slower (Easier to Watch):
Edit `train_pygame.py`, line ~287:
```python
render_delay=20  # Change from 0 to 20 (milliseconds)
```

### Train for Longer:
Edit `train_pygame.py`, line ~285:
```python
num_episodes=1000,  # Change from 500 to 1000
```

### Finer State Discretization:
Edit `q_agent.py`, line ~165:
```python
n_bins=(10, 10, 20, 20),  # More bins = finer detail
```

---

## 📁 Understanding the Output

### Console Messages:
```
Episode 100/500 | Reward: 234.0 | Avg(100): 187.3 | Epsilon: 0.606
```
- **Episode**: Current episode number
- **Reward**: Steps survived this episode
- **Avg(100)**: Average over last 100 episodes
- **Epsilon**: Exploration rate (decreases over time)

### Visual Indicators:

**Pygame Window:**
- Green pole = stable (small angle)
- Red pole = unstable (large angle)
- Left panel = current stats
- Right panel = Q-table statistics

**Web Dashboard:**
- Green text = good performance
- Yellow text = moderate
- Red text = poor
- Pulsing dot = actively training

---

## 🐛 Troubleshooting

### Problem: "Module not found" error
**Solution:**
```powershell
pip install -r requirements.txt --force-reinstall
```

### Problem: Pygame window is black or frozen
**Solution:**
```powershell
pip install --upgrade pygame
# Then restart Python script
```

### Problem: Web page won't load
**Solution:**
1. Check console for errors
2. Try different port:
   ```powershell
   # Edit train_web.py, last line, change 8000 to 8001
   uvicorn.run(app, host="0.0.0.0", port=8001)
   ```
3. Open `http://localhost:8001`

### Problem: Training seems stuck
**Solution:**
- This is normal early on! Pole falls quickly.
- Wait for 50+ episodes to see improvement
- Check epsilon is decreasing (means it's learning)

### Problem: Python version error
**Solution:**
```powershell
python --version  # Should be 3.8 or higher
```
If too old, download from python.org

---

## 📈 What Success Looks Like

### Episode 1-50: Learning Basics
- Reward: 10-50 per episode
- Pole falls quickly
- Random movements
- Epsilon: ~1.0 to 0.7

### Episode 50-200: Getting Better
- Reward: 50-150 per episode
- Occasional good balancing
- Less random movements
- Epsilon: ~0.7 to 0.3

### Episode 200-500: Mastery!
- Reward: 200-500 per episode
- Pole stays up most of the time
- Smooth, controlled movements
- Epsilon: ~0.3 to 0.01

### "Solved" = Average Reward ≥ 195 over 100 episodes

---

## 🎓 Learning Checklist

After running the simulations, you should understand:

- [x] What Q-learning is (value-based RL)
- [x] Why we discretize states (for Q-table)
- [x] How epsilon-greedy works (exploration vs exploitation)
- [x] What the Bellman equation does (Q-value updates)
- [x] How to visualize RL training
- [x] Differences between Q-learning and policy gradient

---

## 🎯 Next Steps

1. **Run Pygame version** - See it work!
2. **Try web version** - Beautiful UI
3. **Read q_agent.py** - Understand the code
4. **Modify hyperparameters** - Experiment!
5. **Compare with policy gradient** - Learn deep RL
6. **Read README.md** - Full theory explanation

---

## 💡 Pro Tips

- **Start simple**: Run Pygame first
- **Read comments**: Every line is explained
- **Experiment**: Change hyperparameters
- **Be patient**: Learning takes 200+ episodes
- **Compare methods**: Q-learning vs Policy Gradient
- **Save agents**: Use evaluate_agent.py to persist

---

## 🆘 Need Help?

1. Check `README.md` for full documentation
2. Read code comments - they explain everything
3. Review `QUICKSTART.md` for quick reference
4. Check `COMMANDS.txt` for all commands
5. Look at `PROJECT_STRUCTURE.md` for overview

---

**You're all set! Pick a method and start watching your AI learn! 🚀**

**Recommended First Run:**
```powershell
cd C:\cartpole-qlearning
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_pygame.py
```

Then just watch the magic happen! ✨
