# Quick Start Guide

## 🚀 Get Running in 3 Steps

### Step 1: Install Dependencies

Open PowerShell and run:

```powershell
# Navigate to the project
cd cartpole-qlearning

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### Step 2: Choose Your Visualization

**Option A: Pygame (Desktop Window)**
```powershell
python train_pygame.py
```

**Option B: Web Dashboard (Browser)**
```powershell
python train_web.py
# Then open: http://localhost:8000
```

### Step 3: Watch the Magic! ✨

You'll see the pole wobbling randomly at first, then gradually stabilizing as the agent learns!

---

## 📁 What Each File Does

| File | Purpose |
|------|---------|
| `q_agent.py` | Core Q-learning implementation |
| `train_pygame.py` | Desktop visualization with Pygame |
| `train_web.py` | Web dashboard with FastAPI + WebSocket |
| `evaluate_agent.py` | Train and evaluate without visualization |
| `policy_gradient.py` | Comparison with policy gradient methods |
| `README.md` | Complete documentation |

---

## 🎯 What You'll Learn

1. **Q-Learning Fundamentals**: State discretization, Q-tables, Bellman equation
2. **Epsilon-Greedy Exploration**: Balance between exploring and exploiting
3. **Real-Time Visualization**: See learning happen step-by-step
4. **Policy Gradients**: How to migrate from Q-learning to deep RL

---

## 💡 Tips

- **First time?** Start with Pygame version - it's simpler and faster
- **Want to share?** Use web version - works on any device with a browser
- **Learning slowly?** Increase `num_episodes` to 1000+ in the training scripts
- **Too fast to see?** Add `render_delay=20` in Pygame version

---

## 🐛 Troubleshooting

**Pygame window won't open?**
```powershell
pip install --upgrade pygame
```

**Web server won't start?**
```powershell
pip install --upgrade fastapi uvicorn
```

**ImportError with gymnasium?**
```powershell
pip install gymnasium
```

**Want to see policy gradient?**
```powershell
python policy_gradient.py
```

---

## 📚 Next Steps

1. Run `evaluate_agent.py` to benchmark performance
2. Experiment with hyperparameters in `q_agent.py`
3. Study `policy_gradient.py` to understand deep RL transition
4. Read full documentation in `README.md`

---

**Happy Learning! 🎓**
