# CartPole Q-Learning Project Structure

```
cartpole-qlearning/
│
├── 📘 Documentation
│   ├── README.md           ⭐ Complete guide with theory & examples
│   ├── QUICKSTART.md       ⚡ Get running in 3 steps
│   └── COMMANDS.txt        📝 All commands reference
│
├── 🧠 Core Q-Learning
│   └── q_agent.py          🎯 Q-learning implementation
│                              - State discretization
│                              - Epsilon-greedy policy
│                              - Q-table updates
│                              - Bellman equation
│
├── 🎮 Visualization Options
│   ├── train_pygame.py     🖥️  Desktop window (Pygame)
│   │                          - 60 FPS animation
│   │                          - Real-time stats
│   │                          - ESC to quit
│   │
│   └── train_web.py        🌐 Web dashboard (FastAPI)
│                              - Browser-based
│                              - WebSocket streaming
│                              - Beautiful UI
│                              - Start/Stop controls
│
├── 🔬 Analysis & Comparison
│   ├── evaluate_agent.py   📊 Benchmark performance
│   │                          - Train without viz
│   │                          - Evaluation metrics
│   │                          - Save/load agents
│   │
│   └── policy_gradient.py  🚀 Deep RL comparison
│                              - Neural network policy
│                              - REINFORCE algorithm
│                              - Side-by-side comparison
│                              - Migration guide
│
└── ⚙️  Configuration
    └── requirements.txt    📦 Python dependencies
                               - gymnasium
                               - numpy
                               - pygame
                               - fastapi/uvicorn
                               - torch
```

## 🔄 Workflow

```
     ┌─────────────────┐
     │  Pick Version   │
     └────────┬────────┘
              │
      ┌───────┴────────┐
      │                │
      ▼                ▼
┌──────────┐    ┌──────────┐
│  Pygame  │    │   Web    │
│ Desktop  │    │ Browser  │
└──────────┘    └──────────┘
      │                │
      └────────┬───────┘
               │
               ▼
      ┌────────────────┐
      │  Q-Learning    │
      │  Agent Trains  │
      └────────┬───────┘
               │
               ▼
      ┌────────────────┐
      │  Pole Learns   │
      │  to Balance!   │
      └────────────────┘
```

## 📚 Learning Path

```
Start Here → QUICKSTART.md
    │
    ├─→ Run train_pygame.py
    │       │
    │       └─→ See it work!
    │
    ├─→ Read q_agent.py
    │       │
    │       └─→ Understand Q-learning
    │
    ├─→ Read README.md
    │       │
    │       └─→ Deep dive theory
    │
    └─→ Study policy_gradient.py
            │
            └─→ Learn deep RL
```

## 🎯 Key Files Explained

### q_agent.py (Core Implementation)
- `QLearningAgent` class
- State discretization logic
- Q-table (numpy array)
- Epsilon-greedy action selection
- Q-value updates (Bellman equation)

### train_pygame.py (Desktop Viz)
- `CartPoleVisualizer` class
- Pygame rendering loop
- Real-time training display
- Statistics overlay

### train_web.py (Web Viz)
- FastAPI server
- WebSocket endpoint
- HTML/CSS/JS frontend
- Canvas rendering
- Live stat updates

### evaluate_agent.py (Analysis)
- Training without visualization
- Performance benchmarking
- Agent save/load functionality
- Statistical analysis

### policy_gradient.py (Comparison)
- PyTorch neural network
- REINFORCE algorithm
- Direct policy learning
- Migration guide from Q-learning

## 🔧 Customization Points

```python
# In q_agent.py
agent = QLearningAgent(
    n_bins=(6, 6, 12, 12),    # ← Increase for finer discretization
    learning_rate=0.1,         # ← Higher = faster learning
    discount_factor=0.99,      # ← Higher = more future-focused
    epsilon_start=1.0,         # ← Start exploration rate
    epsilon_decay=0.995        # ← Lower = slower decay
)

# In train_pygame.py
train_with_visualization(
    num_episodes=500,          # ← More episodes = better learning
    render_delay=0             # ← Higher = slower visualization
)
```

## 🌟 Features Comparison

| Feature | Pygame | Web | Evaluate | Policy Gradient |
|---------|--------|-----|----------|----------------|
| Real-time viz | ✅ | ✅ | ❌ | ❌ |
| Fast training | ⚠️ | ⚠️ | ✅ | ✅ |
| Share easily | ❌ | ✅ | ❌ | ❌ |
| Q-learning | ✅ | ✅ | ✅ | ❌ |
| Deep RL | ❌ | ❌ | ❌ | ✅ |
| Beginner-friendly | ✅ | ⚠️ | ⚠️ | ❌ |

## 🎓 What Each File Teaches

- **q_agent.py**: Tabular Q-learning, discretization, exploration
- **train_pygame.py**: Pygame basics, game loops, rendering
- **train_web.py**: FastAPI, WebSockets, async Python, HTML5 Canvas
- **evaluate_agent.py**: Performance metrics, agent persistence
- **policy_gradient.py**: PyTorch, neural networks, policy methods

## 🚀 Quick Commands

```powershell
# Setup
pip install -r requirements.txt

# Run Pygame
python train_pygame.py

# Run Web (then open http://localhost:8000)
python train_web.py

# Evaluate
python evaluate_agent.py

# Compare with Policy Gradient
python policy_gradient.py
```

---

**Choose your path and start learning! 🎯**
