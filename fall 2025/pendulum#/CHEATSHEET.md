# 🎮 CartPole Q-Learning Cheat Sheet

## 🚀 Quick Commands

```powershell
# Setup (once)
pip install -r requirements.txt

# Run Pygame (Desktop)
python train_pygame.py

# Run Web (Browser)
python train_web.py
# → Open http://localhost:8000

# Evaluate (Fast, no graphics)
python evaluate_agent.py

# Policy Gradient Comparison
python policy_gradient.py
```

---

## 🧠 Q-Learning Formula

**Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]**

- **α (learning_rate)**: How fast to learn (0.1)
- **γ (discount_factor)**: Future reward importance (0.99)
- **r**: Immediate reward (+1 per step)
- **max Q(s',a')**: Best future value

---

## 📊 State Space (CartPole)

| Variable | Range | Bins | Meaning |
|----------|-------|------|---------|
| Position | -4.8 to 4.8 | 6 | Cart location |
| Velocity | -4.0 to 4.0 | 6 | Cart speed |
| Angle | -0.418 to 0.418 | 12 | Pole angle (rad) |
| Ang. Vel | -4.0 to 4.0 | 12 | Pole rotation speed |

**Total Q-table size**: 6×6×12×12×2 = 10,368 values

---

## 🎯 Actions

- **0**: Push cart LEFT
- **1**: Push cart RIGHT

---

## ⚙️ Hyperparameters

```python
# In q_agent.py
n_bins = (6, 6, 12, 12)    # State discretization
learning_rate = 0.1         # α (0.05-0.2)
discount_factor = 0.99      # γ (0.95-0.99)
epsilon_start = 1.0         # Full exploration
epsilon_min = 0.01          # Always explore a bit
epsilon_decay = 0.995       # Per-episode decay
```

---

## 📈 Training Progress

| Episodes | Avg Reward | Epsilon | Status |
|----------|-----------|---------|--------|
| 1-50 | 10-50 | 1.0→0.7 | Learning basics |
| 50-200 | 50-150 | 0.7→0.3 | Improving |
| 200-500 | 200-500 | 0.3→0.01 | Mastery! |

**Solved**: Avg ≥ 195 over 100 episodes

---

## 🎨 Visualization Colors

**Pygame & Web:**
- 🟢 Green pole = Stable (small angle)
- 🟡 Yellow pole = Medium angle
- 🔴 Red pole = Unstable (large angle)

---

## 🔧 Common Tweaks

**Learn faster:**
```python
learning_rate = 0.2  # ↑ from 0.1
```

**Finer discretization:**
```python
n_bins = (10, 10, 20, 20)  # ↑ from (6,6,12,12)
```

**More exploration:**
```python
epsilon_decay = 0.999  # ↑ from 0.995
```

**Slower visualization:**
```python
render_delay = 20  # in train_pygame.py
```

---

## 📁 File Quick Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `q_agent.py` | Core Q-learning | Study algorithm |
| `train_pygame.py` | Desktop viz | Quick local test |
| `train_web.py` | Web dashboard | Share/demo |
| `evaluate_agent.py` | Benchmarking | Fast training |
| `policy_gradient.py` | Deep RL | Learn migration |

---

## 🐛 Quick Fixes

**Module not found:**
```powershell
pip install -r requirements.txt
```

**Pygame won't open:**
```powershell
pip install --upgrade pygame
```

**Web won't start:**
```powershell
pip install --upgrade fastapi uvicorn
```

**Port in use:**
```powershell
# Change port 8000 to 8001 in train_web.py
```

---

## 🎓 Key Concepts

**Exploration vs Exploitation:**
- Exploration = Try random actions (epsilon)
- Exploitation = Use best known action
- Epsilon decays: explore early, exploit later

**Discretization:**
- Q-learning needs discrete states
- Continuous → bins (buckets)
- Trade-off: more bins = finer detail, bigger table

**Epsilon-Greedy:**
- Random action with probability ε
- Best action with probability 1-ε
- Balances learning new vs using knowledge

---

## 🔄 Q-Learning vs Policy Gradient

| Aspect | Q-Learning | Policy Gradient |
|--------|-----------|----------------|
| Learns | Q(s,a) values | π(a\|s) policy |
| States | Discrete | Continuous OK |
| Network | None (table) | Neural network |
| Update | Every step | Per episode |
| Speed | Faster | Slower |
| Scale | Small spaces | Large spaces |

---

## 💡 Pro Tips

1. **Start with Pygame** - Easiest to run
2. **Watch 200+ episodes** - Real learning takes time
3. **Read code comments** - Everything is explained
4. **Experiment** - Change hyperparameters
5. **Compare** - Q-learning vs policy gradient
6. **Be patient** - RL takes time to converge

---

## 📊 What to Monitor

**Good signs:**
- ✅ Average reward increasing
- ✅ Epsilon decreasing
- ✅ Pole staying upright longer
- ✅ Q-values growing

**Bad signs:**
- ❌ Reward stuck at 10-20
- ❌ Epsilon not decreasing
- ❌ No improvement after 100+ episodes

**Fixes:**
- Increase learning rate
- More bins
- Slower epsilon decay
- More episodes

---

## 🎯 Success Metrics

**Episode 100:**
- Target: Avg reward > 50

**Episode 200:**
- Target: Avg reward > 100

**Episode 500:**
- Target: Avg reward > 200
- Goal: Avg reward > 400 (excellent!)

---

## 🔍 Debugging Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All packages installed
- [ ] In correct directory
- [ ] No syntax errors in code
- [ ] Port 8000 available (web version)

---

## 📚 Files to Read (In Order)

1. **HOW_TO_RUN.md** ← Start here!
2. **QUICKSTART.md** ← Quick reference
3. **q_agent.py** ← Core algorithm
4. **README.md** ← Full documentation
5. **policy_gradient.py** ← Deep RL

---

## 🎬 Recommended Flow

```
1. Read HOW_TO_RUN.md
   ↓
2. Run train_pygame.py
   ↓
3. Watch training for 200+ episodes
   ↓
4. Read q_agent.py code
   ↓
5. Try train_web.py
   ↓
6. Read README.md theory
   ↓
7. Compare with policy_gradient.py
   ↓
8. Experiment with hyperparameters!
```

---

**Keep this cheat sheet handy while coding! 📋✨**
