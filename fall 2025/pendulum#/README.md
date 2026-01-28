# 🎮 CartPole Q-Learning: Real-Time RL Visualization

A complete implementation of **Q-learning** (tabular reinforcement learning) for the CartPole environment, with **real-time visualization** of the agent learning to balance an inverted pendulum.

Watch the AI learn from scratch — see the pole wobble chaotically at first, then gradually stabilize as the Q-table converges!

## 🌟 Features

- ✅ **Pure Q-learning** (no deep neural networks)
- ✅ **Discretized state space** with configurable bins
- ✅ **Real-time visualization** as training progresses
- ✅ **Two visualization modes**:
  - **Pygame**: Desktop application with smooth 60 FPS rendering
  - **Web**: Browser-based dashboard with FastAPI + WebSocket streaming
- ✅ **Complete explanations** of every Q-learning component
- ✅ **Epsilon-greedy exploration** with automatic decay
- ✅ **Live training statistics** (rewards, Q-values, state variables)

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Project Structure](#-project-structure)
3. [Q-Learning Explained](#-q-learning-explained)
4. [Running the Simulations](#-running-the-simulations)
5. [Understanding the Code](#-understanding-the-code)
6. [Migrating to Policy Gradient](#-migrating-to-policy-gradient)
7. [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Windows, macOS, or Linux

### Installation

```powershell
# Clone or navigate to the project directory
cd cartpole-qlearning

# Create a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# Or: source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run Pygame Version (Desktop)

```powershell
python train_pygame.py
```

A window will open showing the CartPole learning in real-time!

### Run Web Version (Browser)

```powershell
python train_web.py
```

Then open your browser to: **http://localhost:8000**

The web dashboard supports selecting the agent and the number of episodes before starting training. The server streams real-time updates over WebSocket and the browser renders the CartPole animation on an HTML5 canvas.

Quick web run (recommended):

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the web server (development mode):

```powershell
# Option A (recommended during development)
uvicorn train_web:app --reload --host 127.0.0.1 --port 8000

# Option B (runs the module directly)
python train_web.py
```

4. Open the dashboard in your browser:

```
http://127.0.0.1:8000
```

5. Use the UI:
- Choose the agent from the dropdown: `Q-Learning`, `Policy Gradient`, or `Model-Based (LQR)`.
- Set `Episodes` to control how long the server runs the training/evaluation.
- Click **Start** to begin streaming; click **Stop** to end.

Quick smoke test (command-line WebSocket client):

```powershell
# Runs a 1-episode Q-Learning test and prints received JSON messages
python _test_ws_client.py
```

Notes and tips:
- For a fast demo: select **Model-Based (LQR)** and set episodes to `10` — LQR is deterministic and will show near-perfect balancing immediately.
- Q-Learning and Policy Gradient need many episodes to show learning; start with 50–200 episodes for visible improvement.
- If the server fails to start, ensure the port `8000` is free or change the port with `--port <n>`.
- On Windows, if you get weird Unicode console errors, set the environment variable before running: `$env:PYTHONIOENCODING = 'utf-8'`.
- The web dashboard does not require `pygame`; that is only used by the desktop visualizer.

---

## 📁 Project Structure

```
cartpole-qlearning/
│
├── `q_agent.py`                # Core Q-learning agent with state discretization
├── `q_learning.py`             # Higher-level training loops / helpers for Q-learning
├── `policy_gradient_simple.py` # Simple NumPy Policy-Gradient implementation (used by web UI)
├── `policy_gradient.py`        # Additional PG training scripts and helpers
├── `model_based.py`            # Model-based (LQR) controller and evaluation
├── `train_pygame.py`           # Pygame visualization (desktop window)
├── `train_web.py`              # FastAPI + WebSocket server (web dashboard)
├── `run_animation.py`          # Convenience desktop launcher for Pygame animations
├── `_test_ws_client.py`        # Small test WebSocket client for smoke tests
├── `requirements.txt`          # Python dependencies
└── `README.md`                 # This file
```

---

## 🧠 Q-Learning Explained

### What is Q-Learning?

**Q-learning** is a **model-free** reinforcement learning algorithm that learns the **value** of taking actions in states without needing a model of the environment.

#### Key Concepts

1. **Q-Table**: A table storing Q-values `Q(s, a)` for each state-action pair
   - Represents expected cumulative reward from taking action `a` in state `s`

2. **Epsilon-Greedy Exploration**:
   - With probability `ε`: Choose random action (explore)
   - With probability `1-ε`: Choose best action from Q-table (exploit)
   - `ε` decays over time: explore early, exploit later

3. **Q-Update Rule** (Bellman Equation):
   ```
   Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
   ```
   - `α` (learning_rate): How fast we update (0.1 = gradual, 1.0 = aggressive)
   - `r` (reward): Immediate reward received (+1 per step in CartPole)
   - `γ` (discount_factor): How much we value future rewards (0.99 = far-sighted)
   - `max Q(s',a')`: Best possible future value from next state

4. **State Discretization**:
   - CartPole has **continuous states** (position, velocity, angle, etc.)
   - Q-learning needs **discrete states** for the Q-table
   - We divide each dimension into bins (buckets)
   
   Example: Angle from -0.418 to +0.418 → 12 bins → bin indices 0-11

### Why Discretization?

Imagine trying to create a table with infinitely many rows — impossible! By discretizing:
- Position (-4.8 to 4.8) → 6 bins
- Velocity (-4 to 4) → 6 bins  
- Angle (-0.418 to 0.418) → 12 bins
- Angular velocity (-4 to 4) → 12 bins

Total Q-table size: `6 × 6 × 12 × 12 × 2 actions = 10,368 values`

---

## 🎯 Running the Simulations

### Option 1: Pygame (Desktop Window)

**Best for**: Quick local testing, smooth 60 FPS animation

```powershell
python train_pygame.py
```

**What you'll see**:
- Cart and pole animating in real-time
- Episode number, step count, total reward
- Epsilon (exploration rate) decreasing
- Average reward over last 100 episodes
- State variables (position, velocity, angle, angular velocity)
- Q-table statistics

**Controls**:
- Press `ESC` or close window to stop training

**Customization** (edit `train_pygame.py`):
```python
rewards = train_with_visualization(
    num_episodes=500,        # Number of training episodes
    max_steps_per_episode=500,  # Max steps per episode
    render_delay=10          # Delay per step in ms (0 = fast, 20 = slow)
)
```

---

### Option 2: Web Dashboard (Browser)

**Best for**: Sharing with others, modern UI, remote viewing

```powershell
python train_web.py
```

Then navigate to: **http://localhost:8000**

**What you'll see**:
- Beautiful gradient UI with responsive design
- HTML5 Canvas rendering of CartPole
- Real-time WebSocket streaming of training data
- Color-coded statistics (green = good, red = bad)
- Progress bar showing episode completion
- Start/Stop controls

**How it works**:
1. FastAPI serves HTML page with Canvas and JavaScript
2. Click "Start Training" → WebSocket connection established
3. Server trains agent and streams state updates
4. Browser renders CartPole and updates stats in real-time

---

## 🔍 Understanding the Code

### 1. Q-Agent (`q_agent.py`)

#### Initialization
```python
agent = QLearningAgent(
    n_bins=(6, 6, 12, 12),      # Discretization granularity
    learning_rate=0.1,           # α: Update speed
    discount_factor=0.99,        # γ: Future reward importance
    epsilon_start=1.0,           # Start with full exploration
    epsilon_min=0.01,            # Min exploration (always explore a bit)
    epsilon_decay=0.995          # Decay per episode
)
```

#### State Discretization
```python
def discretize_state(self, state):
    # Converts [2.3, -1.2, 0.15, 0.8] 
    # Into bin indices like (4, 2, 8, 6)
    discrete_state = []
    for i, value in enumerate(state):
        clipped = np.clip(value, self.state_bounds[i][0], self.state_bounds[i][1])
        bin_idx = np.digitize(clipped, self.bin_edges[i]) - 1
        discrete_state.append(bin_idx)
    return tuple(discrete_state)
```

#### Action Selection
```python
def get_action(self, state, explore=True):
    if explore and np.random.random() < self.epsilon:
        return np.random.randint(0, 2)  # Explore
    else:
        return np.argmax(self.q_table[state])  # Exploit
```

#### Q-Table Update
```python
def update(self, state, action, reward, next_state, done):
    current_q = self.q_table[state + (action,)]
    max_future_q = 0 if done else np.max(self.q_table[next_state])
    td_target = reward + self.discount_factor * max_future_q
    td_error = td_target - current_q
    self.q_table[state + (action,)] = current_q + self.learning_rate * td_error
```

---

### 2. Training Loop (Both Versions)

```python
# Reset environment for new episode
state, _ = env.reset()
discrete_state = agent.discretize_state(state)

for step in range(max_steps):
    # 1. Choose action (epsilon-greedy)
    action = agent.get_action(discrete_state, explore=True)
    
    # 2. Take action in environment
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    # 3. Discretize next state
    next_discrete_state = agent.discretize_state(next_state)
    
    # 4. Update Q-table (learn!)
    agent.update(discrete_state, action, reward, next_discrete_state, done)
    
    # 5. Render visualization
    visualizer.render(...)
    
    # 6. Move to next state
    state = next_state
    discrete_state = next_discrete_state
    
    if done:
        break

# Decay exploration for next episode
agent.decay_epsilon()
```

---

### 3. Visualization Details

#### Pygame Rendering
- **Cart**: Blue rectangle with wheels
- **Pole**: Line colored by angle (green = stable, red = unstable)
- **Track**: Black line representing ground
- **Stats**: Text overlays showing all metrics

#### Web Rendering (JavaScript Canvas)
- Similar visual style but rendered in browser
- WebSocket receives JSON updates every 10ms
- Canvas redrawn 60 times per second
- CSS animations for smooth transitions

---

## 🔄 Migrating to Policy Gradient

Ready to upgrade from Q-learning to **policy gradient** methods? Here's how:

### Key Differences

| Aspect | Q-Learning | Policy Gradient |
|--------|-----------|-----------------|
| **What it learns** | Value function Q(s,a) | Policy π(a\|s) directly |
| **State space** | Requires discretization | Handles continuous states |
| **Action space** | Works with discrete actions | Can handle continuous actions |
| **Network** | Q-table (tabular) | Neural network |
| **Update** | Bootstrapping (TD) | Full episode trajectories |

### Implementation Steps

1. **Replace Q-table with Neural Network**:
```python
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # Output action probabilities
        )
    
    def forward(self, state):
        return self.network(state)
```

2. **Replace discretization with raw states**:
```python
# Q-Learning:
discrete_state = agent.discretize_state(state)

# Policy Gradient:
state_tensor = torch.FloatTensor(state)
```

3. **Replace Q-update with Policy Gradient update**:
```python
# REINFORCE algorithm (simplest policy gradient)
def update_policy(episode_states, episode_actions, episode_rewards):
    # Calculate discounted returns
    returns = []
    G = 0
    for r in reversed(episode_rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Calculate policy gradient
    loss = 0
    for state, action, G in zip(episode_states, episode_actions, returns):
        state_tensor = torch.FloatTensor(state)
        probs = policy_network(state_tensor)
        log_prob = torch.log(probs[action])
        loss += -log_prob * G  # Negative because we want gradient ascent
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

4. **Sample actions from policy**:
```python
# Q-Learning:
action = agent.get_action(discrete_state)

# Policy Gradient:
state_tensor = torch.FloatTensor(state)
action_probs = policy_network(state_tensor).detach().numpy()
action = np.random.choice(len(action_probs), p=action_probs)
```

### Advantages of Policy Gradient

- ✅ **No discretization needed** — works with continuous states
- ✅ **Can learn stochastic policies** — useful for partially observable environments
- ✅ **Better for continuous action spaces** — can output real-valued actions
- ✅ **More stable convergence** in some environments

### When to Use Each

- **Q-Learning**: Small state/action spaces, want guaranteed convergence, prefer simplicity
- **Policy Gradient**: Large/continuous spaces, need stochastic policies, have compute resources

---

## 🛠️ Troubleshooting

### Pygame window not opening
- Ensure Pygame is installed: `pip install pygame`
- On some Linux systems: `sudo apt-get install python3-pygame`

### Web server not starting
- Check port 8000 is not in use
- Change port: `uvicorn train_web:app --port 8001`

### Agent not learning (low rewards)
- Increase `num_episodes` (try 1000+)
- Adjust learning rate: try `0.05` or `0.2`
- Increase bins for finer discretization: `n_bins=(10, 10, 20, 20)`
- Check epsilon decay isn't too fast

### Training too slow/fast
- **Pygame**: Adjust `render_delay` parameter
- **Web**: Modify `await asyncio.sleep(0.01)` in `train_web.py`

### ImportError: gymnasium
```powershell
pip install gymnasium
# Or if using older gym:
pip install gym
# Then change imports: gymnasium → gym
```

---

## 📊 Expected Results

After ~200-300 episodes:
- **Average reward**: 150-300 (out of max 500)
- **Epsilon**: ~0.5 (still exploring some)
- **Pole stability**: Noticeably more balanced

After ~500 episodes:
- **Average reward**: 300-500 (near optimal)
- **Epsilon**: ~0.01 (mostly exploiting)
- **Pole stability**: Stays upright for full 500 steps

---

## 📚 Further Reading

- [Sutton & Barto: Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)
- [Q-Learning Tutorial](https://www.freecodecamp.org/news/an-introduction-to-q-learning-reinforcement-learning/)
- [Policy Gradient Methods](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

---

## 🎓 Learning Checklist

After working through this project, you should understand:

- ✅ How Q-learning updates Q-values using the Bellman equation
- ✅ Why and how to discretize continuous state spaces
- ✅ The exploration-exploitation tradeoff (epsilon-greedy)
- ✅ How to visualize RL training in real-time
- ✅ Differences between value-based and policy-based methods
- ✅ When to use Q-learning vs deep RL methods

---

## 🤝 Contributing

Feel free to:
- Add new visualization features
- Implement different RL algorithms (SARSA, DQN, etc.)
- Improve the web UI
- Add more environments (MountainCar, Acrobot, etc.)

---

## 📄 License

MIT License - Feel free to use for learning and teaching!

---

**Enjoy watching your AI learn! 🚀**

Questions? Check the troubleshooting section or review the heavily commented code.
