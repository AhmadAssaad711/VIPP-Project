import gymnasium as gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------
# Q-Network
# --------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# --------------------
# Hyperparameters
# --------------------
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 100_000
MIN_BUFFER = 1_000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 1000
MAX_EPISODES = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Environment
# --------------------
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# --------------------
# Networks
# --------------------
q_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=LR)
replay_buffer = deque(maxlen=BUFFER_SIZE)

# --------------------
# Helpers
# --------------------
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return q_net(state).argmax(dim=1).item()

def sample_batch():
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    return (
        torch.tensor(states, dtype=torch.float32, device=device),
        torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1),
        torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
        torch.tensor(next_states, dtype=torch.float32, device=device),
        torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
    )

# --------------------
# Training Loop
# --------------------
epsilon = EPS_START
step_count = 0
episode_rewards = []

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward
        step_count += 1

        if len(replay_buffer) >= MIN_BUFFER:
            states, actions, rewards, next_states, dones = sample_batch()

            q_values = q_net(states).gather(1, actions)

            with torch.no_grad():
                max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
                targets = rewards + GAMMA * max_next_q * (1 - dones)

            loss = nn.MSELoss()(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step_count % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    episode_rewards.append(episode_reward)

    print(f"Episode {episode:3d} | Reward: {episode_reward:5.1f} | Epsilon: {epsilon:.3f}")

env.close()

# --------------------
# Plot Learning Curve
# --------------------
plt.figure()
plt.plot(episode_rewards, label="Episode reward")

# Moving average for clarity
window = 20
if len(episode_rewards) >= window:
    moving_avg = np.convolve(
        episode_rewards,
        np.ones(window) / window,
        mode="valid"
    )
    plt.plot(range(window - 1, MAX_EPISODES), moving_avg, label="Moving average")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Learning Curve on CartPole")
plt.legend()
plt.show()
