import gymnasium as gym
import numpy as np

# ---------- Environment ----------
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n

# ---------- Q-table ----------
Q = np.zeros((n_states, n_actions))

# ---------- Hyperparameters ----------
alpha = 0.6        # learning rate
gamma = 0.95        # discount factor
epsilon = 1.0        # exploration
epsilon_min = 0.01
episodes = 1000000

epsilon_decay = (epsilon_min / epsilon) ** (1 / episodes)

# ---------- Training ----------
for _ in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# ---------- Evaluation (no exploration) ----------
successes = 0
eval_episodes = 200

for _ in range(eval_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        successes += reward

env.close()
print(f"Success rate over {eval_episodes} episodes: {successes / eval_episodes:.2f}")
print(np.matrix(Q))

