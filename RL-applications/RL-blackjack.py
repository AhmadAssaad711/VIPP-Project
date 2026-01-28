from collections import defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================
# Agent definition
# =====================
class Agent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.gamma = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        self.episode_rewards = []
        self.episode_lengths = []

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_values[state])

    def update(self, state, action, reward, next_state, done):
        future_q = 0.0 if done else np.max(self.q_values[next_state])
        td_error = reward + self.gamma * future_q - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_error
        self.training_error.append(abs(td_error))

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# =====================
# Environment + params
# =====================
env = gym.make("Blackjack-v1")

n_episodes = 100_000
learning_rate = 0.01
initial_epsilon = 1.0
final_epsilon = 0
epsilon_decay = initial_epsilon / (n_episodes / 2)
discount_factor = 0.95

agent = Agent(
    env,
    learning_rate,
    initial_epsilon,
    epsilon_decay,
    final_epsilon,
    discount_factor,
)

# =====================
# Training loop
# =====================
for episode in tqdm(range(n_episodes)):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        steps += 1

    agent.episode_rewards.append(total_reward)
    agent.episode_lengths.append(steps)
    agent.decay_epsilon()

env.close()

# =====================
# Plotting utilities
# =====================
def moving_average(data, window):
    return np.convolve(data, np.ones(window) / window, mode="valid")

window = 500

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Reward curve
axs[0].plot(moving_average(agent.episode_rewards, window))
axs[0].set_title("Average Reward")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")

# Episode length
axs[1].plot(moving_average(agent.episode_lengths, window))
axs[1].set_title("Episode Length")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Steps")

# TD error
axs[2].plot(moving_average(agent.training_error, window))
axs[2].set_title("TD Error")
axs[2].set_xlabel("Update step")
axs[2].set_ylabel("Error")

plt.tight_layout()
plt.show()

# =====================
# Evaluation (no learning)
# =====================
def test_agent(agent, env, n_eval_episodes=5000):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    rewards = []

    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    agent.epsilon = old_epsilon

    rewards = np.array(rewards)
    print("Evaluation results")
    print(f"Win rate: {(rewards > 0).mean():.2%}")
    print(f"Average reward: {rewards.mean():.3f}")
    print(f"Std dev: {rewards.std():.3f}")


test_agent(agent, gym.make("Blackjack-v1"))
