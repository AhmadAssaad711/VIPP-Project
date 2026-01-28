



import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

episodes = 12000
eval_episodes = 200
epsilon_start = 1.0
epsilon_min = 0.01

results = []

for x in range(1, 10):
    for y in range(1, 10):
        learning_rate = x * 0.1
        discount_factor = y * 0.1

        # Reset agent
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        epsilon = epsilon_start
        epsilon_decay = (epsilon_min / epsilon_start) ** (1 / episodes)

        # ---- Training ----
        for _ in range(episodes):
            state, _ = env.reset()
            done = False

            while not done:
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state])

                next_state, reward, terminated, truncated, _ = env.step(action)

                Q[state, action] += learning_rate * (
                    reward + discount_factor * np.max(Q[next_state]) - Q[state, action]
                )

                state = next_state
                done = terminated or truncated

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # ---- Evaluation (no exploration) ----
        total_reward = 0
        for _ in range(eval_episodes):
            state, _ = env.reset()
            done = False

            while not done:
                action = np.argmax(Q[state])
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

        avg_reward = total_reward / eval_episodes
        results.append((avg_reward, learning_rate, discount_factor))
        
    print(f"Completed learning rate {learning_rate} for discount factors.")

env.close()

# ---- Plot ----
avg_rewards = [r[0] for r in results]
learning_rates = [r[1] for r in results]
discount_factors = [r[2] for r in results]

fig, ax1 = plt.subplots()

ax1.set_xlabel("Average Reward")
ax1.set_ylabel("Learning Rate", color="tab:blue")
ax1.scatter(avg_rewards, learning_rates, color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Discount Factor", color="tab:red")
ax2.scatter(avg_rewards, discount_factors, color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.title("FrozenLake Hyperparameter Grid Search")
plt.show()
