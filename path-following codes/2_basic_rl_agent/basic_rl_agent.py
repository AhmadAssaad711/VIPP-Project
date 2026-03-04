"""
Basic RL Agent — Q-Learning with (e_y, e_psi) State
=====================================================
Tabular Q-learning agent that discretizes lateral error and heading
error to learn a steering policy on the highway-env racetrack.
"""

import matplotlib
matplotlib.use("Agg")

import gymnasium as gym
import highway_env  # noqa
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os

# =======================
# OUTPUT DIRECTORY
# =======================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "basic_rl_agent")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =======================
# ENV CONFIG
# =======================
config = {
    "controlled_vehicles": 1,
    "other_vehicles": 0,
    "terminate_off_road": True,
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "heading"],
        "absolute": True,
    },
}

env = gym.make("racetrack-v0", config=config)

# =======================
# ACTION SPACE
# =======================
STEER_MIN, STEER_MAX = -0.5, 0.5
N_ACTIONS = 21
ACTIONS = np.linspace(STEER_MIN, STEER_MAX, N_ACTIONS)

# =======================
# DISCRETIZATION (POSITION + HEADING)
# =======================
EY_MAX = 2.0
EPSI_MAX = np.deg2rad(45)

N_EY = 20
N_EPSI = 20

e_y_bins = np.linspace(-EY_MAX, EY_MAX, N_EY)
e_psi_bins = np.linspace(-EPSI_MAX, EPSI_MAX, N_EPSI)

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def discretize_obs():
    vehicle = env.unwrapped.vehicle
    lane = vehicle.lane

    s, e_y = lane.local_coordinates(vehicle.position)
    psi_lane = lane.heading_at(s)
    e_psi = wrap_angle(vehicle.heading - psi_lane)

    e_y = np.clip(e_y, -EY_MAX, EY_MAX)
    e_psi = np.clip(e_psi, -EPSI_MAX, EPSI_MAX)

    ey_bin = np.digitize(e_y, e_y_bins)
    epsi_bin = np.digitize(e_psi, e_psi_bins)

    return (ey_bin, epsi_bin), e_y, e_psi

# =======================
# Q TABLE
# =======================
Q = defaultdict(lambda: np.zeros(N_ACTIONS))

# =======================
# HYPERPARAMETERS
# =======================
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.05
episodes = 5000

lambda_psi = 0.5
lambda_jerk = 0.1
alive_reward = 0.2

# =======================
# TRAINING METRICS
# =======================
window = 50
avg_returns = []
avg_steps = []
avg_error = []

ret_buf = []
step_buf = []
err_buf = []

# =======================
# TRAINING LOOP
# =======================
import sys, traceback

try:
    for ep in range(1, episodes + 1):
        env.reset()
        s, _, _ = discretize_obs()

        done = False
        prev_action = 0.0
        total_reward = 0.0
        error_sum = 0.0
        steps = 0

        while not done:
            if np.random.rand() < epsilon:
                a_idx = np.random.randint(N_ACTIONS)
            else:
                a_idx = np.argmax(Q[s])

            action = ACTIONS[a_idx]

            _, _, terminated, truncated, _ = env.step([action])
            done = terminated or truncated

            s_next, e_y, e_psi = discretize_obs()

            norm_error = (e_y / EY_MAX) ** 2 + lambda_psi * (e_psi / EPSI_MAX) ** 2
            jerk_cost = (action - prev_action) ** 2
            reward = alive_reward - norm_error

            Q[s][a_idx] += alpha * (
                reward + gamma * np.max(Q[s_next]) - Q[s][a_idx]
            )

            s = s_next
            prev_action = action
            total_reward += reward
            error_sum += norm_error
            steps += 1

            if steps >= 200:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        ret_buf.append(total_reward)
        step_buf.append(steps)
        err_buf.append(error_sum / max(steps, 1))

        if ep % window == 0:
            avg_returns.append(np.mean(ret_buf))
            avg_steps.append(np.mean(step_buf))
            avg_error.append(np.mean(err_buf))
            ret_buf.clear(); step_buf.clear(); err_buf.clear()

            print(
                f"Ep {ep} | "
                f"return {avg_returns[-1]:.2f} | "
                f"steps {avg_steps[-1]:.1f} | "
                f"epsilon {epsilon:.2f}",
                flush=True
            )

except Exception as e:
    print(f"\n*** CRASH at ep {ep}, step {steps}: {e}", flush=True)
    traceback.print_exc()

env.close() # policy evaluation (uncomment to visualise) # ======================= # TRAINING CURVES # =======================
x = np.arange(len(avg_steps)) * window

plt.figure(figsize=(8, 4))
plt.plot(x, avg_steps)
plt.xlabel("Episodes")
plt.ylabel("Avg Steps")
plt.title("Steps vs Training (e_y, e_psi)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "steps_vs_training.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(x, avg_returns)
plt.xlabel("Episodes")
plt.ylabel("Avg Return")
plt.title("Reward vs Training (e_y, e_psi)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "reward_vs_training.png"), dpi=300)
plt.close()

# =======================
# ERROR PER STEP
# =======================
plt.figure(figsize=(8, 4))
plt.plot(x, avg_error)
plt.xlabel("Episodes")
plt.ylabel("Avg Normalized Error/Step")
plt.title("Error per Step vs Training (e_y, e_psi)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "error_per_step_vs_training.png"), dpi=300)
plt.close()

# =======================
# POLICY HEATMAP
# =======================
policy_map = np.zeros((N_EY, N_EPSI))

for i in range(N_EY):
    for j in range(N_EPSI):
        state = (i + 1, j + 1)
        if state in Q:
            policy_map[i, j] = ACTIONS[np.argmax(Q[state])]

plt.figure(figsize=(7, 5))
plt.imshow(
    policy_map,
    origin="lower",
    extent=[
        np.rad2deg(-EPSI_MAX),
        np.rad2deg(EPSI_MAX),
        -EY_MAX,
        EY_MAX,
    ],
    aspect="auto",
)
plt.colorbar(label="Steering [rad]")
plt.xlabel("Heading Error e_psi [deg]")
plt.ylabel("Lateral Error e_y [m]")
plt.title("Learned Policy: Steering vs (e_y, e_psi)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "policy_heatmap.png"), dpi=300)
plt.close()

print("\nTraining complete. Plots saved to", RESULTS_DIR)
