"""
Physics-Informed RL Agent - Q-Learning with (e_y, e_psi, kappa_err) State
==========================================================================
Tabular Q-learning agent that adds curvature error (kappa_err) to the
state representation. Curvature is defined as the rate of change of
heading over an arc-length window.
"""

import matplotlib
matplotlib.use("Agg")

import gymnasium as gym
import highway_env  # noqa
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

# =======================
# OUTPUT DIR
# =======================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "physics_informed_agent")
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

env = gym.make("racetrack-v0", config=config, render_mode=None)
env.reset()

vehicle = env.unwrapped.vehicle
CAR_LENGTH = vehicle.LENGTH
CURVATURE_DS = 2.0 * CAR_LENGTH   # arc-length window

# =======================
# ACTION SPACE (curvature commands)
# =======================
N_ACTIONS = 11
KAPPA_CMD_MAX = 0.2
ACTIONS = np.linspace(-KAPPA_CMD_MAX, KAPPA_CMD_MAX, N_ACTIONS)

# =======================
# STATE DISCRETIZATION
# =======================
EY_MAX = 2.0
EPSI_MAX = np.deg2rad(45)
KAPPA_MAX = 0.2

N_EY = 10
N_EPSI = 10
N_KAPPA = 10

e_y_bins = np.linspace(-EY_MAX, EY_MAX, N_EY)
e_psi_bins = np.linspace(-EPSI_MAX, EPSI_MAX, N_EPSI)
kappa_bins = np.linspace(-KAPPA_MAX, KAPPA_MAX, N_KAPPA)

# =======================
# UTILS
# =======================
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def lane_curvature(lane, s, ds):
    psi1 = lane.heading_at(s)
    psi2 = lane.heading_at(s + ds)
    return wrap_angle(psi2 - psi1) / ds

# =======================
# OBSERVATION FUNCTION
# =======================
def discretize_obs(prev_heading, prev_pos):
    vehicle = env.unwrapped.vehicle
    lane = vehicle.lane

    s, e_y = lane.local_coordinates(vehicle.position)
    psi_lane = lane.heading_at(s)
    e_psi = wrap_angle(vehicle.heading - psi_lane)

    kappa_lane = lane_curvature(lane, s, CURVATURE_DS)

    ds = np.linalg.norm(vehicle.position - prev_pos)
    if ds > 1e-6:
        kappa_vehicle = wrap_angle(vehicle.heading - prev_heading) / ds
    else:
        kappa_vehicle = 0.0

    kappa_err = kappa_vehicle - kappa_lane

    e_y = np.clip(e_y, -EY_MAX, EY_MAX)
    e_psi = np.clip(e_psi, -EPSI_MAX, EPSI_MAX)
    kappa_vehicle = np.clip(kappa_vehicle, -KAPPA_MAX, KAPPA_MAX)
    kappa_err = np.clip(kappa_err, -KAPPA_MAX, KAPPA_MAX)

    return (
        np.digitize(e_y, e_y_bins),
        np.digitize(e_psi, e_psi_bins),
        np.digitize(kappa_err, kappa_bins),
    ), e_y, e_psi, kappa_vehicle, kappa_err

# =======================
# Q TABLE
# =======================
Q = defaultdict(lambda: np.zeros(N_ACTIONS))

# =======================
# HYPERPARAMS
# =======================
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.05
episodes = 5000

lambda_psi = 0.5
lambda_kappa = 1.0
lambda_jerk = 0.1
alive_reward = 0.2

# =======================
# TRAINING METRICS
# =======================
window = 50
avg_return, avg_steps, avg_error = [], [], []
ret_buf, step_buf, err_buf = [], [], []

# =======================
# TRAINING LOOP
# =======================
for ep in range(1, episodes + 1):
    env.reset()
    vehicle = env.unwrapped.vehicle
    prev_heading = vehicle.heading
    prev_pos = vehicle.position.copy()
    s, _, _, _, _ = discretize_obs(prev_heading, prev_pos)

    done = False
    total_reward = 0.0
    error_sum = 0.0
    steps = 0
    prev_kappa_cmd = 0.0

    while not done:
        if np.random.rand() < epsilon:
            a_idx = np.random.randint(N_ACTIONS)
        else:
            a_idx = np.argmax(Q[s])

        kappa_cmd = ACTIONS[a_idx]
        steer_cmd = np.arctan(CAR_LENGTH * kappa_cmd)

        _, _, terminated, truncated, _ = env.step([steer_cmd])
        done = terminated or truncated

        s_next, e_y, e_psi, kappa_vehicle, kappa_err = discretize_obs(prev_heading, prev_pos)

        norm_error = (
            (e_y / EY_MAX) ** 2
            + lambda_psi * (e_psi / EPSI_MAX) ** 2
            + lambda_kappa * (kappa_err / KAPPA_MAX) ** 2
        )
        jerk_cost = (kappa_cmd - prev_kappa_cmd) ** 2
        reward = alive_reward - norm_error - lambda_jerk * jerk_cost

        Q[s][a_idx] += alpha * (
            reward + gamma * np.max(Q[s_next]) - Q[s][a_idx]
        )

        s = s_next
        total_reward += reward
        error_sum += norm_error
        steps += 1

        prev_heading = vehicle.heading
        prev_pos = vehicle.position.copy()
        prev_kappa_cmd = kappa_cmd

        if steps >= 200:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    ret_buf.append(total_reward)
    step_buf.append(steps)
    err_buf.append(error_sum / max(steps, 1))

    if ep % window == 0:
        avg_return.append(np.mean(ret_buf))
        avg_steps.append(np.mean(step_buf))
        avg_error.append(np.mean(err_buf))
        ret_buf.clear(); step_buf.clear(); err_buf.clear()
        print(f"Ep {ep} | return {avg_return[-1]:.2f} | steps {avg_steps[-1]:.1f}")

env.close()

# =======================
# TRAINING CURVES
# =======================
x = np.arange(len(avg_steps)) * window

plt.figure(figsize=(8, 4))
plt.plot(x, avg_steps)
plt.xlabel("Episodes")
plt.ylabel("Avg Steps")
plt.title("Steps vs Training (e_y, e_psi, kappa)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "steps_vs_training.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(x, avg_return)
plt.xlabel("Episodes")
plt.ylabel("Avg Return")
plt.title("Reward vs Training (e_y, e_psi, kappa)")
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
plt.title("Error per Step vs Training (e_y, e_psi, kappa)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "error_per_step_vs_training.png"), dpi=300)
plt.close()

# =======================
# 3D POLICY HEATMAP (kappa slices)
# =======================
for k_bin in [2, N_KAPPA // 2, N_KAPPA - 3]:
    policy = np.zeros((N_EY, N_EPSI))

    for i in range(N_EY):
        for j in range(N_EPSI):
            state = (i + 1, j + 1, k_bin)
            if state in Q:
                policy[i, j] = ACTIONS[np.argmax(Q[state])]

    plt.figure(figsize=(7, 5))
    plt.imshow(
        policy,
        origin="lower",
        extent=[
            np.rad2deg(-EPSI_MAX),
            np.rad2deg(EPSI_MAX),
            -EY_MAX,
            EY_MAX,
        ],
        aspect="auto",
    )
    plt.colorbar(label="Curvature [1/m]")
    plt.xlabel("Heading Error e_psi [deg]")
    plt.ylabel("Lateral Error e_y [m]")
    plt.title(f"Policy Heatmap at Curvature Bin {k_bin}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"policy_heatmap_kappa_bin_{k_bin}.png"), dpi=300)
    plt.close()

# =======================
# 3D Q-TABLE HEATMAP
# =======================
q_max_grid = np.zeros((N_EY, N_EPSI, N_KAPPA))
for i in range(N_EY):
    for j in range(N_EPSI):
        for k in range(N_KAPPA):
            state = (i + 1, j + 1, k)
            if state in Q:
                q_max_grid[i, j, k] = np.max(Q[state])

I, J, K = np.meshgrid(
    np.arange(N_EY),
    np.arange(N_EPSI),
    np.arange(N_KAPPA),
    indexing="ij",
)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    I.flatten(),
    J.flatten(),
    K.flatten(),
    c=q_max_grid.flatten(),
    cmap="viridis",
    s=15,
    alpha=0.8,
)
fig.colorbar(sc, ax=ax, label="Max Q")
ax.set_xlabel("e_y bin")
ax.set_ylabel("e_psi bin")
ax.set_zlabel("kappa bin")
ax.set_title("3D Q-Table Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "q_table_3d_heatmap.png"), dpi=300)
plt.close()

print("\nTraining complete. Plots saved to", RESULTS_DIR)
