"""Physics-Informed RL Agent - Q-Learning with (e_y, e_psi, kappa_err, kappa_near, kappa_la) State
=================================================================================================
Tabular Q-learning agent that adds curvature error (kappa_err), lane
curvature at the nearest point (kappa_near), and lane curvature at a
lookahead distance along arc length (kappa_la) to the state representation.
Curvature is defined as the rate of change of heading over an arc-length window.
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
import pickle
import sys

from highway_env.envs.racetrack_env import RacetrackEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import CircularLane, StraightLane, LineType

# =======================
# OUTPUT DIR
# =======================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "physics_informed_agent")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =======================
# RANDOMIZED RACETRACK ENVIRONMENT
# =======================
class RandomizedRacetrackEnv(RacetrackEnv):
    """
    Subclass of RacetrackEnv that generates a new random closed-loop
    track on every reset by overriding _make_road().

    Randomization affects:
      - Lengths of straight sections
      - Radii (curvature) of circular arcs
      - Arc sweep angles
    The topology (number of sections, connectivity) is kept fixed so
    that the track always forms a valid loop.
    """

    def _make_road(self) -> None:
        rng = self.np_random
        net = RoadNetwork()
        width = 5
        speed = 10.0

        # --- 1. Horizontal straight (a -> b) ---
        straight1_len = rng.uniform(40, 70)
        x_start = 42
        x_end = x_start + straight1_len
        net.add_lane("a", "b", StraightLane(
            [x_start, 0], [x_end, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=width, speed_limit=speed))
        net.add_lane("a", "b", StraightLane(
            [x_start, width], [x_end, width],
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            width=width, speed_limit=speed))

        # --- 2. Circular arc #1 (b -> c) — right turn ---
        radii1 = rng.uniform(15, 30)
        center1 = [x_end, -radii1]
        arc1_end = np.deg2rad(rng.uniform(-5, 5))
        net.add_lane("b", "c", CircularLane(
            center1, radii1, np.deg2rad(90), arc1_end,
            width=width, clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=speed))
        net.add_lane("b", "c", CircularLane(
            center1, radii1 + width, np.deg2rad(90), arc1_end,
            width=width, clockwise=False,
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            speed_limit=speed))

        # endpoint of arc1
        cx1, cy1 = center1
        arc1_x = cx1 + radii1 * np.cos(arc1_end)
        arc1_y = cy1 + radii1 * np.sin(arc1_end)

        # --- 3. Short vertical straight (c -> d) ---
        straight2_len = rng.uniform(8, 18)
        net.add_lane("c", "d", StraightLane(
            [cx1 + radii1, cy1], [cx1 + radii1, cy1 - straight2_len],
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            width=width, speed_limit=speed))
        net.add_lane("c", "d", StraightLane(
            [cx1 + radii1 + width, cy1],
            [cx1 + radii1 + width, cy1 - straight2_len],
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            width=width, speed_limit=speed))

        # --- 4. Circular arc #2 (d -> e) — U-turn ---
        radii2 = rng.uniform(12, 22)
        center2_x = cx1 + radii1 - radii2
        center2_y = cy1 - straight2_len
        center2 = [center2_x, center2_y]
        net.add_lane("d", "e", CircularLane(
            center2, radii2, np.deg2rad(0), np.deg2rad(-181),
            width=width, clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=speed))
        net.add_lane("d", "e", CircularLane(
            center2, radii2 + width, np.deg2rad(0), np.deg2rad(-181),
            width=width, clockwise=False,
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            speed_limit=speed))

        # --- 5. Circular arc #3 (e -> f) — S-curve ---
        radii3 = rng.uniform(12, 22)
        center3_x = center2_x - radii2 - radii3
        center3_y = center2_y
        center3 = [center3_x, center3_y]
        arc3_end_deg = rng.uniform(125, 145)
        net.add_lane("e", "f", CircularLane(
            center3, radii3 + width, np.deg2rad(0), np.deg2rad(arc3_end_deg),
            width=width, clockwise=True,
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            speed_limit=speed))
        net.add_lane("e", "f", CircularLane(
            center3, radii3, np.deg2rad(0), np.deg2rad(arc3_end_deg + 1),
            width=width, clockwise=True,
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            speed_limit=speed))

        # endpoint of arc3 (outer lane)
        arc3_rad = np.deg2rad(arc3_end_deg)
        f_x = center3_x + (radii3 + width) * np.cos(arc3_rad)
        f_y = center3_y + (radii3 + width) * np.sin(arc3_rad)

        # --- 6. Diagonal straight (f -> g) ---
        slant_len = rng.uniform(18, 32)
        angle_slant = np.deg2rad(-135)
        g_x = f_x + slant_len * np.cos(angle_slant)
        g_y = f_y + slant_len * np.sin(angle_slant)

        # perpendicular offset for second lane
        perp = np.array([np.cos(angle_slant + np.pi / 2),
                         np.sin(angle_slant + np.pi / 2)]) * width

        net.add_lane("f", "g", StraightLane(
            [f_x, f_y], [g_x, g_y],
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            width=width, speed_limit=speed))
        net.add_lane("f", "g", StraightLane(
            [f_x + perp[0], f_y + perp[1]],
            [g_x + perp[0], g_y + perp[1]],
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            width=width, speed_limit=speed))

        # --- 7. Large return arc (g -> h) ---
        radii4 = rng.uniform(20, 32)
        # place center so arc starts heading from slant direction
        center4_x = g_x + radii4 * np.cos(angle_slant + np.pi / 2)
        center4_y = g_y + radii4 * np.sin(angle_slant + np.pi / 2)
        center4 = [center4_x, center4_y]
        net.add_lane("g", "h", CircularLane(
            center4, radii4,
            np.deg2rad(315), np.deg2rad(170),
            width=width, clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=speed))
        net.add_lane("g", "h", CircularLane(
            center4, radii4 + width,
            np.deg2rad(315), np.deg2rad(165),
            width=width, clockwise=False,
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            speed_limit=speed))

        # --- 8. Continuation arc (h -> i) ---
        net.add_lane("h", "i", CircularLane(
            center4, radii4,
            np.deg2rad(170), np.deg2rad(56),
            width=width, clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=speed))
        net.add_lane("h", "i", CircularLane(
            center4, radii4 + width,
            np.deg2rad(170), np.deg2rad(58),
            width=width, clockwise=False,
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            speed_limit=speed))

        # --- 9. Final arc reconnecting to start (i -> a) ---
        radii5 = rng.uniform(15, 25)
        center5 = [x_start + 1.2, radii5 + 4.0]
        net.add_lane("i", "a", CircularLane(
            center5, radii5 + width,
            np.deg2rad(240), np.deg2rad(270),
            width=width, clockwise=True,
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            speed_limit=speed))
        net.add_lane("i", "a", CircularLane(
            center5, radii5,
            np.deg2rad(238), np.deg2rad(268),
            width=width, clockwise=True,
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            speed_limit=speed))

        # --- Build road ---
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road


# Register randomized environment
gym.register(
    id="randomized-racetrack-v0",
    entry_point=RandomizedRacetrackEnv,
)

# =======================
# MULTI-MAP TRAINING CONFIG
# =======================
MAP_SEEDS = [42, 123, 456, 789, 1024]   # 5 fixed seeds → 5 distinct tracks
MAP_SWITCH_INTERVAL = 100               # switch map every 100 episodes

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

env = gym.make("randomized-racetrack-v0", config=config, render_mode=None)
env.reset(seed=MAP_SEEDS[0])

vehicle = env.unwrapped.vehicle
CAR_LENGTH = vehicle.LENGTH
CURVATURE_DS = 2.0 * CAR_LENGTH   # arc-length window
LOOKAHEAD_DIST = 5.0 * CAR_LENGTH  # lookahead distance for curvature preview

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
N_KAPPA_NEAR = 5     # bins for lane curvature at nearest point
N_KAPPA_LA = 5       # bins for lane curvature at lookahead

e_y_bins = np.linspace(-EY_MAX, EY_MAX, N_EY)
e_psi_bins = np.linspace(-EPSI_MAX, EPSI_MAX, N_EPSI)
kappa_bins = np.linspace(-KAPPA_MAX, KAPPA_MAX, N_KAPPA)
kappa_near_bins = np.linspace(-KAPPA_MAX, KAPPA_MAX, N_KAPPA_NEAR)
kappa_la_bins = np.linspace(-KAPPA_MAX, KAPPA_MAX, N_KAPPA_LA)

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
def discretize_obs(prev_heading, prev_pos, target_env=None):
    if target_env is None:
        target_env = env
    vehicle = target_env.unwrapped.vehicle
    lane = vehicle.lane

    s, e_y = lane.local_coordinates(vehicle.position)
    psi_lane = lane.heading_at(s)
    e_psi = wrap_angle(vehicle.heading - psi_lane)

    kappa_near = lane_curvature(lane, s, CURVATURE_DS)          # curvature at nearest point
    kappa_la = lane_curvature(lane, s + LOOKAHEAD_DIST, CURVATURE_DS)  # curvature at lookahead

    ds = np.linalg.norm(vehicle.position - prev_pos)
    if ds > 1e-6:
        kappa_vehicle = wrap_angle(vehicle.heading - prev_heading) / ds
    else:
        kappa_vehicle = 0.0

    kappa_err = kappa_vehicle - kappa_near

    e_y = np.clip(e_y, -EY_MAX, EY_MAX)
    e_psi = np.clip(e_psi, -EPSI_MAX, EPSI_MAX)
    kappa_vehicle = np.clip(kappa_vehicle, -KAPPA_MAX, KAPPA_MAX)
    kappa_err = np.clip(kappa_err, -KAPPA_MAX, KAPPA_MAX)
    kappa_near = np.clip(kappa_near, -KAPPA_MAX, KAPPA_MAX)
    kappa_la = np.clip(kappa_la, -KAPPA_MAX, KAPPA_MAX)

    return (
        np.digitize(e_y, e_y_bins),
        np.digitize(e_psi, e_psi_bins),
        np.digitize(kappa_err, kappa_bins),
        np.digitize(kappa_near, kappa_near_bins),
        np.digitize(kappa_la, kappa_la_bins),
    ), e_y, e_psi, kappa_vehicle, kappa_err

# =======================
# Q TABLE
# =======================
Q = defaultdict(lambda: np.zeros(N_ACTIONS))

Q_TABLE_PATH = os.path.join(RESULTS_DIR, "q_table.pkl")
SKIP_TRAINING = "--eval-only" in sys.argv or os.path.exists(Q_TABLE_PATH)

# =======================
# HYPERPARAMS
# =======================
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.99985
epsilon_min = 0.05
episodes = 20000

lambda_psi = 0.5
lambda_kappa = 1.0
lambda_jerk = 0.1
alive_reward = 0.2

# =======================
# TRAINING (skip if Q-table already saved)
# =======================
if SKIP_TRAINING and os.path.exists(Q_TABLE_PATH):
    print(f"Loading Q-table from {Q_TABLE_PATH} (skipping training)")
    with open(Q_TABLE_PATH, "rb") as f:
        saved = pickle.load(f)
    # Restore into defaultdict
    Q = defaultdict(lambda: np.zeros(N_ACTIONS))
    Q.update(saved)
    print(f"  Loaded {len(Q)} states")
else:
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
        # --- Cycle through maps every MAP_SWITCH_INTERVAL episodes ---
        current_map_idx = ((ep - 1) // MAP_SWITCH_INTERVAL) % len(MAP_SEEDS)
        env.reset(seed=MAP_SEEDS[current_map_idx])
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
            print(f"Ep {ep} [Map {current_map_idx}] | return {avg_return[-1]:.2f} | steps {avg_steps[-1]:.1f}")

    env.close()

    # --- Save Q-table ---
    with open(Q_TABLE_PATH, "wb") as f:
        pickle.dump(dict(Q), f)
    print(f"\nQ-table saved to {Q_TABLE_PATH} ({len(Q)} states)")

    # =======================
    # TRAINING CURVES
    # =======================
    x = np.arange(len(avg_steps)) * window

    plt.figure(figsize=(8, 4))
    plt.plot(x, avg_steps)
    plt.xlabel("Episodes")
    plt.ylabel("Avg Steps")
    plt.title("Steps vs Training (Multi-Map)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "steps_vs_training.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, avg_return)
    plt.xlabel("Episodes")
    plt.ylabel("Avg Return")
    plt.title("Reward vs Training (Multi-Map)")
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
    plt.title("Error per Step vs Training (Multi-Map)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_per_step_vs_training.png"), dpi=300)
    plt.close()

    # =======================
    # 3D POLICY HEATMAP (kappa slices, mid kappa_near & kappa_la)
    # =======================
    mid_kn = N_KAPPA_NEAR // 2
    mid_kla = N_KAPPA_LA // 2
    for k_bin in [2, N_KAPPA // 2, N_KAPPA - 3]:
        policy = np.zeros((N_EY, N_EPSI))

        for i in range(N_EY):
            for j in range(N_EPSI):
                state = (i + 1, j + 1, k_bin, mid_kn, mid_kla)
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
        plt.title(f"Policy Heatmap at kappa_err bin {k_bin} (kn={mid_kn}, kla={mid_kla})")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"policy_heatmap_kappa_bin_{k_bin}.png"), dpi=300)
        plt.close()

    # =======================
    # 3D Q-TABLE HEATMAP (marginalised over kappa_near & kappa_la at mid bins)
    # =======================
    q_max_grid = np.zeros((N_EY, N_EPSI, N_KAPPA))
    for i in range(N_EY):
        for j in range(N_EPSI):
            for k in range(N_KAPPA):
                state = (i + 1, j + 1, k, mid_kn, mid_kla)
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
    ax.set_zlabel("kappa_err bin")
    ax.set_title("3D Q-Table Heatmap (kn=mid, kla=mid)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "q_table_3d_heatmap.png"), dpi=300)
    plt.close()

    print("\nTraining complete. Plots saved to", RESULTS_DIR)

# =======================
# EVALUATION ON RANDOM MAPS
# =======================
print("\n" + "=" * 60)
print("  EVALUATION: 200 Episodes on Randomly Generated Maps")
print("=" * 60)

EVAL_EPISODES = 200
EVAL_SEED = 9999  # different from training seeds

eval_env = gym.make("randomized-racetrack-v0", config=config, render_mode=None)

eval_errors = []
eval_steps = []

for ep in range(EVAL_EPISODES):
    eval_env.reset(seed=EVAL_SEED + ep)  # unique random map each episode
    vehicle = eval_env.unwrapped.vehicle
    prev_heading = vehicle.heading
    prev_pos = vehicle.position.copy()
    s, _, _, _, _ = discretize_obs(prev_heading, prev_pos, eval_env)

    done = False
    total_error = 0.0
    steps = 0

    while not done:
        # Greedy policy (no exploration)
        if s in Q:
            a_idx = int(np.argmax(Q[s]))
        else:
            a_idx = N_ACTIONS // 2  # straight if unseen state

        kappa_cmd = ACTIONS[a_idx]
        steer_cmd = np.arctan(CAR_LENGTH * kappa_cmd)

        _, _, terminated, truncated, _ = eval_env.step([steer_cmd])
        done = terminated or truncated

        s_next, e_y, e_psi, _, _ = discretize_obs(prev_heading, prev_pos, eval_env)

        total_error += abs(e_y)
        steps += 1
        s = s_next

        vehicle = eval_env.unwrapped.vehicle
        prev_heading = vehicle.heading
        prev_pos = vehicle.position.copy()

        if steps >= 200:
            break

    avg_err = total_error / max(steps, 1)
    eval_errors.append(avg_err)
    eval_steps.append(steps)

    if (ep + 1) % 20 == 0:
        print(
            f"  Eval ep {ep + 1:3d}/{EVAL_EPISODES} | "
            f"avg_lat_err={avg_err:.4f} m | steps={steps}"
        )

eval_env.close()

print("\n" + "-" * 60)
print("EVALUATION SUMMARY (200 random maps)")
print("-" * 60)
print(f"  Mean lateral error : {np.mean(eval_errors):.4f} m")
print(f"  Std  lateral error : {np.std(eval_errors):.4f} m")
print(f"  Mean steps survived: {np.mean(eval_steps):.1f}")
print(f"  Std  steps survived: {np.std(eval_steps):.1f}")
print(f"  Q-table states     : {len(Q)}")

# --- Evaluation plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Lateral error per episode
kernel = np.ones(10) / 10
err_smooth = np.convolve(eval_errors, kernel, mode="valid")
axes[0].plot(eval_errors, alpha=0.3, label="Raw")
axes[0].plot(np.arange(9, len(eval_errors)), err_smooth, linewidth=2, label="Smoothed (w=10)")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Avg Lateral Error [m]")
axes[0].set_title("Evaluation — Lateral Error per Episode")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Steps survived per episode
steps_smooth = np.convolve(eval_steps, kernel, mode="valid")
axes[1].plot(eval_steps, alpha=0.3, label="Raw")
axes[1].plot(np.arange(9, len(eval_steps)), steps_smooth, linewidth=2, label="Smoothed (w=10)")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Steps Survived")
axes[1].set_title("Evaluation — Steps per Episode")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "evaluation_random_maps.png"), dpi=300)
plt.close()
print(f"\nEvaluation plot saved to {RESULTS_DIR}")
print("Done.")
