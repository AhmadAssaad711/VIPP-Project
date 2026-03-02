"""
Generalization Test — Evaluate Q-Learning Agents on Randomized Racetracks
==========================================================================
Trains both the basic and physics-informed RL agents using their original
procedures, then evaluates each on 100 randomized racetrack episodes.
Collects average lateral tracking error and survival steps per episode,
and saves comparison plots to disk.
"""

import matplotlib
matplotlib.use("Agg")

import sys
import os
import copy
import runpy
import numpy as np
import gymnasium as gym
import highway_env  # noqa
import matplotlib.pyplot as plt
from collections import defaultdict

from highway_env.envs.racetrack_env import RacetrackEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import CircularLane, StraightLane, LineType

# ============================================================
# GLOBAL SEED
# ============================================================
SEED = 42

# ============================================================
# RANDOMIZED RACETRACK ENVIRONMENT
# ============================================================
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


# ============================================================
# AGENT CONFIGURATION CONTAINERS
# ============================================================
class BasicAgentConfig:
    """Holds all parameters needed to run the basic (e_y, e_psi) agent."""
    STEER_MIN, STEER_MAX = -0.5, 0.5
    N_ACTIONS = 21
    ACTIONS = np.linspace(STEER_MIN, STEER_MAX, N_ACTIONS)
    EY_MAX = 2.0
    EPSI_MAX = np.deg2rad(45)
    N_EY = 20
    N_EPSI = 20
    e_y_bins = np.linspace(-EY_MAX, EY_MAX, N_EY)
    e_psi_bins = np.linspace(-EPSI_MAX, EPSI_MAX, N_EPSI)


class PhysicsAgentConfig:
    """Holds all parameters needed to run the physics-informed agent."""
    N_ACTIONS = 11
    KAPPA_CMD_MAX = 0.2
    ACTIONS = np.linspace(-KAPPA_CMD_MAX, KAPPA_CMD_MAX, N_ACTIONS)
    EY_MAX = 2.0
    EPSI_MAX = np.deg2rad(45)
    KAPPA_MAX = 0.2
    N_EY = 10
    N_EPSI = 10
    N_KAPPA = 10
    e_y_bins = np.linspace(-EY_MAX, EY_MAX, N_EY)
    e_psi_bins = np.linspace(-EPSI_MAX, EPSI_MAX, N_EPSI)
    kappa_bins = np.linspace(-KAPPA_MAX, KAPPA_MAX, N_KAPPA)
    CAR_LENGTH = 5.0  # default; updated after first env reset


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def wrap_angle(a):
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def lane_curvature(lane, s, ds):
    """Estimate lane curvature at arc-length s over a window ds."""
    psi1 = lane.heading_at(s)
    psi2 = lane.heading_at(s + ds)
    return wrap_angle(psi2 - psi1) / ds


def discretize_basic(env):
    """Discretize observation for the basic agent."""
    cfg = BasicAgentConfig
    vehicle = env.unwrapped.vehicle
    lane = vehicle.lane

    s, e_y = lane.local_coordinates(vehicle.position)
    psi_lane = lane.heading_at(s)
    e_psi = wrap_angle(vehicle.heading - psi_lane)

    e_y = np.clip(e_y, -cfg.EY_MAX, cfg.EY_MAX)
    e_psi = np.clip(e_psi, -cfg.EPSI_MAX, cfg.EPSI_MAX)

    ey_bin = np.digitize(e_y, cfg.e_y_bins)
    epsi_bin = np.digitize(e_psi, cfg.e_psi_bins)

    return (ey_bin, epsi_bin), abs(e_y)


def discretize_physics(env, prev_heading, prev_pos, car_length):
    """Discretize observation for the physics-informed agent."""
    cfg = PhysicsAgentConfig
    vehicle = env.unwrapped.vehicle
    lane = vehicle.lane

    s, e_y = lane.local_coordinates(vehicle.position)
    psi_lane = lane.heading_at(s)
    e_psi = wrap_angle(vehicle.heading - psi_lane)

    curvature_ds = 2.0 * car_length
    kappa_lane = lane_curvature(lane, s, curvature_ds)

    ds = np.linalg.norm(vehicle.position - prev_pos)
    if ds > 1e-6:
        kappa_vehicle = wrap_angle(vehicle.heading - prev_heading) / ds
    else:
        kappa_vehicle = 0.0

    kappa_err = kappa_vehicle - kappa_lane

    e_y = np.clip(e_y, -cfg.EY_MAX, cfg.EY_MAX)
    e_psi = np.clip(e_psi, -cfg.EPSI_MAX, cfg.EPSI_MAX)
    kappa_err = np.clip(kappa_err, -cfg.KAPPA_MAX, cfg.KAPPA_MAX)

    state = (
        np.digitize(e_y, cfg.e_y_bins),
        np.digitize(e_psi, cfg.e_psi_bins),
        np.digitize(kappa_err, cfg.kappa_bins),
    )
    return state, abs(e_y)


# ============================================================
# TRAINING VIA runpy (executes the original scripts)
# ============================================================
ENV_CONFIG = {
    "controlled_vehicles": 1,
    "other_vehicles": 0,
    "terminate_off_road": True,
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "heading"],
        "absolute": True,
    },
}


def train_basic_agent():
    """
    Train the basic RL agent by executing its original script.
    Returns the trained Q-table (dict).
    """
    print("=" * 60)
    print("  TRAINING: Basic RL Agent (e_y, e_psi)")
    print("=" * 60)
    script_path = os.path.join(
        os.path.dirname(__file__), "2_basic_rl_agent", "basic_rl_agent.py"
    )
    ns = runpy.run_path(script_path, run_name="__main__")
    # Convert defaultdict to regular dict for portability
    return dict(ns["Q"])


def train_physics_agent():
    """
    Train the physics-informed agent by executing its original script.
    Returns the trained Q-table (dict) and the car length used.
    """
    print("=" * 60)
    print("  TRAINING: Physics-Informed Agent (e_y, e_psi, kappa)")
    print("=" * 60)
    script_path = os.path.join(
        os.path.dirname(__file__),
        "3_physics_informed_agent",
        "physics_informed_agent.py",
    )
    ns = runpy.run_path(script_path, run_name="__main__")
    return dict(ns["Q"]), ns["CAR_LENGTH"]


# ============================================================
# EVALUATION
# ============================================================
def evaluate_agent(q_table, agent_type, episodes=100, seed=SEED, car_length=5.0):
    """
    Evaluate a trained agent on randomized racetrack environments.

    Parameters
    ----------
    q_table : dict
        Trained Q-table mapping state tuples to action-value arrays.
    agent_type : str
        Either "basic" or "physics".
    episodes : int
        Number of evaluation episodes.
    seed : int
        Base random seed for reproducibility.
    car_length : float
        Vehicle length (used only by physics agent).

    Returns
    -------
    episode_avg_errors : list[float]
        Average lateral tracking error per step for each episode.
    episode_steps : list[int]
        Number of steps survived in each episode.
    """
    env = gym.make("randomized-racetrack-v0", config=ENV_CONFIG)

    episode_avg_errors = []
    episode_steps = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)

        if agent_type == "basic":
            state, lat_err = discretize_basic(env)
        else:
            vehicle = env.unwrapped.vehicle
            prev_heading = vehicle.heading
            prev_pos = vehicle.position.copy()
            state, lat_err = discretize_physics(
                env, prev_heading, prev_pos, car_length
            )

        done = False
        total_error = 0.0
        steps = 0

        while not done:
            # Greedy action from Q-table (epsilon = 0)
            if state in q_table:
                a_idx = int(np.argmax(q_table[state]))
            else:
                # Unseen state: pick middle action (go straight)
                if agent_type == "basic":
                    a_idx = BasicAgentConfig.N_ACTIONS // 2
                else:
                    a_idx = PhysicsAgentConfig.N_ACTIONS // 2

            # Convert action index to steering command
            if agent_type == "basic":
                action = BasicAgentConfig.ACTIONS[a_idx]
            else:
                kappa_cmd = PhysicsAgentConfig.ACTIONS[a_idx]
                action = np.arctan(car_length * kappa_cmd)

            _, _, terminated, truncated, _ = env.step([action])
            done = terminated or truncated

            # Observe new state
            if agent_type == "basic":
                state, lat_err = discretize_basic(env)
            else:
                state, lat_err = discretize_physics(
                    env, prev_heading, prev_pos, car_length
                )
                vehicle = env.unwrapped.vehicle
                prev_heading = vehicle.heading
                prev_pos = vehicle.position.copy()

            total_error += lat_err
            steps += 1

            if steps >= 200:
                break

        avg_err = total_error / max(steps, 1)
        episode_avg_errors.append(avg_err)
        episode_steps.append(steps)

        if (ep + 1) % 20 == 0:
            print(
                f"  [{agent_type.upper():>7s}] Eval ep {ep + 1:3d}/{episodes} | "
                f"avg_err={avg_err:.4f} | steps={steps}"
            )

    env.close()
    return episode_avg_errors, episode_steps


# ============================================================
# SMOOTHING UTILITY
# ============================================================
def moving_average(data, window=10):
    """Compute a simple moving average over a 1-D sequence."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


# ============================================================
# PLOTTING
# ============================================================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SMOOTH_WINDOW = 10


def plot_comparison(basic_errors, basic_steps, physics_errors, physics_steps):
    """Generate and save comparison plots for both agents."""

    episodes_x = np.arange(1, len(basic_errors) + 1)

    # --- Smoothed average error per episode ---
    basic_err_smooth = moving_average(basic_errors, SMOOTH_WINDOW)
    physics_err_smooth = moving_average(physics_errors, SMOOTH_WINDOW)
    x_smooth = np.arange(SMOOTH_WINDOW, len(basic_errors) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x_smooth, basic_err_smooth, label="Standard Agent", linewidth=1.5)
    plt.plot(x_smooth, physics_err_smooth, label="Physics-Informed Agent", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Average Lateral Error per Step [m]")
    plt.title("Generalization Test — Average Tracking Error (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_err = os.path.join(RESULTS_DIR, "generalization_error.png")
    plt.savefig(path_err, dpi=300)
    plt.close()
    print(f"  Saved: {path_err}")

    # --- Steps per episode ---
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, basic_steps, label="Standard Agent", linewidth=1.5, alpha=0.8)
    plt.plot(episodes_x, physics_steps, label="Physics-Informed Agent", linewidth=1.5, alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Steps Survived")
    plt.title("Generalization Test — Steps per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_steps = os.path.join(RESULTS_DIR, "generalization_steps.png")
    plt.savefig(path_steps, dpi=300)
    plt.close()
    print(f"  Saved: {path_steps}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    np.random.seed(SEED)

    # ------ Train both agents using their original procedures ------
    Q_basic = train_basic_agent()
    Q_physics, car_length = train_physics_agent()

    # ------ Evaluate on randomized tracks ------
    print("\n" + "=" * 60)
    print("  EVALUATION: 100 Randomized Racetrack Episodes")
    print("=" * 60)

    basic_errors, basic_steps = evaluate_agent(
        Q_basic, agent_type="basic", episodes=100, seed=SEED
    )
    physics_errors, physics_steps = evaluate_agent(
        Q_physics, agent_type="physics", episodes=100, seed=SEED,
        car_length=car_length,
    )

    # ------ Summary statistics ------
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"  Standard Agent        | mean error: {np.mean(basic_errors):.4f} m  "
          f"| mean steps: {np.mean(basic_steps):.1f}")
    print(f"  Physics-Informed Agent| mean error: {np.mean(physics_errors):.4f} m  "
          f"| mean steps: {np.mean(physics_steps):.1f}")

    # ------ Plot and save ------
    plot_comparison(basic_errors, basic_steps, physics_errors, physics_steps)

    print("\nDone.")
