"""Configurable baseline DQN runner for the laned HighwayEnv experiment.

The module preserves the Leurent-style discrete DQN formulation for
highway-v0 and adds the operational pieces needed for repeatable research:
CLI/configuration parsing, deterministic run metadata, vectorized training
environments, per-run artifact folders, TensorBoard logging, checkpoint
saving, and evaluation summaries. The notebook-facing helpers can override
the environment profile and wrapper configuration without changing the
baseline learner itself.

Use this module to establish the structured reference before enabling the
attention extractor or any custom reward/safety wrapper. Results are only
comparable when the environment profile, reward configuration, timestep
budget, seeds, and evaluation protocol are recorded together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401 - registers highway environments
import matplotlib
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptive_longitudinal import (
    ADAPTIVE_LONGITUDINAL_CONFIG_KEY,
    build_adaptive_longitudinal_config,
    compute_same_lane_ttc,
    make_highway_env_with_adaptive_longitudinal,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "artifacts" / "dqn"
DEFAULT_N_ENVS = 24
NOTEBOOK_SAFE_MAX_ENVS = 8
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "dqn" / "baseline" / "models" / "elurant_dqn.zip"
POLICY_PANEL_RENDER_CONFIG = {
    "screen_width": 900,
    "screen_height": 220,
}


def _safe_path_token(value: str, max_length: int = 48) -> str:
    token = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    token = token.strip("_")
    return (token or "run")[:max_length]


def make_tensorboard_dir(run_dir: Path, run_name: str) -> Path:
    """Use a short TensorBoard path on Windows to avoid MAX_PATH failures."""
    override = os.environ.get("HIGHWAY_RL_TB_ROOT")
    if override:
        root = Path(override).expanduser()
    elif os.name == "nt":
        root = Path(os.environ.get("TEMP", r"C:\tmp")) / "highway_rl_tb"
    else:
        root = run_dir / "tb"

    digest = hashlib.sha1(str(run_dir).encode("utf-8")).hexdigest()[:10]
    return root / f"{_safe_path_token(run_name)}_{digest}"


def next_tensorboard_run_dir(tb_dir: Path) -> Path:
    run_id = 1
    while (tb_dir / f"r{run_id}").exists():
        run_id += 1
    return tb_dir / f"r{run_id}"


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except Exception:
        return False

    shell = get_ipython()
    if shell is None:
        return False
    return "ipykernel" in sys.modules or shell.__class__.__name__ == "ZMQInteractiveShell"


class TimestepProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, every_n_steps: int = 5000) -> None:
        super().__init__(verbose=0)
        self.total_timesteps = max(1, int(total_timesteps))
        self.every_n_steps = max(1, int(every_n_steps))
        self._next_print = self.every_n_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_print:
            progress = min(100.0, 100.0 * self.num_timesteps / self.total_timesteps)
            print(
                f"[train] timesteps={self.num_timesteps}/{self.total_timesteps} "
                f"({progress:.1f}%)",
                flush=True,
            )
            while self._next_print <= self.num_timesteps:
                self._next_print += self.every_n_steps
        return True


def make_config() -> dict:
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 3,
        "vehicles_count": 20,
        "duration": 40,
        ADAPTIVE_LONGITUDINAL_CONFIG_KEY: build_adaptive_longitudinal_config(),
    }


def make_env(render_mode: str | None = None, config: dict | None = None):
    return make_highway_env_with_adaptive_longitudinal(
        render_mode=render_mode,
        config=dict(config or make_config()),
    )


def _update_overtake_tracker(
    env: gym.Env,
    seen_ahead_ids: set[int],
    overtaken_ids: set[int],
) -> None:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if vehicle is None or road is None:
        return

    ego_x = float(vehicle.position[0])
    for other in getattr(road, "vehicles", []):
        if other is vehicle:
            continue
        other_id = id(other)
        dx = float(other.position[0] - ego_x)
        if dx > 0.0:
            seen_ahead_ids.add(other_id)
        elif other_id in seen_ahead_ids:
            overtaken_ids.add(other_id)


def evaluate_with_metrics(
    model: DQN,
    *,
    episodes: int,
    seed: int,
    render_mode: str | None = None,
    config: dict | None = None,
    ttc_cap: float = 10.0,
) -> list[dict[str, float | bool]]:
    env = Monitor(make_env(render_mode=render_mode, config=config))
    summaries: list[dict[str, float | bool]] = []

    try:
        for episode_idx in range(int(episodes)):
            obs, _ = env.reset(seed=seed + episode_idx)
            terminated = False
            truncated = False
            total_reward = 0.0
            speed_trace: list[float] = []
            ttc_trace: list[float] = []
            seen_ahead_ids: set[int] = set()
            overtaken_ids: set[int] = set()
            final_info: dict = {}
            adaptive_delta_trace: list[float] = []
            adaptive_target_speed_trace: list[float] = []
            adaptive_ttc_trace: list[float] = []
            adaptive_safety_penalty_trace: list[float] = []
            adaptive_unsafe_request_count = 0
            adaptive_longitudinal_steps = 0
            traffic_penalty_trace: list[float] = []
            traffic_rear_ttc_trace: list[float] = []
            traffic_flow_speed_trace: list[float] = []
            traffic_speed_deficit_trace: list[float] = []
            safety_shaping_trace: list[float] = []
            safety_ttc_bonus_trace: list[float] = []
            safety_low_ttc_penalty_trace: list[float] = []
            safety_lag_penalty_trace: list[float] = []
            safety_flow_speed_trace: list[float] = []
            safety_speed_deficit_trace: list[float] = []
            safety_rear_ttc_trace: list[float] = []
            potential_field_cost_trace: list[float] = []
            potential_field_penalty_trace: list[float] = []
            potential_field_vehicle_count_trace: list[float] = []
            potential_field_max_vehicle_cost_trace: list[float] = []
            potential_field_closest_longitudinal_gap_trace: list[float] = []
            potential_field_closest_lateral_gap_trace: list[float] = []
            lane_change_safety_penalty_trace: list[float] = []
            lane_change_safety_risky_count = 0

            while not (terminated or truncated):
                _update_overtake_tracker(env, seen_ahead_ids, overtaken_ids)
                speed_trace.append(float(getattr(env.unwrapped.vehicle, "speed", 0.0)))
                ttc_trace.append(compute_same_lane_ttc(env, ttc_cap=ttc_cap))

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                final_info = dict(info)
                if "adaptive_speed_delta" in final_info:
                    adaptive_delta_trace.append(float(final_info["adaptive_speed_delta"]))
                    adaptive_target_speed_trace.append(
                        float(final_info.get("adaptive_target_speed_after", np.nan))
                    )
                    adaptive_ttc_trace.append(float(final_info.get("adaptive_ttc", np.nan)))
                    adaptive_safety_penalty_trace.append(
                        float(final_info.get("adaptive_safety_override_penalty", 0.0))
                    )
                    adaptive_unsafe_request_count += int(final_info.get("adaptive_unsafe_speed_request", False))
                    if final_info.get("adaptive_requested_action") in {"FASTER", "SLOWER"}:
                        adaptive_longitudinal_steps += 1
                if "traffic_flow_penalty" in final_info:
                    traffic_penalty_trace.append(float(final_info.get("traffic_flow_penalty", 0.0)))
                    traffic_rear_ttc_trace.append(float(final_info.get("traffic_rear_ttc", np.nan)))
                    traffic_flow_speed_trace.append(float(final_info.get("traffic_flow_speed", np.nan)))
                    traffic_speed_deficit_trace.append(
                        float(final_info.get("traffic_speed_deficit", 0.0))
                    )
                if "safety_reward_shaping" in final_info:
                    safety_shaping_trace.append(float(final_info.get("safety_reward_shaping", 0.0)))
                    safety_ttc_bonus_trace.append(float(final_info.get("safety_ttc_bonus", 0.0)))
                    safety_low_ttc_penalty_trace.append(
                        float(final_info.get("safety_low_ttc_penalty", 0.0))
                    )
                    safety_lag_penalty_trace.append(float(final_info.get("safety_lag_penalty", 0.0)))
                    safety_flow_speed_trace.append(float(final_info.get("safety_flow_speed", np.nan)))
                    safety_speed_deficit_trace.append(
                        float(final_info.get("safety_speed_deficit", 0.0))
                    )
                    safety_rear_ttc_trace.append(float(final_info.get("safety_rear_ttc", np.nan)))
                if "potential_field_cost" in final_info:
                    potential_field_cost_trace.append(float(final_info.get("potential_field_cost", 0.0)))
                    potential_field_penalty_trace.append(float(final_info.get("potential_field_penalty", 0.0)))
                    potential_field_vehicle_count_trace.append(
                        float(final_info.get("potential_field_vehicle_count", 0.0))
                    )
                    potential_field_max_vehicle_cost_trace.append(
                        float(final_info.get("potential_field_max_vehicle_cost", 0.0))
                    )
                    potential_field_closest_longitudinal_gap_trace.append(
                        float(final_info.get("potential_field_closest_longitudinal_gap", np.nan))
                    )
                    potential_field_closest_lateral_gap_trace.append(
                        float(final_info.get("potential_field_closest_lateral_gap", np.nan))
                    )
                if "lane_change_safety_penalty" in final_info:
                    lane_change_penalty = float(final_info.get("lane_change_safety_penalty", 0.0))
                    lane_change_safety_penalty_trace.append(lane_change_penalty)
                    lane_change_safety_risky_count += int(lane_change_penalty > 0.0)

            collision = bool(
                final_info.get("crashed", getattr(env.unwrapped.vehicle, "crashed", False))
            )
            summary = {
                "episode": int(episode_idx + 1),
                "reward": float(total_reward),
                "collision": bool(collision),
                "avg_speed": float(np.mean(speed_trace)) if speed_trace else 0.0,
                "overtakes": int(len(overtaken_ids)),
                "avg_ttc": float(np.mean(ttc_trace)) if ttc_trace else float(ttc_cap),
                "min_ttc": float(np.min(ttc_trace)) if ttc_trace else float(ttc_cap),
            }
            if "driver_aggressiveness_mean" in final_info:
                summary.update(
                    {
                        "driver_aggressiveness_mean": float(final_info["driver_aggressiveness_mean"]),
                        "driver_aggressiveness_min": float(final_info["driver_aggressiveness_min"]),
                        "driver_aggressiveness_max": float(final_info["driver_aggressiveness_max"]),
                    }
                )
            if adaptive_delta_trace:
                summary.update(
                    {
                        "adaptive_longitudinal_steps": int(adaptive_longitudinal_steps),
                        "adaptive_avg_speed_delta": float(np.mean(adaptive_delta_trace)),
                        "adaptive_avg_target_speed": float(np.nanmean(adaptive_target_speed_trace)),
                        "adaptive_avg_controller_ttc": float(np.nanmean(adaptive_ttc_trace)),
                        "adaptive_avg_safety_penalty": float(np.mean(adaptive_safety_penalty_trace)),
                        "adaptive_unsafe_speed_requests": int(adaptive_unsafe_request_count),
                    }
                )
            if traffic_penalty_trace:
                summary.update(
                    {
                        "traffic_avg_flow_penalty": float(np.nanmean(traffic_penalty_trace)),
                        "traffic_avg_rear_ttc": float(np.nanmean(traffic_rear_ttc_trace)),
                        "traffic_avg_flow_speed": float(np.nanmean(traffic_flow_speed_trace)),
                        "traffic_avg_speed_deficit": float(np.nanmean(traffic_speed_deficit_trace)),
                    }
                )
            if safety_shaping_trace:
                summary.update(
                    {
                        "safety_avg_reward_shaping": float(np.nanmean(safety_shaping_trace)),
                        "safety_avg_ttc_bonus": float(np.nanmean(safety_ttc_bonus_trace)),
                        "safety_avg_low_ttc_penalty": float(np.nanmean(safety_low_ttc_penalty_trace)),
                        "safety_avg_lag_penalty": float(np.nanmean(safety_lag_penalty_trace)),
                        "safety_avg_flow_speed": float(np.nanmean(safety_flow_speed_trace)),
                        "safety_avg_speed_deficit": float(np.nanmean(safety_speed_deficit_trace)),
                        "safety_avg_rear_ttc": float(np.nanmean(safety_rear_ttc_trace)),
                    }
                )
            if potential_field_cost_trace:
                summary.update(
                    {
                        "potential_field_avg_cost": float(np.nanmean(potential_field_cost_trace)),
                        "potential_field_avg_penalty": float(np.nanmean(potential_field_penalty_trace)),
                        "potential_field_avg_vehicle_count": float(
                            np.nanmean(potential_field_vehicle_count_trace)
                        ),
                        "potential_field_avg_max_vehicle_cost": float(
                            np.nanmean(potential_field_max_vehicle_cost_trace)
                        ),
                        "potential_field_avg_closest_longitudinal_gap": float(
                            np.nanmean(potential_field_closest_longitudinal_gap_trace)
                        ),
                        "potential_field_avg_closest_lateral_gap": float(
                            np.nanmean(potential_field_closest_lateral_gap_trace)
                        ),
                    }
                )
            if lane_change_safety_penalty_trace:
                summary.update(
                    {
                        "lane_change_safety_avg_penalty": float(
                            np.mean(lane_change_safety_penalty_trace)
                        ),
                        "lane_change_safety_risky_actions": int(lane_change_safety_risky_count),
                    }
                )
            summaries.append(summary)
    finally:
        env.close()

    return summaries


def plot_evaluation_metrics(
    summaries: list[dict[str, float | bool]],
    save_path: Path,
) -> None:
    if not summaries:
        return

    episodes = np.arange(1, len(summaries) + 1)
    avg_speed = np.array([float(item["avg_speed"]) for item in summaries], dtype=float)
    overtakes = np.array([float(item["overtakes"]) for item in summaries], dtype=float)
    avg_ttc = np.array([float(item["avg_ttc"]) for item in summaries], dtype=float)
    min_ttc = np.array([float(item["min_ttc"]) for item in summaries], dtype=float)
    collisions = np.array([float(bool(item["collision"])) for item in summaries], dtype=float)
    running_collision_rate = 100.0 * np.cumsum(collisions) / np.arange(1, len(collisions) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(episodes, avg_speed, marker="o", color="tab:green")
    axes[0, 0].set_title("Average Speed")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("m/s")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(episodes, running_collision_rate, marker="o", color="crimson")
    axes[0, 1].set_title("Running Collision Rate")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("%")
    axes[0, 1].set_ylim(0.0, 100.0)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].bar(episodes, overtakes, color="tab:orange")
    axes[1, 0].set_title("Overtakes")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].plot(episodes, min_ttc, marker="o", label="Min TTC", color="tab:blue")
    axes[1, 1].plot(episodes, avg_ttc, marker="o", label="Avg TTC", color="tab:purple")
    axes[1, 1].set_title("Time To Collision")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def resolve_baseline_model_path(model_path: str | Path | None = None) -> Path:
    candidate = Path(model_path or DEFAULT_MODEL_PATH).expanduser().resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find baseline DQN model at {candidate}")


def _policy_panel_env_config() -> dict:
    config = make_config()
    config.update(POLICY_PANEL_RENDER_CONFIG)
    return config


def _viewer_closed(env: gym.Env) -> bool:
    return bool(getattr(env.unwrapped, "done", False)) or getattr(env.unwrapped, "viewer", None) is None


def _safe_render(env: gym.Env) -> bool:
    try:
        env.render()
    except Exception:
        if _viewer_closed(env):
            return False
        raise
    return not _viewer_closed(env)


def _safe_step(env: gym.Env, action) -> tuple[np.ndarray, float, bool, bool, dict] | None:
    try:
        return env.step(action)
    except Exception:
        if _viewer_closed(env):
            return None
        raise


class SB3BaselineAgentAdapter:
    """
    Small adapter that exposes the state-action values needed by the viewer panel.
    """

    def __init__(self, model: DQN, env: gym.Env) -> None:
        self.model = model
        self.env = env
        self.device = model.device
        self.config = {"gamma": float(getattr(model, "gamma", 0.8))}
        self.previous_state: np.ndarray | None = None
        self.last_action: int | None = None
        raw_actions = getattr(env.unwrapped.action_type, "actions", {})
        if isinstance(raw_actions, dict):
            self.action_labels = {int(key): str(value) for key, value in raw_actions.items()}
        else:
            self.action_labels = {index: str(value) for index, value in enumerate(raw_actions)}

    def update(self, state: np.ndarray, action: int | None = None) -> None:
        self.previous_state = np.asarray(state, dtype=np.float32)
        self.last_action = action

    def get_state_action_values(self, state: np.ndarray) -> np.ndarray:
        obs_tensor, _ = self.model.policy.obs_to_tensor(np.asarray(state, dtype=np.float32))
        with torch.no_grad():
            q_values = self.model.policy.q_net(obs_tensor).detach().cpu().numpy()[0]
        return q_values

    def action_distribution(self, state: np.ndarray) -> np.ndarray:
        q_values = self.get_state_action_values(state)
        shifted = q_values - np.max(q_values)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)
        return probabilities


class BaselineDQNGraphics:
    """
    Graphical visualization of the SB3 DQN state-action values.
    """

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (100, 255, 120)

    @classmethod
    def display(cls, agent: SB3BaselineAgentAdapter, surface, sim_surface=None, display_text: bool = True) -> None:
        import pygame

        if agent.previous_state is None:
            return

        q_values = agent.get_state_action_values(agent.previous_state)
        action_distribution = agent.action_distribution(agent.previous_state)
        labels = [agent.action_labels.get(index, str(index)) for index in range(len(q_values))]

        width = surface.get_width()
        height = surface.get_height()
        cell_width = max(width // len(q_values), 1)

        pygame.draw.rect(surface, cls.BLACK, (0, 0, width, height), 0)

        q_min = float(np.min(q_values))
        q_max = float(np.max(q_values))
        if np.isclose(q_min, q_max):
            q_min -= 1.0
            q_max += 1.0
        norm = matplotlib.colors.Normalize(vmin=q_min, vmax=q_max)
        color_map = matplotlib.cm.get_cmap("viridis")

        for action, value in enumerate(q_values):
            color = color_map(norm(float(value)), bytes=True)
            left = cell_width * action
            rect = (left, 0, cell_width, height)
            pygame.draw.rect(surface, color, rect, 0)

            border_color = cls.GREEN if action == int(np.argmax(q_values)) else cls.WHITE
            border_width = 4 if action == agent.last_action else 2
            pygame.draw.rect(surface, border_color, rect, border_width)

            if display_text:
                font = pygame.font.Font(None, 20)
                text_lines = [
                    labels[action],
                    f"Q={value:.2f}",
                    f"p={action_distribution[action]:.2f}",
                ]
                for row_index, line in enumerate(text_lines):
                    text = font.render(line, True, (10, 10, 10), cls.WHITE)
                    surface.blit(text, (left + 10, 12 + row_index * 22))

        footer = pygame.font.Font(None, 22)
        caption = footer.render(
            "Green border = argmax Q, thicker border = executed action",
            True,
            cls.WHITE,
            cls.BLACK,
        )
        surface.blit(caption, (12, height - 28))


def run_policy_panel_visualization(
    model: DQN | None = None,
    *,
    model_path: str | Path | None = None,
    episodes: int = 5,
    max_steps: int | None = 300,
    seed: int = 42,
    stochastic: bool = False,
    display_text: bool = True,
    config: dict | None = None,
) -> list[dict[str, float | bool]]:
    """
    Render the saved baseline DQN with a live policy panel showing Q-values.
    """

    active_model = model
    if active_model is None:
        resolved_model_path = resolve_baseline_model_path(model_path)
        print(f"Loading model from {resolved_model_path} ...", flush=True)
        active_model = DQN.load(str(resolved_model_path))

    env = make_env(
        render_mode="human",
        config=dict(config or _policy_panel_env_config()),
    )
    graphics_agent = SB3BaselineAgentAdapter(model=active_model, env=env)
    episode_summaries: list[dict[str, float | bool]] = []

    try:
        obs, _ = env.reset(seed=seed)
        graphics_agent.update(obs)

        if not _safe_render(env):
            print("Viewer closed before visualization started.", flush=True)
            return episode_summaries

        env.unwrapped.viewer.set_agent_display(
            lambda surface, sim_surface: BaselineDQNGraphics.display(
                graphics_agent,
                surface,
                sim_surface,
                display_text=display_text,
            )
        )

        if not _safe_render(env):
            print("Viewer closed before visualization started.", flush=True)
            return episode_summaries

        for episode in range(int(episodes)):
            if episode > 0:
                obs, _ = env.reset(seed=seed + episode)
                graphics_agent.update(obs)
                if not _safe_render(env):
                    print("Viewer closed by user. Exiting visualization.", flush=True)
                    return episode_summaries

            done = False
            truncated = False
            total_reward = 0.0
            step_count = 0
            final_info: dict = {}

            while not (done or truncated):
                if _viewer_closed(env):
                    print("Viewer closed by user. Exiting visualization.", flush=True)
                    return episode_summaries

                action, _ = active_model.predict(obs, deterministic=not stochastic)
                action_index = int(np.asarray(action).item())
                graphics_agent.update(obs, action_index)

                step_result = _safe_step(env, action)
                if step_result is None:
                    print("Viewer closed by user. Exiting visualization.", flush=True)
                    return episode_summaries

                obs, reward, done, truncated, info = step_result
                total_reward += float(reward)
                final_info = dict(info)
                step_count += 1

                if max_steps is not None and step_count >= int(max_steps):
                    truncated = True

            summary = {
                "episode": int(episode + 1),
                "reward": float(total_reward),
                "steps": int(step_count),
                "collision": bool(final_info.get("crashed", getattr(env.unwrapped.vehicle, "crashed", False))),
                "avg_speed": float(getattr(env.unwrapped.vehicle, "speed", 0.0)),
                "ttc": compute_same_lane_ttc(env),
            }
            if "adaptive_speed_delta" in final_info:
                summary.update(
                    {
                        "adaptive_requested_action": str(final_info.get("adaptive_requested_action")),
                        "adaptive_forwarded_action": str(final_info.get("adaptive_forwarded_action")),
                        "adaptive_speed_delta": float(final_info.get("adaptive_speed_delta", 0.0)),
                        "adaptive_target_speed_after": float(
                            final_info.get("adaptive_target_speed_after", np.nan)
                        ),
                    }
                )
            episode_summaries.append(summary)
            message = (
                f"[visual] episode={summary['episode']}/{episodes} reward={summary['reward']:.2f} "
                f"steps={summary['steps']} collision={summary['collision']} "
                f"avg_speed={summary['avg_speed']:.2f} ttc={summary['ttc']:.2f}"
            )
            if "adaptive_speed_delta" in summary:
                message += (
                    f" adaptive_action={summary['adaptive_requested_action']}->"
                    f"{summary['adaptive_forwarded_action']} delta={summary['adaptive_speed_delta']:.2f}"
                )
            print(message, flush=True)
    finally:
        env.close()

    return episode_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline Leurent DQN on highway-v0")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total training timesteps")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="DQN learning rate")
    parser.add_argument("--buffer-size", type=int, default=15000, help="Replay buffer size")
    parser.add_argument("--learning-starts", type=int, default=200, help="Warmup steps before gradient updates")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.8, help="Discount factor")
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=50,
        help="Target network update interval",
    )
    parser.add_argument("--tau", type=float, default=1.0, help="Soft update coefficient")
    parser.add_argument(
        "--train-freq",
        type=int,
        default=4,
        help="Collect this many environment steps before each DQN training phase",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Gradient steps per DQN training phase",
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.1,
        help="Fraction of total timesteps used for linear epsilon decay",
    )
    parser.add_argument(
        "--exploration-initial-eps",
        type=float,
        default=1.0,
        help="Initial epsilon for epsilon-greedy exploration",
    )
    parser.add_argument(
        "--exploration-final-eps",
        type=float,
        default=0.05,
        help="Final epsilon for epsilon-greedy exploration",
    )
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=DEFAULT_N_ENVS,
        help="Number of parallel highway environments for rollout collection (default: 24)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device passed to SB3 (default: auto)",
    )
    parser.add_argument("--run-name", default="baseline", help="Run name for logs and result folders")
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory for run artifacts",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print a notebook-friendly training progress update every N timesteps",
    )
    parser.add_argument("--verbose", type=int, default=1, help="SB3 verbosity level")
    parser.add_argument("--disable-tensorboard", action="store_true", help="Disable TensorBoard logging")
    return parser.parse_args()


def train_and_evaluate(args: argparse.Namespace, config: dict | None = None) -> dict:
    results_root = Path(args.results_root).resolve()
    run_dir = results_root / args.run_name
    models_dir = run_dir / "models"
    tb_dir = make_tensorboard_dir(run_dir, args.run_name)
    models_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_enabled = not bool(getattr(args, "disable_tensorboard", False))
    if tensorboard_enabled:
        tb_dir.mkdir(parents=True, exist_ok=True)

    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")

    env_config = dict(config or make_config())
    env_kwargs = {"render_mode": None, "config": env_config}
    requested_num_envs = int(args.num_envs)
    use_notebook_fallback = requested_num_envs > 1 and os.name == "nt" and _running_in_notebook()
    effective_num_envs = requested_num_envs
    vec_env_cls = SubprocVecEnv

    if use_notebook_fallback:
        effective_num_envs = min(requested_num_envs, NOTEBOOK_SAFE_MAX_ENVS)
        vec_env_cls = DummyVecEnv
        print(
            "[train] Windows notebook session detected; using "
            f"DummyVecEnv with {effective_num_envs} in-process env(s) instead of "
            f"{requested_num_envs} subprocess env(s) to avoid BrokenPipeError/EOFError.",
            flush=True,
        )
    elif effective_num_envs == 1:
        vec_env_cls = DummyVecEnv

    # Default to subprocess vectorization for script runs, but keep notebooks
    # on in-process environments because Windows Jupyter kernels do not
    # reliably survive SubprocVecEnv worker startup.
    if effective_num_envs == 1:
        env = make_vec_env(
            make_env,
            n_envs=1,
            seed=args.seed,
            env_kwargs=env_kwargs,
            vec_env_cls=DummyVecEnv,
        )
    else:
        print(f"Spawning {effective_num_envs} parallel DQN environments", flush=True)
        env = make_vec_env(
            make_env,
            n_envs=effective_num_envs,
            seed=args.seed,
            env_kwargs=env_kwargs,
            vec_env_cls=vec_env_cls,
        )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=getattr(args, "tau", 1.0),
        gamma=args.gamma,
        train_freq=getattr(args, "train_freq", 4),
        gradient_steps=getattr(args, "gradient_steps", 1),
        target_update_interval=args.target_update_interval,
        exploration_fraction=getattr(args, "exploration_fraction", 0.1),
        exploration_initial_eps=getattr(args, "exploration_initial_eps", 1.0),
        exploration_final_eps=getattr(args, "exploration_final_eps", 0.05),
        tensorboard_log=str(tb_dir) if tensorboard_enabled else None,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
    )

    tb_run_dir = None
    if tensorboard_enabled:
        tb_run_dir = next_tensorboard_run_dir(tb_dir)
        tb_run_dir.mkdir(parents=True, exist_ok=True)
        log_formats = ["stdout", "tensorboard"] if args.verbose >= 1 else ["tensorboard"]
        model.set_logger(configure_logger(str(tb_run_dir), format_strings=log_formats))

    print(f"Starting training for {args.timesteps} timesteps...", flush=True)
    progress_callback = TimestepProgressCallback(
        total_timesteps=args.timesteps,
        every_n_steps=args.progress_every,
    )
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=args.run_name,
        callback=progress_callback,
        progress_bar=False,
    )

    model_path = models_dir / "elurant_dqn"
    model.save(str(model_path))
    print(f"Model saved to {model_path}.zip")

    evaluation_details = evaluate_with_metrics(
        model,
        episodes=args.eval_episodes,
        seed=args.seed + 1000,
        config=env_config,
    )
    mean_reward = float(np.mean([float(item["reward"]) for item in evaluation_details]))
    std_reward = float(np.std([float(item["reward"]) for item in evaluation_details]))
    eval_metrics_path = run_dir / "evaluation_metrics.json"
    eval_plot_path = run_dir / "evaluation_metrics.png"
    eval_metrics_path.write_text(json.dumps(evaluation_details, indent=2), encoding="utf-8")
    plot_evaluation_metrics(evaluation_details, eval_plot_path)
    print(
        f"Evaluation over {args.eval_episodes} episodes: "
        f"mean reward = {mean_reward:.2f}, std = {std_reward:.2f}"
    )

    summary = {
        "run_name": args.run_name,
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "tau": float(getattr(args, "tau", 1.0)),
        "gamma": args.gamma,
        "train_freq": int(getattr(args, "train_freq", 4)),
        "gradient_steps": int(getattr(args, "gradient_steps", 1)),
        "target_update_interval": args.target_update_interval,
        "exploration_fraction": float(getattr(args, "exploration_fraction", 0.1)),
        "exploration_initial_eps": float(getattr(args, "exploration_initial_eps", 1.0)),
        "exploration_final_eps": float(getattr(args, "exploration_final_eps", 0.05)),
        "seed": args.seed,
        "num_envs": requested_num_envs,
        "effective_num_envs": effective_num_envs,
        "vec_env_type": vec_env_cls.__name__,
        "device": args.device,
        "env_config": env_config,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "collision_rate_percent": float(
            100.0 * np.mean([float(bool(item["collision"])) for item in evaluation_details])
        ),
        "mean_avg_speed": float(np.mean([float(item["avg_speed"]) for item in evaluation_details])),
        "mean_overtakes": float(np.mean([float(item["overtakes"]) for item in evaluation_details])),
        "mean_avg_ttc": float(np.mean([float(item["avg_ttc"]) for item in evaluation_details])),
        "mean_min_ttc": float(np.mean([float(item["min_ttc"]) for item in evaluation_details])),
        "evaluation_metrics_path": str(eval_metrics_path),
        "eval_plot_path": str(eval_plot_path),
        "model_path": str(model_path.with_suffix(".zip")),
        "tensorboard_dir": str(tb_run_dir) if tb_run_dir is not None else None,
    }
    for optional_key in [
        "adaptive_longitudinal_steps",
        "adaptive_avg_speed_delta",
        "adaptive_avg_target_speed",
        "adaptive_avg_controller_ttc",
        "traffic_avg_flow_penalty",
        "traffic_avg_rear_ttc",
        "traffic_avg_flow_speed",
        "traffic_avg_speed_deficit",
        "safety_avg_reward_shaping",
        "safety_avg_ttc_bonus",
        "safety_avg_low_ttc_penalty",
        "safety_avg_lag_penalty",
        "safety_avg_flow_speed",
        "safety_avg_speed_deficit",
        "safety_avg_rear_ttc",
        "potential_field_avg_cost",
        "potential_field_avg_penalty",
        "potential_field_avg_vehicle_count",
        "potential_field_avg_max_vehicle_cost",
        "potential_field_avg_closest_longitudinal_gap",
        "potential_field_avg_closest_lateral_gap",
        "lane_change_safety_avg_penalty",
        "lane_change_safety_risky_actions",
    ]:
        values = [float(item[optional_key]) for item in evaluation_details if optional_key in item]
        if values:
            summary[f"mean_{optional_key}"] = float(np.nanmean(values))
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")

    env.close()
    return summary


def main() -> None:
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
