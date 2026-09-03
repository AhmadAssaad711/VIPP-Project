"""Shared orchestration helpers for the structured DQN notebooks.

This module is the boundary between the notebook narrative and the reusable
laned experiment code. It defines the structured environment profiles, merges
native HighwayEnv settings with optional reward/observation wrappers, loads
the selected DQN backend, and provides common training, evaluation, plotting,
and result-table helpers.

The profiles named semi_unstructured and unstructured_stress remain in this
module for historical experiments, but they still use lane counts and
lane-based HighwayEnv semantics. They therefore belong to the structured
track until a genuinely lane-free environment is selected.
"""
from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Image, display
from stable_baselines3 import DQN


ENVIRONMENT_PROFILES: dict[str, dict[str, Any]] = {
    "structured_baseline": {
        "lanes_count": 3,
        "vehicles_count": 20,
        "duration": 40,
        "ego_spacing": 2.0,
        "vehicles_density": 1.0,
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "initial_lane_id": None,
        "offroad_terminal": False,
    },
    "semi_unstructured": {
        "lanes_count": 4,
        "vehicles_count": 28,
        "duration": 45,
        "ego_spacing": 1.5,
        "vehicles_density": 1.3,
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "other_vehicles_type": "highway_env.vehicle.behavior.LinearVehicle",
        "initial_lane_id": None,
        "offroad_terminal": False,
    },
    "unstructured_stress": {
        "lanes_count": 4,
        "vehicles_count": 36,
        "duration": 50,
        "ego_spacing": 1.0,
        "vehicles_density": 1.7,
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
        "initial_lane_id": None,
        "offroad_terminal": False,
    },
}


DEFAULT_REWARD_CONFIG: dict[str, Any] = {
    "collision_reward": -1.0,
    "right_lane_reward": 0.05,
    "high_speed_reward": 0.8,
    "lane_change_reward": -0.05,
    "normalize_reward": True,
}


DEFAULT_SPEED_CONFIG: dict[str, Any] = {
    "reward_speed_range": [25.0, 30.0],
}


DEFAULT_OBSERVATION_CONFIG: dict[str, Any] = {
    "vehicles_count": 5,
    "features": ["presence", "x", "y", "vx", "vy"],
    "absolute": False,
}


DEFAULT_ACTION_CONFIG: dict[str, Any] = {
    "type": "DiscreteMetaAction",
}


DEFAULT_ADAPTIVE_LONGITUDINAL_CONFIG: dict[str, Any] = {
    "enabled": False,
    "mode": "delta",
    "ttc_midpoint": 4.0,
    "ttc_temperature": 1.0,
    "ttc_cap": 10.0,
    "min_target_speed": 10.0,
    "max_target_speed": 35.0,
    "faster_max_delta": 1.25,
    "slower_min_delta": 1.25,
    "slower_max_delta": 2.5,
    "cruise_speed": 28.0,
    "action_speed_delta": 3.0,
}


NOTEBOOK_SUBDIR_ALIASES: dict[str, Path] = {
    "attention_dqn": Path("structured_highway") / "attention_dqn",
    "baseline_dqn": Path("structured_highway") / "baseline_dqn",
    "congested_traffic_policy": Path("congested_traffic"),
}


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src").exists() and (candidate / "notebooks").exists():
            return candidate
        nested = candidate / "highway-rl-decision-making"
        if (nested / "src").exists() and (nested / "notebooks").exists():
            return nested
    raise RuntimeError("Could not locate the project root from the current working directory.")


def resolve_notebook_dir(project_root: Path, notebook_subdir: str) -> Path:
    alias = NOTEBOOK_SUBDIR_ALIASES.get(notebook_subdir, Path(notebook_subdir))
    candidates = [
        project_root / "notebooks" / alias,
        project_root / "notebooks" / notebook_subdir,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_dqn_backend(backend_module: str, notebook_subdir: str, results_subdir: str):
    project_root = find_project_root()
    notebook_dir = resolve_notebook_dir(project_root, notebook_subdir)
    results_dir = project_root / "artifacts" / "dqn" / results_subdir
    script_dir = project_root / "src" / "deep_learning" / "DQN"

    for path in [str(script_dir), str(project_root / "notebooks" / "_shared")]:
        sys.path = [path] + [existing for existing in sys.path if existing != path]

    for module_name in ("adaptive_longitudinal", backend_module):
        sys.modules.pop(module_name, None)

    trainer = importlib.import_module(backend_module)
    trainer = importlib.reload(trainer)
    results_dir.mkdir(parents=True, exist_ok=True)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Notebook dir:", notebook_dir)
    print("Script dir:", script_dir)
    print("Results dir:", results_dir)
    print(f"Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {default_device}")
    return trainer, project_root, notebook_dir, results_dir, default_device


def resolve_environment_config(profile_name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if profile_name not in ENVIRONMENT_PROFILES:
        raise KeyError(f"Unknown environment profile {profile_name!r}. Choose from {sorted(ENVIRONMENT_PROFILES)}")
    return {**ENVIRONMENT_PROFILES[profile_name], **dict(overrides or {})}


def build_env_config(
    *,
    profile_name: str,
    profile_overrides: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
    reward: dict[str, Any] | None = None,
    speed: dict[str, Any] | None = None,
    adaptive_longitudinal: dict[str, Any] | None = None,
    rear_flow: dict[str, Any] | None = None,
    traffic_flow_reward: dict[str, Any] | None = None,
    safety_ttc_flow_reward: dict[str, Any] | None = None,
    potential_field_reward: dict[str, Any] | None = None,
    driver_aggressiveness: dict[str, Any] | None = None,
    driver_aggressiveness_observation: dict[str, Any] | None = None,
    ttc_observation: dict[str, Any] | None = None,
    lane_change_safety: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "observation": {
            "type": "Kinematics",
            **DEFAULT_OBSERVATION_CONFIG,
            **dict(observation or {}),
        },
        "action": {
            **DEFAULT_ACTION_CONFIG,
            **dict(action or {}),
        },
        **resolve_environment_config(profile_name, profile_overrides),
        **DEFAULT_REWARD_CONFIG,
        **dict(reward or {}),
        **DEFAULT_SPEED_CONFIG,
        **dict(speed or {}),
        "adaptive_longitudinal": {
            **DEFAULT_ADAPTIVE_LONGITUDINAL_CONFIG,
            **dict(adaptive_longitudinal or {}),
        },
        "rear_flow": dict(rear_flow or {"enabled": False}),
        "traffic_flow_reward": dict(traffic_flow_reward or {"enabled": False}),
        "safety_ttc_flow_reward": dict(safety_ttc_flow_reward or {"enabled": False}),
        "potential_field_reward": dict(potential_field_reward or {"enabled": False}),
        "driver_aggressiveness": dict(driver_aggressiveness or {"enabled": False}),
        "driver_aggressiveness_observation": dict(driver_aggressiveness_observation or {"enabled": False}),
        "ttc_observation": dict(ttc_observation or {"enabled": False}),
        "lane_change_safety": dict(lane_change_safety or {"enabled": False}),
    }


def build_dqn_args(
    *,
    results_dir: Path,
    run_name: str,
    timesteps: int,
    eval_episodes: int,
    seed: int,
    num_envs: int,
    device: str,
    hyperparameters: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Namespace:
    defaults = {
        "learning_rate": 2.5e-4,
        "buffer_size": 50_000,
        "learning_starts": 2_000,
        "batch_size": 128,
        "gamma": 0.99,
        "target_update_interval": 1_000,
        "tau": 1.0,
        "train_freq": 4,
        "gradient_steps": 4,
        "exploration_fraction": 0.25,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,
        "progress_every": 5_000,
        "verbose": 0,
    }
    return Namespace(
        timesteps=int(timesteps),
        eval_episodes=int(eval_episodes),
        seed=int(seed),
        num_envs=int(num_envs),
        device=device,
        run_name=run_name,
        results_root=str(results_dir),
        **{**defaults, **dict(hyperparameters or {}), **dict(extra or {})},
    )


def metric_columns(eval_df: pd.DataFrame) -> list[str]:
    columns = ["episode", "collision", "avg_speed", "overtakes", "avg_ttc", "min_ttc", "reward", "steps"]
    return [column for column in columns if column in eval_df.columns]


def adaptive_metric_columns(eval_df: pd.DataFrame) -> list[str]:
    columns = [
        "adaptive_longitudinal_steps",
        "adaptive_avg_speed_delta",
        "adaptive_avg_target_speed",
        "adaptive_avg_controller_ttc",
        "adaptive_avg_safety_penalty",
        "adaptive_unsafe_speed_requests",
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
        "driver_aggressiveness_mean",
        "driver_aggressiveness_min",
        "driver_aggressiveness_max",
    ]
    return [column for column in columns if column in eval_df.columns]


def build_metric_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, label, scale in [
        ("avg_speed", "Average Speed (m/s)", 1.0),
        ("collision", "Collision Rate (%)", 100.0),
        ("overtakes", "Overtakes", 1.0),
        ("avg_ttc", "Average TTC (s)", 1.0),
        ("min_ttc", "Minimum TTC (s)", 1.0),
        ("reward", "Reward", 1.0),
    ]:
        values = pd.to_numeric(eval_df[column], errors="coerce").fillna(0.0).astype(float) * scale
        std = float(values.std(ddof=0))
        standard_error = float(std / np.sqrt(max(len(values), 1)))
        row = {
            "metric": label,
            "mean": float(values.mean()),
            "std": std,
            "standard_error": standard_error,
        }
        if column == "collision":
            row["collisions"] = int(pd.to_numeric(eval_df[column], errors="coerce").fillna(0).astype(bool).sum())
            row["episodes"] = int(len(values))
        rows.append(row)
    for column, label in [
        ("lane_change_safety_avg_penalty", "Lane-Change Safety Penalty"),
        ("lane_change_safety_risky_actions", "Risky Lane-Change Actions"),
        ("potential_field_avg_cost", "Potential Field Cost"),
        ("potential_field_avg_penalty", "Potential Field Penalty"),
    ]:
        if column not in eval_df.columns:
            continue
        values = pd.to_numeric(eval_df[column], errors="coerce").fillna(0.0).astype(float)
        std = float(values.std(ddof=0))
        rows.append(
            {
                "metric": label,
                "mean": float(values.mean()),
                "std": std,
                "standard_error": float(std / np.sqrt(max(len(values), 1))),
            }
        )
    return pd.DataFrame(rows)


def plot_metric_summary(eval_df: pd.DataFrame, save_path: Path, title: str) -> pd.DataFrame:
    specs = [
        ("avg_speed", "Average Speed (m/s)", "tab:green", 1.0),
        ("collision", "Collision Rate (%)", "crimson", 100.0),
        ("overtakes", "Overtakes", "tab:orange", 1.0),
        ("avg_ttc", "Average TTC (s)", "tab:purple", 1.0),
        ("min_ttc", "Minimum TTC (s)", "tab:blue", 1.0),
        ("reward", "Reward", "tab:gray", 1.0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    episode_label = f"{len(eval_df)} eps"
    for ax, (column, label, color, scale) in zip(axes.flat, specs):
        values = pd.to_numeric(eval_df[column], errors="coerce").fillna(0.0).astype(float) * scale
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        standard_error = float(std / np.sqrt(max(len(values), 1)))
        error = standard_error if column == "collision" else std
        error_label = "se" if column == "collision" else "std"
        ax.bar([0], [mean], yerr=[error], capsize=8, color=color, alpha=0.85)
        ax.scatter(np.zeros(len(values)), values, color=color, alpha=0.08, s=8)
        ax.set_xticks([0], [episode_label])
        ax.set_title(f"{label}\nmean={mean:.2f}, {error_label}={error:.2f}")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return build_metric_summary(eval_df)


def display_image_if_exists(path: str | Path) -> None:
    image_path = Path(path)
    if image_path.exists():
        display(Image(filename=str(image_path)))
    else:
        print(f"Image preview skipped; file not found: {image_path}")


def train_and_display(trainer, args: Namespace, env_config: dict[str, Any], label: str) -> dict[str, Any]:
    print("Training env config:")
    print(json.dumps(env_config, indent=2))
    summary = trainer.train_and_evaluate(args, config=env_config)
    print(json.dumps(summary, indent=2))
    display_image_if_exists(summary["eval_plot_path"])

    eval_df = pd.read_json(summary["evaluation_metrics_path"])
    run_dir = Path(args.results_root) / args.run_name
    summary_plot_path = run_dir / "training_evaluation_summary.png"
    summary_df = plot_metric_summary(eval_df, summary_plot_path, f"{label} Training Evaluation")
    display_image_if_exists(summary_plot_path)
    display(summary_df.round(3))
    display(eval_df[metric_columns(eval_df)])
    adaptive_columns = adaptive_metric_columns(eval_df)
    if adaptive_columns:
        display(eval_df[["episode", *adaptive_columns]].round(3))
    return summary


def evaluate_saved_model(
    trainer,
    *,
    summary_path: Path,
    env_config: dict[str, Any],
    episodes: int,
    seed: int,
    name: str,
    label: str,
) -> dict[str, Any]:
    summary_path = Path(summary_path).expanduser().resolve()
    if not summary_path.exists():
        raise RuntimeError("Run the training cell once so a saved model exists.")

    print("Saved-model evaluation env config:")
    print(json.dumps(env_config, indent=2))

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_model_path = Path(saved_summary["model_path"]).expanduser()
    model_candidates = [raw_model_path]
    if not raw_model_path.is_absolute():
        model_candidates.append(summary_path.parent / raw_model_path)
    model_candidates.append(summary_path.parent / "models" / raw_model_path.name)
    if raw_model_path.suffix != ".zip":
        model_candidates.extend(Path(f"{candidate}.zip") for candidate in list(model_candidates))
    model_path = next((candidate.resolve() for candidate in model_candidates if candidate.exists()), None)
    if model_path is None:
        searched = "\n".join(f"  - {candidate}" for candidate in model_candidates)
        raise FileNotFoundError(f"Saved model file was not found. Checked:\n{searched}")

    eval_dir = (summary_path.parent / name).resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    model = DQN.load(model_path)
    eval_rows = trainer.evaluate_with_metrics(model, episodes=episodes, seed=seed, render_mode=None, config=env_config)
    eval_df = pd.DataFrame(eval_rows)

    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "metrics.json"
    detail_plot_path = eval_dir / "metrics.png"
    summary_plot_path = eval_dir / "summary.png"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(eval_df.to_json(orient="records", indent=2), encoding="utf-8")
    trainer.plot_evaluation_metrics(eval_rows, detail_plot_path)
    summary_df = plot_metric_summary(eval_df, summary_plot_path, f"{label} Saved-Model Evaluation")

    output = {
        "episodes": int(episodes),
        "seed": int(seed),
        "model_path": str(model_path),
        "env_config": env_config,
        "evaluation_metrics_path": str(metrics_path),
        "detailed_plot_path": str(detail_plot_path),
        "summary_plot_path": str(summary_plot_path),
        "metric_summary": summary_df.to_dict(orient="records"),
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    display_image_if_exists(detail_plot_path)
    display_image_if_exists(summary_plot_path)
    display(summary_df.round(3))
    display(eval_df[metric_columns(eval_df)].head(10))
    adaptive_columns = adaptive_metric_columns(eval_df)
    if adaptive_columns:
        display(eval_df[["episode", *adaptive_columns]].head(10).round(3))
        display(eval_df[adaptive_columns].agg(["mean", "std"]).round(3))
    return output


def show_policy_panel(
    trainer,
    *,
    summary_path: Path,
    env_config: dict[str, Any],
    episodes: int,
    max_steps: int,
    seed: int,
    stochastic: bool,
) -> list[dict[str, Any]]:
    if not summary_path.exists():
        raise RuntimeError("Run the training cell once so a saved model exists.")
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = trainer.run_policy_panel_visualization(
        model_path=saved_summary["model_path"],
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        stochastic=stochastic,
        config=env_config,
    )
    display(pd.DataFrame(rows))
    return rows
