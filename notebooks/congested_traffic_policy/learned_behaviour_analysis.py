"""Learned-behaviour plots for congested traffic policy v2 runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_SPECS: list[dict[str, Any]] = [
    {
        "run_name": "congested_lane_safety_baseline_dqn_v2_100k",
        "label": "Baseline",
        "short_label": "Baseline",
        "backend_module": "elurant_dqn",
        "lane_change_safety": True,
    },
    {
        "run_name": "congested_baseline_dqn_safety_reward_v2_100k",
        "label": "Baseline + safety reward",
        "short_label": "Baseline + safety",
        "backend_module": "elurant_dqn",
        "safety_reward": True,
        "lane_change_safety": False,
    },
    {
        "run_name": "congested_lane_safety_attention_dqn_v2_100k",
        "label": "Attention",
        "short_label": "Attention",
        "backend_module": "attention_dqn",
        "attention": True,
        "lane_change_safety": True,
    },
    {
        "run_name": "congested_lane_safety_attention_safety_reward_v2_100k",
        "label": "Attention + safety reward",
        "short_label": "Attn + safety",
        "backend_module": "attention_dqn",
        "attention": True,
        "safety_reward": True,
        "lane_change_safety": True,
    },
    {
        "run_name": "congested_lane_safety_adaptive_attention_safety_reward_v2_100k",
        "label": "Adaptive + safety reward",
        "short_label": "Adaptive + safety",
        "backend_module": "attention_dqn",
        "attention": True,
        "adaptive": True,
        "safety_reward": True,
        "lane_change_safety": True,
    },
]

METRIC_SPECS: list[tuple[str, str]] = [
    ("collision_rate_percent", "Collision rate (%)"),
    ("avg_speed", "Average speed (m/s)"),
    ("overtakes", "Overtakes"),
    ("avg_ttc", "Average TTC (s)"),
    ("min_ttc", "Minimum TTC (s)"),
    ("reward", "Reward"),
]

DISTRIBUTION_SPECS: list[tuple[str, str]] = [
    ("min_ttc", "Minimum TTC (s)"),
    ("avg_speed", "Average speed (m/s)"),
    ("overtakes", "Overtakes"),
    ("reward", "Reward"),
]

MODEL_COLORS = {
    "Baseline": "#4C78A8",
    "Baseline + safety reward": "#72B7B2",
    "Attention": "#F58518",
    "Attention + safety reward": "#54A24B",
    "Adaptive + safety reward": "#B279A2",
}


def _ordered_labels(run_specs: list[dict[str, Any]]) -> list[str]:
    return [str(spec["label"]) for spec in run_specs]


def _short_labels(run_specs: list[dict[str, Any]]) -> list[str]:
    return [str(spec.get("short_label", spec["label"])) for spec in run_specs]


def _color_for(label: str) -> str:
    return MODEL_COLORS.get(label, "#6B7280")


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _ensure_dqn_script_path(project_root: Path) -> None:
    dqn_script_dir = project_root / "src" / "deep_learning" / "DQN"
    dqn_script_dir_str = str(dqn_script_dir)
    if dqn_script_dir_str not in sys.path:
        sys.path.insert(0, dqn_script_dir_str)


def load_saved_eval_metrics(
    results_dir: Path,
    run_specs: list[dict[str, Any]],
    saved_eval_name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in run_specs:
        metrics_path = (
            results_dir
            / str(spec["run_name"])
            / saved_eval_name
            / "evaluation_metrics.json"
        )
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing saved-evaluation metrics: {metrics_path}")
        frame = pd.read_json(metrics_path)
        frame["model"] = str(spec["label"])
        frame["model_short"] = str(spec.get("short_label", spec["label"]))
        frame["run_name"] = str(spec["run_name"])
        frames.append(frame)

    eval_df = pd.concat(frames, ignore_index=True)
    eval_df["collision_rate_percent"] = (
        pd.to_numeric(eval_df["collision"], errors="coerce").fillna(0.0).astype(float)
        * 100.0
    )
    return eval_df


def build_metric_summary(
    eval_df: pd.DataFrame,
    run_specs: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in run_specs:
        model = str(spec["label"])
        model_df = eval_df[eval_df["model"] == model]
        for metric_key, metric_label in METRIC_SPECS:
            values = pd.to_numeric(model_df[metric_key], errors="coerce").dropna()
            std = float(values.std(ddof=0)) if len(values) else 0.0
            rows.append(
                {
                    "model": model,
                    "model_short": str(spec.get("short_label", model)),
                    "run_name": str(spec["run_name"]),
                    "metric_key": metric_key,
                    "metric": metric_label,
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "std": std,
                    "standard_error": float(std / np.sqrt(max(len(values), 1))),
                    "episodes": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def plot_metric_dashboard(
    metric_summary: pd.DataFrame,
    run_specs: list[dict[str, Any]],
    save_path: Path,
) -> None:
    labels = _ordered_labels(run_specs)
    short_labels = _short_labels(run_specs)
    colors = [_color_for(label) for label in labels]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (metric_key, metric_label) in zip(axes.flat, METRIC_SPECS):
        sub = (
            metric_summary[metric_summary["metric_key"] == metric_key]
            .set_index("model")
            .reindex(labels)
        )
        means = sub["mean"].to_numpy(dtype=float)
        errors = 1.96 * sub["standard_error"].to_numpy(dtype=float)
        ax.bar(short_labels, means, yerr=errors, color=colors, alpha=0.88, capsize=5)
        ax.set_title(metric_label)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle("Saved 1k-Episode Evaluation: Behaviour and Safety Metrics")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_episode_distributions(
    eval_df: pd.DataFrame,
    run_specs: list[dict[str, Any]],
    save_path: Path,
) -> None:
    labels = _ordered_labels(run_specs)
    short_labels = _short_labels(run_specs)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (metric_key, metric_label) in zip(axes.flat, DISTRIBUTION_SPECS):
        data = [
            pd.to_numeric(
                eval_df.loc[eval_df["model"] == label, metric_key],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
            for label in labels
        ]
        box = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(_color_for(label))
            patch.set_alpha(0.65)
        ax.set_xticks(range(1, len(short_labels) + 1))
        ax.set_xticklabels(short_labels, rotation=18)
        ax.set_title(metric_label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Per-Episode Metric Distributions")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_safety_tradeoffs(
    eval_df: pd.DataFrame,
    run_specs: list[dict[str, Any]],
    save_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in run_specs:
        model = str(spec["label"])
        model_df = eval_df[eval_df["model"] == model]
        rows.append(
            {
                "model": model,
                "model_short": str(spec.get("short_label", model)),
                "collision_rate_percent": float(model_df["collision_rate_percent"].mean()),
                "avg_speed": float(pd.to_numeric(model_df["avg_speed"], errors="coerce").mean()),
                "overtakes": float(pd.to_numeric(model_df["overtakes"], errors="coerce").mean()),
                "min_ttc": float(pd.to_numeric(model_df["min_ttc"], errors="coerce").mean()),
                "reward": float(pd.to_numeric(model_df["reward"], errors="coerce").mean()),
            }
        )
    tradeoff_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    tradeoff_specs = [
        ("avg_speed", "collision_rate_percent", "Average speed (m/s)", "Collision rate (%)"),
        ("overtakes", "min_ttc", "Overtakes", "Minimum TTC (s)"),
        ("collision_rate_percent", "reward", "Collision rate (%)", "Reward"),
    ]
    for ax, (x_key, y_key, x_label, y_label) in zip(axes, tradeoff_specs):
        for row in tradeoff_df.itertuples(index=False):
            color = _color_for(str(row.model))
            ax.scatter(getattr(row, x_key), getattr(row, y_key), s=90, color=color)
            ax.annotate(
                str(row.model_short),
                (getattr(row, x_key), getattr(row, y_key)),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
    fig.suptitle("Safety-Performance Tradeoffs")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return tradeoff_df


def _resolve_model_path(run_dir: Path, summary: dict[str, Any], spec: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    raw_model_path = summary.get("model_path")
    if raw_model_path:
        candidates.append(Path(str(raw_model_path)))
    backend = str(spec.get("backend_module", ""))
    if backend == "attention_dqn":
        candidates.append(run_dir / "models" / "attention_dqn.zip")
    candidates.append(run_dir / "models" / "elurant_dqn.zip")
    candidates.extend(sorted((run_dir / "models").glob("*.zip")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve model checkpoint for {run_dir}")


def _action_label(env: Any, action: Any) -> str:
    action_index = int(np.asarray(action).item())
    actions = getattr(env.unwrapped.action_type, "actions", None)
    if isinstance(actions, dict):
        if action_index in actions:
            return str(actions[action_index])
        for key, value in actions.items():
            try:
                if int(value) == action_index:
                    return str(key)
            except (TypeError, ValueError):
                continue
    elif actions is not None:
        try:
            return str(actions[action_index])
        except (IndexError, TypeError):
            pass
    return str(action_index)


def _lane_id(env: Any) -> float:
    lane_index = getattr(getattr(env.unwrapped, "vehicle", None), "lane_index", None)
    if lane_index is None or len(lane_index) < 3:
        return np.nan
    return _safe_float(lane_index[2])


def _step_context(
    env: Any,
    action: Any,
    diagnostic_config: dict[str, float],
    lane_indices_fn: Callable,
    lane_metrics_fn: Callable,
) -> dict[str, Any]:
    lanes = lane_indices_fn(env)
    metrics = {
        key: lane_metrics_fn(env, lane_index, diagnostic_config)
        for key, lane_index in lanes.items()
    }
    action_name = _action_label(env, action)
    target_key = {"LANE_LEFT": "left", "LANE_RIGHT": "right"}.get(action_name, "current")
    current = metrics["current"]
    target = metrics[target_key]
    vehicle = getattr(env.unwrapped, "vehicle", None)

    return {
        "action": action_name,
        "speed": _safe_float(getattr(vehicle, "speed", np.nan)),
        "lane": _lane_id(env),
        "current_front_ttc": _safe_float(current.get("front_ttc")),
        "current_rear_ttc": _safe_float(current.get("rear_ttc")),
        "current_front_gap": _safe_float(current.get("front_gap")),
        "current_rear_gap": _safe_float(current.get("rear_gap")),
        "current_lane_quality": _safe_float(current.get("quality")),
        "left_front_ttc": _safe_float(metrics["left"].get("front_ttc")),
        "left_rear_ttc": _safe_float(metrics["left"].get("rear_ttc")),
        "left_lane_change_safe": bool(metrics["left"].get("lane_change_safe", False)),
        "right_front_ttc": _safe_float(metrics["right"].get("front_ttc")),
        "right_rear_ttc": _safe_float(metrics["right"].get("rear_ttc")),
        "right_lane_change_safe": bool(metrics["right"].get("lane_change_safe", False)),
        "target_front_ttc": _safe_float(target.get("front_ttc")),
        "target_rear_ttc": _safe_float(target.get("rear_ttc")),
        "target_lane_change_safe": bool(target.get("lane_change_safe", False)),
        "unsafe_lane_change_action": bool(
            action_name in {"LANE_LEFT", "LANE_RIGHT"}
            and not bool(target.get("lane_change_safe", False))
        ),
    }


def run_step_replay(
    *,
    project_root: Path,
    results_dir: Path,
    results_subdir: str,
    run_specs: list[dict[str, Any]],
    make_congested_env_config: Callable[..., dict[str, Any]],
    load_dqn_backend: Callable[..., tuple[Any, Any, Any, Path, str]],
    episodes: int,
    seed: int,
) -> pd.DataFrame:
    _ensure_dqn_script_path(project_root)
    from stable_baselines3 import DQN

    from congestion_diagnostics import lane_indices, lane_metrics, merge_diagnostic_config

    diagnostic_config = merge_diagnostic_config(
        {
            "front_ttc_safe": 4.0,
            "front_ttc_critical": 1.5,
            "rear_ttc_safe": 4.0,
            "rear_ttc_critical": 1.5,
            "lane_gap_safe": 12.0,
        }
    )
    info_float_fields = [
        "adaptive_speed_delta",
        "adaptive_target_speed_after",
        "adaptive_ttc",
        "adaptive_safety_override_penalty",
        "safety_reward_shaping",
        "safety_ttc_bonus",
        "safety_low_ttc_penalty",
        "safety_lag_penalty",
        "safety_flow_speed",
        "safety_speed_deficit",
        "safety_rear_ttc",
        "lane_change_safety_penalty",
    ]
    rows: list[dict[str, Any]] = []

    for spec in run_specs:
        trainer, _, _, _, default_device = load_dqn_backend(
            backend_module=str(spec["backend_module"]),
            notebook_subdir="congested_traffic_policy",
            results_subdir=results_subdir,
        )
        run_dir = results_dir / str(spec["run_name"])
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing run summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        env_config = dict(
            summary.get("env_config")
            or make_congested_env_config(
                safety_reward=bool(spec.get("safety_reward", False)),
                adaptive=bool(spec.get("adaptive", False)),
                aggressiveness_state=bool(spec.get("aggressiveness_state", False)),
                lane_change_safety=bool(spec.get("lane_change_safety", True)),
            )
        )
        max_steps = int(
            env_config.get(
                "max_policy_steps",
                int(env_config.get("duration", 40))
                * int(env_config.get("policy_frequency", 1)),
            )
        )
        model = DQN.load(
            str(_resolve_model_path(run_dir, summary, spec)),
            device=default_device,
        )

        for episode_idx in range(int(episodes)):
            env = trainer.make_env(render_mode=None, config=env_config)
            try:
                obs, _ = env.reset(seed=int(seed) + episode_idx)
                terminated = False
                truncated = False
                step = 0
                while not (terminated or truncated) and step < max_steps:
                    action, _ = model.predict(obs, deterministic=True)
                    row = _step_context(
                        env,
                        action,
                        diagnostic_config,
                        lane_indices,
                        lane_metrics,
                    )
                    row.update(
                        {
                            "model": str(spec["label"]),
                            "model_short": str(spec.get("short_label", spec["label"])),
                            "run_name": str(spec["run_name"]),
                            "episode": int(episode_idx + 1),
                            "step": int(step),
                        }
                    )
                    obs, reward, terminated, truncated, info = env.step(action)
                    row["step_reward"] = _safe_float(reward)
                    row["crashed_after_step"] = bool(
                        info.get(
                            "crashed",
                            getattr(getattr(env.unwrapped, "vehicle", None), "crashed", False),
                        )
                    )
                    row["adaptive_requested_action"] = info.get("adaptive_requested_action")
                    row["adaptive_applied_action"] = info.get("adaptive_applied_action")
                    for field in info_float_fields:
                        if field in info:
                            row[field] = _safe_float(info[field])
                    rows.append(row)
                    step += 1
            finally:
                env.close()

            if (episode_idx + 1) % 50 == 0:
                print(
                    f"{spec['label']}: replayed {episode_idx + 1}/{episodes} episodes",
                    flush=True,
                )

    return pd.DataFrame(rows)


def summarize_step_behaviour(step_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = step_df.copy()
    working["is_lane_change"] = working["action"].isin(["LANE_LEFT", "LANE_RIGHT"])
    working["is_faster"] = working["action"].eq("FASTER")
    working["is_slower"] = working["action"].eq("SLOWER")
    working["is_idle"] = working["action"].eq("IDLE")
    working["unsafe_lane_change_action"] = working["unsafe_lane_change_action"].astype(bool)
    working["crashed_after_step"] = working["crashed_after_step"].astype(bool)

    summary = (
        working.groupby(["model", "model_short", "run_name"], sort=False)
        .agg(
            episodes=("episode", "nunique"),
            steps=("step", "count"),
            mean_speed=("speed", "mean"),
            mean_current_front_ttc=("current_front_ttc", "mean"),
            min_current_front_ttc=("current_front_ttc", "min"),
            lane_change_rate=("is_lane_change", "mean"),
            faster_rate=("is_faster", "mean"),
            slower_rate=("is_slower", "mean"),
            idle_rate=("is_idle", "mean"),
            unsafe_lane_change_rate=("unsafe_lane_change_action", "mean"),
            collision_episode_rate=("crashed_after_step", lambda s: s.groupby(working.loc[s.index, "episode"]).max().mean()),
        )
        .reset_index()
    )

    optional_cols = [
        "adaptive_speed_delta",
        "adaptive_target_speed_after",
        "adaptive_ttc",
        "safety_reward_shaping",
        "safety_low_ttc_penalty",
        "safety_rear_ttc",
        "lane_change_safety_penalty",
    ]
    grouped = working.groupby("model", sort=False)
    for column in optional_cols:
        if column in working.columns:
            summary[column + "_mean"] = summary["model"].map(grouped[column].mean())

    episode_summary = (
        working.groupby(["model", "model_short", "run_name", "episode"], sort=False)
        .agg(
            mean_speed=("speed", "mean"),
            min_front_ttc=("current_front_ttc", "min"),
            min_rear_ttc=("current_rear_ttc", "min"),
            lane_changes=("is_lane_change", "sum"),
            unsafe_lane_changes=("unsafe_lane_change_action", "sum"),
            collision=("crashed_after_step", "max"),
            steps=("step", "count"),
        )
        .reset_index()
    )
    return summary, episode_summary


def plot_action_frequencies(
    step_df: pd.DataFrame,
    run_specs: list[dict[str, Any]],
    save_path: Path,
) -> None:
    labels = _ordered_labels(run_specs)
    action_counts = pd.crosstab(
        step_df["model"],
        step_df["action"],
        normalize="index",
    ).reindex(labels)
    canonical_actions = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
    action_order = [
        action
        for action in canonical_actions
        if action in action_counts.columns
    ] + [
        action
        for action in sorted(action_counts.columns)
        if action not in canonical_actions
    ]
    action_counts = action_counts[action_order]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    action_counts.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xticklabels(_short_labels(run_specs), rotation=15)
    ax.set_ylabel("Share of decisions")
    ax.set_title("Deterministic Replay Action Frequencies")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Action", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_ttc_conditioned_behaviour(
    step_df: pd.DataFrame,
    run_specs: list[dict[str, Any]],
    save_path: Path,
) -> pd.DataFrame:
    working = step_df.copy()
    working["front_ttc_bin"] = pd.cut(
        pd.to_numeric(working["current_front_ttc"], errors="coerce"),
        bins=[-0.01, 1.5, 3.0, 4.0, 6.0, 10.01],
        labels=["<=1.5", "1.5-3", "3-4", "4-6", "6-10"],
        include_lowest=True,
    )
    working["lane_change"] = working["action"].isin(["LANE_LEFT", "LANE_RIGHT"])
    working["faster"] = working["action"].eq("FASTER")
    working["slower"] = working["action"].eq("SLOWER")
    working["unsafe_lane_change"] = working["unsafe_lane_change_action"].astype(bool)

    rate_df = (
        working.groupby(["model", "model_short", "front_ttc_bin"], observed=True)
        [["lane_change", "faster", "slower", "unsafe_lane_change"]]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    plot_specs = [
        ("faster", "FASTER rate"),
        ("slower", "SLOWER rate"),
        ("lane_change", "Lane-change rate"),
        ("unsafe_lane_change", "Unsafe lane-change rate"),
    ]
    for ax, (column, title) in zip(axes.flat, plot_specs):
        for spec in run_specs:
            label = str(spec["label"])
            sub = rate_df[rate_df["model"] == label]
            ax.plot(
                sub["front_ttc_bin"].astype(str),
                sub[column],
                marker="o",
                label=str(spec.get("short_label", label)),
                color=_color_for(label),
            )
        ax.set_title(title)
        ax.set_ylabel("Decision rate")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper right")
    axes[1, 0].set_xlabel("Current front TTC bin (s)")
    axes[1, 1].set_xlabel("Current front TTC bin (s)")
    fig.suptitle("Action Choice Conditioned on Front TTC")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return rate_df


def plot_matched_episode_trace(
    step_df: pd.DataFrame,
    episode_summary: pd.DataFrame,
    run_specs: list[dict[str, Any]],
    save_path: Path,
) -> int:
    baseline_label = str(run_specs[0]["label"])
    baseline_episodes = episode_summary[episode_summary["model"] == baseline_label].copy()
    baseline_episodes = baseline_episodes.sort_values(
        ["collision", "min_front_ttc"],
        ascending=[False, True],
    )
    trace_episode = int(baseline_episodes.iloc[0]["episode"])
    trace_df = step_df[step_df["episode"] == trace_episode].copy()

    canonical_actions = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
    action_order = [
        action
        for action in canonical_actions
        if action in set(trace_df["action"])
    ] + [
        action
        for action in sorted(set(trace_df["action"]))
        if action not in canonical_actions
    ]
    action_to_y = {action: idx for idx, action in enumerate(action_order)}
    trace_df["action_y"] = trace_df["action"].map(action_to_y)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for spec in run_specs:
        model = str(spec["label"])
        sub = trace_df[trace_df["model"] == model]
        color = _color_for(model)
        short_label = str(spec.get("short_label", model))
        axes[0].plot(sub["step"], sub["speed"], label=short_label, color=color)
        axes[1].plot(sub["step"], sub["current_front_ttc"], label=short_label, color=color)
        axes[2].step(sub["step"], sub["lane"], where="post", label=short_label, color=color)
        axes[3].step(sub["step"], sub["action_y"], where="post", label=short_label, color=color)

    axes[0].set_ylabel("Speed (m/s)")
    axes[1].set_ylabel("Front TTC (s)")
    axes[1].axhline(1.5, color="crimson", linestyle="--", linewidth=1, alpha=0.6)
    axes[1].axhline(4.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    axes[2].set_ylabel("Lane")
    axes[3].set_ylabel("Action")
    axes[3].set_yticks(list(action_to_y.values()))
    axes[3].set_yticklabels(list(action_to_y.keys()))
    axes[3].set_xlabel("Decision step")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=2)
    fig.suptitle(f"Matched Replay Episode {trace_episode}: Policy Behaviour Trace")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return trace_episode


def run_learned_behaviour_analysis(
    *,
    project_root: Path,
    results_dir: Path,
    results_subdir: str,
    make_congested_env_config: Callable[..., dict[str, Any]],
    load_dqn_backend: Callable[..., tuple[Any, Any, Any, Path, str]],
    seed: int,
    replay_episodes: int = 200,
    force_replay: bool = False,
    saved_eval_name: str = "saved_model_eval_1000_episodes",
    run_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    specs = list(run_specs or DEFAULT_RUN_SPECS)
    output_dir = results_dir / "learned_behaviour_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = load_saved_eval_metrics(results_dir, specs, saved_eval_name)
    eval_metrics_path = output_dir / "saved_eval_episode_metrics.csv"
    eval_df.to_csv(eval_metrics_path, index=False)

    metric_summary = build_metric_summary(eval_df, specs)
    metric_summary_path = output_dir / "saved_eval_metric_summary.csv"
    metric_summary.to_csv(metric_summary_path, index=False)

    dashboard_path = output_dir / "saved_eval_metric_dashboard.png"
    distributions_path = output_dir / "episode_metric_distributions.png"
    tradeoffs_path = output_dir / "safety_performance_tradeoffs.png"
    plot_metric_dashboard(metric_summary, specs, dashboard_path)
    plot_episode_distributions(eval_df, specs, distributions_path)
    tradeoff_df = plot_safety_tradeoffs(eval_df, specs, tradeoffs_path)
    tradeoff_path = output_dir / "safety_performance_tradeoffs.csv"
    tradeoff_df.to_csv(tradeoff_path, index=False)

    outputs: dict[str, Any] = {
        "output_dir": output_dir,
        "eval_metrics": eval_df,
        "eval_metrics_path": eval_metrics_path,
        "metric_summary": metric_summary,
        "metric_summary_path": metric_summary_path,
        "tradeoff_summary": tradeoff_df,
        "tradeoff_summary_path": tradeoff_path,
        "plot_paths": [dashboard_path, distributions_path, tradeoffs_path],
    }

    if int(replay_episodes) <= 0:
        return outputs

    replay_seed = int(seed) + 70_000
    step_trace_path = output_dir / f"learned_behaviour_step_traces_{replay_episodes}_episodes.csv"
    if step_trace_path.exists() and not force_replay:
        step_df = pd.read_csv(step_trace_path)
    else:
        step_df = run_step_replay(
            project_root=project_root,
            results_dir=results_dir,
            results_subdir=results_subdir,
            run_specs=specs,
            make_congested_env_config=make_congested_env_config,
            load_dqn_backend=load_dqn_backend,
            episodes=int(replay_episodes),
            seed=replay_seed,
        )
        step_df.to_csv(step_trace_path, index=False)

    replay_summary, episode_summary = summarize_step_behaviour(step_df)
    replay_summary_path = output_dir / f"learned_behaviour_replay_summary_{replay_episodes}_episodes.csv"
    episode_summary_path = output_dir / f"learned_behaviour_episode_summary_{replay_episodes}_episodes.csv"
    replay_summary.to_csv(replay_summary_path, index=False)
    episode_summary.to_csv(episode_summary_path, index=False)

    action_frequency_path = output_dir / "replay_action_frequencies.png"
    ttc_conditioned_path = output_dir / "ttc_conditioned_action_rates.png"
    matched_trace_path = output_dir / "matched_episode_behaviour_trace.png"
    plot_action_frequencies(step_df, specs, action_frequency_path)
    ttc_conditioned_df = plot_ttc_conditioned_behaviour(step_df, specs, ttc_conditioned_path)
    ttc_conditioned_path_csv = output_dir / "ttc_conditioned_action_rates.csv"
    ttc_conditioned_df.to_csv(ttc_conditioned_path_csv, index=False)
    trace_episode = plot_matched_episode_trace(
        step_df,
        episode_summary,
        specs,
        matched_trace_path,
    )

    outputs.update(
        {
            "replay_seed": replay_seed,
            "step_traces": step_df,
            "step_trace_path": step_trace_path,
            "replay_summary": replay_summary,
            "replay_summary_path": replay_summary_path,
            "episode_summary": episode_summary,
            "episode_summary_path": episode_summary_path,
            "ttc_conditioned_rates": ttc_conditioned_df,
            "ttc_conditioned_rates_path": ttc_conditioned_path_csv,
            "matched_trace_episode": trace_episode,
        }
    )
    outputs["plot_paths"].extend(
        [action_frequency_path, ttc_conditioned_path, matched_trace_path]
    )
    return outputs
