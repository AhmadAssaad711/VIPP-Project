from __future__ import annotations

"""Evaluation-only critical-damping sweep for the learned DDPG-CBF policy.

The active shield implements

    h_ddot + k1 h_dot + k0 h >= 0.

For the requested critical-damping restriction c1 = c2 = c, this script
evaluates k0 = c**2 and k1 = 2*c while keeping the learned policy, paired
scenario seeds, environment, and evaluation horizon fixed.
"""

import argparse
import faulthandler
import json
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


DEFAULT_C_VALUES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
DEFAULT_BOUNDARY_THRESHOLD_M = 0.5

# The notebook contains markdown cells between the executable CBF cells.  Keep
# the indices explicit and adjacent to the implementation so a notebook edit
# does not silently execute the markdown-only slots used by the old script.
NOTEBOOK_CODE_CELLS = (2, 3, 5, 6, 8, 36, 38, 40, 42, 44)


def set_stable_native_defaults() -> None:
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ]:
        os.environ.setdefault(key, "1")


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
        nested = candidate / "highway-rl-decision-making"
        if (nested / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return nested
    raise RuntimeError("Could not find project root containing notebooks/lanelessKaralakou.ipynb")


def exec_notebook_cells(notebook_path: Path, cell_indices: Iterable[int], namespace: dict[str, Any]) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in cell_indices:
        cell = notebook["cells"][cell_index]
        if cell.get("cell_type") != "code":
            raise RuntimeError(f"Expected executable notebook cell at index {cell_index}")
        source = "".join(cell.get("source", []))
        print(f"[cbf_critical_c_sweep] executing notebook cell {cell_index}", flush=True)
        exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def parse_c_values(value: str | None) -> list[float]:
    raw_values = DEFAULT_C_VALUES if value is None else tuple(float(part.strip()) for part in value.split(",") if part.strip())
    values = [float(item) for item in raw_values]
    if not values:
        raise ValueError("At least one c value is required.")
    if any(not np.isfinite(item) or item <= 0.0 for item in values):
        raise ValueError("All c values must be finite and strictly positive.")
    if len(set(values)) != len(values):
        raise ValueError("c values must be unique so each row identifies one controller.")
    return values


def critical_damping_candidates(c_values: Iterable[float]) -> list[dict[str, float | str]]:
    """Return the exact HOCBF gain mapping for c1=c2=c."""
    candidates: list[dict[str, float | str]] = []
    for c in c_values:
        c = float(c)
        candidates.append(
            {
                "label": f"c={c:g}",
                "c": c,
                "c1": c,
                "c2": c,
                "k0": c**2,
                "k1": 2.0 * c,
            }
        )
    return candidates


def set_cbf_gains(env: Any, k0: float, k1: float) -> None:
    current = env
    while current is not None:
        if hasattr(current, "k0") and hasattr(current, "k1"):
            current.k0 = float(k0)
            current.k1 = float(k1)
            return
        current = getattr(current, "env", None)
    raise RuntimeError("Could not find SafetyFilteredAccelerationWrapper to set k0/k1.")


def configure_observation_layout_for_model(namespace: dict[str, Any], model: Any) -> bool:
    """Match the current environment's flat observation layout to the checkpoint.

    The legacy DDPG-CBF checkpoint bundled with this project is a 42-feature
    policy (six rows x seven features), while the current notebook default is
    the newer 30-feature layout (six rows x five features).  The two layouts
    use the same physical environment and filter; selecting the layout from
    the frozen checkpoint prevents an accidental policy/environment mismatch.
    """
    shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if not shape:
        raise ValueError("Learned policy does not expose a flat observation-space shape.")
    observation_size = int(np.prod(shape))
    env_config = namespace["ENV_CONFIG"]
    rows = int(env_config.get("neighbors_count", 5)) + 1
    expected_layouts = {
        rows * 5: False,
        rows * 7: True,
    }
    if observation_size not in expected_layouts:
        raise ValueError(
            f"Checkpoint expects {shape}, but this evaluator supports only "
            f"{rows * 5}D or {rows * 7}D lane-free observations."
        )
    include_dimensions = expected_layouts[observation_size]
    env_config["observation_include_vehicle_dimensions"] = include_dimensions
    return include_dimensions


def _info_float(info: dict[str, Any], keys: tuple[str, ...], default: float = np.nan) -> float:
    for key in keys:
        value = info.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return float(default)


def _finite_min(values: list[float], default: float = np.nan) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(min(finite)) if finite else float(default)


def _finite_mean(values: list[float], default: float = np.nan) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float(default)


def _finite_max(values: list[float], default: float = np.nan) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else float(default)


def physical_clearance_metrics(env: Any) -> tuple[float, float, float]:
    """Return signed rectangle clearance for pairs, boundary, and overall.

    The environment declares a collision when both the longitudinal and
    lateral rectangle gaps are negative.  ``max(gap_x, gap_y)`` is therefore
    the signed distance to that collision set in the axis-aligned geometry:
    positive means separated, zero is contact, and negative means overlap.
    """
    base = env.unwrapped
    ego = getattr(base, "vehicle", None)
    if ego is None:
        return np.nan, np.nan, np.nan

    road_width = float(base.config["road_width"])
    boundary_clearance = min(
        float(ego.position[1]) - 0.5 * float(ego.width),
        road_width - 0.5 * float(ego.width) - float(ego.position[1]),
    )
    pairwise_clearances: list[float] = []
    for vehicle in getattr(base.road, "vehicles", []):
        if vehicle is ego:
            continue
        dx = abs(float(base._signed_distance(ego.position[0], vehicle.position[0])))
        dy = abs(float(vehicle.position[1] - ego.position[1]))
        longitudinal_gap = dx - 0.5 * (float(ego.length) + float(vehicle.length))
        lateral_gap = dy - 0.5 * (float(ego.width) + float(vehicle.width))
        pairwise_clearances.append(max(longitudinal_gap, lateral_gap))
    pairwise_clearance = min(pairwise_clearances) if pairwise_clearances else np.nan
    overall_clearance = min(
        [value for value in (pairwise_clearance, boundary_clearance) if np.isfinite(value)],
        default=np.nan,
    )
    return float(pairwise_clearance), float(boundary_clearance), float(overall_clearance)


def evaluate_candidate(
    namespace: dict[str, Any],
    model: Any,
    candidate: dict[str, float | str],
    episodes: int,
    seed: int,
    boundary_threshold_m: float,
    evaluation_steps: int,
    deterministic: bool = True,
) -> pd.DataFrame:
    """Evaluate one c value on paired seeds and return one row per episode."""
    rows: list[dict[str, float | str]] = []
    c = float(candidate["c"])
    k0 = float(candidate["k0"])
    k1 = float(candidate["k1"])

    for episode in range(int(episodes)):
        episode_seed = int(seed) + episode
        env = namespace["make_cbf_single_env"](
            seed=episode_seed,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
        )
        set_cbf_gains(env, k0, k1)
        # Keep every candidate on the same fixed-length, non-terminating-on-
        # collision horizon. Collision events are still counted in info.
        namespace["configure_paper_evaluation_env"](
            env,
            steps=int(evaluation_steps),
            terminate_on_collision=False,
        )
        obs, _ = env.reset(seed=episode_seed)

        done = False
        step_count = 0
        rewards: list[float] = []
        speeds: list[float] = []
        abs_speed_errors: list[float] = []
        signed_speed_errors: list[float] = []
        corrections: list[float] = []
        jerk_norms: list[float] = []
        hocbf_pairwise_clearances: list[float] = []
        pairwise_physical_clearances: list[float] = []
        boundary_clearances: list[float] = []
        physical_clearances: list[float] = []
        progress_steps: list[float] = []
        near_boundary_steps = 0
        qp_failures = 0
        raw_infeasible_steps = 0
        numerical_interventions = 0
        meaningful_interventions = 0
        ego_collision_events = 0.0
        total_collision_events = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, raw_info = env.step(action)
            info = dict(raw_info)
            base = env.unwrapped

            speed = _info_float(info, ("kpi_speed_mps",), default=float(base.vehicle.vx))
            target_speed = _info_float(
                info,
                ("kpi_target_speed_mps", "karalakou_target_speed"),
                default=float(base.vehicle.desired_speed),
            )
            speed_error = speed - target_speed
            pairwise_h = _info_float(
                info,
                ("kpi_pairwise_h_min", "cbf_min_h", "kpi_h_min"),
            )
            boundary_h = _info_float(
                info,
                ("kpi_boundary_h_min", "cbf_min_boundary_h"),
            )
            if np.isfinite(pairwise_h):
                hocbf_pairwise_clearances.append(pairwise_h)
            physical_pairwise, physical_boundary, physical_overall = physical_clearance_metrics(env)
            if np.isfinite(physical_pairwise):
                pairwise_physical_clearances.append(physical_pairwise)
            if np.isfinite(physical_boundary):
                boundary_clearances.append(physical_boundary)
            if np.isfinite(physical_overall):
                physical_clearances.append(physical_overall)
            if np.isfinite(physical_boundary) and physical_boundary < boundary_threshold_m:
                near_boundary_steps += 1

            correction = _info_float(
                info,
                ("cbf_correction_norm", "kpi_correction_norm"),
                default=0.0,
            )
            jerk_norm = _info_float(info, ("kpi_jerk_norm",))
            progress_step = _info_float(
                info,
                ("kpi_step_progress_m", "karalakou_progress_m"),
                default=0.0,
            )
            qp_success = bool(info.get("cbf_qp_success", True))
            raw_feasible = bool(info.get("cbf_raw_feasible", True))
            numerical_intervention = bool(info.get("cbf_intervened", correction > 1e-6))
            meaningful_intervention = bool(
                info.get("kpi_meaningful_intervention", info.get("cbf_event_intervened", correction > 0.03))
            )

            rewards.append(float(reward))
            speeds.append(speed)
            signed_speed_errors.append(speed_error)
            abs_speed_errors.append(abs(speed_error))
            corrections.append(correction)
            if np.isfinite(jerk_norm):
                jerk_norms.append(jerk_norm)
            progress_steps.append(max(progress_step, 0.0))
            qp_failures += int(not qp_success)
            raw_infeasible_steps += int(not raw_feasible)
            numerical_interventions += int(numerical_intervention)
            meaningful_interventions += int(meaningful_intervention)
            ego_collision_events += max(
                _info_float(info, ("ego_collision_events", "kpi_ego_collision_events"), default=0.0),
                0.0,
            )
            total_collision_events += max(
                _info_float(info, ("collisions", "kpi_total_collision_events"), default=0.0),
                0.0,
            )

            step_count += 1
            done = bool(terminated or truncated)

        episode_time_s = _info_float(
            info,
            ("kpi_episode_time_s",),
            default=float(step_count) * _info_float(info, ("kpi_dt_s",), default=np.nan),
        )
        distance_traveled_m = _info_float(
            info,
            ("kpi_distance_traveled_m", "task_distance_traveled_m"),
            default=float(np.sum(progress_steps)),
        )
        ego_collision_events = max(
            ego_collision_events,
            _info_float(info, ("kpi_episode_ego_collisions",), default=0.0),
        )
        total_collision_events = max(
            total_collision_events,
            _info_float(info, ("kpi_episode_total_collision_events",), default=0.0),
        )
        rows.append(
            {
                "label": str(candidate["label"]),
                "c": c,
                "c1": c,
                "c2": c,
                "k0": k0,
                "k1": k1,
                "episode": float(episode),
                "seed": float(episode_seed),
                "steps": float(step_count),
                "return": float(np.sum(rewards)),
                "collision": float(ego_collision_events > 0.0),
                "ego_collision_events": float(ego_collision_events),
                "total_collision_events": float(total_collision_events),
                "mean_speed_mps": _finite_mean(speeds, default=0.0),
                "mean_signed_speed_error_mps": _finite_mean(signed_speed_errors, default=0.0),
                "mean_abs_speed_error_mps": _finite_mean(abs_speed_errors, default=0.0),
                "distance_traveled_m": float(distance_traveled_m),
                "ego_collisions_per_km": (
                    float(ego_collision_events / (distance_traveled_m / 1000.0))
                    if distance_traveled_m > 0.0
                    else np.nan
                ),
                "total_collisions_per_km": (
                    float(total_collision_events / (distance_traveled_m / 1000.0))
                    if distance_traveled_m > 0.0
                    else np.nan
                ),
                "progress_rate_mps": (
                    float(distance_traveled_m / episode_time_s)
                    if np.isfinite(episode_time_s) and episode_time_s > 0.0
                    else np.nan
                ),
                "episode_time_s": float(episode_time_s),
                "min_hocbf_pairwise_clearance_m": _finite_min(hocbf_pairwise_clearances),
                "min_pairwise_physical_clearance_m": _finite_min(pairwise_physical_clearances),
                "min_boundary_clearance_m": _finite_min(boundary_clearances),
                "min_physical_clearance_m": _finite_min(physical_clearances),
                "raw_action_feasible_rate": float(1.0 - raw_infeasible_steps / step_count) if step_count else 0.0,
                "qp_infeasibility_rate": float(qp_failures / step_count) if step_count else 0.0,
                "qp_failure_steps": float(qp_failures),
                "intervention_rate": float(numerical_interventions / step_count) if step_count else 0.0,
                "meaningful_intervention_rate": (
                    float(meaningful_interventions / step_count) if step_count else 0.0
                ),
                "mean_correction_norm": _finite_mean(corrections, default=0.0),
                "max_correction_norm": _finite_max(corrections, default=0.0),
                "mean_jerk_norm_mps3": _finite_mean(jerk_norms),
                "max_jerk_norm_mps3": _finite_max(jerk_norms),
                "time_near_boundary_fraction": float(near_boundary_steps / step_count) if step_count else 0.0,
                "time_near_boundary_s": (
                    float(near_boundary_steps / step_count * episode_time_s)
                    if step_count and np.isfinite(episode_time_s)
                    else np.nan
                ),
            }
        )
        env.close()
    return pd.DataFrame(rows)


def summarize(episodes: pd.DataFrame, candidates: list[dict[str, float | str]]) -> pd.DataFrame:
    grouped = episodes.groupby(["label", "c", "c1", "c2", "k0", "k1"], as_index=False)
    summary = grouped.agg(
        episodes=("episode", "count"),
        return_mean=("return", "mean"),
        return_std=("return", "std"),
        collision_rate=("collision", "mean"),
        ego_collision_events_mean=("ego_collision_events", "mean"),
        total_collision_events_mean=("total_collision_events", "mean"),
        ego_collision_events_sum=("ego_collision_events", "sum"),
        total_collision_events_sum=("total_collision_events", "sum"),
        distance_traveled_m_sum=("distance_traveled_m", "sum"),
        ego_collisions_per_km_mean=("ego_collisions_per_km", "mean"),
        total_collisions_per_km_mean=("total_collisions_per_km", "mean"),
        mean_speed_mps=("mean_speed_mps", "mean"),
        mean_abs_speed_error_mps=("mean_abs_speed_error_mps", "mean"),
        distance_traveled_m=("distance_traveled_m", "mean"),
        progress_rate_mps=("progress_rate_mps", "mean"),
        min_hocbf_pairwise_clearance_m=("min_hocbf_pairwise_clearance_m", "min"),
        min_pairwise_physical_clearance_m=("min_pairwise_physical_clearance_m", "min"),
        min_boundary_clearance_m=("min_boundary_clearance_m", "min"),
        min_physical_clearance_m=("min_physical_clearance_m", "min"),
        raw_action_feasible_rate=("raw_action_feasible_rate", "mean"),
        qp_infeasibility_rate=("qp_infeasibility_rate", "mean"),
        qp_failure_steps_mean=("qp_failure_steps", "mean"),
        intervention_rate=("intervention_rate", "mean"),
        meaningful_intervention_rate=("meaningful_intervention_rate", "mean"),
        mean_correction_norm=("mean_correction_norm", "mean"),
        max_correction_norm=("max_correction_norm", "max"),
        mean_jerk_norm_mps3=("mean_jerk_norm_mps3", "mean"),
        max_jerk_norm_mps3=("max_jerk_norm_mps3", "max"),
        time_near_boundary_fraction=("time_near_boundary_fraction", "mean"),
        time_near_boundary_s=("time_near_boundary_s", "mean"),
    )
    order = {float(candidate["c"]): index for index, candidate in enumerate(candidates)}
    summary["_order"] = summary["c"].map(lambda value: order.get(float(value), 999))
    summary["ego_collisions_per_km"] = np.where(
        summary["distance_traveled_m_sum"] > 0.0,
        1000.0 * summary["ego_collision_events_sum"] / summary["distance_traveled_m_sum"],
        np.nan,
    )
    summary["total_collisions_per_km"] = np.where(
        summary["distance_traveled_m_sum"] > 0.0,
        1000.0 * summary["total_collision_events_sum"] / summary["distance_traveled_m_sum"],
        np.nan,
    )
    return summary.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, output_path: Path, boundary_threshold_m: float) -> None:
    x = summary["c"].to_numpy(dtype=float)
    panels = [
        ("collision_rate", "Collision Rate", "percent"),
        ("min_physical_clearance_m", "Minimum Physical Clearance (m)", None),
        ("qp_infeasibility_rate", "QP Infeasibility Rate", "percent"),
        ("intervention_rate", "Numerical Intervention Rate", "percent"),
        ("mean_correction_norm", "Mean ||u_safe - u_RL||", None),
        ("progress_rate_mps", "Progress Rate (m/s)", None),
        ("mean_jerk_norm_mps3", "Mean Jerk Norm (m/s³)", None),
        ("time_near_boundary_fraction", f"Time with Boundary Clearance < {boundary_threshold_m:.2f} m", "percent"),
        ("return_mean", "Mean Return", None),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for axis, (column, title, scale) in zip(axes.flat, panels):
        values = summary[column].to_numpy(dtype=float)
        axis.plot(x, values, marker="o", linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("c (c1=c2)")
        axis.grid(True, alpha=0.3)
        if scale == "percent":
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.suptitle("Critical-Damping HOCBF Sweep on a Frozen Learned Policy", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation-only HOCBF critical-damping c sweep.")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=190_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--c-values",
        default=None,
        help="Comma-separated positive c values. Default: 0.5,1,1.5,2,2.5,3,4,5,6,8",
    )
    parser.add_argument("--boundary-threshold-m", type=float, default=DEFAULT_BOUNDARY_THRESHOLD_M)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Physics integration steps per episode. Default: notebook PAPER_EVAL_STEPS.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive when provided.")
    if not np.isfinite(args.boundary_threshold_m) or args.boundary_threshold_m < 0.0:
        raise ValueError("--boundary-threshold-m must be finite and nonnegative.")

    c_values = parse_c_values(args.c_values)
    candidates = critical_damping_candidates(c_values)
    project_root = find_project_root(args.project_root or Path.cwd())
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec_notebook_cells(notebook_path, NOTEBOOK_CODE_CELLS, namespace)
    namespace["DEVICE"] = args.device
    evaluation_steps = int(namespace["PAPER_EVAL_STEPS"] if args.steps is None else args.steps)

    artifact_dir: Path = namespace["ARTIFACT_DIR"]
    default_steps = int(namespace["PAPER_EVAL_STEPS"])
    output_name = "cbf_critical_c_sweep" if evaluation_steps == default_steps else f"cbf_critical_c_sweep_{evaluation_steps}steps"
    output_dir = artifact_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path or Path(namespace["DDPG_CBF_MODEL_PATH"])
    if not model_path.exists():
        raise FileNotFoundError(f"Learned policy not found: {model_path}")

    print(f"[sweep] loading frozen learned policy {model_path}", flush=True)
    model = namespace["DDPG"].load(str(model_path), device=args.device)
    include_dimensions = configure_observation_layout_for_model(namespace, model)
    print(
        f"[sweep] policy observation shape={model.observation_space.shape}; "
        f"observation_include_vehicle_dimensions={include_dimensions}",
        flush=True,
    )
    print(
        f"[sweep] critical mapping: k0=c², k1=2c; candidates={c_values}; "
        f"episodes={args.episodes}; physics_steps={evaluation_steps}; "
        f"paired_seed_start={args.seed}",
        flush=True,
    )

    episode_frames = []
    for candidate in candidates:
        print(
            f"[sweep] {candidate['label']}: c1=c2={float(candidate['c']):.3f} "
            f"k0={float(candidate['k0']):.3f} k1={float(candidate['k1']):.3f}",
            flush=True,
        )
        episode_frames.append(
            evaluate_candidate(
                namespace,
                model,
                candidate,
                args.episodes,
                int(args.seed),
                boundary_threshold_m=float(args.boundary_threshold_m),
                evaluation_steps=evaluation_steps,
            )
        )

    episodes = pd.concat(episode_frames, ignore_index=True)
    summary = summarize(episodes, candidates)
    episodes_path = output_dir / "episodes.csv"
    summary_path = output_dir / "summary.csv"
    config_path = output_dir / "config.json"
    plot_path = output_dir / "summary.png"
    episodes.to_csv(episodes_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, plot_path, float(args.boundary_threshold_m))
    config_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "model_observation_shape": list(model.observation_space.shape),
                "observation_include_vehicle_dimensions": bool(include_dimensions),
                "episodes_per_c": int(args.episodes),
                "physics_steps_per_episode": evaluation_steps,
                "policy_steps_per_episode": int(round(evaluation_steps * float(namespace["ENV_CONFIG"]["policy_frequency"]) / float(namespace["ENV_CONFIG"]["simulation_frequency"]))),
                "seed_start": int(args.seed),
                "paired_seeds": [int(args.seed) + i for i in range(int(args.episodes))],
                "c_values": c_values,
                "critical_damping_mapping": {"c1": "c", "c2": "c", "k0": "c**2", "k1": "2*c"},
                "boundary_threshold_m": float(args.boundary_threshold_m),
                "notebook_code_cells": list(NOTEBOOK_CODE_CELLS),
                "qp_infeasibility_definition": "fraction of evaluation steps with cbf_qp_success=False",
                "intervention_definition": "fraction of steps with ||u_safe-u_RL|| > 1e-6",
                "ego_collisions_per_km_definition": "ego-involved collision-pair events divided by ego distance traveled",
                "total_collisions_per_km_definition": "all new collision-pair events divided by ego distance traveled",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[sweep] wrote {episodes_path}", flush=True)
    print(f"[sweep] wrote {summary_path}", flush=True)
    print(f"[sweep] wrote {config_path}", flush=True)
    print(f"[sweep] wrote {plot_path}", flush=True)
    print(
        summary[
            [
                "c",
                "k0",
                "k1",
                "return_mean",
                "collision_rate",
                "ego_collisions_per_km",
                "total_collisions_per_km",
                "min_physical_clearance_m",
                "qp_infeasibility_rate",
                "intervention_rate",
                "mean_correction_norm",
                "mean_speed_mps",
                "distance_traveled_m",
                "progress_rate_mps",
                "mean_jerk_norm_mps3",
                "time_near_boundary_fraction",
            ]
        ].to_string(index=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
