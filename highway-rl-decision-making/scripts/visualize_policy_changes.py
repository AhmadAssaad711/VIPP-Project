"""Visualize how the saved PPO/CBF policies differ as functions and rollouts.

The script deliberately produces two complementary views:

* common-state probes: every checkpoint sees the same simulator states, so
  action differences are differences in the learned policy map;
* same-seed raw rollouts: every checkpoint starts from the same traffic state,
  then the closed-loop trajectories are allowed to diverge.

The external CBF is disabled during raw rollouts.  Its projection is computed
as a shadow signal and plotted separately from the action that is actually
passed to the simulator.  The ordinary simulator traffic guard remains on.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

for _thread_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(_thread_key, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_final_results_policies as base_analysis
import run_cbf_filter_ablation as protocol
import run_ppo_cbf_progression as progression
from cbf_projection import project_polytope_2d_numpy, split_cbf_context_numpy
from ppo_reward_safety import install_cbf_violation_reward


VARIANT_ORDER = base_analysis.VARIANT_ORDER
VARIANT_LABELS = base_analysis.VARIANT_LABELS
VARIANT_COLORS = base_analysis.VARIANT_COLORS
PAIRS = (
    ("B2_1", "B3_1", "reward only"),
    ("B2_2", "B3_2", "reward + actor feedback"),
    ("B2_3", "B3_3", "actor only"),
    ("B3_1", "B3_2", "explicit differentiable mean term"),
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "final_Results" / "policy_analysis" / "visualizations"
)
DEFAULT_PROBE_SEEDS = 12
DEFAULT_PROBE_STEPS = 80
DEFAULT_PROBE_SEED_START = 1_300_000
DEFAULT_ROLLOUT_SEEDS = (1_300_000, 1_300_001, 1_300_002)
DEFAULT_ROLLOUT_STEPS = 120
DEFAULT_SENSITIVITY_STATES = 500
RAW_TOL = 1e-6
INTERVENTION_THRESHOLD = 0.03


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _as_action(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)[:2]


def _time_to_contact(dx: float, dy: float, dvx: float, dvy: float, radius: float) -> float:
    """First positive constant-velocity contact time for a circular proxy."""

    position = np.asarray([dx, dy], dtype=float)
    velocity = np.asarray([dvx, dvy], dtype=float)
    radius = max(float(radius), 0.0)
    c = float(position @ position - radius * radius)
    if c <= 0.0:
        return 0.0
    a = float(velocity @ velocity)
    if a <= 1e-12:
        return np.inf
    b = float(2.0 * position @ velocity)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return np.inf
    root = float(np.sqrt(max(discriminant, 0.0)))
    candidates = [
        value
        for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
        if value >= 0.0
    ]
    return float(min(candidates)) if candidates else np.inf


def _bootstrap_namespace() -> dict[str, Any]:
    protocol.set_stable_native_defaults()
    try:
        th.set_num_threads(1)
        th.set_num_interop_threads(1)
    except RuntimeError:
        pass
    namespace = protocol.bootstrap_notebook_namespace(PROJECT_ROOT)
    protocol.exec_required_notebook_cells(
        PROJECT_ROOT / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    install_cbf_violation_reward(namespace)
    namespace["DEVICE"] = "cpu"
    return namespace


def _probe_config(specs: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(specs["B3_2"]["config"]["env_config"])
    config["cbf_substep_filtering"] = False
    config["cbf_require_initial_safe_set"] = True
    config["real_time_rendering"] = False
    reward_config = copy.deepcopy(specs["B3_2"]["config"]["reward_config"])
    return config, reward_config


def _make_raw_env(
    namespace: dict[str, Any], config: dict[str, Any], reward_config: dict[str, Any]
) -> Any:
    return progression.make_ppo_cbf_env(
        namespace,
        env_config=copy.deepcopy(config),
        reward_config=copy.deepcopy(reward_config),
        project_inputs=False,
        lambda_delta=0.0,
        lambda_intervention=0.0,
        correction_epsilon=INTERVENTION_THRESHOLD,
        action_rate_penalty_lambda=0.0,
    )


def _find_context_wrapper(env: Any) -> Any:
    current = env
    for _ in range(20):
        if hasattr(current, "current_constraint_system"):
            return current
        if not hasattr(current, "env"):
            break
        current = current.env
    raise RuntimeError("Could not find the CBF context wrapper")


def _neighbor_metrics(
    namespace: dict[str, Any],
    ego: dict[str, float],
    neighbors: list[dict[str, float]],
    *,
    eps_side: float,
    k0: float,
    k1: float,
    road_width: float,
) -> dict[str, Any]:
    """Extract interpretable traffic coordinates from one exact simulator state."""

    records: list[dict[str, Any]] = []
    for neighbor_index, neighbor in enumerate(neighbors):
        try:
            dx, dy, dvx, dvy = namespace["pairwise_relative_state"](ego, neighbor)
            other_acc = np.asarray(
                [float(neighbor.get("ax", 0.0)), float(neighbor.get("ay", 0.0))],
                dtype=float,
            )
            _row, _bound, h_value, center_distance, required_distance = namespace[
                "pairwise_hocbf_constraint"
            ](
                ego,
                neighbor,
                eps_side=float(eps_side),
                k0=float(k0),
                k1=float(k1),
                other_acc=other_acc,
            )
        except (KeyError, TypeError, ValueError, FloatingPointError, ZeroDivisionError):
            continue
        ego_length = float(ego.get("length", 3.5))
        neighbor_length = float(neighbor.get("length", 3.5))
        ego_width = float(ego.get("width", 1.8))
        neighbor_width = float(neighbor.get("width", 1.8))
        footprint_gap = float(dx) - 0.5 * (ego_length + neighbor_length)
        lateral_gap = abs(float(dy)) - 0.5 * (ego_width + neighbor_width)
        records.append(
            {
                "neighbor_index": int(neighbor_index),
                "signed_dx_m": float(dx),
                "dy_m": float(dy),
                "abs_dy_m": abs(float(dy)),
                "dvx_mps": float(dvx),
                "dvy_mps": float(dvy),
                "footprint_gap_m": footprint_gap,
                "lateral_gap_m": lateral_gap,
                "h": float(h_value),
                "center_distance_m": float(center_distance),
                "required_distance_m": float(required_distance),
                "clearance_m": float(center_distance) - float(required_distance),
                "ttc_s": _time_to_contact(
                    float(dx),
                    float(dy),
                    float(dvx),
                    float(dvy),
                    float(required_distance),
                ),
            }
        )

    road_half_width = 0.5 * float(ego.get("width", 1.8))
    ego_y = float(ego.get("y", np.nan))
    ego_vy = float(ego.get("vy", np.nan))
    left_h = ego_y - road_half_width
    right_h = float(road_width) - road_half_width - ego_y
    all_h = [item["h"] for item in records] + [left_h, right_h]
    finite_h = [value for value in all_h if np.isfinite(value)]
    overall_h = float(min(finite_h)) if finite_h else np.nan

    critical = min(records, key=lambda item: item["h"]) if records else None
    front_candidates = [
        item
        for item in records
        if item["signed_dx_m"] >= 0.0 and item["abs_dy_m"] <= 3.0
    ]
    if not front_candidates:
        front_candidates = [item for item in records if item["signed_dx_m"] >= 0.0]
    front = min(front_candidates, key=lambda item: item["footprint_gap_m"]) if front_candidates else None

    def pick(item: dict[str, Any] | None, key: str) -> float:
        return np.nan if item is None else _finite(item.get(key))

    return {
        "overall_h": overall_h,
        "neighbor_h_min": pick(critical, "h"),
        "critical_neighbor_index": -1 if critical is None else int(critical["neighbor_index"]),
        "critical_dx_m": pick(critical, "signed_dx_m"),
        "critical_dy_m": pick(critical, "dy_m"),
        "critical_lateral_gap_m": pick(critical, "lateral_gap_m"),
        "critical_abs_dy_m": pick(critical, "abs_dy_m"),
        "critical_dvx_mps": pick(critical, "dvx_mps"),
        "critical_dvy_mps": pick(critical, "dvy_mps"),
        "critical_clearance_m": pick(critical, "clearance_m"),
        "critical_ttc_s": pick(critical, "ttc_s"),
        "front_gap_m": pick(front, "footprint_gap_m"),
        "front_closing_speed_mps": pick(front, "dvx_mps"),
        "front_dy_m": pick(front, "dy_m"),
        "front_ttc_s": pick(front, "ttc_s"),
        "left_boundary_h": left_h,
        "right_boundary_h": right_h,
        "neighbor_count": int(len(records)),
        "critical_h_type": "neighbor" if critical is not None else "road_boundary",
        "neighbor_records": records,
    }


def _probe_command(controller: str, step: int) -> np.ndarray:
    if controller == "coast":
        return np.asarray([0.0, 0.0], dtype=np.float32)
    if controller == "accelerate":
        return np.asarray([1.5, 0.0], dtype=np.float32)
    if controller == "brake":
        return np.asarray([-1.5, 0.0], dtype=np.float32)
    if controller == "lateral_sweep":
        lateral = 0.65 if (int(step) // 10) % 2 == 0 else -0.65
        return np.asarray([0.40, lateral], dtype=np.float32)
    raise ValueError(f"Unknown probe controller {controller!r}")


def collect_enriched_probe_states(
    env: Any,
    namespace: dict[str, Any],
    *,
    seeds: int,
    steps_per_seed: int,
    seed_start: int,
) -> pd.DataFrame:
    """Collect common, physically grounded states for policy probing."""

    wrapper = _find_context_wrapper(env)
    config = env.unwrapped.config
    controllers = ("coast", "accelerate", "brake", "lateral_sweep")
    records: list[dict[str, Any]] = []
    state_id = 0
    for controller in controllers:
        for seed_index in range(int(seeds)):
            scenario_seed = int(seed_start) + int(seed_index)
            observation, _info = env.reset(seed=scenario_seed)
            observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            for step in range(int(steps_per_seed)):
                ego = dict(namespace["get_ego_state"](env))
                neighbors = [
                    dict(item)
                    for item in namespace["get_neighbor_states"](
                        env, neighbor_range=float(config["sensing_range"])
                    )
                ]
                system = wrapper.current_constraint_system()
                geometry = _neighbor_metrics(
                    namespace,
                    ego,
                    neighbors,
                    eps_side=float(wrapper.eps_side),
                    k0=float(wrapper.k0),
                    k1=float(wrapper.k1),
                    road_width=float(config["road_width"]),
                )
                neutral = project_polytope_2d_numpy(
                    np.zeros(2, dtype=np.float32),
                    system["rows"],
                    system["bounds"],
                    action_low=wrapper.physical_low,
                    action_high=wrapper.physical_high,
                )
                command = _probe_command(controller, step)
                row = {
                    "state_id": int(state_id),
                    "controller": controller,
                    "scenario_seed": int(scenario_seed),
                    "policy_step": int(step),
                    "time_s": float(step) / max(float(config["policy_frequency"]), 1e-9),
                    "action_set_feasible": bool(neutral.feasible),
                    "constraint_count": int(system["rows"].shape[0]),
                    "cbf_row_count": int(len(system["cbf_rows"])),
                    "cbf_min_h": _finite(system.get("min_h")),
                    "cbf_min_boundary_h": _finite(system.get("min_boundary_h")),
                    "command_ax": float(command[0]),
                    "command_ay": float(command[1]),
                    "ego_x_m": float(ego.get("x", np.nan)),
                    "ego_y_m": float(ego.get("y", np.nan)),
                    "ego_vx_mps": float(ego.get("vx", np.nan)),
                    "ego_vy_mps": float(ego.get("vy", np.nan)),
                    "ego_desired_speed_mps": float(ego.get("desired_speed", np.nan)),
                    "observation": observation.copy(),
                }
                row.update(
                    {
                        key: value
                        for key, value in geometry.items()
                        if key != "neighbor_records"
                    }
                )
                records.append(row)
                state_id += 1
                observation, _reward, terminated, truncated, _step_info = env.step(command)
                observation = np.asarray(observation, dtype=np.float32).reshape(-1)
                if bool(terminated) or bool(truncated):
                    break
    if not records:
        raise RuntimeError("The common probe state bank is empty")
    return pd.DataFrame(records)


def _query_stages(model: Any, probe_observation: np.ndarray) -> dict[str, np.ndarray]:
    model_observation = base_analysis._model_observation_from_probe(
        model, np.asarray(probe_observation, dtype=np.float32)
    )
    stages = model.predict_action_stages(model_observation, deterministic=True)
    return {
        "model_observation": model_observation,
        "mu_raw": _as_action(stages["mu_raw"]),
        "mu_safe": _as_action(stages["mu_safe"]),
    }


def _raw_actor_batch(model: Any, observations: np.ndarray) -> np.ndarray:
    """Vectorized raw network means for the feature sensitivity calculation."""

    policy = model.policy
    obs_tensor, _vectorized = policy.obs_to_tensor(
        np.asarray(observations, dtype=np.float32)
    )
    with th.no_grad():
        if hasattr(policy, "_latents"):
            latent_pi, _latent_vf = policy._latents(obs_tensor)
        else:
            features = policy.extract_features(obs_tensor)
            if bool(getattr(policy, "share_features_extractor", True)):
                latent_pi, _latent_vf = policy.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        action = policy.action_net(latent_pi)
    return np.asarray(action.detach().cpu().numpy(), dtype=np.float32).reshape(-1, 2)


def _hocbf_margin(system: dict[str, Any], action: np.ndarray) -> float:
    rows = np.asarray(system.get("cbf_rows", ()), dtype=float).reshape(-1, 2)
    bounds = np.asarray(system.get("cbf_bounds", ()), dtype=float).reshape(-1)
    if rows.size == 0 or bounds.size == 0:
        return np.inf
    slack = bounds - rows @ np.asarray(action, dtype=float).reshape(2)
    return float(np.min(slack))


def _probe_action_table(
    states: pd.DataFrame, models: dict[str, Any]
) -> pd.DataFrame:
    low = np.asarray([-3.0, -3.0], dtype=np.float32)
    high = np.asarray([3.0, 3.0], dtype=np.float32)
    half_range = np.maximum(0.5 * (high - low), 1e-6)
    rows_out: list[dict[str, Any]] = []
    for state in states.itertuples(index=False):
        observation = np.asarray(getattr(state, "observation"), dtype=np.float32)
        _base, rows, bounds, mask = split_cbf_context_numpy(
            observation, layout=base_analysis.PROBE_LAYOUT
        )
        active = mask > 0.5
        common = {
            column: getattr(state, column)
            for column in states.columns
            if column != "observation"
        }
        for variant_id in VARIANT_ORDER:
            stage = _query_stages(models[variant_id], observation)
            mu_raw = stage["mu_raw"].astype(float)
            mu_safe = stage["mu_safe"].astype(float)
            raw_box = np.clip(mu_raw, low, high)
            safe_box = np.clip(mu_safe, low, high)
            external_raw = project_polytope_2d_numpy(
                mu_raw, rows, bounds, mask, action_low=low, action_high=high
            )
            external_internal = project_polytope_2d_numpy(
                mu_safe, rows, bounds, mask, action_low=low, action_high=high
            )
            external_raw_action = np.asarray(external_raw.action, dtype=float)
            external_internal_action = np.asarray(external_internal.action, dtype=float)
            raw_in_box = bool(np.all(mu_raw >= low - RAW_TOL) and np.all(mu_raw <= high + RAW_TOL))
            active_rows = rows[active]
            active_bounds = bounds[active]
            cbf_row_count = int(getattr(state, "cbf_row_count"))
            cbf_rows = active_rows[:cbf_row_count]
            cbf_bounds = active_bounds[:cbf_row_count]
            raw_margin = (
                float(np.min(cbf_bounds - cbf_rows @ mu_raw))
                if cbf_row_count and cbf_rows.size
                else np.inf
            )
            internal_margin = (
                float(np.min(cbf_bounds - cbf_rows @ mu_safe))
                if cbf_row_count and cbf_rows.size
                else np.inf
            )
            external_margin = (
                float(np.min(cbf_bounds - cbf_rows @ external_internal_action))
                if cbf_row_count and cbf_rows.size
                else np.inf
            )
            raw_delta = external_raw_action - raw_box
            internal_delta = mu_safe - mu_raw
            external_internal_delta = external_internal_action - safe_box
            rows_out.append(
                {
                    **common,
                    "variant_id": variant_id,
                    "label": VARIANT_LABELS[variant_id],
                    "raw_ax": float(mu_raw[0]),
                    "raw_ay": float(mu_raw[1]),
                    "internal_ax": float(mu_safe[0]),
                    "internal_ay": float(mu_safe[1]),
                    "raw_box_ax": float(raw_box[0]),
                    "raw_box_ay": float(raw_box[1]),
                    "external_raw_ax": float(external_raw_action[0]),
                    "external_raw_ay": float(external_raw_action[1]),
                    "external_internal_ax": float(external_internal_action[0]),
                    "external_internal_ay": float(external_internal_action[1]),
                    "raw_in_box": raw_in_box,
                    "raw_hocbf_margin": raw_margin,
                    "internal_hocbf_margin": internal_margin,
                    "external_hocbf_margin": external_margin,
                    "raw_feasible": bool(
                        external_raw.feasible and raw_in_box and raw_margin >= -RAW_TOL
                    ),
                    "internal_feasible": bool(
                        external_internal.feasible and internal_margin >= -RAW_TOL
                    ),
                    "internal_mean_delta_ax": float(internal_delta[0]),
                    "internal_mean_delta_ay": float(internal_delta[1]),
                    "internal_mean_correction_norm": float(
                        np.linalg.norm(internal_delta / half_range)
                    ),
                    "external_raw_correction_norm": float(
                        np.linalg.norm(raw_delta / half_range)
                    ),
                    "external_internal_correction_norm": float(
                        np.linalg.norm(external_internal_delta / half_range)
                    ),
                    "external_intervention": bool(
                        np.linalg.norm(external_raw_delta := raw_delta / half_range)
                        > INTERVENTION_THRESHOLD
                    ),
                    "external_projection_feasible": bool(external_raw.feasible),
                    "external_projection_fallback": bool(external_raw.fallback_used),
                    "external_projection_source": str(external_raw.source),
                    "internal_external_projection_error": float(
                        np.linalg.norm(external_internal_action - safe_box)
                    ),
                }
            )
    return pd.DataFrame(rows_out)


def _paired_effects(actions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = (
        "raw_ax",
        "raw_ay",
        "internal_mean_correction_norm",
        "external_raw_correction_norm",
        "raw_hocbf_margin",
        "raw_feasible",
    )
    for left_id, right_id, label in PAIRS:
        left = actions.loc[actions["variant_id"].eq(left_id)].set_index("state_id")
        right = actions.loc[actions["variant_id"].eq(right_id)].set_index("state_id")
        shared = left.join(right, how="inner", lsuffix="_left", rsuffix="_right")
        for metric in metrics:
            left_values = pd.to_numeric(shared[f"{metric}_left"], errors="coerce").to_numpy(dtype=float)
            right_values = pd.to_numeric(shared[f"{metric}_right"], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            difference = right_values[finite] - left_values[finite]
            if not difference.size:
                continue
            rows.append(
                {
                    "pair": f"{left_id}_vs_{right_id}",
                    "left_variant": left_id,
                    "right_variant": right_id,
                    "comparison": label,
                    "metric": metric,
                    "paired_states": int(difference.size),
                    "left_mean": float(np.mean(left_values[finite])),
                    "right_mean": float(np.mean(right_values[finite])),
                    "right_minus_left": float(np.mean(difference)),
                    "right_median_minus_left": float(np.median(difference)),
                }
            )
    return pd.DataFrame(rows)


def _bin_metric(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = frame[[x, y, value]].copy()
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return (
            np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan, dtype=float),
            np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=int),
        )
    data["x_bin"] = pd.cut(data[x], bins=x_edges, labels=False, include_lowest=True)
    data["y_bin"] = pd.cut(data[y], bins=y_edges, labels=False, include_lowest=True)
    data = data.dropna(subset=["x_bin", "y_bin"])
    matrix = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan, dtype=float)
    counts = np.zeros_like(matrix, dtype=int)
    if data.empty:
        return matrix, counts
    grouped = data.groupby(["y_bin", "x_bin"], observed=True)[value].agg(["mean", "count"])
    for (y_index, x_index), item in grouped.iterrows():
        yi = int(y_index)
        xi = int(x_index)
        if 0 <= yi < matrix.shape[0] and 0 <= xi < matrix.shape[1]:
            matrix[yi, xi] = float(item["mean"])
            counts[yi, xi] = int(item["count"])
    return matrix, counts


def _metric_limits(frame: pd.DataFrame, metric: str) -> tuple[float, float]:
    fixed = {
        "raw_ax": (-3.0, 3.0),
        "raw_ay": (-3.0, 3.0),
        "internal_mean_correction_norm": (0.0, 1.5),
        "external_raw_correction_norm": (0.0, 1.5),
    }
    if metric in fixed:
        return fixed[metric]
    values = pd.to_numeric(frame[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return (-1.0, 1.0)
    limit = max(abs(float(values.quantile(0.05))), abs(float(values.quantile(0.95))), 1e-3)
    return (-limit, limit)


def _draw_heatmap(
    axis: Any,
    matrix: np.ndarray,
    counts: np.ndarray,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    margin_zero: bool = False,
) -> Any:
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("#eeeeee")
    image = axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(matrix),
        shading="auto",
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
    )
    if margin_zero and np.isfinite(matrix).any():
        try:
            x_mids = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_mids = 0.5 * (y_edges[:-1] + y_edges[1:])
            axis.contour(x_mids, y_mids, matrix, levels=[0.0], colors="#111111", linewidths=0.8)
        except (ValueError, TypeError):
            pass
    observed = int(np.sum(counts))
    axis.text(
        0.98,
        0.02,
        f"n={observed}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#222222",
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.5},
    )
    axis.grid(alpha=0.12)
    return image


def plot_policy_atlas(
    actions: pd.DataFrame,
    states: pd.DataFrame,
    output_path: Path,
    *,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    title: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    metadata = states.drop(columns=["observation"]).copy()
    frame = actions.merge(metadata, on="state_id", how="left", suffixes=("", "_state"))
    metrics = (
        ("raw_ax", "raw $a_x$", "coolwarm"),
        ("raw_ay", "raw $a_y$", "coolwarm"),
        ("internal_mean_correction_norm", "internal CBF shift", "viridis"),
        ("external_raw_correction_norm", "external CBF correction", "viridis"),
        ("raw_hocbf_margin", "raw HOCBF margin", "RdYlGn"),
    )
    figure, axes = plt.subplots(
        len(VARIANT_ORDER), len(metrics),
        figsize=(20, 23),
        squeeze=False,
        constrained_layout=True,
    )
    for column, (metric, label, cmap) in enumerate(metrics):
        vmin, vmax = _metric_limits(frame, metric)
        for row, variant_id in enumerate(VARIANT_ORDER):
            axis = axes[row, column]
            subset = frame.loc[frame["variant_id"].eq(variant_id)]
            matrix, counts = _bin_metric(
                subset, x=x, y=y, value=metric, x_edges=x_edges, y_edges=y_edges
            )
            image = _draw_heatmap(
                axis,
                matrix,
                counts,
                x_edges=x_edges,
                y_edges=y_edges,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                margin_zero=metric == "raw_hocbf_margin",
            )
            if row == 0:
                axis.set_title(label, fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{VARIANT_LABELS[variant_id]}\n{y_label}", fontsize=8)
            if row == len(VARIANT_ORDER) - 1:
                axis.set_xlabel(x_label)
        figure.colorbar(image, ax=axes[:, column].tolist(), shrink=0.65, pad=0.015)
    figure.suptitle(title, fontsize=16)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_difference_atlas(
    actions: pd.DataFrame,
    states: pd.DataFrame,
    output_path: Path,
    *,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    metadata = states.drop(columns=["observation"]).copy()
    metrics = (
        ("raw_ax", r"$\Delta a_x$"),
        ("raw_ay", r"$\Delta a_y$"),
        ("external_raw_correction_norm", r"$\Delta$ external correction"),
    )
    differences: dict[tuple[str, str], pd.DataFrame] = {}
    global_limits: dict[str, float] = {metric: 1e-3 for metric, _label in metrics}
    for left_id, right_id, label in PAIRS:
        left = actions.loc[actions["variant_id"].eq(left_id), ["state_id", *[item[0] for item in metrics]]]
        right = actions.loc[actions["variant_id"].eq(right_id), ["state_id", *[item[0] for item in metrics]]]
        merged = left.merge(right, on="state_id", suffixes=("_left", "_right")).merge(metadata, on="state_id")
        differences[(left_id, right_id)] = merged
        for metric, _metric_label in metrics:
            values = pd.to_numeric(merged[f"{metric}_right"], errors="coerce") - pd.to_numeric(
                merged[f"{metric}_left"], errors="coerce"
            )
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            if not values.empty:
                global_limits[metric] = max(global_limits[metric], float(np.quantile(np.abs(values), 0.95)))
    figure, axes = plt.subplots(
        len(PAIRS), len(metrics),
        figsize=(16, 15),
        squeeze=False,
        constrained_layout=True,
    )
    for row, (left_id, right_id, label) in enumerate(PAIRS):
        frame = differences[(left_id, right_id)]
        for column, (metric, metric_label) in enumerate(metrics):
            diff_name = f"{metric}_difference"
            frame = frame.copy()
            frame[diff_name] = pd.to_numeric(frame[f"{metric}_right"], errors="coerce") - pd.to_numeric(
                frame[f"{metric}_left"], errors="coerce"
            )
            matrix, counts = _bin_metric(
                frame,
                x=x,
                y=y,
                value=diff_name,
                x_edges=x_edges,
                y_edges=y_edges,
            )
            limit = global_limits[metric]
            image = _draw_heatmap(
                axes[row, column],
                matrix,
                counts,
                x_edges=x_edges,
                y_edges=y_edges,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            if row == 0:
                axes[row, column].set_title(metric_label)
            if column == 0:
                axes[row, column].set_ylabel(f"{right_id} − {left_id}\n{y_label}", fontsize=9)
            if row == len(PAIRS) - 1:
                axes[row, column].set_xlabel(x_label)
        figure.colorbar(image, ax=axes[row, :].tolist(), shrink=0.75, pad=0.02)
    figure.suptitle("Policy deformation maps (right policy minus left policy)", fontsize=16)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _binned_action_difference(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    left_ax: str,
    right_ax: str,
    left_ay: str,
    right_ay: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = frame[[x, y, left_ax, right_ax, left_ay, right_ay]].copy()
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    shape = (len(y_edges) - 1, len(x_edges) - 1)
    dx = np.full(shape, np.nan)
    dy = np.full(shape, np.nan)
    norm = np.full(shape, np.nan)
    mx = np.full(shape, np.nan)
    my = np.full(shape, np.nan)
    if data.empty:
        return dx, dy, norm, mx, my
    data["x_bin"] = pd.cut(data[x], bins=x_edges, labels=False, include_lowest=True)
    data["y_bin"] = pd.cut(data[y], bins=y_edges, labels=False, include_lowest=True)
    data["delta_ax"] = data[right_ax] - data[left_ax]
    data["delta_ay"] = data[right_ay] - data[left_ay]
    for (yi, xi), group in data.dropna(subset=["x_bin", "y_bin"]).groupby(["y_bin", "x_bin"], observed=True):
        yi = int(yi)
        xi = int(xi)
        if 0 <= yi < shape[0] and 0 <= xi < shape[1]:
            dx[yi, xi] = float(group["delta_ax"].mean())
            dy[yi, xi] = float(group["delta_ay"].mean())
            norm[yi, xi] = float(np.hypot(dx[yi, xi], dy[yi, xi]))
            mx[yi, xi] = float(group[left_ax].mean())
            my[yi, xi] = float(group[left_ay].mean())
    return dx, dy, norm, mx, my


def plot_deformation_vectors(
    actions: pd.DataFrame,
    states: pd.DataFrame,
    output_path: Path,
    *,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    metadata = states.drop(columns=["observation"]).copy()
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), squeeze=False, constrained_layout=True)
    x_mids = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_mids = 0.5 * (y_edges[:-1] + y_edges[1:])
    grid_x, grid_y = np.meshgrid(x_mids, y_mids)
    for axis, (left_id, right_id, label) in zip(axes.flat, PAIRS):
        left = actions.loc[actions["variant_id"].eq(left_id)]
        right = actions.loc[actions["variant_id"].eq(right_id)]
        merged = left.merge(right, on="state_id", suffixes=("_left", "_right")).merge(metadata, on="state_id")
        dx, dy, norm, _mx, _my = _binned_action_difference(
            merged,
            x=x,
            y=y,
            x_edges=x_edges,
            y_edges=y_edges,
            left_ax="raw_ax_left",
            right_ax="raw_ax_right",
            left_ay="raw_ay_left",
            right_ay="raw_ay_right",
        )
        background = np.ma.masked_invalid(norm)
        image = axis.pcolormesh(
            x_edges,
            y_edges,
            background,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=max(float(np.nanpercentile(norm, 95)) if np.isfinite(norm).any() else 1.0, 1e-3),
        )
        valid = np.isfinite(dx) & np.isfinite(dy)
        if np.any(valid):
            axis.quiver(
                grid_x[valid],
                grid_y[valid],
                dx[valid],
                dy[valid],
                color="#f5f5f5",
                edgecolor="#111111",
                linewidth=0.35,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.004,
            )
        axis.set_title(f"{right_id} − {left_id}: {label}")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.15)
        figure.colorbar(image, ax=axis, label="$||\\Delta u||$")
    figure.suptitle("Policy deformation vector field over common traffic states", fontsize=16)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _select_snapshot_states(states: pd.DataFrame, count: int = 4) -> list[int]:
    candidates: list[int] = []
    for column, ascending in (
        ("overall_h", True),
        ("front_gap_m", True),
        ("front_closing_speed_mps", False),
        ("critical_clearance_m", True),
    ):
        values = pd.to_numeric(states[column], errors="coerce")
        order = values.sort_values(ascending=ascending).index
        for index in order:
            state_id = int(states.loc[index, "state_id"])
            if state_id not in candidates:
                candidates.append(state_id)
            if len(candidates) >= int(count):
                return candidates
    return candidates[: int(count)]


def plot_action_space_snapshots(
    states: pd.DataFrame,
    actions: pd.DataFrame,
    output_path: Path,
    selected_path: Path,
) -> list[int]:
    selected = _select_snapshot_states(states, count=4)
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), squeeze=False, constrained_layout=True)
    selected_rows: list[dict[str, Any]] = []
    raw_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant], markersize=6)
        for variant in VARIANT_ORDER
    ]
    stage_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="#111111", label="internal mean (B3)", markersize=7),
        Line2D([0], [0], marker="*", linestyle="", color="#111111", label="external projected action", markersize=9),
        Line2D([0, 1], [0, 1], color="#777777", label="correction vector"),
    ]
    for axis, state_id in zip(axes.flat, selected):
        state = states.loc[states["state_id"].eq(state_id)].iloc[0]
        observation = np.asarray(state["observation"], dtype=np.float32)
        _base, rows, bounds, mask = split_cbf_context_numpy(
            observation, layout=base_analysis.PROBE_LAYOUT
        )
        active_rows = rows[mask > 0.5]
        active_bounds = bounds[mask > 0.5]
        grid = np.linspace(-3.0, 3.0, 121)
        gx, gy = np.meshgrid(grid, grid)
        points = np.column_stack([gx.reshape(-1), gy.reshape(-1)])
        feasible = (
            np.all(points @ active_rows.T <= active_bounds.reshape(1, -1) + 1e-6, axis=1)
            if active_rows.size
            else np.ones(points.shape[0], dtype=bool)
        ).reshape(gx.shape)
        axis.contourf(
            gx,
            gy,
            feasible.astype(float),
            levels=(-0.1, 0.5, 1.1),
            colors=("#f4cccc", "#d9ead3"),
            alpha=0.7,
        )
        state_actions = actions.loc[actions["state_id"].eq(state_id)]
        for variant_id in VARIANT_ORDER:
            row = state_actions.loc[state_actions["variant_id"].eq(variant_id)].iloc[0]
            color = VARIANT_COLORS[variant_id]
            raw = np.asarray([row["raw_ax"], row["raw_ay"]], dtype=float)
            external = np.asarray([row["external_internal_ax"], row["external_internal_ay"]], dtype=float)
            axis.scatter(raw[0], raw[1], color=color, s=40, zorder=5)
            axis.plot([raw[0], external[0]], [raw[1], external[1]], color=color, alpha=0.65, linewidth=1.1, zorder=3)
            axis.scatter(external[0], external[1], marker="*", color=color, s=80, zorder=6)
            if variant_id.startswith("B3"):
                internal = np.asarray([row["internal_ax"], row["internal_ay"]], dtype=float)
                axis.scatter(internal[0], internal[1], facecolors="none", edgecolors="#111111", s=62, linewidths=1.2, zorder=7)
            selected_rows.append(
                {
                    "state_id": state_id,
                    "controller": state["controller"],
                    "scenario_seed": state["scenario_seed"],
                    "policy_step": state["policy_step"],
                    "overall_h": state["overall_h"],
                    "front_gap_m": state["front_gap_m"],
                    "front_closing_speed_mps": state["front_closing_speed_mps"],
                    "variant_id": variant_id,
                    "raw_ax": row["raw_ax"],
                    "raw_ay": row["raw_ay"],
                    "internal_ax": row["internal_ax"],
                    "internal_ay": row["internal_ay"],
                    "external_internal_ax": row["external_internal_ax"],
                    "external_internal_ay": row["external_internal_ay"],
                }
            )
        axis.set_xlim(-3.0, 3.0)
        axis.set_ylim(-3.0, 3.0)
        axis.set_xlabel("longitudinal action $a_x$")
        axis.set_ylabel("lateral action $a_y$")
        axis.set_title(
            f"state {state_id} | {state['controller']} | h={float(state['overall_h']):.2f}\n"
            f"front gap={float(state['front_gap_m']):.2f} m, closing={float(state['front_closing_speed_mps']):.2f} m/s"
        )
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    if selected:
        axes.flat[0].legend(handles=raw_handles + stage_handles, fontsize=7, loc="upper left")
    figure.suptitle("Action-space snapshots: raw proposals, internal means, and external projections", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
    return selected


def _feature_names(spec: dict[str, Any]) -> list[str]:
    config = spec["config"]["env_config"]
    features = progression._base_observation_features(config)
    rows = int(config.get("neighbors_count", 5)) + 1
    names = []
    for row in range(rows):
        prefix = "ego" if row == 0 else f"neighbor_{row}"
        names.extend([f"{prefix}_{feature}" for feature in features])
    base_dim = int(spec["signature"]["model_contract"].get("base_observation_dim", len(names)))
    if base_dim > len(names):
        names.extend(["previous_action_ax", "previous_action_ay"][: base_dim - len(names)])
    return names[:base_dim]


def build_feature_sensitivity(
    states: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    models: dict[str, Any],
    *,
    state_count: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    available = states.index.to_numpy()
    selected_indices = rng.choice(available, size=min(int(state_count), len(available)), replace=False)
    selected_observations = [np.asarray(states.loc[index, "observation"], dtype=np.float32) for index in selected_indices]
    rows_out: list[dict[str, Any]] = []
    epsilon = 0.01
    for variant_id in VARIANT_ORDER:
        model = models[variant_id]
        spec = specs[variant_id]
        model_observations = np.stack(
            [base_analysis._model_observation_from_probe(model, observation) for observation in selected_observations]
        ).astype(np.float32)
        base_dim = int(spec["signature"]["model_contract"].get("base_observation_dim", 30))
        names = _feature_names(spec)
        for feature_index in range(base_dim):
            plus = model_observations.copy()
            minus = model_observations.copy()
            plus[:, feature_index] += epsilon
            minus[:, feature_index] -= epsilon
            plus_action = _raw_actor_batch(model, plus)
            minus_action = _raw_actor_batch(model, minus)
            derivative = (plus_action - minus_action) / (2.0 * epsilon)
            for action_index, action_name in enumerate(("ax", "ay")):
                values = derivative[:, action_index]
                rows_out.append(
                    {
                        "variant_id": variant_id,
                        "label": VARIANT_LABELS[variant_id],
                        "feature_index": int(feature_index),
                        "feature": names[feature_index] if feature_index < len(names) else f"feature_{feature_index}",
                        "action": action_name,
                        "states": int(values.size),
                        "mean_signed_derivative": float(np.mean(values)),
                        "median_signed_derivative": float(np.median(values)),
                        "mean_absolute_derivative": float(np.mean(np.abs(values))),
                        "median_absolute_derivative": float(np.median(np.abs(values))),
                    }
                )
    return pd.DataFrame(rows_out)


def plot_feature_sensitivity(sensitivity: pd.DataFrame, output_path: Path) -> None:
    variant_labels = list(VARIANT_ORDER)
    features = list(dict.fromkeys(sensitivity["feature"].tolist()))
    figure, axes = plt.subplots(2, 2, figsize=(22, 12), squeeze=False, constrained_layout=True)
    for row, action in enumerate(("ax", "ay")):
        for column, value_column, title, cmap in (
            (0, "mean_signed_derivative", "mean signed sensitivity", "coolwarm"),
            (1, "median_absolute_derivative", "median absolute sensitivity", "viridis"),
        ):
            axis = axes[row, column]
            matrices = []
            for variant_id in variant_labels:
                row_values = []
                for feature in features:
                    pair = sensitivity.loc[
                        sensitivity["variant_id"].eq(variant_id) & sensitivity["feature"].eq(feature),
                    ]
                    pair = pair.loc[pair["action"].eq(action)]
                    row_values.append(float(pair[value_column].iloc[0]) if not pair.empty else np.nan)
                matrices.append(row_values)
            matrix = np.asarray(matrices, dtype=float)
            if value_column == "mean_signed_derivative":
                limit = max(float(np.nanpercentile(np.abs(matrix), 95)) if np.isfinite(matrix).any() else 1.0, 1e-3)
                vmin, vmax = -limit, limit
            else:
                vmin, vmax = 0.0, max(float(np.nanpercentile(matrix, 95)) if np.isfinite(matrix).any() else 1.0, 1e-3)
            cmap_object = plt.get_cmap(cmap).copy()
            cmap_object.set_bad("#eeeeee")
            image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", interpolation="nearest", cmap=cmap_object, vmin=vmin, vmax=vmax)
            axis.set_yticks(np.arange(len(variant_labels)), variant_labels)
            axis.set_xticks(np.arange(len(features)), features, rotation=55, ha="right", fontsize=8)
            axis.set_title(f"${action}$: {title}")
            axis.set_ylabel("policy")
            axis.grid(False)
            figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle("Which observation features move the raw actor? (finite differences)", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def select_frozen_group(states: pd.DataFrame) -> tuple[str, int]:
    grouped: list[tuple[float, str, int]] = []
    for (controller, seed), group in states.groupby(["controller", "scenario_seed"], sort=False):
        front = pd.to_numeric(group["front_gap_m"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(front):
            grouped.append((float(front.min()), str(controller), int(seed)))
    if grouped:
        _score, controller, seed = min(grouped)
        return controller, seed
    fallback = states.groupby(["controller", "scenario_seed"], sort=False).first().reset_index().iloc[0]
    return str(fallback["controller"]), int(fallback["scenario_seed"])


def plot_frozen_replay(
    states: pd.DataFrame,
    actions: pd.DataFrame,
    output_path: Path,
    matrix_path: Path,
) -> dict[str, Any]:
    controller, seed = select_frozen_group(states)
    selected_states = states.loc[
        states["controller"].eq(controller) & states["scenario_seed"].eq(seed)
    ].sort_values("policy_step")
    selected_ids = selected_states["state_id"].tolist()
    # The action table already carries the common-state metadata.  Avoid a
    # second merge here because duplicate columns (policy_step, time_s, and
    # geometry) would be suffixed and make the replay ordering ambiguous.
    frame = actions.loc[actions["state_id"].isin(selected_ids)].copy()
    frame = frame.sort_values(["variant_id", "policy_step"])
    figure, axes = plt.subplots(3, 2, figsize=(16, 12), squeeze=False, constrained_layout=True)
    panels = (
        ("raw_ax", "raw proposed $a_x$"),
        ("raw_ay", "raw proposed $a_y$"),
        ("internal_mean_correction_norm", "internal projection shift"),
        ("external_raw_correction_norm", "shadow external correction"),
        ("front_gap_m", "front footprint gap (m)"),
        ("raw_hocbf_margin", "raw HOCBF margin"),
    )
    for axis, (metric, label) in zip(axes.flat, panels):
        for variant_id in VARIANT_ORDER:
            subset = frame.loc[frame["variant_id"].eq(variant_id)]
            if subset.empty:
                continue
            axis.plot(
                subset["time_s"],
                subset[metric],
                color=VARIANT_COLORS[variant_id],
                linewidth=1.8,
                label=VARIANT_LABELS[variant_id],
            )
        if "margin" in metric:
            axis.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        axis.set_title(label)
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.suptitle(
        f"Frozen-state action proposals: {controller}, seed {seed} (all policies see identical observations)",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.6), squeeze=False, constrained_layout=True)
    axes = axes[0]
    for axis, metric, title, cmap in (
        (axes[0], "raw_ax", "raw $a_x$", "coolwarm"),
        (axes[1], "raw_ay", "raw $a_y$", "coolwarm"),
        (axes[2], "external_raw_correction_norm", "shadow correction", "viridis"),
    ):
        matrix = np.full((len(VARIANT_ORDER), len(selected_states)), np.nan)
        for row, variant_id in enumerate(VARIANT_ORDER):
            subset = frame.loc[frame["variant_id"].eq(variant_id)].set_index("state_id")
            for col, state_id in enumerate(selected_ids):
                if state_id in subset.index:
                    matrix[row, col] = float(subset.loc[state_id, metric])
        image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", interpolation="nearest", cmap=cmap)
        axis.set_yticks(np.arange(len(VARIANT_ORDER)), VARIANT_ORDER)
        tick_indices = np.linspace(0, max(len(selected_states) - 1, 0), min(8, len(selected_states)), dtype=int)
        axis.set_xticks(tick_indices, [f"{float(selected_states.iloc[index]['time_s']):.1f}" for index in tick_indices], rotation=30)
        axis.set_xlabel("frozen replay time (s)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle("Policy-by-time action proposal matrices", fontsize=15)
    figure.savefig(matrix_path, dpi=180)
    plt.close(figure)
    return {"controller": controller, "scenario_seed": seed, "state_count": len(selected_states)}


def _match_neighbor_token(env: Any, ego_object: Any, neighbor: dict[str, Any]) -> str:
    best_index = -1
    best_distance = np.inf
    for index, vehicle in enumerate(env.road.vehicles):
        if vehicle is ego_object:
            continue
        try:
            dx = float(env._signed_distance(ego_object.position[0], vehicle.position[0]))
            dy = float(vehicle.position[1] - ego_object.position[1])
            distance = (dx - float(neighbor.get("signed_dx", 0.0))) ** 2 + (dy - float(neighbor.get("y", 0.0) - ego_object.position[1])) ** 2
        except (AttributeError, TypeError, ValueError):
            continue
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return f"v{best_index}" if best_index >= 0 else "unknown"


def _vehicle_snapshot(env: Any, ego_object: Any, *, variant_id: str, seed: int, step: int, time_s: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, vehicle in enumerate(env.road.vehicles):
        try:
            relative_x = float(env._signed_distance(ego_object.position[0], vehicle.position[0]))
            relative_y = float(vehicle.position[1] - ego_object.position[1])
            absolute_x = float(vehicle.position[0])
            absolute_y = float(vehicle.position[1])
            vx = float(vehicle.velocity[0])
            vy = float(vehicle.velocity[1])
        except (AttributeError, TypeError, ValueError, IndexError):
            continue
        rows.append(
            {
                "variant_id": variant_id,
                "scenario_seed": int(seed),
                "policy_step": int(step),
                "time_s": float(time_s),
                "vehicle_index": int(index),
                "vehicle_token": f"v{index}",
                "is_ego": bool(vehicle is ego_object),
                "relative_x_m": relative_x,
                "relative_y_m": relative_y,
                "absolute_x_m": absolute_x,
                "absolute_y_m": absolute_y,
                "vx_mps": vx,
                "vy_mps": vy,
            }
        )
    return rows


def collect_closed_loop_traces(
    namespace: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    models: dict[str, Any],
    *,
    scenario_seeds: Iterable[int],
    max_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config, reward_config = _probe_config(specs)
    policy_frequency = float(config["policy_frequency"])
    low = np.asarray([-3.0, -3.0], dtype=np.float32)
    high = np.asarray([3.0, 3.0], dtype=np.float32)
    half_range = np.maximum(0.5 * (high - low), 1e-6)
    trace_rows: list[dict[str, Any]] = []
    vehicle_rows: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        print(f"[visualization] closed-loop trace {variant_id}", flush=True)
        for scenario_seed in scenario_seeds:
            env = _make_raw_env(namespace, config, reward_config)
            try:
                observation, _reset_info = env.reset(seed=int(scenario_seed))
                observation = np.asarray(observation, dtype=np.float32).reshape(-1)
                context_wrapper = _find_context_wrapper(env)
                base_env = env.unwrapped
                last_guard: dict[str, float] = {}
                terminal_outcome = "horizon"
                for step in range(int(max_steps)):
                    ego = dict(namespace["get_ego_state"](env))
                    neighbors = [
                        dict(item)
                        for item in namespace["get_neighbor_states"](
                            env, neighbor_range=float(config["sensing_range"])
                        )
                    ]
                    system = context_wrapper.current_constraint_system()
                    geometry = _neighbor_metrics(
                        namespace,
                        ego,
                        neighbors,
                        eps_side=float(context_wrapper.eps_side),
                        k0=float(context_wrapper.k0),
                        k1=float(context_wrapper.k1),
                        road_width=float(config["road_width"]),
                    )
                    stage = _query_stages(models[variant_id], observation)
                    raw = stage["mu_raw"].astype(float)
                    internal = stage["mu_safe"].astype(float)
                    operational = internal if variant_id.startswith("B3") else raw
                    raw_box = np.clip(raw, low, high)
                    operational_box = np.clip(operational, low, high)
                    external_raw = project_polytope_2d_numpy(
                        raw, system["rows"], system["bounds"], action_low=low, action_high=high
                    )
                    external_operational = project_polytope_2d_numpy(
                        operational, system["rows"], system["bounds"], action_low=low, action_high=high
                    )
                    critical_index = int(geometry["critical_neighbor_index"])
                    critical_token = "unknown"
                    if 0 <= critical_index < len(neighbors):
                        critical_token = _match_neighbor_token(base_env, base_env.vehicle, neighbors[critical_index])
                    time_s = float(step) / max(policy_frequency, 1e-9)
                    vehicle_rows.extend(
                        _vehicle_snapshot(
                            base_env,
                            base_env.vehicle,
                            variant_id=variant_id,
                            seed=int(scenario_seed),
                            step=step,
                            time_s=time_s,
                        )
                    )
                    raw_margin = _hocbf_margin(system, raw)
                    operational_margin = _hocbf_margin(system, operational)
                    external_margin = _hocbf_margin(system, external_operational.action)
                    guard_values: dict[str, float] = {}
                    guard_step_values: dict[str, float] = {}
                    for key in (
                        "traffic_guard_brakes",
                        "traffic_guard_traffic_only",
                        "traffic_guard_lateral_yields",
                        "traffic_guard_ego_emergency_interventions",
                        "traffic_guard_traffic_constraints",
                    ):
                        current = _finite(_reset_info.get(key, 0.0), 0.0)
                        # The info dict from the previous transition is replaced below.
                        guard_values[key] = current
                        guard_step_values[f"{key}_step"] = max(current - last_guard.get(key, 0.0), 0.0)
                    row = {
                        "variant_id": variant_id,
                        "variant_label": VARIANT_LABELS[variant_id],
                        "scenario_seed": int(scenario_seed),
                        "policy_step": int(step),
                        "time_s": time_s,
                        "raw_ax": float(raw[0]),
                        "raw_ay": float(raw[1]),
                        "internal_ax": float(internal[0]),
                        "internal_ay": float(internal[1]),
                        "operational_ax": float(operational[0]),
                        "operational_ay": float(operational[1]),
                        "executed_box_ax": float(operational_box[0]),
                        "executed_box_ay": float(operational_box[1]),
                        "internal_mean_correction_norm": float(np.linalg.norm((internal - raw) / half_range)),
                        "shadow_external_correction_norm": float(np.linalg.norm((np.asarray(external_operational.action) - operational_box) / half_range)),
                        "shadow_external_raw_correction_norm": float(np.linalg.norm((np.asarray(external_raw.action) - raw_box) / half_range)),
                        "raw_hocbf_margin": raw_margin,
                        "operational_hocbf_margin": operational_margin,
                        "shadow_external_hocbf_margin": external_margin,
                        "raw_feasible": bool(external_raw.feasible and raw_margin >= -RAW_TOL),
                        "operational_feasible": bool(external_operational.feasible and operational_margin >= -RAW_TOL),
                        "shadow_external_intervention": bool(np.linalg.norm((np.asarray(external_operational.action) - operational_box) / half_range) > INTERVENTION_THRESHOLD),
                        "critical_vehicle_token": critical_token,
                        "guard_counters_source": "transition_info",
                        **guard_values,
                        **guard_step_values,
                        **{key: value for key, value in geometry.items() if key != "neighbor_records"},
                        "ego_x_m": float(ego.get("x", np.nan)),
                        "ego_y_m": float(ego.get("y", np.nan)),
                        "ego_vx_mps": float(ego.get("vx", np.nan)),
                        "ego_vy_mps": float(ego.get("vy", np.nan)),
                        "collision": False,
                        "outcome": "running",
                    }
                    observation, _reward, terminated, truncated, info = env.step(operational_box.astype(np.float32))
                    info = dict(info)
                    for key in guard_values:
                        guard_values[key] = _finite(info.get(key, guard_values[key]), guard_values[key])
                    # Store the actual post-transition guard counters and the corresponding step deltas.
                    for key, value in guard_values.items():
                        row[key] = value
                        row[f"{key}_step"] = max(value - last_guard.get(key, 0.0), 0.0)
                        last_guard[key] = value
                    collision = bool(
                        info.get("ego_collision", False)
                        or int(info.get("ego_collision_events", 0)) > 0
                        or info.get("task_collision_terminated", False)
                    )
                    row["collision"] = collision
                    trace_rows.append(row)
                    if bool(terminated) or bool(truncated):
                        terminal_outcome = "collision" if collision else ("terminated" if terminated else "truncated")
                        trace_rows[-1]["outcome"] = terminal_outcome
                        break
                if trace_rows:
                    matching = [
                        item
                        for item in trace_rows
                        if item["variant_id"] == variant_id and item["scenario_seed"] == int(scenario_seed)
                    ]
                    if matching:
                        matching[-1]["outcome"] = terminal_outcome
            finally:
                env.close()
    trace = pd.DataFrame(trace_rows)
    vehicles = pd.DataFrame(vehicle_rows)
    if trace.empty:
        raise RuntimeError("Closed-loop trace collection returned no rows")
    summaries: list[dict[str, Any]] = []
    for (variant_id, scenario_seed), group in trace.groupby(["variant_id", "scenario_seed"], sort=False):
        last = group.iloc[-1]
        summaries.append(
            {
                "variant_id": variant_id,
                "variant_label": VARIANT_LABELS[variant_id],
                "scenario_seed": int(scenario_seed),
                "policy_steps": int(len(group)),
                "final_time_s": float(last["time_s"]),
                "episode_min_h": float(pd.to_numeric(group["overall_h"], errors="coerce").min()),
                "episode_min_critical_clearance_m": float(pd.to_numeric(group["critical_clearance_m"], errors="coerce").min()),
                "episode_min_raw_margin": float(pd.to_numeric(group["raw_hocbf_margin"], errors="coerce").min()),
                "episode_min_operational_margin": float(pd.to_numeric(group["operational_hocbf_margin"], errors="coerce").min()),
                "mean_shadow_external_correction": float(pd.to_numeric(group["shadow_external_correction_norm"], errors="coerce").mean()),
                "shadow_intervention_rate": float(pd.to_numeric(group["shadow_external_intervention"], errors="coerce").mean()),
                "mean_internal_mean_correction": float(pd.to_numeric(group["internal_mean_correction_norm"], errors="coerce").mean()),
                "traffic_guard_step_interventions": float(pd.to_numeric(group["traffic_guard_brakes_step"], errors="coerce").sum()),
                "collision": bool(last["collision"]),
                "outcome": str(last["outcome"]),
            }
        )
    return trace, vehicles, pd.DataFrame(summaries)


def _plot_lines(axis: Any, frame: pd.DataFrame, metric: str, *, variants: Iterable[str] = VARIANT_ORDER) -> None:
    for variant_id in variants:
        data = frame.loc[frame["variant_id"].eq(variant_id)].sort_values("policy_step")
        if data.empty:
            continue
        axis.plot(data["time_s"], data[metric], color=VARIANT_COLORS[variant_id], linewidth=1.7, label=VARIANT_LABELS[variant_id])


def plot_closed_loop_all(trace: pd.DataFrame, output_path: Path) -> None:
    seeds = sorted(int(value) for value in trace["scenario_seed"].unique())
    panels = (
        ("critical_clearance_m", "critical geometric clearance (m)"),
        ("critical_ttc_s", "critical TTC (s)"),
        ("operational_hocbf_margin", "operational HOCBF margin"),
        ("shadow_external_correction_norm", "shadow external correction"),
    )
    figure, axes = plt.subplots(len(panels), len(seeds), figsize=(17, 12), squeeze=False, constrained_layout=True)
    for row, (metric, label) in enumerate(panels):
        for column, seed in enumerate(seeds):
            axis = axes[row, column]
            subset = trace.loc[trace["scenario_seed"].eq(seed)]
            _plot_lines(axis, subset, metric)
            if "margin" in metric:
                axis.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
            axis.set_title(f"seed {seed}" if row == 0 else "")
            axis.set_ylabel(label)
            axis.set_xlabel("time (s)")
            axis.grid(alpha=0.2)
            if row == 0 and column == 0:
                axis.legend(fontsize=7, ncol=2)
    figure.suptitle("Same-seed closed-loop raw-policy safety and traffic signals", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_primary_storyboard(trace: pd.DataFrame, output_path: Path) -> None:
    left_id, right_id = "B2_2", "B3_2"
    seeds = sorted(int(value) for value in trace["scenario_seed"].unique())
    figure, axes = plt.subplots(len(seeds), 4, figsize=(18, max(6, 4.2 * len(seeds))), squeeze=False, constrained_layout=True)
    for row, seed in enumerate(seeds):
        subset = trace.loc[trace["scenario_seed"].eq(seed)]
        pair = subset.loc[subset["variant_id"].isin((left_id, right_id))]
        for variant_id in (left_id, right_id):
            data = pair.loc[pair["variant_id"].eq(variant_id)].sort_values("policy_step")
            color = VARIANT_COLORS[variant_id]
            label = "B2.2 non-diff" if variant_id == left_id else "B3.2 diff"
            axes[row, 0].plot(data["time_s"], data["critical_clearance_m"], color=color, linewidth=1.9, label=label)
            axes[row, 1].plot(data["time_s"], data["raw_ax"], color=color, linewidth=1.5, linestyle="-", label=f"{label} raw $a_x$")
            axes[row, 1].plot(data["time_s"], data["operational_ay"], color=color, linewidth=1.4, linestyle="--", label=f"{label} operational $a_y$")
            axes[row, 2].plot(data["time_s"], data["raw_hocbf_margin"], color=color, linewidth=1.4, linestyle="-", label=f"{label} raw margin")
            axes[row, 2].plot(data["time_s"], data["operational_hocbf_margin"], color=color, linewidth=1.4, linestyle="--", label=f"{label} operational margin")
            axes[row, 3].plot(data["time_s"], data["internal_mean_correction_norm"], color=color, linewidth=1.5, linestyle="-", label=f"{label} internal shift")
            axes[row, 3].plot(data["time_s"], data["shadow_external_correction_norm"], color=color, linewidth=1.5, linestyle="--", label=f"{label} shadow external")
        axes[row, 0].axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        axes[row, 2].axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        titles = ("critical clearance", "actions", "raw vs operational HOCBF margin", "policy/filter correction")
        for column, title in enumerate(titles):
            axes[row, column].set_title(f"seed {seed}: {title}")
            axes[row, column].set_xlabel("time (s)")
            axes[row, column].grid(alpha=0.2)
        axes[row, 0].set_ylabel("clearance (m)")
        axes[row, 1].set_ylabel("action")
        axes[row, 2].set_ylabel("margin")
        axes[row, 3].set_ylabel("normalized shift")
        if row == 0:
            axes[row, 0].legend(fontsize=7)
            axes[row, 1].legend(fontsize=6, ncol=2)
            axes[row, 2].legend(fontsize=6, ncol=2)
            axes[row, 3].legend(fontsize=6, ncol=2)
    figure.suptitle("Primary differentiable/non-differentiable traffic interaction: B2.2 vs B3.2", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_closed_loop_actions(trace: pd.DataFrame, output_path: Path) -> None:
    seeds = sorted(int(value) for value in trace["scenario_seed"].unique())
    figure, axes = plt.subplots(2, len(seeds), figsize=(17, 7), squeeze=False, constrained_layout=True)
    for column, seed in enumerate(seeds):
        subset = trace.loc[trace["scenario_seed"].eq(seed)]
        for variant_id in VARIANT_ORDER:
            data = subset.loc[subset["variant_id"].eq(variant_id)].sort_values("policy_step")
            axes[0, column].plot(data["time_s"], data["raw_ax"], color=VARIANT_COLORS[variant_id], linewidth=1.5, label=VARIANT_LABELS[variant_id])
            axes[1, column].plot(data["time_s"], data["raw_ay"], color=VARIANT_COLORS[variant_id], linewidth=1.5, label=VARIANT_LABELS[variant_id])
        axes[0, column].set_title(f"seed {seed}: raw longitudinal proposals")
        axes[1, column].set_title(f"seed {seed}: raw lateral proposals")
        for axis in axes[:, column]:
            axis.set_xlabel("time (s)")
            axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("raw $a_x$")
    axes[1, 0].set_ylabel("raw $a_y$")
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.suptitle("Raw actor proposals during divergent same-seed rollouts", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_space_time(vehicles: pd.DataFrame, output_path: Path) -> None:
    seeds = sorted(int(value) for value in vehicles["scenario_seed"].unique())
    pair_panels = (("B2_2", "B2.2 non-diff"), ("B3_2", "B3.2 diff"))
    figure, axes = plt.subplots(len(seeds), len(pair_panels), figsize=(15, max(5, 4.2 * len(seeds))), squeeze=False, constrained_layout=True)
    for row, seed in enumerate(seeds):
        for column, (variant_id, label) in enumerate(pair_panels):
            axis = axes[row, column]
            subset = vehicles.loc[vehicles["scenario_seed"].eq(seed) & vehicles["variant_id"].eq(variant_id)]
            if subset.empty:
                axis.text(0.5, 0.5, "no vehicle trace", ha="center", va="center")
                continue
            # Keep vehicles that are near the ego initially or become the critical vehicle.
            first = subset.sort_values("time_s").groupby("vehicle_token", as_index=False).first()
            chosen = set(first.sort_values("relative_x_m").head(10)["vehicle_token"].tolist())
            chosen.update(subset.loc[subset["vehicle_token"].isin(set(subset["vehicle_token"])) & ~subset["is_ego"], "vehicle_token"].unique()[:4])
            for token, data in subset.loc[subset["vehicle_token"].isin(chosen) & ~subset["is_ego"]].groupby("vehicle_token", sort=False):
                axis.plot(data["time_s"], data["relative_x_m"], linewidth=0.9, alpha=0.55)
            ego = subset.loc[subset["is_ego"]]
            axis.plot(ego["time_s"], ego["relative_x_m"], color="#111111", linewidth=2.3, label="ego")
            axis.axhline(0.0, color="#111111", linestyle="--", linewidth=0.8)
            axis.set_title(f"seed {seed}: {label}")
            axis.set_xlabel("time (s)")
            axis.set_ylabel("vehicle longitudinal position relative to ego (m)")
            axis.grid(alpha=0.2)
    figure.suptitle("Traffic space–time diagrams (vehicle positions relative to ego)", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_traffic_guard(trace: pd.DataFrame, output_path: Path) -> None:
    seeds = sorted(int(value) for value in trace["scenario_seed"].unique())
    metrics = (
        ("traffic_guard_brakes_step", "traffic-guard brake events / step"),
        ("traffic_guard_traffic_only_step", "traffic-only guard events / step"),
        ("traffic_guard_lateral_yields_step", "traffic-guard lateral yields / step"),
    )
    figure, axes = plt.subplots(len(metrics), len(seeds), figsize=(17, 9), squeeze=False, constrained_layout=True)
    for row, (metric, label) in enumerate(metrics):
        for column, seed in enumerate(seeds):
            axis = axes[row, column]
            subset = trace.loc[trace["scenario_seed"].eq(seed)]
            _plot_lines(axis, subset, metric)
            axis.set_title(f"seed {seed}" if row == 0 else "")
            axis.set_xlabel("time (s)")
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            if row == 0 and column == 0:
                axis.legend(fontsize=7, ncol=2)
    figure.suptitle("Ordinary simulator traffic-guard activity during raw-policy rollouts", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_rollout_summary(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False, constrained_layout=True)
    panels = (
        ("episode_min_h", "episode minimum h"),
        ("episode_min_critical_clearance_m", "episode minimum critical clearance (m)"),
        ("mean_shadow_external_correction", "mean shadow external correction"),
        ("collision", "collision indicator"),
    )
    for axis, (metric, label) in zip(axes.flat, panels):
        for index, variant_id in enumerate(VARIANT_ORDER):
            values = pd.to_numeric(summary.loc[summary["variant_id"].eq(variant_id), metric], errors="coerce").dropna().to_numpy()
            if not len(values):
                continue
            x = np.full(len(values), index, dtype=float) + np.linspace(-0.12, 0.12, len(values))
            axis.scatter(x, values, color=VARIANT_COLORS[variant_id], s=38, alpha=0.9)
            axis.plot([index - 0.16, index + 0.16], [np.mean(values), np.mean(values)], color="#111111", linewidth=2.0)
        axis.set_xticks(np.arange(len(VARIANT_ORDER)), VARIANT_ORDER, rotation=25)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Closed-loop summary across paired same-seed rollouts", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_interpretation(
    output_path: Path,
    *,
    probe_summary: pd.DataFrame,
    paired_effects: pd.DataFrame,
    rollout_summary: pd.DataFrame,
    frozen: dict[str, Any],
    selected_snapshots: list[int],
) -> None:
    def mean_probe(variant: str, metric: str) -> float:
        values = pd.to_numeric(probe_summary.loc[probe_summary["variant_id"].eq(variant), metric], errors="coerce")
        return float(values.iloc[0]) if not values.empty else np.nan

    b31_feasible = mean_probe("B3_1", "raw_feasible")
    b32_feasible = mean_probe("B3_2", "raw_feasible")
    b31_corr = mean_probe("B3_1", "external_raw_correction_norm")
    b32_corr = mean_probe("B3_2", "external_raw_correction_norm")
    b32_delta = paired_effects.loc[
        paired_effects["pair"].eq("B3_1_vs_B3_2") & paired_effects["metric"].eq("external_raw_correction_norm"),
        "right_minus_left",
    ]
    b32_delta_value = float(b32_delta.iloc[0]) if not b32_delta.empty else np.nan
    lines = [
        "# Policy-change visualization results",
        "",
        "The visualizations use a common state bank for policy-function comparisons and same-seed closed-loop raw rollouts for behavioral comparisons.",
        "The external CBF is disabled during rollouts; its action is retained as a shadow projection. The ordinary simulator traffic guard remains enabled.",
        "The shared evaluation environment used the B3.2 checkpoint contract: 100 Hz physics, 10 Hz policy, and 10 physics substeps per policy action. B1 is evaluated in this shared environment for comparability.",
        "",
        "## Probe result",
        "",
        f"- Common probe states: {len(probe_summary) and int(probe_summary['state_count'].max())} per policy; the frozen sequence is `{frozen['controller']}`, seed `{frozen['scenario_seed']}`, with `{frozen['state_count']}` states.",
        f"- B3.1 raw-feasible fraction: {b31_feasible:.3f}; B3.2: {b32_feasible:.3f}.",
        f"- B3.1 mean normalized external correction: {b31_corr:.4f}; B3.2: {b32_corr:.4f}.",
        f"- B3.2 minus B3.1 correction difference: {b32_delta_value:.4f} normalized action units.",
        f"- Action-space snapshot states: {', '.join(map(str, selected_snapshots))}.",
        "",
        "## Reading the figures",
        "",
        "- The policy atlas shows the average raw actor output in physically interpretable traffic coordinates. Gray cells are unobserved; the figures should not be read as evidence outside the occupied state region.",
        "- The deformation atlas and quiver field show where the right-hand policy changes its raw proposal relative to the left-hand policy.",
        "- The action-space snapshots separate raw proposals, B3 internal projected means, and the common external projection.",
        "- The frozen-replay plots isolate action-map differences because every policy receives the same observation at each timestamp.",
        "- The closed-loop plots show consequences after policies begin changing the state and traffic interaction.",
        "- The feature-sensitivity plot is a finite-difference diagnostic of the raw actor. It is not a causal proof by itself.",
        "",
        "## Guardrails",
        "",
        "- These checkpoints use one training seed (307).",
        "- B1 has the older 30-dimensional learned base observation; the context is adapted only for the common probe.",
        "- The B3 operational action in closed-loop plots is the differentiable internal mean; B1/B2 operational action is their raw mean. Raw proposals are retained for every policy.",
        "- Collision and margin traces are diagnostic results from these short scenarios, not replacements for the formal 50/200-episode KPI evaluations.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _probe_summary(actions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        data = actions.loc[actions["variant_id"].eq(variant_id)]
        feasible = data.loc[data["action_set_feasible"].astype(bool)]
        row = {
            "variant_id": variant_id,
            "label": VARIANT_LABELS[variant_id],
            "state_count": int(len(data)),
            "feasible_set_state_count": int(len(feasible)),
            "raw_feasible": float(feasible["raw_feasible"].mean()) if len(feasible) else np.nan,
            "mean_raw_ax": float(data["raw_ax"].mean()),
            "mean_raw_ay": float(data["raw_ay"].mean()),
            "mean_internal_mean_correction_norm": float(feasible["internal_mean_correction_norm"].mean()) if len(feasible) else np.nan,
            "external_raw_correction_norm": float(feasible["external_raw_correction_norm"].mean()) if len(feasible) else np.nan,
            "external_intervention_rate": float(feasible["external_intervention"].mean()) if len(feasible) else np.nan,
            "raw_hocbf_margin": float(feasible["raw_hocbf_margin"].mean()) if len(feasible) else np.nan,
            "raw_hocbf_margin_min": float(feasible["raw_hocbf_margin"].min()) if len(feasible) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe-seeds", type=int, default=DEFAULT_PROBE_SEEDS)
    parser.add_argument("--probe-steps", type=int, default=DEFAULT_PROBE_STEPS)
    parser.add_argument("--probe-seed-start", type=int, default=DEFAULT_PROBE_SEED_START)
    parser.add_argument("--rollout-seeds", type=int, nargs="+", default=list(DEFAULT_ROLLOUT_SEEDS))
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--sensitivity-states", type=int, default=DEFAULT_SENSITIVITY_STATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[visualization] output={output_dir}", flush=True)
    specs = base_analysis.load_specs(PROJECT_ROOT)
    namespace = _bootstrap_namespace()
    config, reward_config = _probe_config(specs)
    env = _make_raw_env(namespace, config, reward_config)
    try:
        print("[visualization] collecting enriched common-state bank", flush=True)
        states = collect_enriched_probe_states(
            env,
            namespace,
            seeds=int(args.probe_seeds),
            steps_per_seed=int(args.probe_steps),
            seed_start=int(args.probe_seed_start),
        )
    finally:
        env.close()
    states.drop(columns=["observation"]).to_csv(output_dir / "common_probe_states.csv", index=False)
    np.savez_compressed(
        output_dir / "common_probe_observations.npz",
        state_id=states["state_id"].to_numpy(dtype=np.int64),
        observation=np.stack(states["observation"].to_numpy()).astype(np.float32),
    )
    print("[visualization] loading models and querying common states", flush=True)
    models = base_analysis.load_models(specs)
    actions = _probe_action_table(states, models)
    actions.to_csv(output_dir / "common_probe_actions.csv", index=False)
    summary = _probe_summary(actions)
    summary.to_csv(output_dir / "common_probe_summary.csv", index=False)
    paired = _paired_effects(actions)
    paired.to_csv(output_dir / "paired_policy_effects.csv", index=False)

    front_x_edges = np.linspace(0.0, 60.0, 25)
    front_y_edges = np.linspace(-10.0, 15.0, 25)
    lateral_x_edges = np.linspace(-30.0, 50.0, 25)
    lateral_y_edges = np.linspace(0.0, 10.0, 25)
    plot_policy_atlas(
        actions,
        states,
        output_dir / "policy_response_atlas_front_gap_closing_speed.png",
        x="front_gap_m",
        y="front_closing_speed_mps",
        x_label="front footprint gap (m)",
        y_label="front closing speed (m/s)",
        title="Policy response atlas: incoming front traffic",
        x_edges=front_x_edges,
        y_edges=front_y_edges,
    )
    plot_policy_atlas(
        actions,
        states,
        output_dir / "policy_response_atlas_critical_geometry.png",
        x="critical_dx_m",
        y="critical_abs_dy_m",
        x_label="critical vehicle longitudinal offset (m)",
        y_label="critical vehicle lateral offset (m)",
        title="Policy response atlas: critical-neighbor geometry",
        x_edges=lateral_x_edges,
        y_edges=lateral_y_edges,
    )
    plot_difference_atlas(
        actions,
        states,
        output_dir / "policy_deformation_difference_atlas.png",
        x="front_gap_m",
        y="front_closing_speed_mps",
        x_label="front footprint gap (m)",
        y_label="front closing speed (m/s)",
        x_edges=front_x_edges,
        y_edges=front_y_edges,
    )
    plot_deformation_vectors(
        actions,
        states,
        output_dir / "policy_deformation_vector_fields.png",
        x="front_gap_m",
        y="front_closing_speed_mps",
        x_label="front footprint gap (m)",
        y_label="front closing speed (m/s)",
        x_edges=front_x_edges,
        y_edges=front_y_edges,
    )
    selected_snapshots = plot_action_space_snapshots(
        states,
        actions,
        output_dir / "action_space_projection_snapshots.png",
        output_dir / "action_space_snapshot_data.csv",
    )
    print("[visualization] finite-difference feature sensitivity", flush=True)
    sensitivity = build_feature_sensitivity(
        states,
        specs,
        models,
        state_count=int(args.sensitivity_states),
        seed=20260826,
    )
    sensitivity.to_csv(output_dir / "feature_sensitivity.csv", index=False)
    plot_feature_sensitivity(sensitivity, output_dir / "feature_sensitivity_heatmap.png")
    frozen = plot_frozen_replay(
        states,
        actions,
        output_dir / "frozen_state_action_proposals.png",
        output_dir / "frozen_state_action_matrix.png",
    )
    print("[visualization] collecting same-seed closed-loop traces", flush=True)
    trace, vehicles, rollout_summary = collect_closed_loop_traces(
        namespace,
        specs,
        models,
        scenario_seeds=tuple(dict.fromkeys(int(seed) for seed in args.rollout_seeds)),
        max_steps=int(args.rollout_steps),
    )
    trace.to_csv(output_dir / "closed_loop_trace.csv", index=False)
    vehicles.to_csv(output_dir / "closed_loop_vehicle_trace.csv", index=False)
    rollout_summary.to_csv(output_dir / "closed_loop_summary.csv", index=False)
    plot_closed_loop_all(trace, output_dir / "closed_loop_safety_timeseries_all_policies.png")
    plot_primary_storyboard(trace, output_dir / "closed_loop_primary_B2_2_vs_B3_2_storyboard.png")
    plot_closed_loop_actions(trace, output_dir / "closed_loop_raw_action_proposals.png")
    plot_space_time(vehicles, output_dir / "closed_loop_traffic_space_time.png")
    plot_traffic_guard(trace, output_dir / "closed_loop_traffic_guard_activity.png")
    plot_rollout_summary(rollout_summary, output_dir / "closed_loop_summary_plot.png")
    write_interpretation(
        output_dir / "INTERPRETATION.md",
        probe_summary=summary,
        paired_effects=paired,
        rollout_summary=rollout_summary,
        frozen=frozen,
        selected_snapshots=selected_snapshots,
    )
    metadata = {
        "script": str(Path(__file__).resolve()),
        "variants": list(VARIANT_ORDER),
        "probe_states": int(len(states)),
        "probe_seeds": int(args.probe_seeds),
        "probe_steps": int(args.probe_steps),
        "rollout_seeds": [int(seed) for seed in args.rollout_seeds],
        "rollout_steps": int(args.rollout_steps),
        "sensitivity_states": int(args.sensitivity_states),
        "shared_eval_env": {
            "physics_hz": float(config.get("simulation_frequency", np.nan)),
            "policy_hz": float(config.get("policy_frequency", np.nan)),
            "physics_substeps_per_policy_action": int(round(float(config.get("simulation_frequency", 1.0)) / max(float(config.get("policy_frequency", 1.0)), 1e-9))),
            "cbf_k0": 5.29,
            "cbf_k1": 3.68,
        },
        "external_cbf_during_rollouts": False,
        "external_cbf_shadow_projection": True,
        "ordinary_traffic_guard": True,
        "frozen_sequence": frozen,
        "action_space_snapshot_states": selected_snapshots,
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("[visualization] complete", flush=True)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print(rollout_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
