"""Create a transparent, agent-relevant view of true-CBF-free evaluation data.

This post-processor deliberately does not delete collision episodes.  The
saved evaluator records ego-involved collisions but not collision causality,
so removing a collision because a traffic vehicle may have hit the ego would
be unsupported.  It creates a primary table with CBF-only diagnostics moved
out of scope, plus explicit attribution and protocol warnings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/final_Results/eval/true_cbf_free"),
        help="Directory containing episodes_true_cbf_free.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final_Results/eval/postprocessed_true_cbf_free"),
    )
    return parser.parse_args()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _mean(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.mean()) if len(values) else np.nan


def _sd(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else np.nan)


def _weighted_mean(frame: pd.DataFrame, value_column: str, weight_column: str = "timesteps") -> float:
    values = _numeric(frame, value_column).to_numpy(dtype=float)
    weights = _numeric(frame, weight_column).to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    return float(np.average(values[valid], weights=weights[valid])) if np.any(valid) else np.nan


def _protocol_by_variant(input_dir: Path, frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    protocols: dict[str, dict[str, Any]] = {}
    for variant in sorted(frame["variant"].dropna().unique()):
        manifest_path = input_dir / str(variant) / "seed_307" / "manifest.json"
        if manifest_path.exists():
            protocols[str(variant)] = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            protocols[str(variant)] = {}
    return protocols


def _build_primary_episode_view(frame: pd.DataFrame) -> pd.DataFrame:
    selected = [
        "comparison_label",
        "variant",
        "variant_label",
        "training_seed",
        "episode_index",
        "episode_seed",
        "timesteps",
        "episode_return",
        "episode_length_steps",
        "total_distance_m",
        "task_distance_m",
        "task_completed",
        "task_timeout",
        "task_collision_terminated",
        "distance_completion_rate",
        "full_horizon_survival_rate",
        "ego_collisions_per_km",
        "distinct_ego_collision_events",
        "mean_abs_speed_deviation",
        "mean_lat_y_error_m",
        "mean_jerk_norm",
        "mean_action_saturation",
    ]
    view = frame[[column for column in selected if column in frame.columns]].copy()
    collisions = pd.to_numeric(view["distinct_ego_collision_events"], errors="coerce").fillna(0.0)
    view["collision_attribution"] = np.where(
        collisions > 0.0,
        "unavailable_ego_involved_collision",
        "no_ego_collision_observed",
    )
    view["row_retained_in_primary_view"] = True
    return view.sort_values(["variant", "episode_index"], kind="stable").reset_index(drop=True)


def _build_primary_summary(frame: pd.DataFrame, protocols: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in frame.groupby("variant", sort=True):
        collision_events = _numeric(group, "distinct_ego_collision_events").fillna(0.0)
        distance = _numeric(group, "total_distance_m").fillna(0.0)
        completed = group["task_completed"].astype(bool)
        collision_terminated = group["task_collision_terminated"].astype(bool)
        protocol = protocols.get(str(variant), {})
        rows.append(
            {
                "comparison_label": str(group["comparison_label"].iloc[0]),
                "variant": str(variant),
                "variant_label": str(group["variant_label"].iloc[0]),
                "training_seed": int(group["training_seed"].iloc[0]),
                "episodes": int(len(group)),
                "task_distance_m": _mean(group, "task_distance_m"),
                "physics_frequency_hz": protocol.get("environment_simulation_frequency_hz"),
                "policy_frequency_hz": protocol.get("policy_frequency_hz"),
                "episode_return_mean": _mean(group, "episode_return"),
                "episode_return_sd": _sd(group, "episode_return"),
                "episode_length_steps_mean": _mean(group, "episode_length_steps"),
                "episode_length_steps_sd": _sd(group, "episode_length_steps"),
                "distance_m_mean": _mean(group, "total_distance_m"),
                "distance_m_sd": _sd(group, "total_distance_m"),
                "task_completion_rate": float(completed.mean()),
                "task_collision_termination_rate": float(collision_terminated.mean()),
                "ego_collision_episode_rate": float((collision_events > 0.0).mean()),
                "ego_collision_free_episode_rate": float((collision_events <= 0.0).mean()),
                "ego_collision_events_total": int(collision_events.sum()),
                "ego_collision_events_per_km_pooled": (
                    float(1000.0 * collision_events.sum() / distance.sum())
                    if distance.sum() > 1e-12
                    else np.nan
                ),
                "abs_speed_error_mps_weighted": _weighted_mean(group, "mean_abs_speed_deviation"),
                "lateral_error_m_weighted": _weighted_mean(group, "mean_lat_y_error_m"),
                "jerk_norm_weighted": _weighted_mean(group, "mean_jerk_norm"),
                "action_saturation_mean": _weighted_mean(group, "mean_action_saturation"),
                "collision_causality_status": (
                    "unavailable_in_saved_logs" if (collision_events > 0.0).any() else "not_applicable"
                ),
                "rows_removed": 0,
                "protocol_comparison_warning": (
                    "B1 uses a different task distance and simulator frequency than the CBF variants"
                    if str(variant) == "ppo_nominal"
                    else "B1 uses a different task distance and simulator frequency; cross-variant comparison is confounded"
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_secondary_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in frame.groupby("variant", sort=True):
        rows.append(
            {
                "comparison_label": str(group["comparison_label"].iloc[0]),
                "variant": str(variant),
                "variant_label": str(group["variant_label"].iloc[0]),
                "minimum_h_mean": _mean(group, "h_min"),
                "minimum_h_sd": _sd(group, "h_min"),
                "h_dot_mean": _mean(group, "h_dot"),
                "ttc_cbf_linearized_s_mean": _mean(group, "ttc_cbf_linearized_s"),
                "neighbor_count_mean": _mean(group, "neighbor_count"),
                "note": "Geometric/CBF diagnostics retained separately; they are not primary agent-performance outcomes in a CBF-free run.",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = input_dir / "episodes_true_cbf_free.csv"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing evaluation episode file: {episodes_path}")

    frame = pd.read_csv(episodes_path)
    protocols = _protocol_by_variant(input_dir, frame)
    primary_episodes = _build_primary_episode_view(frame)
    primary_summary = _build_primary_summary(frame, protocols)
    secondary_diagnostics = _build_secondary_diagnostics(frame)

    primary_episodes.to_csv(output_dir / "episodes_agent_relevant.csv", index=False)
    primary_summary.to_csv(output_dir / "summary_agent_relevant.csv", index=False)
    secondary_diagnostics.to_csv(output_dir / "secondary_cbf_diagnostics.csv", index=False)

    handling = pd.DataFrame(
        [
            {
                "metric_or_field": "qp_failure_rate",
                "primary_handling": "excluded",
                "reason": "CBF/QP was removed from this evaluation; field is not applicable.",
            },
            {
                "metric_or_field": "event_intervention_rate",
                "primary_handling": "excluded",
                "reason": "External CBF was removed; intervention rate is not applicable.",
            },
            {
                "metric_or_field": "mean_correction_norm",
                "primary_handling": "excluded",
                "reason": "No CBF action correction was applied; correction norm is not applicable.",
            },
            {
                "metric_or_field": "h_min / h_dot / ttc_cbf_linearized_s",
                "primary_handling": "secondary diagnostic",
                "reason": "These describe geometric/CBF-potential proximity, not direct policy performance or causality.",
            },
            {
                "metric_or_field": "distinct_ego_collision_events",
                "primary_handling": "retained",
                "reason": "Physical ego-involved safety outcome; saved logs do not identify which actor initiated contact.",
            },
            {
                "metric_or_field": "collision episodes",
                "primary_handling": "not removed",
                "reason": "Deleting potentially exogenous collisions would bias the safety result without collision provenance.",
            },
        ]
    )
    handling.to_csv(output_dir / "metric_handling.csv", index=False)

    manifest = {
        "source": str(episodes_path),
        "output": str(output_dir),
        "episodes_in_source": int(len(frame)),
        "episodes_in_primary_view": int(len(primary_episodes)),
        "rows_removed": 0,
        "collision_causality_available": False,
        "pure_background_collision_metric": "not present in saved true-CBF-free episode rows",
        "primary_view_policy": "CBF-only/N-A diagnostics excluded; all physical ego outcomes retained",
        "protocol_warning": "B1 and CBF variants in this legacy package were evaluated with different task distances and simulator frequencies.",
        "source_untouched": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        """# Post-processed true-CBF-free evaluation

This is a transparent reporting view of the saved evaluation results.
The original files under `../true_cbf_free/` are unchanged.

## What was changed

- CBF-only or not-applicable fields (QP failure, intervention rate, and correction norm) were removed from the primary table.
- Geometric CBF diagnostics (`h_min`, `h_dot`, linearized TTC, and neighbor count) were moved to a secondary table.
- Ego-involved collision outcomes were retained. The saved rows do not include collision partner, relative closing direction, or causal attribution, so no collision episode was deleted.
- The primary summary uses pooled ego collision events divided by pooled distance, rather than averaging per-episode rates.

## Important protocol caveat

The legacy package is not a fully controlled cross-variant comparison: B1 uses a 380 m task and different simulator frequency, while the CBF variants use 1,000 m and 100 Hz physics. This post-processing cannot repair that confound; a fair comparison requires rerunning all variants under one protocol.

For agent-versus-traffic responsibility, the next required experiment is a provenance-enabled rerun that logs the collision pair and a paired counterfactual with a fixed safe ego controller.
""",
        encoding="utf-8",
    )
    print(f"Wrote post-processed evaluation views to {output_dir}")
    print(primary_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
