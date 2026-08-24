from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_ppo_cbf_gain_grid_stage1 import (  # noqa: E402
    evaluation_env_config,
    expected_episode_seeds,
    grid_values,
    make_candidates,
    summarize_episode_metrics,
)


def test_stage1_grid_has_11_values_and_66_symmetry_reduced_pairs():
    values = grid_values(0.0, 5.0, 0.5)
    candidates = make_candidates(0.0, 5.0, 0.5)

    assert values == [0.5 * index for index in range(11)]
    assert len(candidates) == 66
    assert all(float(row["c1"]) <= float(row["c2"]) for row in candidates)
    assert len({(row["c1"], row["c2"]) for row in candidates}) == 66
    assert candidates[0]["c1"] == 0.0
    assert candidates[0]["c2"] == 0.0
    assert candidates[0]["k0"] == 0.0
    assert candidates[0]["k1"] == 0.0
    assert candidates[-1]["c1"] == 5.0
    assert candidates[-1]["c2"] == 5.0
    assert candidates[-1]["k0"] == 25.0
    assert candidates[-1]["k1"] == 10.0
    assert sum(row["c1"] == 0.0 and row["c2"] == 5.0 for row in candidates) == 1
    assert not any(row["c1"] == 5.0 and row["c2"] == 0.0 for row in candidates)


def test_stage1_episode_seeds_are_shared_and_sequential():
    assert expected_episode_seeds(1_200_000, 20) == list(range(1_200_000, 1_200_020))


def test_stage1_disables_only_runtime_reset_rejection_and_keeps_source_spawn_gain():
    source = {
        "env_config": {
            "cbf_require_initial_safe_set": True,
            "traffic_safety": {
                "spawn_cbf_safe_set": True,
                "spawn_cbf_k1": 3.68,
            },
        }
    }

    evaluation = evaluation_env_config(source)

    assert evaluation["cbf_require_initial_safe_set"] is False
    assert evaluation["traffic_safety"]["spawn_cbf_safe_set"] is True
    assert evaluation["traffic_safety"]["spawn_cbf_k1"] == 3.68
    assert source["env_config"]["cbf_require_initial_safe_set"] is True


def test_stage1_summary_reports_pooled_collision_rate_and_weighted_metrics():
    candidates = make_candidates(0.0, 0.5, 0.5)[:2]
    rows = []
    for candidate, distance, collisions, completion in (
        (candidates[0], 100.0, 1, 0.0),
        (candidates[0], 900.0, 0, 1.0),
        (candidates[1], 500.0, 0, 1.0),
    ):
        rows.append(
            {
                **candidate,
                "episode_return": distance / 10.0,
                "total_distance_m": distance,
                "distinct_ego_collision_events": collisions,
                "distance_completion_rate": completion,
                "timesteps": 10 if distance == 100.0 else 90,
                "mean_abs_speed_deviation": 1.0,
                "mean_lat_y_error_m": 2.0,
                "event_intervention_rate": 0.1,
                "mean_correction_norm": 0.2,
                "qp_failure_rate": 0.0,
                "mean_jerk_norm": 3.0,
                "h_min": -1.0,
                "shadow_event_intervention_rate": 0.4,
            }
        )

    summary = summarize_episode_metrics(pd.DataFrame(rows), candidates)
    first = summary.iloc[0]
    second = summary.iloc[1]

    assert int(first["episodes_completed"]) == 2
    assert np.isclose(first["collision_episode_rate"], 0.5)
    assert np.isclose(first["collision_events_per_km_pooled"], 1_000.0 / 1_000.0)
    assert np.isclose(first["distance_completion_rate_mean"], 0.5)
    assert np.isclose(first["event_intervention_rate_weighted"], 0.1)
    assert int(second["episodes_completed"]) == 1
