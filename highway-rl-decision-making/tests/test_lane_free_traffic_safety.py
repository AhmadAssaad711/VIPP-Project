from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = PROJECT_ROOT / "laneless highway env"
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from lane_free_env import LaneFreeTrafficEnv  # noqa: E402


def _dense_env() -> LaneFreeTrafficEnv:
    return LaneFreeTrafficEnv(
        config={
            "road_length": 380.0,
            "road_width": 10.2,
            "dt": 0.05,
            "simulation_frequency": 20,
            "policy_frequency": 10,
            "vehicles_count": 55,
            "desired_speed_range": [15.0, 25.0],
            "initial_speed_fraction_range": [0.55, 1.10],
            "bounds": {
                "ax_min": -3.0,
                "ax_max": 3.0,
                "ay_min": -3.0,
                "ay_max": 3.0,
            },
        }
    )


def test_safe_spawn_has_no_overlapping_pairs_across_dense_seeds():
    for seed in range(12):
        env = _dense_env()
        try:
            _obs, info = env.reset(seed=seed)
            env._detect_collisions()
            assert env._last_active_collision_count == 0
            assert env._last_ego_collision_count == 0
            assert info["traffic_safe_spawn"] is True
        finally:
            env.close()


def test_guard_brakes_social_follower_without_changing_ego_action():
    env = LaneFreeTrafficEnv(
        config={
            "road_length": 380.0,
            "vehicles_count": 2,
            "bounds": {
                "ax_min": -3.0,
                "ax_max": 3.0,
                "ay_min": -3.0,
                "ay_max": 3.0,
            },
        }
    )
    try:
        env.reset(seed=1)
        ego, social = env.road.vehicles
        ego.position[:] = [100.0, 5.1]
        ego.vx, ego.vy = 15.0, 0.0
        social.position[:] = [90.0, 5.1]
        social.vx, social.vy = 25.0, 0.0
        requested = np.asarray([[1.2, -0.5], [2.0, 0.0]], dtype=float)

        guarded = env._apply_traffic_safety_guard(requested, dt=0.05)

        assert np.allclose(guarded[0], requested[0])
        assert guarded[1, 0] == -3.0
        assert env._last_traffic_safety_diagnostics["traffic_brakes"] == 1.0
    finally:
        env.close()


def test_guard_brakes_a_social_follower_for_social_traffic():
    env = LaneFreeTrafficEnv(
        config={
            "road_length": 380.0,
            "vehicles_count": 3,
            "bounds": {
                "ax_min": -3.0,
                "ax_max": 3.0,
                "ay_min": -3.0,
                "ay_max": 3.0,
            },
        }
    )
    try:
        env.reset(seed=3)
        ego, follower, leader = env.road.vehicles
        ego.position[:] = [250.0, 1.0]
        follower.position[:] = [90.0, 5.1]
        leader.position[:] = [100.0, 5.1]
        follower.vx, follower.vy = 25.0, 0.0
        leader.vx, leader.vy = 15.0, 0.0
        requested = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]], dtype=float)

        guarded = env._apply_traffic_safety_guard(requested, dt=0.05)

        assert np.allclose(guarded[0], requested[0])
        assert guarded[1, 0] == -3.0
        assert env._last_traffic_safety_diagnostics["traffic_brakes"] == 1.0
    finally:
        env.close()


def test_guard_makes_social_leader_yield_instead_of_overwriting_ego_action():
    env = LaneFreeTrafficEnv(
        config={
            "road_length": 380.0,
            "vehicles_count": 2,
            "bounds": {
                "ax_min": -3.0,
                "ax_max": 3.0,
                "ay_min": -3.0,
                "ay_max": 3.0,
            },
        }
    )
    try:
        env.reset(seed=2)
        ego, social = env.road.vehicles
        ego.position[:] = [90.0, 5.1]
        ego.vx, ego.vy = 25.0, 0.0
        social.position[:] = [100.0, 5.1]
        social.vx, social.vy = 15.0, 0.0
        requested = np.asarray([[-2.0, 0.7], [-3.0, 0.0]], dtype=float)

        guarded = env._apply_traffic_safety_guard(requested, dt=0.05)

        assert np.allclose(guarded[0], requested[0])
        assert guarded[1, 0] > requested[1, 0]
        assert env._last_traffic_safety_diagnostics["ego_leader_yields"] == 1.0
    finally:
        env.close()
