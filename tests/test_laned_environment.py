"""Regression checks for the structured/laned HighwayEnv contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT / "src" / "deep_learning" / "DQN"),
    str(PROJECT_ROOT / "notebooks" / "_shared"),
]

from adaptive_longitudinal import validate_laned_environment_config  # noqa: E402
from dqn_notebook_utils import ENVIRONMENT_PROFILES, build_env_config  # noqa: E402
import elurant_dqn  # noqa: E402


class LanedEnvironmentContractTests(unittest.TestCase):
    def test_profiles_declare_lane_based_dqn_settings(self) -> None:
        expected_lanes = {
            "structured_baseline": 3,
            "semi_unstructured": 4,
            "unstructured_stress": 4,
        }

        self.assertEqual(set(ENVIRONMENT_PROFILES), set(expected_lanes))
        for profile_name, lanes_count in expected_lanes.items():
            config = build_env_config(profile_name=profile_name)
            self.assertEqual(config["lanes_count"], lanes_count)
            self.assertEqual(config["observation"]["type"], "Kinematics")
            self.assertEqual(config["action"]["type"], "DiscreteMetaAction")
            validate_laned_environment_config(config)

    def test_each_profile_resets_steps_and_reseeds_deterministically(self) -> None:
        for profile_name in ENVIRONMENT_PROFILES:
            with self.subTest(profile_name=profile_name):
                env = elurant_dqn.make_env(config=build_env_config(profile_name=profile_name))
                try:
                    observation, _ = env.reset(seed=123)
                    repeated_observation, _ = env.reset(seed=123)
                    np.testing.assert_array_equal(observation, repeated_observation)

                    self.assertEqual(observation.shape, (5, 5))
                    self.assertTrue(env.observation_space.contains(observation))
                    self.assertTrue(np.isfinite(observation).all())
                    self.assertGreaterEqual(env.action_space.n, 3)

                    next_observation, reward, terminated, truncated, _ = env.step(0)
                    self.assertEqual(next_observation.shape, (5, 5))
                    self.assertTrue(env.observation_space.contains(next_observation))
                    self.assertTrue(np.isfinite(next_observation).all())
                    self.assertTrue(np.isfinite(float(reward)))
                    self.assertIsInstance(terminated, bool)
                    self.assertIsInstance(truncated, bool)
                finally:
                    env.close()

    def test_optional_wrappers_keep_observations_inside_their_spaces(self) -> None:
        cases = {
            "ttc": (
                {"ttc_observation": {"enabled": True}},
                (5, 6),
                None,
            ),
            "ttc_lane_context": (
                {"ttc_observation": {"enabled": True, "include_lane_context": True}},
                (5, 16),
                None,
            ),
            "adaptive": (
                {"adaptive_longitudinal": {"enabled": True, "mode": "safe_speed_limiter"}},
                (5, 5),
                "adaptive_mode",
            ),
            "lane_change_safety": (
                {"lane_change_safety": {"enabled": True}},
                (5, 5),
                "lane_change_safety_penalty",
            ),
        }

        for case_name, (overrides, expected_shape, info_key) in cases.items():
            with self.subTest(case_name=case_name):
                config = build_env_config(profile_name="structured_baseline", **overrides)
                env = elurant_dqn.make_env(config=config)
                try:
                    observation, _ = env.reset(seed=123)
                    self.assertEqual(observation.shape, expected_shape)
                    self.assertTrue(env.observation_space.contains(observation))
                    self.assertTrue(np.isfinite(observation).all())

                    next_observation, _, _, _, info = env.step(0)
                    self.assertEqual(next_observation.shape, expected_shape)
                    self.assertTrue(env.observation_space.contains(next_observation))
                    self.assertTrue(np.isfinite(next_observation).all())
                    if info_key is not None:
                        self.assertIn(info_key, info)
                finally:
                    env.close()

    def test_invalid_laned_contracts_fail_before_environment_creation(self) -> None:
        invalid_configs = (
            ({"observation": {"type": "Occupancy"}}, "observation.type"),
            ({"action": {"type": "ContinuousAction"}}, "action.type"),
            ({"lanes_count": 0}, "lanes_count"),
            ({"duration": 0}, "duration"),
            ({"reward_speed_range": [30.0, 20.0]}, "reward_speed_range"),
        )

        for config, message in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    elurant_dqn.make_env(config=config)


if __name__ == "__main__":
    unittest.main()
