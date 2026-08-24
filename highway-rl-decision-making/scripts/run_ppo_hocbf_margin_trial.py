"""Standalone PPO Variant C trial: normalized raw-action HOCBF margin reward.

This script deliberately leaves the notebook and the canonical progression
script unchanged.  It reuses their environment/training machinery, but
inserts one side-trial reward wrapper between the CBF context wrapper and the
metrics/monitor wrappers.

Variant C uses the raw physical PPO action before any CBF projection and the
exact non-box HOCBF rows from the shared QP context:

    A_i a_raw <= b_i
    m_i = (b_i - A_i a_raw) / (||A_i||_2 + epsilon)
    m_min = min_i m_i
    r_C = lambda_h * clip(m_min / sigma_h, -1, 1)

No state-only clearance reward and no raw-to-safe correction reward are used.
Training executes the raw box-bounded action.  Post-training evaluation can
still compare raw deployment with external CBF deployment.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_cbf_filter_ablation as protocol  # noqa: E402
import run_ppo_cbf_progression as progression  # noqa: E402


TRIAL_VARIANT = "ppo_hocbf_margin"
TRIAL_PARAMS: dict[str, float] = {
    "lambda_h": 0.10,
    "sigma_h": 0.50,
    "margin_epsilon": 1e-8,
}


class HOCBFMarginRewardWrapper(gym.Wrapper):
    """Add the normalized worst raw-action HOCBF margin reward."""

    def __init__(
        self,
        env: gym.Env,
        *,
        lambda_h: float,
        sigma_h: float,
        margin_epsilon: float,
    ) -> None:
        super().__init__(env)
        if not hasattr(env, "current_constraint_system"):
            raise TypeError(
                "HOCBFMarginRewardWrapper requires the shared CBF context wrapper"
            )
        if not np.isfinite(lambda_h) or lambda_h < 0.0:
            raise ValueError("lambda_h must be finite and non-negative")
        if not np.isfinite(sigma_h) or sigma_h <= 0.0:
            raise ValueError("sigma_h must be finite and positive")
        if not np.isfinite(margin_epsilon) or margin_epsilon <= 0.0:
            raise ValueError("margin_epsilon must be finite and positive")
        self.lambda_h = float(lambda_h)
        self.sigma_h = float(sigma_h)
        self.margin_epsilon = float(margin_epsilon)

    def _margin_reward(
        self, action: Any, system: dict[str, Any]
    ) -> tuple[float, float, float, int, bool]:
        rows = np.asarray(system.get("cbf_rows", ()), dtype=float)
        bounds = np.asarray(system.get("cbf_bounds", ()), dtype=float).reshape(-1)
        raw_action = np.asarray(action, dtype=float).reshape(-1)[:2]
        if rows.size == 0 or bounds.size == 0:
            return 0.0, 0.0, 0.0, 0, True
        rows = rows.reshape(-1, 2)
        if rows.shape[0] != bounds.size:
            raise RuntimeError(
                "CBF row/bound count mismatch: "
                f"{rows.shape[0]} rows vs {bounds.size} bounds"
            )
        row_norms = np.linalg.norm(rows, axis=1)
        valid = (
            np.all(np.isfinite(rows), axis=1)
            & np.isfinite(bounds)
            & np.isfinite(row_norms)
            & (row_norms > 0.0)
        )
        if not np.any(valid):
            return 0.0, 0.0, 0.0, 0, True
        margins = (bounds[valid] - rows[valid] @ raw_action) / (
            row_norms[valid] + self.margin_epsilon
        )
        normalized_min_margin = float(np.min(margins))
        raw_min_margin = float(
            np.min(bounds[valid] - rows[valid] @ raw_action)
        )
        clipped = float(
            np.clip(normalized_min_margin / self.sigma_h, -1.0, 1.0)
        )
        return (
            float(self.lambda_h * clipped),
            normalized_min_margin,
            raw_min_margin,
            int(np.count_nonzero(valid)),
            bool(normalized_min_margin >= -1e-5),
        )

    def step(self, action: Any):
        # This is the pre-step system represented by the current observation.
        # It is exactly the same CBF system that the shared QP would use for
        # this policy action; no safe action or QP correction is consulted.
        system = self.env.current_constraint_system()
        (
            hocbf_reward,
            normalized_min_margin,
            raw_min_margin,
            constraint_count,
            condition_satisfied,
        ) = self._margin_reward(action, system)

        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        reward = float(reward) + hocbf_reward
        info.update(
            {
                "hocbf_margin_reward": float(hocbf_reward),
                "hocbf_normalized_min_margin_raw": float(normalized_min_margin),
                "hocbf_raw_min_margin": float(raw_min_margin),
                "hocbf_margin_constraint_count": int(constraint_count),
                "hocbf_raw_condition_satisfied": bool(condition_satisfied),
                "hocbf_margin_lambda_h": float(self.lambda_h),
                "hocbf_margin_sigma_h": float(self.sigma_h),
            }
        )
        return observation, reward, terminated, truncated, info


def make_hocbf_env(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    project_inputs: bool,
    lambda_delta: float,
    lambda_intervention: float,
    correction_epsilon: float,
    action_rate_penalty_lambda: float = 0.0,
    monitor_path: Path | None = None,
) -> gym.Env:
    """Build the canonical environment plus only the Variant C reward term."""

    env = progression._base_environment(
        namespace, env_config=env_config, reward_config=reward_config
    )
    env = progression.CBFContextPhysicalActionWrapper(
        env,
        namespace=namespace,
        ax_bounds=namespace["CBF_AX_BOUNDS"],
        ay_bounds=namespace["CBF_AY_BOUNDS"],
        neighbor_range=float(namespace["CBF_NEIGHBOR_RANGE"]),
        eps_side=float(namespace["CBF_EPS_SIDE"]),
        k0=float(namespace["CBF_K0"]),
        k1=float(namespace["CBF_K1"]),
        max_neighbor_constraints=int(namespace["CBF_MAX_NEIGHBOR_CONSTRAINTS"]),
        base_observation_dim=int(np.prod(env.observation_space.shape)),
        project_inputs=bool(project_inputs),
        # Variant C intentionally has no correction/intervention reward.
        lambda_delta=float(lambda_delta),
        lambda_intervention=float(lambda_intervention),
        correction_epsilon=float(correction_epsilon),
        action_rate_penalty_lambda=float(action_rate_penalty_lambda),
    )
    env = HOCBFMarginRewardWrapper(
        env,
        lambda_h=float(reward_config["hocbf_margin_lambda_h"]),
        sigma_h=float(reward_config["hocbf_margin_sigma_h"]),
        margin_epsilon=float(reward_config["hocbf_margin_epsilon"]),
    )
    if "KPIInfoWrapper" in namespace:
        env = namespace["KPIInfoWrapper"](
            env, intervention_threshold=float(correction_epsilon)
        )
    env = protocol.ProtocolMetricsWrapper(env)
    if monitor_path is not None:
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=protocol.TRAINING_MONITOR_INFO_KEYS,
        )
    return env


def make_hocbf_worker_env(
    *,
    project_root: str,
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    cbf_snapshot: dict[str, Any],
    lambda_delta: float,
    lambda_intervention: float,
    correction_epsilon: float,
    action_rate_penalty_lambda: float,
    monitor_path: str,
) -> gym.Env:
    """Spawn-safe worker constructor matching the canonical pipeline API."""

    worker_project_root = Path(project_root)
    protocol.set_stable_native_defaults()
    worker_namespace = protocol.bootstrap_notebook_namespace(worker_project_root)
    protocol.exec_required_notebook_cells(
        worker_project_root / "notebooks" / "lanelessKaralakou.ipynb",
        worker_namespace,
    )
    worker_namespace.update(copy.deepcopy(cbf_snapshot))
    return make_hocbf_env(
        worker_namespace,
        env_config=copy.deepcopy(env_config),
        reward_config=copy.deepcopy(reward_config),
        project_inputs=False,
        lambda_delta=float(lambda_delta),
        lambda_intervention=float(lambda_intervention),
        correction_epsilon=float(correction_epsilon),
        action_rate_penalty_lambda=float(action_rate_penalty_lambda),
        monitor_path=Path(monitor_path),
    )


def install_side_trial_patches() -> None:
    """Patch only in-memory module references; no repository source is changed."""

    original_make_base_reward_config = progression.protocol.make_base_reward_config

    def make_variant_c_reward_config(namespace: dict[str, Any]) -> dict[str, float]:
        config = original_make_base_reward_config(namespace)
        # Disable the old nominal potential.  The new safety signal is added
        # by HOCBFMarginRewardWrapper after the base reward is calculated.
        config.update(
            {
                "wf": 0.0,
                "use_current_potential": 0.0,
                "use_safety_potential": 0.0,
                "w_safe": 0.0,
                "hocbf_margin_reward_enabled": float(
                    TRIAL_PARAMS["lambda_h"] > 0.0
                ),
                "hocbf_margin_lambda_h": float(TRIAL_PARAMS["lambda_h"]),
                "hocbf_margin_sigma_h": float(TRIAL_PARAMS["sigma_h"]),
                "hocbf_margin_epsilon": float(TRIAL_PARAMS["margin_epsilon"]),
                "hocbf_margin_definition": (
                    "min_i((b_i-A_i-a_raw)/(||A_i||_2+epsilon)); "
                    "non-box HOCBF rows only"
                ),
                "hocbf_margin_action": "raw physical PPO action before CBF projection",
                "hocbf_margin_evaluation_rate": "policy-step pre-action constraint system",
            }
        )
        return config

    progression.protocol.make_base_reward_config = make_variant_c_reward_config
    progression.make_ppo_cbf_env = make_hocbf_env
    progression._make_ppo_worker_env = make_hocbf_worker_env

    variant_spec = copy.deepcopy(progression.VARIANT_SPECS["ppo_nominal"])
    variant_spec.update(
        {
            "label": (
                "PPO without dense safety reward"
                if TRIAL_PARAMS["lambda_h"] == 0.0
                else "PPO with normalized raw-action HOCBF margin reward"
            ),
            "execution_mode": "box",
            "reward_penalty": False,
        }
    )
    progression.VARIANT_SPECS[TRIAL_VARIANT] = variant_spec
    progression.TENSORBOARD_VARIANT_IDS[TRIAL_VARIANT] = "hocbfm"


def parse_side_arguments() -> argparse.Namespace:
    """Consume only side-trial options, leaving canonical CLI options intact."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lambda-h", type=float, default=TRIAL_PARAMS["lambda_h"])
    parser.add_argument("--sigma-h", type=float, default=TRIAL_PARAMS["sigma_h"])
    parser.add_argument(
        "--margin-epsilon",
        type=float,
        default=TRIAL_PARAMS["margin_epsilon"],
    )
    side_args, remaining = parser.parse_known_args()
    if not np.isfinite(side_args.lambda_h) or side_args.lambda_h < 0.0:
        raise ValueError("--lambda-h must be finite and non-negative")
    if not np.isfinite(side_args.sigma_h) or side_args.sigma_h <= 0.0:
        raise ValueError("--sigma-h must be finite and positive")
    if not np.isfinite(side_args.margin_epsilon) or side_args.margin_epsilon <= 0.0:
        raise ValueError("--margin-epsilon must be finite and positive")
    sys.argv = [sys.argv[0], *remaining]
    return side_args


def main() -> int:
    side_args = parse_side_arguments()
    TRIAL_PARAMS.update(
        {
            "lambda_h": float(side_args.lambda_h),
            "sigma_h": float(side_args.sigma_h),
            "margin_epsilon": float(side_args.margin_epsilon),
        }
    )
    install_side_trial_patches()
    return progression.main()


if __name__ == "__main__":
    raise SystemExit(main())
