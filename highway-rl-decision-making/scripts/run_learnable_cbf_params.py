"""Pilot runner for the paired fixed/learnable HOCBF-parameter study.

The runner intentionally starts from the same projected-PPO checkpoint for both
arms.  The fixed arm holds ``p1=p2=2.3`` and ``nu=0``; the learnable arm uses
the separate Gamma network.  The notebook section ``learnable cbf parms`` is a
thin front end for this script.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

import run_cbf_filter_ablation as protocol
from learnable_hocbf_params import (
    AugmentedParameterStateWrapper,
    LearnableCBFConfig,
    LearnableCBFContextPhysicalActionWrapper,
    LearnableProjectedCBFPPO,
    config_from_dict,
    migrate_projected_checkpoint,
)
from ppo_observation_variants import install_previous_action_observation


def load_notebook_namespace(project_root: Path) -> dict[str, Any]:
    """Load only the source definitions required by the existing environment."""

    namespace = protocol.bootstrap_notebook_namespace(project_root)
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    protocol.exec_required_notebook_cells(notebook_path, namespace)
    return namespace


def learnable_env_config(namespace: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(namespace["ENV_CONFIG"])
    safety = config.setdefault("traffic_safety", {})
    safety["spawn_cbf_safe_set"] = True
    safety["spawn_cbf_k1"] = 2.3
    config["cbf_require_initial_safe_set"] = True
    config["cbf_substep_filtering"] = True
    return config


def make_single_env(
    *,
    project_root: Path,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    parameter_config: LearnableCBFConfig,
    seed: int,
    monitor_path: Path | None = None,
) -> gym.Env:
    namespace = load_notebook_namespace(project_root)
    # The canonical PPO state is the 30D target-y table plus the previous
    # normalized executed action.  Install this variant before constructing the
    # reward wrapper so the learnable adapter sees the same 32D base state as
    # B3.2.
    install_previous_action_observation(namespace)
    env = gym.make("lane-free-v0", render_mode=None, config=copy.deepcopy(env_config))
    env = namespace["KaralakouRewardWrapper"](
        env, reward_config=copy.deepcopy(reward_config)
    )
    if namespace.get("NORMALIZE_RL_OBSERVATIONS", False):
        env = namespace["LaneFreeObservationNormalizationWrapper"](
            env, clip=namespace["OBSERVATION_CLIP"]
        )
    base_observation_dim = int(np.prod(env.observation_space.shape))
    if base_observation_dim != 32:
        raise ValueError(
            "The learnable pilot expects the canonical 32D base state; "
            f"got {base_observation_dim}"
        )
    env = AugmentedParameterStateWrapper(
        env,
        p_nominal=parameter_config.p_nominal,
        p_min=parameter_config.p_min,
        p_max=parameter_config.p_max,
        dt_policy=parameter_config.dt_policy,
    )
    env = LearnableCBFContextPhysicalActionWrapper(
        env,
        namespace=namespace,
        config=parameter_config,
        ax_bounds=namespace["CBF_AX_BOUNDS"],
        ay_bounds=namespace["CBF_AY_BOUNDS"],
        neighbor_range=float(namespace["CBF_NEIGHBOR_RANGE"]),
        eps_side=float(namespace["CBF_EPS_SIDE"]),
        max_neighbor_constraints=int(namespace["CBF_MAX_NEIGHBOR_CONSTRAINTS"]),
        max_constraints=18,
        project_inputs=False,
        # Preserve the canonical B3.2 PPO reward shaping.  The separate
        # parameter learner has its own intervention loss; these coefficients
        # remain part of the driving-policy objective and are identical in the
        # fixed and learnable arms.
        lambda_delta=0.05,
        lambda_intervention=0.10,
        correction_epsilon=0.03,
        action_rate_penalty_lambda=0.0,
    )
    if "KPIInfoWrapper" in namespace:
        env = namespace["KPIInfoWrapper"](env, intervention_threshold=0.03)
    env = protocol.ProtocolMetricsWrapper(env)
    if monitor_path is not None:
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_path))
    # Seed once here; SB3 will perform the normal reset before collection.
    env.reset(seed=int(seed))
    return env


def _worker_env_factory(
    project_root: str,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    parameter_config: dict[str, Any],
    seed: int,
    monitor_path: str | None,
):
    return make_single_env(
        project_root=Path(project_root),
        env_config=env_config,
        reward_config=reward_config,
        parameter_config=config_from_dict(parameter_config),
        seed=int(seed),
        monitor_path=None if monitor_path is None else Path(monitor_path),
    )


def make_vec_env(
    *,
    project_root: Path,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    parameter_config: LearnableCBFConfig,
    n_envs: int,
    seed: int,
    subproc: bool,
    output_dir: Path,
) -> VecEnv:
    factories = [
        lambda index=index: _worker_env_factory(
            str(project_root),
            copy.deepcopy(env_config),
            copy.deepcopy(reward_config),
            asdict_config(parameter_config),
            int(seed + index),
            str(output_dir / f"monitor_{index}.csv"),
        )
        for index in range(int(n_envs))
    ]
    if subproc and int(n_envs) > 1:
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


def asdict_config(config: LearnableCBFConfig) -> dict[str, Any]:
    return {
        "p_nominal": list(config.p_nominal),
        "p_min": list(config.p_min),
        "p_max": list(config.p_max),
        "nu_max": list(config.nu_max),
        "gamma_lower": list(config.gamma_lower),
        "gamma_upper": list(config.gamma_upper),
        "dt_policy": config.dt_policy,
        "unroll_horizon": config.unroll_horizon,
        "lambda_feas": config.lambda_feas,
        "lambda_intervention": config.lambda_intervention,
        "lambda_smooth": config.lambda_smooth,
        "lambda_reg": config.lambda_reg,
        "feasibility_epsilon": config.feasibility_epsilon,
        "bound_hit_tolerance": config.bound_hit_tolerance,
        "action_scale": list(config.action_scale),
    }


def make_model(
    env: VecEnv,
    *,
    config: LearnableCBFConfig,
    seed: int,
    device: str,
    learnable: bool,
    tensorboard_dir: Path | None,
) -> LearnableProjectedCBFPPO:
    model = LearnableProjectedCBFPPO(
        "ProjectedCBFPolicy",
        env,
        parameter_config=asdict_config(config),
        parameter_learning_rate=1e-4,
        parameter_unroll_horizon=int(config.unroll_horizon),
        learnable_parameters=bool(learnable),
        lambda_mean=0.10,
        lambda_sample=0.0,
        lambda_critic=0.0,
        learning_rate=3e-4,
        n_steps=int(256),
        batch_size=int(64),
        n_epochs=int(4),
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        execution_mode="cbf",
        cbf_base_observation_dim=34,
        cbf_max_constraints=18,
        policy_kwargs={
            "net_arch": {"pi": [256, 128], "vf": [256, 128]},
            "activation_fn": th.nn.Tanh,
            "ortho_init": True,
            "log_std_init": -0.5,
            "use_safety_critic": False,
            "cbf_base_observation_dim": 34,
            "cbf_max_constraints": 18,
        },
        tensorboard_log=None if tensorboard_dir is None else str(tensorboard_dir),
        seed=int(seed),
        device=str(device),
        verbose=1,
    )
    return model


def run_arm(
    *,
    arm: str,
    args: argparse.Namespace,
    project_root: Path,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    config: LearnableCBFConfig,
    output_dir: Path,
) -> dict[str, Any]:
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(
        project_root=project_root,
        env_config=env_config,
        reward_config=reward_config,
        parameter_config=config,
        n_envs=int(args.n_envs),
        seed=int(args.seed),
        subproc=bool(args.subproc),
        output_dir=arm_dir,
    )
    model = make_model(
        env,
        config=config,
        seed=int(args.seed),
        device=str(args.device),
        learnable=arm == "learnable_p",
        tensorboard_dir=(arm_dir / "tensorboard") if args.tensorboard else None,
    )
    migration: dict[str, Any] | None = None
    if args.init_checkpoint:
        migration = migrate_projected_checkpoint(
            args.init_checkpoint, model, device=str(args.device)
        )
        (arm_dir / "checkpoint_migration.json").write_text(
            json.dumps(migration, indent=2), encoding="utf-8"
        )
    model.learn(total_timesteps=int(args.timesteps), progress_bar=False)
    model.save(str(arm_dir / "model"))
    diagnostics = pd.DataFrame(model.parameter_learning_diagnostics)
    if not diagnostics.empty:
        diagnostics.to_csv(arm_dir / "parameter_learning_diagnostics.csv", index=False)
    # Keep the most recent unrolled rollout in a flat, human-readable table.
    # The in-memory object also retains the padded rows/primitives used for the
    # differentiable learner; this export exposes the quantities requested for
    # pilot auditing without serializing a large NumPy object graph.
    if model.parameter_rollout is not None:
        rollout = model.parameter_rollout
        records: list[dict[str, Any]] = []
        for step in range(rollout.n_steps):
            for env_index in range(rollout.n_envs):
                records.append(
                    {
                        "step": step,
                        "env_index": env_index,
                        "p1": float(rollout.p[step, env_index, 0]),
                        "p2": float(rollout.p[step, env_index, 1]),
                        "nu1_raw": float(rollout.nu_raw[step, env_index, 0]),
                        "nu2_raw": float(rollout.nu_raw[step, env_index, 1]),
                        "nu1": float(rollout.nu_safe[step, env_index, 0]),
                        "nu2": float(rollout.nu_safe[step, env_index, 1]),
                        "nu1_lower": float(rollout.nu_lower[step, env_index, 0]),
                        "nu1_upper": float(rollout.nu_upper[step, env_index, 0]),
                        "nu2_lower": float(rollout.nu_lower[step, env_index, 1]),
                        "nu2_upper": float(rollout.nu_upper[step, env_index, 1]),
                        "p1_next": float(rollout.p_next[step, env_index, 0]),
                        "p2_next": float(rollout.p_next[step, env_index, 1]),
                        "mu_raw_ax": float(rollout.mu_raw[step, env_index, 0]),
                        "mu_raw_ay": float(rollout.mu_raw[step, env_index, 1]),
                        "mu_safe_ax": float(rollout.mu_safe[step, env_index, 0]),
                        "mu_safe_ay": float(rollout.mu_safe[step, env_index, 1]),
                        "a_raw_ax": float(rollout.latent_raw[step, env_index, 0]),
                        "a_raw_ay": float(rollout.latent_raw[step, env_index, 1]),
                        "a_safe_ax": float(rollout.executed[step, env_index, 0]),
                        "a_safe_ay": float(rollout.executed[step, env_index, 1]),
                        "slack": float(rollout.slack[step, env_index]),
                        "qp_infeasible": bool(rollout.qp_infeasible[step, env_index]),
                        "feasible": bool(rollout.feasible[step, env_index]),
                        "fallback": bool(rollout.fallback[step, env_index]),
                        "p_clipped": bool(rollout.p_clipped[step, env_index]),
                        "correction": float(rollout.correction[step, env_index]),
                        "sample_correction": float(
                            np.linalg.norm(
                                rollout.latent_raw[step, env_index]
                                - rollout.executed[step, env_index]
                            )
                        ),
                        "hocbf_margin": float(rollout.hocbf_margin[step, env_index]),
                        "done": bool(rollout.done[step, env_index]),
                    }
                )
        pd.DataFrame(records).to_csv(arm_dir / "parameter_rollout_last.csv", index=False)
    env.close()
    return {
        "arm": arm,
        "model": str(arm_dir / "model.zip"),
        "timesteps": int(model.num_timesteps),
        "parameter_updates": int(len(model.parameter_learning_diagnostics)),
        "migration": migration,
        "last_parameter_diagnostics": (
            model.parameter_learning_diagnostics[-1]
            if model.parameter_learning_diagnostics
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="enable TensorBoard output (CSV/console diagnostics are always written)",
    )
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--skip-learnable", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    if args.init_checkpoint is None:
        default_checkpoint = (
            project_root
            / "artifacts"
            / "B3_50k_v2"
            / "iao"
            / "ppo_cbf_integrated_actor_only"
            / "seed_307"
            / "model_final.zip"
        )
        if default_checkpoint.is_file():
            args.init_checkpoint = default_checkpoint
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    namespace = load_notebook_namespace(project_root)
    env_config = learnable_env_config(namespace)
    reward_config = copy.deepcopy(namespace["REWARD_CONFIG"])
    config = LearnableCBFConfig().validate()
    env_dt_policy = float(env_config.get("dt", 0.01)) * int(
        env_config.get("simulation_frequency", 100)
    ) / max(float(env_config.get("policy_frequency", 20)), 1.0)
    if env_dt_policy <= 0.0:
        raise RuntimeError("invalid policy timing in environment configuration")
    if abs(env_dt_policy - float(config.dt_policy)) > 1e-8:
        raise RuntimeError(
            "learnable-CBF dt_policy disagrees with the simulator policy step: "
            f"{config.dt_policy} vs {env_dt_policy}"
        )
    if args.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest = {
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "config": asdict_config(config),
        "timesteps": int(args.timesteps),
        "n_envs": int(args.n_envs),
        "seed": int(args.seed),
        "device": str(args.device),
        "init_checkpoint": None
        if args.init_checkpoint is None
        else str(args.init_checkpoint),
        "baseline": "fixed p1=p2=2.3, nu=0",
        "learnable": "Gamma_psi with H=16 truncated unroll",
    }
    (output_dir / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    results: list[dict[str, Any]] = []
    if not args.skip_fixed:
        results.append(
            run_arm(
                arm="fixed_p_nominal",
                args=args,
                project_root=project_root,
                env_config=env_config,
                reward_config=reward_config,
                config=config,
                output_dir=output_dir,
            )
        )
    if not args.skip_learnable:
        results.append(
            run_arm(
                arm="learnable_p",
                args=args,
                project_root=project_root,
                env_config=env_config,
                reward_config=reward_config,
                config=config,
                output_dir=output_dir,
            )
        )
    (output_dir / "pilot_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
