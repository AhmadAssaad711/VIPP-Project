"""Evaluate the 50k PPO checkpoints with CBF removed from runtime entirely.

This evaluator is deliberately separate from the canonical CBF evaluator.  It
does not construct ``CBFContextPhysicalActionWrapper`` and it disables the
CBF-specific reset/substep settings.  The projected B3 policies are queried at
their neural-network actor mean directly, so their architectural
``mu_raw -> mu_safe`` projection is not executed either.  The base simulator's
ordinary traffic-safety guard remains unchanged; this is a CBF ablation, not a
removal of the simulator's generic collision-avoidance dynamics guard.

The seven checkpoints are evaluated sequentially, while each variant's
episodes are distributed over the requested worker pool.  A CBF context is
zero-padded only at the policy boundary because the trained networks were
saved with a wider observation space; their feature extractors consume only
the original base observation.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

# Keep one BLAS/OpenMP thread per evaluation worker.  These are set before
# importing torch through the project modules.
for _native_thread_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(_native_thread_key, "1")

import numpy as np
import pandas as pd
import torch as th

import run_cbf_filter_ablation as protocol
import run_ppo_cbf_progression as progression
from ppo_reward_safety import install_cbf_violation_reward
from projected_ppo_cbf import ProjectedCBFActorCriticPolicy


TRAINING_SEED = 307
DEFAULT_EPISODES = 30
DEFAULT_WORKERS = 20
DEFAULT_SEED_START = 1_100_000
OUTPUT_NAME = "50k_cbf_free_eval_v1"


MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "comparison_label": "B1",
        "variant": "ppo_nominal",
        "label": "B1 nominal (p50n)",
        "model": "artifacts/p50n/ppo_nominal/seed_307/model_final.zip",
        "run_config": "artifacts/p50n/ppo_nominal/seed_307/run_config.json",
    },
    {
        "comparison_label": "B2.1",
        "variant": "ppo_cbf_reward",
        "label": "B2.1 non-differentiable reward",
        "model": "artifacts/B2_50k_q1_stable_307/ppo_cbf_reward/seed_307/model_final.zip",
        "run_config": "artifacts/B2_50k_q1_stable_307/ppo_cbf_reward/seed_307/run_config.json",
    },
    {
        "comparison_label": "B2.2",
        "variant": "ppo_cbf_nd_reward_actor",
        "label": "B2.2 non-differentiable reward + detached actor",
        "model": "artifacts/B2_50k_q1_stable_307/ppo_cbf_nd_reward_actor/seed_307/model_final.zip",
        "run_config": "artifacts/B2_50k_q1_stable_307/ppo_cbf_nd_reward_actor/seed_307/run_config.json",
    },
    {
        "comparison_label": "B2.3",
        "variant": "ppo_cbf_nd_actor_only",
        "label": "B2.3 non-differentiable detached actor only",
        "model": "artifacts/B2_50k_q1_stable_307/ppo_cbf_nd_actor_only/seed_307/model_final.zip",
        "run_config": "artifacts/B2_50k_q1_stable_307/ppo_cbf_nd_actor_only/seed_307/run_config.json",
    },
    {
        "comparison_label": "B3.1",
        "variant": "ppo_cbf_diff_reward_only",
        "label": "B3.1 differentiable reward only",
        "model": "artifacts/B3_50k_v2/dfro/ppo_cbf_diff_reward_only/seed_307/model_final.zip",
        "run_config": "artifacts/B3_50k_v2/dfro/ppo_cbf_diff_reward_only/seed_307/run_config.json",
    },
    {
        "comparison_label": "B3.2",
        "variant": "ppo_cbf_integrated_actor_only",
        "label": "B3.2 differentiable reward + actor",
        "model": "artifacts/B3_50k_v2/iao/ppo_cbf_integrated_actor_only/seed_307/model_final.zip",
        "run_config": "artifacts/B3_50k_v2/iao/ppo_cbf_integrated_actor_only/seed_307/run_config.json",
    },
    {
        "comparison_label": "B3.3",
        "variant": "ppo_cbf_projected_reward_off",
        "label": "B3.3 differentiable actor only",
        "model": "artifacts/B3_50k_v2/dfao/ppo_cbf_projected_reward_off/seed_307/model_final.zip",
        "run_config": "artifacts/B3_50k_v2/dfao/ppo_cbf_projected_reward_off/seed_307/run_config.json",
    },
)

KPI_SPECS: tuple[tuple[str, str], ...] = progression.POST_TRAIN_KPI_SPECS
WEIGHTED_COLUMNS = (
    "mean_abs_speed_deviation",
    "mean_lat_y_error_m",
    "mean_jerk_norm",
)

_WORKER_STATE: dict[str, Any] | None = None


def _finite_mean(values: Iterable[float], default: float = np.nan) -> float:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float(default)


def _finite_min(values: Iterable[float], default: float = np.nan) -> float:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    return float(np.min(array)) if array.size else float(default)


def _read_run_config(root: Path, spec: dict[str, str]) -> dict[str, Any]:
    path = root / spec["run_config"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing run configuration: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    env_config = payload.get("env_config")
    reward_config = payload.get("reward_config")
    if not isinstance(env_config, dict) or not isinstance(reward_config, dict):
        raise ValueError(f"Run configuration lacks env_config/reward_config: {path}")
    return {
        "path": path,
        "payload": payload,
        "env_config": copy.deepcopy(env_config),
        "reward_config": copy.deepcopy(reward_config),
    }


def _cbf_free_env_config(env_config: dict[str, Any]) -> dict[str, Any]:
    """Disable CBF-specific simulator behavior while preserving base dynamics."""

    config = copy.deepcopy(env_config)
    # These are read by the base environment's optional substep diagnostics;
    # false ensures it cannot act as an action filter even without the wrapper.
    config["cbf_substep_filtering"] = False
    config["cbf_require_initial_safe_set"] = False
    traffic_safety = copy.deepcopy(config.get("traffic_safety", {}))
    if not isinstance(traffic_safety, dict):
        traffic_safety = {}
    for key in list(traffic_safety):
        if str(key).startswith("spawn_cbf_"):
            traffic_safety.pop(key, None)
    config["traffic_safety"] = traffic_safety
    return config


def _make_cbf_free_env(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    task_distance_m: float,
    max_policy_steps: int,
) -> Any:
    """Construct the plain reward-wrapped simulator with no CBF wrapper."""

    env = progression._base_environment(
        namespace,
        env_config=copy.deepcopy(env_config),
        reward_config=copy.deepcopy(reward_config),
    )
    env = protocol.ProtocolMetricsWrapper(env)
    return progression.DistanceTaskEvaluationWrapper(
        env,
        task_distance_m=float(task_distance_m),
        max_policy_steps=int(max_policy_steps),
    )


def _policy_observation(model: Any, base_observation: np.ndarray) -> np.ndarray:
    """Pad only for checkpoint compatibility; the extractor ignores the pad."""

    base = np.asarray(base_observation, dtype=np.float32).reshape(-1)
    expected = int(np.prod(model.observation_space.shape))
    if base.size == expected:
        return base
    extractor = getattr(model.policy, "features_extractor", None)
    base_dim = getattr(extractor, "base_observation_dim", None)
    if base_dim is not None and int(base.size) != int(base_dim):
        raise RuntimeError(
            "CBF-free base observation width does not match the checkpoint "
            f"extractor ({base.size} != {int(base_dim)})"
        )
    if base.size > expected:
        raise RuntimeError(
            f"Base observation is wider than checkpoint space ({base.size} > {expected})"
        )
    padded = np.zeros(expected, dtype=np.float32)
    padded[: base.size] = base
    return padded


def _raw_physical_action(model: Any, observation: np.ndarray) -> tuple[np.ndarray, str]:
    """Return a deterministic physical action without any CBF projection."""

    if isinstance(model.policy, ProjectedCBFActorCriticPolicy):
        # Calling model.predict() on a projected policy invokes its normal
        # distribution centered on mu_safe.  Read action_net(mu_raw) directly
        # instead, without calling _distribution_and_stages/project_actions.
        obs_tensor, vectorized = model.policy.obs_to_tensor(observation)
        with th.no_grad():
            latent_pi, _latent_vf = model.policy._latents(obs_tensor)
            action_tensor = model.policy.action_net(latent_pi)
        action = action_tensor.detach().cpu().numpy()
        if not vectorized:
            action = np.asarray(action).squeeze(axis=0)
        source = "raw_actor_mean"
    else:
        action, _ = model.predict(observation, deterministic=True)
        source = "policy_deterministic_mean"
    action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
    low = np.asarray(model.action_space.low, dtype=np.float32).reshape(-1)[:2]
    high = np.asarray(model.action_space.high, dtype=np.float32).reshape(-1)[:2]
    return np.clip(action, low, high).astype(np.float32), source


def _initialize_worker(
    project_root: str,
    model_path: str,
    variant: str,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    task_distance_m: float,
    max_policy_steps: int,
) -> None:
    """Load one checkpoint and one plain environment in an isolated worker."""

    protocol.set_stable_native_defaults()
    try:
        th.set_num_threads(1)
        th.set_num_interop_threads(1)
    except RuntimeError:
        pass

    root = Path(project_root).resolve()
    # Notebook setup prints a number of definition-cell messages.  Suppress
    # those repeated messages from 20 workers while retaining parent progress.
    with contextlib.redirect_stdout(io.StringIO()):
        namespace = protocol.bootstrap_notebook_namespace(root)
        protocol.exec_required_notebook_cells(
            root / "notebooks" / "lanelessKaralakou.ipynb", namespace
        )
        install_cbf_violation_reward(namespace)
    namespace["DEVICE"] = "cpu"
    model = progression.load_model(variant, Path(model_path), "cpu")
    model.policy.set_training_mode(False)
    env = _make_cbf_free_env(
        namespace,
        env_config=env_config,
        reward_config=reward_config,
        task_distance_m=float(task_distance_m),
        max_policy_steps=int(max_policy_steps),
    )

    global _WORKER_STATE
    _WORKER_STATE = {
        "namespace": namespace,
        "model": model,
        "variant": variant,
        "env": env,
        "env_config": copy.deepcopy(env_config),
        "task_distance_m": float(task_distance_m),
        "max_policy_steps": int(max_policy_steps),
    }


def _evaluate_one(task: tuple[int, int]) -> dict[str, Any]:
    """Evaluate one seeded complete distance-task episode."""

    state = _WORKER_STATE
    if state is None:
        raise RuntimeError("CBF-free evaluation worker was not initialized")
    episode_index, episode_seed = task
    namespace = state["namespace"]
    model = state["model"]
    env = state["env"]
    variant = str(state["variant"])
    spec = next(item for item in MODEL_SPECS if item["variant"] == variant)
    env_config = state["env_config"]
    task_distance_m = float(state["task_distance_m"])
    max_policy_steps = int(state["max_policy_steps"])

    observation, reset_info = env.reset(seed=int(episode_seed))
    del reset_info
    policy_dt = protocol._policy_dt(env)
    rewards: list[float] = []
    speed_errors: list[float] = []
    lateral_errors: list[float] = []
    jerk_norms: list[float] = []
    h_values: list[float] = []
    h_dots: list[float] = []
    ttc_values: list[float] = []
    neighbor_counts: list[float] = []
    action_saturations: list[float] = []
    total_distance_m = 0.0
    collision_events = 0
    previous_acceleration = np.zeros(2, dtype=float)
    last_info: dict[str, Any] = {}
    action_source = "policy_deterministic_mean"
    diagnostic_errors = 0
    completed = False
    forced_timeout = False

    for _step_index in range(max_policy_steps):
        try:
            pre_state = protocol.cbf_state_occupancy_metrics(
                namespace,
                env,
                eps_side=float(namespace.get("CBF_EPS_SIDE", 0.10)),
                ttc_cap_s=30.0,
            )
        except Exception:
            # h is a read-only diagnostic in this experiment.  A diagnostic
            # failure must not reintroduce a safety action or reset behavior.
            pre_state = {}
            diagnostic_errors += 1
        h_values.append(float(pre_state.get("h_min", np.nan)))
        h_dots.append(float(pre_state.get("h_dot", np.nan)))
        ttc_values.append(float(pre_state.get("ttc_cbf_linearized_s", np.nan)))
        neighbor_counts.append(float(pre_state.get("neighbor_count", np.nan)))

        policy_observation = _policy_observation(model, observation)
        physical_action, action_source = _raw_physical_action(
            model, policy_observation
        )
        normalized_action = np.asarray(
            namespace["_physical_to_normalized_action"](env, physical_action),
            dtype=np.float32,
        ).reshape(-1)[:2]
        observation, reward, terminated, truncated, info = env.step(normalized_action)
        info = dict(info)
        last_info = info
        rewards.append(float(reward))

        step_distance_m = float(
            info.get(
                "task_distance_step_m",
                info.get("pipeline_distance_step_m", 0.0),
            )
        )
        if np.isfinite(step_distance_m):
            total_distance_m += max(step_distance_m, 0.0)

        step_collision_events = max(int(info.get("ego_collision_events", 0)), 0)
        if step_collision_events == 0 and bool(
            info.get("ego_collision", False)
            or info.get("task_collision_terminated", False)
        ):
            step_collision_events = 1
        collision_events += step_collision_events

        base = env.unwrapped
        speed_errors.append(
            float(
                info.get(
                    "karalakou_abs_speed_deviation",
                    abs(float(base.vehicle.vx) - float(base.vehicle.desired_speed)),
                )
            )
        )
        lateral_error = float(info.get("karalakou_lat_y_error_m", np.nan))
        if np.isfinite(lateral_error):
            lateral_errors.append(lateral_error)

        accelerations = np.asarray(
            getattr(base, "_last_accelerations", np.empty((0, 2))),
            dtype=float,
        )
        if accelerations.ndim == 2 and accelerations.shape[0] > 0:
            acceleration = accelerations[0, :2]
            jerk_norms.append(
                float(
                    np.linalg.norm(acceleration - previous_acceleration)
                    / max(policy_dt, 1e-6)
                )
            )
            previous_acceleration = acceleration.copy()
        action_saturations.append(
            float(info.get("pipeline_action_saturation", np.nan))
        )

        if bool(terminated or truncated):
            completed = bool(info.get("task_completed", False))
            break
    else:
        forced_timeout = True

    if not last_info:
        raise RuntimeError("Evaluation episode produced no transition")
    if forced_timeout and not bool(last_info.get("task_completed", False)):
        last_info["task_timeout"] = True

    completion = progression._distance_completion_flag(
        total_distance_m=total_distance_m,
        collision_events=collision_events,
        env_config=env_config,
        task_distance_m=task_distance_m,
    )
    # A true CBF-free run has no QP, intervention, or correction observations.
    # Keep these as NaN so the summary reports N/A rather than falsely claiming
    # zero safety activity.
    return {
        "schema_version": progression.PROGRESSION_SCHEMA_VERSION,
        "evaluation_kind": "true_cbf_free_complete_episodes",
        "variant": variant,
        "variant_label": spec["label"],
        "comparison_label": spec["comparison_label"],
        "mode": "cbf_free",
        "external_cbf": "REMOVED",
        "policy_cbf_projection": "REMOVED",
        "action_source": action_source,
        "training_seed": TRAINING_SEED,
        "episode_index": int(episode_index),
        "scenario_seed": int(episode_seed),
        "episode_seed": int(episode_seed),
        "timesteps": int(len(rewards)),
        "episode_return": float(np.sum(rewards)),
        "episode_length_steps": float(len(rewards)),
        "ego_collisions_per_km": (
            1000.0 * float(collision_events) / float(total_distance_m)
            if total_distance_m > 1e-9
            else np.nan
        ),
        "h_min": _finite_min(h_values),
        "h_dot": _finite_min(h_dots),
        "ttc_cbf_linearized_s": _finite_min(ttc_values),
        "neighbor_count": _finite_mean(neighbor_counts),
        "qp_failure_rate": np.nan,
        "mean_abs_speed_deviation": _finite_mean(speed_errors, default=0.0),
        "mean_lat_y_error_m": _finite_mean(lateral_errors),
        "event_intervention_rate": np.nan,
        "mean_correction_norm": np.nan,
        "mean_jerk_norm": _finite_mean(jerk_norms, default=0.0),
        "full_horizon_survival_rate": progression._full_horizon_survival_flag(
            episode_length_steps=len(rewards),
            collision_events=collision_events,
            env_config=env_config,
        ),
        "distance_completion_rate": float(completion),
        "task_distance_m": task_distance_m,
        "task_completed": bool(last_info.get("task_completed", completed)),
        "task_timeout": bool(last_info.get("task_timeout", forced_timeout)),
        "task_collision_terminated": bool(
            last_info.get("task_collision_terminated", False)
        ),
        "shadow_event_intervention_rate": np.nan,
        "shadow_mean_correction_norm": np.nan,
        "mean_action_saturation": _finite_mean(action_saturations),
        "h_diagnostic_error_count": int(diagnostic_errors),
        "total_return": float(np.sum(rewards)),
        "total_distance_m": float(total_distance_m),
        "distinct_ego_collision_events": int(collision_events),
    }


def _weighted_mean(group: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(group["timesteps"], errors="coerce").to_numpy(
        dtype=float
    )
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    return float(np.average(values[valid], weights=weights[valid])) if np.any(valid) else np.nan


def summarize_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool equal episode blocks using the canonical post-training semantics."""

    rows = rows.sort_values("episode_index", kind="stable").reset_index(drop=True)
    block_count, episodes_per_block = progression._post_training_summary_geometry(
        len(rows)
    )
    block_rows: list[dict[str, Any]] = []
    for block_index in range(block_count):
        block = rows.iloc[
            block_index * episodes_per_block : (block_index + 1) * episodes_per_block
        ]
        distance = float(pd.to_numeric(block["total_distance_m"], errors="coerce").sum())
        collisions = int(
            pd.to_numeric(
                block["distinct_ego_collision_events"], errors="coerce"
            ).fillna(0.0).sum()
        )
        steps = int(pd.to_numeric(block["timesteps"], errors="coerce").fillna(0.0).sum())
        row: dict[str, Any] = {
            "schema_version": progression.PROGRESSION_SCHEMA_VERSION,
            "evaluation_kind": "true_cbf_free_pooled_episode_block",
            "variant": str(block["variant"].iloc[0]),
            "variant_label": str(block["variant_label"].iloc[0]),
            "comparison_label": str(block["comparison_label"].iloc[0]),
            "mode": "cbf_free",
            "external_cbf": "REMOVED",
            "training_seed": int(block["training_seed"].iloc[0]),
            "summary_block": int(block_index + 1),
            "episode_seed_start": int(block["episode_seed"].iloc[0]),
            "episode_seed_end": int(block["episode_seed"].iloc[-1]),
            "episodes_in_block": int(len(block)),
            "timesteps": steps,
            "episode_return": _finite_mean(block["episode_return"]),
            "episode_length_steps": _finite_mean(block["episode_length_steps"]),
            "distance_completion_rate": _finite_mean(
                block["distance_completion_rate"]
            ),
            "ego_collisions_per_km": (
                1000.0 * float(collisions) / distance if distance > 1e-9 else np.nan
            ),
            "h_min": _finite_min(block["h_min"]),
            "qp_failure_rate": np.nan,
            "mean_abs_speed_deviation": _weighted_mean(
                block, "mean_abs_speed_deviation"
            ),
            "mean_lat_y_error_m": _weighted_mean(block, "mean_lat_y_error_m"),
            "event_intervention_rate": np.nan,
            "mean_correction_norm": np.nan,
            "mean_jerk_norm": _weighted_mean(block, "mean_jerk_norm"),
            "total_return": float(
                pd.to_numeric(block["episode_return"], errors="coerce").sum()
            ),
            "total_distance_m": distance,
            "distinct_ego_collision_events": collisions,
        }
        block_rows.append(row)

    block_metrics = pd.DataFrame(block_rows)
    table_rows: list[dict[str, Any]] = []
    for label, column in KPI_SPECS:
        values = pd.to_numeric(block_metrics[column], errors="coerce").dropna()
        table_rows.append(
            {
                "comparison_label": str(block_metrics["comparison_label"].iloc[0]),
                "variant": str(block_metrics["variant"].iloc[0]),
                "variant_label": str(block_metrics["variant_label"].iloc[0]),
                "training_seed": int(block_metrics["training_seed"].iloc[0]),
                "external_cbf": "REMOVED",
                "mode": "cbf_free",
                "KPI": label,
                "Mean": float(values.mean()) if len(values) else np.nan,
                "SD": float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else np.nan),
                "N": int(len(values)),
                "episodes_per_mode": int(len(rows)),
                "summary_blocks": int(block_count),
                "episodes_per_summary_block": int(episodes_per_block),
                "aggregation": "pooled per block; Mean/SD across equal complete-episode blocks",
            }
        )
    return block_metrics, pd.DataFrame(table_rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _variant_output_dir(output_dir: Path, variant: str) -> Path:
    return output_dir / variant / f"seed_{TRAINING_SEED}"


def _run_variant(
    *,
    root: Path,
    output_dir: Path,
    spec: dict[str, str],
    run_config: dict[str, Any],
    episodes: int,
    workers: int,
    seed_start: int,
    task_distance_override: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant = spec["variant"]
    model_path = root / spec["model"]
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    clean_env_config = _cbf_free_env_config(run_config["env_config"])
    reward_config = copy.deepcopy(run_config["reward_config"])
    task_distance_m = (
        float(task_distance_override)
        if task_distance_override is not None
        else (380.0 if variant == "ppo_nominal" else 1_000.0)
    )
    max_policy_steps = progression._evaluation_horizon_steps(clean_env_config)
    run_dir = _variant_output_dir(output_dir, variant)
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = run_dir / "episodes.csv"
    progress_path = run_dir / "episodes_progress.csv"
    status_path = run_dir / "progress.json"
    started = time.perf_counter()
    tasks = [(index, seed_start + index - 1) for index in range(1, episodes + 1)]

    _write_json(
        status_path,
        {
            "state": "running",
            "variant": variant,
            "expected_episodes": int(episodes),
            "completed_episodes": 0,
            "workers": int(workers),
            "seed_start": int(seed_start),
            "updated_at_epoch_s": time.time(),
        },
    )
    print(
        f"[cbf-free] start {spec['comparison_label']} {variant} | "
        f"episodes={episodes} workers={workers} task_distance={task_distance_m:g}m "
        f"max_policy_steps={max_policy_steps}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    executor = ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=mp.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(
            str(root.resolve()),
            str(model_path.resolve()),
            variant,
            clean_env_config,
            reward_config,
            float(task_distance_m),
            int(max_policy_steps),
        ),
    )
    try:
        futures = [executor.submit(_evaluate_one, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            ordered = pd.DataFrame(rows).sort_values(
                "episode_index", kind="stable"
            )
            ordered.to_csv(progress_path, index=False)
            _write_json(
                status_path,
                {
                    "state": "running",
                    "variant": variant,
                    "expected_episodes": int(episodes),
                    "completed_episodes": int(len(rows)),
                    "workers": int(workers),
                    "seed_start": int(seed_start),
                    "last_episode_index": int(row["episode_index"]),
                    "last_episode_seed": int(row["episode_seed"]),
                    "last_steps": int(row["timesteps"]),
                    "last_distance_m": float(row["total_distance_m"]),
                    "last_collisions": int(row["distinct_ego_collision_events"]),
                    "updated_at_epoch_s": time.time(),
                },
            )
            print(
                f"[cbf-free] {spec['comparison_label']} "
                f"{len(rows)}/{episodes} seed={int(row['episode_seed'])} "
                f"steps={int(row['timesteps'])} distance={float(row['total_distance_m']):.1f}m "
                f"collisions={int(row['distinct_ego_collision_events'])} "
                f"complete={bool(row['distance_completion_rate'])}",
                flush=True,
            )
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    ordered = pd.DataFrame(rows).sort_values("episode_index", kind="stable").reset_index(drop=True)
    if len(ordered) != episodes:
        raise RuntimeError(f"{variant} produced {len(ordered)} rows; expected {episodes}")
    blocks, table = summarize_rows(ordered)
    ordered.to_csv(episodes_path, index=False)
    blocks.to_csv(run_dir / "blocks.csv", index=False)
    table.to_csv(run_dir / "kpi.csv", index=False)
    manifest = {
        "schema_version": progression.PROGRESSION_SCHEMA_VERSION,
        "evaluation_kind": "true_cbf_free_complete_episodes",
        "variant": variant,
        "variant_label": spec["label"],
        "comparison_label": spec["comparison_label"],
        "model_path": str(model_path.resolve()),
        "model_sha256": protocol.file_sha256(model_path),
        "source_run_config": str(run_config["path"].resolve()),
        "training_seed": TRAINING_SEED,
        "external_cbf": "REMOVED",
        "policy_cbf_projection": "REMOVED",
        "episodes_per_variant": int(episodes),
        "workers": int(workers),
        "episode_seed_start": int(seed_start),
        "task_distance_m": float(task_distance_m),
        "max_policy_steps": int(max_policy_steps),
        "policy_calls_per_environment_step": 1,
        "environment_simulation_frequency_hz": float(
            clean_env_config.get("simulation_frequency", np.nan)
        ),
        "policy_frequency_hz": float(clean_env_config.get("policy_frequency", np.nan)),
        "cbf_free_changes": [
            "CBFContextPhysicalActionWrapper was not constructed",
            "CBF-specific safe reset flags were disabled/removed",
            "CBF substep action filtering was disabled",
            "projected-policy mu_raw was used directly for B3 checkpoints",
            "CBF context was zero-padded only at the checkpoint input boundary",
        ],
        "generic_traffic_safety_guard_preserved": bool(
            clean_env_config.get("traffic_safety", {}).get("dynamics_guard", True)
        ),
        "episode_metrics_path": str(episodes_path.resolve()),
        "pooled_block_metrics_path": str((run_dir / "blocks.csv").resolve()),
        "kpi_table_path": str((run_dir / "kpi.csv").resolve()),
        "elapsed_s": float(max(time.perf_counter() - started, 0.0)),
        "complete": True,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(
        status_path,
        {
            "state": "complete",
            "variant": variant,
            "expected_episodes": int(episodes),
            "completed_episodes": int(episodes),
            "workers": int(workers),
            "seed_start": int(seed_start),
            "elapsed_s": manifest["elapsed_s"],
            "updated_at_epoch_s": time.time(),
        },
    )
    return ordered, table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--output-dir", type=Path, default=default_root / "artifacts" / OUTPUT_NAME
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--training-seed", type=int, default=TRAINING_SEED)
    parser.add_argument(
        "--task-distance-m",
        type=float,
        default=None,
        help="Override the completion distance for every variant; legacy defaults are used when omitted.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[spec["variant"] for spec in MODEL_SPECS],
        choices=[spec["variant"] for spec in MODEL_SPECS],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive")
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")
    if int(args.training_seed) != TRAINING_SEED:
        raise ValueError("This artifact set contains only seed_307 checkpoints")
    if args.task_distance_m is not None and float(args.task_distance_m) <= 0.0:
        raise ValueError("--task-distance-m must be positive")

    root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output_dir.iterdir() if path.name != ".gitkeep"]
    if existing:
        raise FileExistsError(
            f"Output directory is non-empty; preserving existing results: {output_dir}"
        )

    selected_specs = [
        next(spec for spec in MODEL_SPECS if spec["variant"] == variant)
        for variant in args.variants
    ]
    source_metadata: list[dict[str, Any]] = []
    for spec in selected_specs:
        config = _read_run_config(root, spec)
        model_path = root / spec["model"]
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
        source_metadata.append(
            {
                **spec,
                "model_path": str(model_path.resolve()),
                "model_sha256": protocol.file_sha256(model_path),
                "run_config_path": str(config["path"].resolve()),
                "base_observation_dim": progression._base_observation_dim(
                    config["env_config"]
                ),
                    "task_distance_m": (
                        float(args.task_distance_m)
                        if args.task_distance_m is not None
                        else (380.0 if spec["variant"] == "ppo_nominal" else 1000.0)
                    ),
            }
        )
    _write_json(
        output_dir / "metadata.json",
        {
            "schema_version": progression.PROGRESSION_SCHEMA_VERSION,
            "evaluation_kind": "true_cbf_free_complete_episodes",
            "description": "Seven 50k PPO checkpoints evaluated with CBF removed from runtime env and policy.",
            "training_seed": TRAINING_SEED,
            "episodes_per_variant": int(args.episodes),
            "workers": int(args.workers),
            "episode_seed_start": int(args.seed_start),
            "same_seed_range_across_variants": True,
            "policy_frequency_matches_environment": True,
            "cbf_removed": True,
            "generic_traffic_safety_guard_preserved": True,
            "source_pilot_comparison_note": "30 episodes per variant gives one common comparison set; the older nominal pilot used 50 episodes, while the B2/B3 pilot comparison used 30.",
            "task_distance_override_m": (
                float(args.task_distance_m) if args.task_distance_m is not None else None
            ),
            "models": source_metadata,
        },
    )

    all_episode_rows: list[pd.DataFrame] = []
    all_tables: list[pd.DataFrame] = []
    for spec in selected_specs:
        config = _read_run_config(root, spec)
        episodes, table = _run_variant(
            root=root,
            output_dir=output_dir,
            spec=spec,
            run_config=config,
            episodes=int(args.episodes),
            workers=int(args.workers),
            seed_start=int(args.seed_start),
            task_distance_override=(
                float(args.task_distance_m) if args.task_distance_m is not None else None
            ),
        )
        all_episode_rows.append(episodes)
        all_tables.append(table)

    combined_episodes = pd.concat(all_episode_rows, ignore_index=True)
    combined_table = pd.concat(all_tables, ignore_index=True)
    combined_episodes.to_csv(output_dir / "episodes_true_cbf_free.csv", index=False)
    combined_table.to_csv(
        output_dir / "post_train_true_cbf_free_kpis.csv", index=False
    )
    _write_json(
        output_dir / "metadata.json",
        {
            "schema_version": progression.PROGRESSION_SCHEMA_VERSION,
            "evaluation_kind": "true_cbf_free_complete_episodes",
            "description": "Seven 50k PPO checkpoints evaluated with CBF removed from runtime env and policy.",
            "training_seed": TRAINING_SEED,
            "episodes_per_variant": int(args.episodes),
            "workers": int(args.workers),
            "episode_seed_start": int(args.seed_start),
            "same_seed_range_across_variants": True,
            "policy_frequency_matches_environment": True,
            "cbf_removed": True,
            "generic_traffic_safety_guard_preserved": True,
            "source_pilot_comparison_note": "30 episodes per variant gives one common comparison set; the older nominal pilot used 50 episodes, while the B2/B3 pilot comparison used 30.",
            "task_distance_override_m": (
                float(args.task_distance_m) if args.task_distance_m is not None else None
            ),
            "models": source_metadata,
            "episodes_path": str((output_dir / "episodes_true_cbf_free.csv").resolve()),
            "kpi_path": str((output_dir / "post_train_true_cbf_free_kpis.csv").resolve()),
            "complete": True,
        },
    )
    print("\n[cbf-free] all variants complete", flush=True)
    print(
        combined_table[
            ["comparison_label", "variant_label", "KPI", "Mean", "SD", "N"]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        flush=True,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
