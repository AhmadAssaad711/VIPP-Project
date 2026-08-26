"""Render a trained PPO checkpoint with CBF removed from runtime.

The renderer follows ``evaluate_ppo_cbf_free.py``: it constructs the plain
reward-wrapped simulator, retains only the generic traffic-safety dynamics
guard, and feeds the policy one action per policy step.  For projected B3
policies it reads the raw actor mean directly, bypassing the architectural CBF
projection.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

for _native_thread_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(_native_thread_key, "1")

import cv2
import gymnasium as gym
import numpy as np
import torch as th

import run_cbf_filter_ablation as protocol
import run_ppo_cbf_progression as progression
from evaluate_ppo_cbf_free import (
    MODEL_SPECS,
    _cbf_free_env_config,
    _policy_observation,
    _raw_physical_action,
    _read_run_config,
)
from ppo_reward_safety import install_cbf_violation_reward


def _spec_for_variant(variant: str) -> dict[str, str]:
    for spec in MODEL_SPECS:
        if spec["variant"] == variant:
            return spec
    raise ValueError(f"Unknown variant: {variant}")


def _build_namespace(project_root: Path) -> dict[str, Any]:
    protocol.set_stable_native_defaults()
    namespace = protocol.bootstrap_notebook_namespace(project_root)
    protocol.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    install_cbf_violation_reward(namespace)
    namespace["DEVICE"] = "cpu"
    return namespace


def _make_render_env(
    namespace: dict[str, Any],
    *,
    env_config: dict[str, Any],
    reward_config: dict[str, Any],
    task_distance_m: float,
    max_policy_steps: int,
) -> gym.Env:
    config = _cbf_free_env_config(env_config)
    config.update(
        {
            "real_time_rendering": False,
            "offscreen_rendering": True,
            "screen_width": 1200,
            "screen_height": 320,
            "show_trajectories": False,
        }
    )
    # The canonical PPO checkpoint was trained with the target-y plus previous
    # executed-action observation.  Install that wrapper before constructing
    # the reward wrapper so a CBF-free render still receives the same 32 base
    # features; only the CBF context is omitted and padded at inference.
    progression._ensure_ppo_observation_variant(namespace, config)
    env = gym.make("lane-free-v0", render_mode="rgb_array", config=config)
    env = namespace["KaralakouRewardWrapper"](
        env, reward_config=copy.deepcopy(reward_config)
    )
    if namespace.get("NORMALIZE_RL_OBSERVATIONS", False):
        env = namespace["LaneFreeObservationNormalizationWrapper"](
            env, clip=namespace["OBSERVATION_CLIP"]
        )
    env = protocol.ProtocolMetricsWrapper(env)
    return progression.DistanceTaskEvaluationWrapper(
        env,
        task_distance_m=float(task_distance_m),
        max_policy_steps=int(max_policy_steps),
    )


def _frame_rgb(env: gym.Env) -> np.ndarray:
    frame = np.asarray(env.render(), dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Unexpected render frame shape: {frame.shape}")
    return frame


def _annotate(
    frame_rgb: np.ndarray,
    *,
    variant_label: str,
    seed: int,
    step: int,
    policy_dt: float,
    action_physical: np.ndarray,
    info: dict[str, Any],
    total_distance_m: float,
    collisions: int,
) -> np.ndarray:
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    top = 76
    frame = cv2.copyMakeBorder(
        frame, top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(22, 22, 22)
    )
    base = info.get("_base_env")
    if base is not None:
        speed = float(base.vehicle.vx)
        desired = float(base.vehicle.desired_speed)
        ego_y = float(base.vehicle.position[1])
    else:
        speed = float(info.get("karalakou_ego_speed", np.nan))
        desired = float(info.get("karalakou_desired_speed", np.nan))
        ego_y = float(info.get("karalakou_ego_y", np.nan))
    line_1 = f"{variant_label} | TRUE CBF-FREE | seed {seed}"
    line_2 = (
        f"t={step * policy_dt:5.1f}s  step={step:04d}  v={speed:5.2f}  "
        f"v_des={desired:5.2f}  y={ego_y:5.2f}  collisions={collisions}"
    )
    line_3 = (
        f"a_phys=[{float(action_physical[0]):+.2f}, {float(action_physical[1]):+.2f}]  "
        f"distance={total_distance_m:6.1f} m  CBF projection=REMOVED"
    )
    for y, text_value in ((22, line_1), (47, line_2), (70, line_3)):
        cv2.putText(
            frame,
            text_value,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (238, 238, 238),
            1,
            cv2.LINE_AA,
        )
    return frame


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--variant", choices=[spec["variant"] for spec in MODEL_SPECS], default="ppo_nominal"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Explicit PPO checkpoint path; otherwise use the legacy variant spec.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Explicit run_config.json; otherwise use the legacy variant spec.",
    )
    parser.add_argument("--seed", type=int, default=1_100_001)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--task-distance-m",
        type=float,
        default=None,
        help="Distance-task target in metres; defaults to the legacy renderer target.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=root / "artifacts" / "50k_cbf_free_eval_v1" / "renders"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    project_root = Path(args.project_root).resolve()
    spec = _spec_for_variant(args.variant)
    if args.run_config is None:
        config = _read_run_config(project_root, spec)
    else:
        run_config_path = args.run_config
        if not run_config_path.is_absolute():
            run_config_path = project_root / run_config_path
        run_config_path = run_config_path.resolve()
        if not run_config_path.is_file():
            raise FileNotFoundError(f"Missing run configuration: {run_config_path}")
        payload = json.loads(run_config_path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("env_config"), dict) or not isinstance(
            payload.get("reward_config"), dict
        ):
            raise ValueError(
                f"Run configuration lacks env_config/reward_config: {run_config_path}"
            )
        config = payload
    model_path = args.model_path
    if model_path is None:
        model_path = project_root / spec["model"]
    elif not model_path.is_absolute():
        model_path = project_root / model_path
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    default_task_distance_m = 380.0 if args.variant == "ppo_nominal" else 1_000.0
    task_distance_m = float(
        args.task_distance_m
        if args.task_distance_m is not None
        else config.get("evaluation_task_distance_m", default_task_distance_m)
    )
    if not np.isfinite(task_distance_m) or task_distance_m <= 0.0:
        raise ValueError("--task-distance-m must be positive and finite")
    clean_env_config = _cbf_free_env_config(config["env_config"])
    max_policy_steps = progression._evaluation_horizon_steps(clean_env_config)
    steps_requested = int(args.steps if args.steps is not None else max_policy_steps)
    if steps_requested > max_policy_steps:
        raise ValueError(
            f"--steps={steps_requested} exceeds the CBF-free task horizon {max_policy_steps}"
        )

    namespace = _build_namespace(project_root)
    model = progression.load_model(args.variant, model_path, str(args.device))
    model.policy.set_training_mode(False)
    variant_label = str(config.get("variant_spec", {}).get("label", spec["label"]))
    if args.model_path is not None:
        variant_label = f"{variant_label} | {int(config.get('timesteps', 0)):,} steps"
    env = _make_render_env(
        namespace,
        env_config=clean_env_config,
        reward_config=config["reward_config"],
        task_distance_m=task_distance_m,
        max_policy_steps=max_policy_steps,
    )

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}_cbf_free_seed{int(args.seed)}"
    video_path = output_dir / f"{stem}.mp4"
    preview_path = output_dir / f"{stem}_preview.png"
    summary_path = output_dir / f"{stem}_summary.json"
    if video_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing render: {video_path}")

    policy_dt = protocol._policy_dt(env)
    writer: cv2.VideoWriter | None = None
    observation, info = env.reset(seed=int(args.seed))
    info = dict(info)
    total_return = 0.0
    total_distance_m = 0.0
    collisions = 0
    rendered_steps = 0
    terminated = False
    truncated = False
    last_info = info
    started = time.perf_counter()

    try:
        for step in range(1, steps_requested + 1):
            policy_observation = _policy_observation(model, observation)
            action_physical, action_source = _raw_physical_action(
                model, policy_observation
            )
            action_normalized = np.asarray(
                namespace["_physical_to_normalized_action"](env, action_physical),
                dtype=np.float32,
            ).reshape(-1)[:2]
            observation, reward, terminated, truncated, info = env.step(
                action_normalized
            )
            info = dict(info)
            last_info = info
            total_return += float(reward)
            step_distance = float(
                info.get(
                    "task_distance_step_m",
                    info.get("pipeline_distance_step_m", 0.0),
                )
            )
            if np.isfinite(step_distance):
                total_distance_m += max(step_distance, 0.0)
            step_events = max(int(info.get("ego_collision_events", 0)), 0)
            if step_events == 0 and bool(
                info.get("ego_collision", False)
                or info.get("task_collision_terminated", False)
            ):
                step_events = 1
            collisions += step_events
            frame_info = dict(info)
            frame_info["_base_env"] = env.unwrapped
            frame = _annotate(
                _frame_rgb(env),
                variant_label=variant_label,
                seed=int(args.seed),
                step=step,
                policy_dt=policy_dt,
                action_physical=action_physical,
                info=frame_info,
                total_distance_m=total_distance_m,
                collisions=collisions,
            )
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(args.fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {video_path}")
            writer.write(frame)
            rendered_steps += 1
            if terminated or truncated:
                break
        if writer is not None and rendered_steps:
            if not cv2.imwrite(str(preview_path), frame):
                raise RuntimeError(f"Could not write preview: {preview_path}")
    finally:
        if writer is not None:
            writer.release()
        env.close()
        del model
        if th.cuda.is_available():
            th.cuda.empty_cache()

    if rendered_steps <= 0 or not video_path.is_file():
        raise RuntimeError("No video frames were produced")
    summary = {
        "evaluation_kind": "true_cbf_free_render",
        "variant": args.variant,
        "variant_label": variant_label,
        "model_path": str(model_path.resolve()),
        "seed": int(args.seed),
        "external_cbf": "REMOVED",
        "policy_cbf_projection": "REMOVED",
        "generic_traffic_safety_guard_preserved": bool(
            clean_env_config.get("traffic_safety", {}).get("dynamics_guard", True)
        ),
        "task_distance_m": task_distance_m,
        "steps_requested": steps_requested,
        "steps_rendered": rendered_steps,
        "policy_dt_s": policy_dt,
        "video_fps": int(args.fps),
        "action_source": action_source,
        "total_return": total_return,
        "total_distance_m": total_distance_m,
        "distinct_ego_collision_events": collisions,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "video_path": str(video_path.resolve()),
        "preview_path": str(preview_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "elapsed_s": time.perf_counter() - started,
        "final_info": {
            str(key): value
            for key, value in last_info.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
