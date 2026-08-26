"""Run a small timing ablation for the external PPO CBF.

The pilot keeps the completed nominal policy, source environment, paired
episode seeds, and deployment gains fixed. It evaluates the CBF once per
20 Hz policy action while the lane-free dynamics still integrate at 100 Hz.
The surrounding-vehicle controller is refreshed at every 100 Hz physics tick.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_cbf_filter_ablation as protocol
import run_ppo_cbf_progression as progression
from evaluate_ppo_cbf_gain_grid_stage1 import (
    _source_cbf_snapshot,
    evaluation_env_config,
    load_source_run,
)


DEFAULT_C1 = 0.5
DEFAULT_C2 = 3.0
DEFAULT_EPISODES = 20
DEFAULT_SEED_START = 1_200_000
DEFAULT_WORKERS = 20
DEFAULT_CORRECTION_EPSILON = 0.03
DEFAULT_EPS_SIDE = 0.10
DEFAULT_TTC_CAP = 30.0
DEFAULT_TASK_DISTANCE_M = 600.0
DEFAULT_TASK_MAX_POLICY_STEPS = 6_000
DEFAULT_PHYSICS_HZ = 100
DEFAULT_POLICY_HZ = 20
DEFAULT_RUN_DIR = Path("artifacts/1MRun/nom/ppo_nominal/seed_307")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/1MRun/nom/external_cbf_timing_pilot_fcbf20_fpolicy20_fphysics100"
)


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(frame["timesteps"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    return float(np.average(values[valid], weights=weights[valid])) if np.any(valid) else float("nan")


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    total_distance = float(pd.to_numeric(frame["total_distance_m"], errors="coerce").sum())
    total_collisions = int(pd.to_numeric(frame["distinct_ego_collision_events"], errors="coerce").sum())
    row: dict[str, Any] = {
        "episodes": int(len(frame)),
        "timesteps": int(pd.to_numeric(frame["timesteps"], errors="coerce").sum()),
        "return_mean": float(pd.to_numeric(frame["episode_return"], errors="coerce").mean()),
        "distance_mean_m": float(pd.to_numeric(frame["total_distance_m"], errors="coerce").mean()),
        "completion_rate": float(pd.to_numeric(frame["distance_completion_rate"], errors="coerce").mean()),
        "collision_events": total_collisions,
        "collision_events_per_km_pooled": (
            1000.0 * total_collisions / total_distance if total_distance > 0.0 else float("nan")
        ),
        "h_min_min": float(pd.to_numeric(frame["h_min"], errors="coerce").min()),
        "event_intervention_rate_weighted": _weighted_mean(frame, "event_intervention_rate"),
        "shadow_intervention_rate_weighted": _weighted_mean(frame, "shadow_event_intervention_rate"),
        "mean_correction_norm_weighted": _weighted_mean(frame, "mean_correction_norm"),
        "qp_failure_rate_weighted": _weighted_mean(frame, "qp_failure_rate"),
        "mean_jerk_norm_weighted": _weighted_mean(frame, "mean_jerk_norm"),
    }
    return pd.DataFrame([row])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--c1", type=float, default=DEFAULT_C1)
    parser.add_argument("--c2", type=float, default=DEFAULT_C2)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--physics-hz", type=float, default=DEFAULT_PHYSICS_HZ)
    parser.add_argument("--policy-hz", type=float, default=DEFAULT_POLICY_HZ)
    parser.add_argument("--eps-side", type=float, default=DEFAULT_EPS_SIDE)
    parser.add_argument(
        "--max-neighbor-constraints",
        type=int,
        default=None,
        help="Override the source CBF neighbor cap for this pilot; omitted keeps the source value.",
    )
    parser.add_argument(
        "--task-max-policy-steps",
        type=int,
        default=None,
        help="Maximum policy steps; defaults to the 20 Hz horizon converted to the selected policy rate.",
    )
    parser.add_argument(
        "--correction-epsilon",
        type=float,
        default=DEFAULT_CORRECTION_EPSILON,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = (
        args.project_root.resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    run_dir = _resolve(root, args.run_dir)
    output_dir = _resolve(root, args.output_dir)
    if args.episodes <= 0 or args.workers <= 0:
        raise ValueError("episodes and workers must be positive")
    if args.c1 < 0.0 or args.c2 < 0.0:
        raise ValueError("c1 and c2 must be non-negative")
    if args.physics_hz <= 0.0 or args.policy_hz <= 0.0:
        raise ValueError("physics-hz and policy-hz must be positive")
    if args.max_neighbor_constraints is not None and args.max_neighbor_constraints <= 0:
        raise ValueError("max-neighbor-constraints must be positive when provided")
    if args.eps_side < 0.0:
        raise ValueError("eps-side must be non-negative")
    substeps = args.physics_hz / args.policy_hz
    if abs(substeps - round(substeps)) > 1e-9 or substeps < 1.0:
        raise ValueError("physics-hz must be an integer multiple of policy-hz")

    run_config, model_path, model_sha256 = load_source_run(run_dir)
    env_config = evaluation_env_config(run_config)
    env_config.update(
        {
            "dt": 1.0 / float(args.physics_hz),
            "simulation_frequency": float(args.physics_hz),
            "policy_frequency": float(args.policy_hz),
            "vehicle_policy_frequency": float(args.physics_hz),
            "cbf_substep_filtering": False,
            "cbf_require_initial_safe_set": False,
        }
    )
    if int(round(float(env_config["simulation_frequency"]))) != int(round(args.physics_hz)):
        raise RuntimeError("timing pilot requires 100 Hz physics")

    namespace = protocol.bootstrap_notebook_namespace(root)
    protocol.exec_required_notebook_cells(
        root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    namespace["DEVICE"] = "cpu"
    namespace.update(copy.deepcopy(_source_cbf_snapshot(run_config)))
    if args.max_neighbor_constraints is not None:
        namespace["CBF_MAX_NEIGHBOR_CONSTRAINTS"] = int(args.max_neighbor_constraints)
    namespace["CBF_EPS_SIDE"] = float(args.eps_side)
    namespace["CBF_K0"] = float(args.c1 * args.c2)
    namespace["CBF_K1"] = float(args.c1 + args.c2)

    eval_args = argparse.Namespace(
        device="cpu",
        training_seed=int(run_config["training_seed"]),
        correction_epsilon=float(args.correction_epsilon),
        ttc_cap=DEFAULT_TTC_CAP,
        task_distance_m=DEFAULT_TASK_DISTANCE_M,
        task_max_policy_steps=(
            int(args.task_max_policy_steps)
            if args.task_max_policy_steps is not None
            else int(round(DEFAULT_TASK_MAX_POLICY_STEPS * args.policy_hz / DEFAULT_POLICY_HZ))
        ),
        post_train_eval_workers=int(args.workers),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "episode_metrics.csv"
    status_path = output_dir / "status.json"
    started = time.perf_counter()
    rows = progression._evaluate_complete_episode_rows(
        namespace,
        model_path=model_path,
        variant="ppo_nominal",
        training_seed=int(run_config["training_seed"]),
        env_config=env_config,
        reward_config=copy.deepcopy(run_config["reward_config"]),
        args=eval_args,
        modes=("cbf",),
        episode_count=int(args.episodes),
        seed_start=int(args.seed_start),
        action_source="policy",
        progress_path=progress_path,
        status_path=status_path,
        progress_started=started,
        progress_variant="ppo_nominal:timing_pilot",
    )

    public_rows = [
        {key: value for key, value in row.items() if not str(key).startswith("_")}
        for row in rows
    ]
    frame = pd.DataFrame(public_rows)
    frame.to_csv(progress_path, index=False)
    summary = _summary(frame)
    summary.insert(0, "c1", float(args.c1))
    summary.insert(1, "c2", float(args.c2))
    summary.insert(2, "k1", float(args.c1 + args.c2))
    summary.insert(3, "k0", float(args.c1 * args.c2))
    summary.to_csv(output_dir / "summary.csv", index=False)

    manifest = {
        "evaluation_kind": "external_cbf_timing_pilot",
        "source_run_dir": str(run_dir),
        "source_model": str(model_path),
        "source_model_sha256": model_sha256,
        "training_seed": int(run_config["training_seed"]),
        "episode_seeds": [int(args.seed_start) + i for i in range(int(args.episodes))],
        "episodes": int(args.episodes),
        "workers": int(args.workers),
        "c1": float(args.c1),
        "c2": float(args.c2),
        "k1": float(args.c1 + args.c2),
        "k0": float(args.c1 * args.c2),
        "correction_epsilon": float(args.correction_epsilon),
        "eps_side": float(args.eps_side),
        "physics_hz": float(args.physics_hz),
        "policy_hz": float(args.policy_hz),
        "cbf_hz": float(args.policy_hz),
        "physics_dt_s": 1.0 / float(args.physics_hz),
        "policy_dt_s": 1.0 / float(args.policy_hz),
        "physics_substeps_per_policy_action": int(round(substeps)),
        "max_neighbor_constraints": int(namespace["CBF_MAX_NEIGHBOR_CONSTRAINTS"]),
        "task_max_policy_steps": int(eval_args.task_max_policy_steps),
        "cbf_substep_filtering": False,
        "vehicle_policy_frequency_hz": float(args.physics_hz),
        "action_semantics": (
            "CBF-filtered physical action held for "
            f"{int(round(substeps))} physics frames per policy action"
        ),
        "complete": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# External-CBF timing pilot\n\n"
        "This pilot evaluates the nominal PPO policy with the external "
        "CBF applied once per 20 Hz policy/action update. The 100 Hz simulator "
        "holds that filtered physical action for five physics frames, while "
        "social vehicle controllers update at 100 Hz. It uses "
        "the same paired seeds and deployment gains as the Stage 1 candidate "
        "specified in `manifest.json`.\n",
        encoding="utf-8",
    )
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    status.update({"state": "complete", "summary_path": str((output_dir / "summary.csv").resolve())})
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("[timing-pilot] complete", flush=True)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
