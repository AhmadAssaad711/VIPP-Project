"""Stage 1 deployment-gain grid for the completed 1M nominal PPO policy.

The nominal policy is kept fixed while the external CBF deployment gains are
varied through

    k1 = c1 + c2
    k0 = c1 * c2

Stage 1 defaults to c1,c2 in {0, 0.5, ..., 5.0}.  Because the CBF only uses
k0 and k1, (c1, c2) and (c2, c1) are identical; only c1 <= c2 is evaluated.
That produces 66 candidates and, with the default 20 episodes per candidate,
1320 evaluation episodes.

The runner uses the canonical PPO progression evaluator for the environment,
external CBF projection, distance-task termination, and episode metrics.  It
writes a manifest and checkpoints progress after every episode so a long sweep
can be resumed safely after interruption.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

# Keep one native numerical thread per process.  The evaluator is CPU-bound and
# this avoids multiplying BLAS/OpenMP pools across repeated environment builds.
for _native_thread_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(_native_thread_key, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

import run_cbf_filter_ablation as protocol
import run_ppo_cbf_progression as progression


STAGE_SCHEMA_VERSION = 1
VARIANT = "ppo_nominal"
TRAINING_SEED = 307
DEFAULT_MIN_C = 0.0
DEFAULT_MAX_C = 5.0
DEFAULT_GRID_STEP = 0.5
DEFAULT_EPISODES = 20
DEFAULT_SEED_START = 1_200_000
DEFAULT_WORKERS = 20
DEFAULT_DEVICE = "cpu"
DEFAULT_TASK_DISTANCE_M = 600.0
DEFAULT_TASK_MAX_POLICY_STEPS = 3_000
DEFAULT_TTC_CAP = 30.0
DEFAULT_CORRECTION_EPSILON = 0.03
DEFAULT_RUN_DIR = Path(
    "artifacts/1MRun/nom/ppo_nominal/seed_307"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/1MRun/nom/external_cbf_gain_grid_stage1"
)


def _round_grid_value(value: float) -> float:
    """Avoid binary floating-point tails in IDs, manifests, and CSV files."""

    return float(round(float(value), 10))


def grid_values(min_c: float, max_c: float, step: float) -> list[float]:
    """Return an inclusive, evenly spaced c-value grid.

    The range must be divisible by the step.  Enforcing this makes the number
    of candidates deterministic and prevents a nearly-equal endpoint from
    being silently omitted.
    """

    min_c = float(min_c)
    max_c = float(max_c)
    step = float(step)
    if not np.isfinite(min_c) or not np.isfinite(max_c) or not np.isfinite(step):
        raise ValueError("Grid bounds and step must be finite")
    if min_c < 0.0 or max_c < min_c:
        raise ValueError("Grid requires 0 <= min_c <= max_c")
    if step <= 0.0:
        raise ValueError("Grid step must be positive")
    span = (max_c - min_c) / step
    count = int(round(span))
    if count < 0 or not np.isclose(span, count, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Grid range [{min_c}, {max_c}] is not divisible by step {step}"
        )
    return [_round_grid_value(min_c + index * step) for index in range(count + 1)]


def make_candidates(
    min_c: float = DEFAULT_MIN_C,
    max_c: float = DEFAULT_MAX_C,
    step: float = DEFAULT_GRID_STEP,
) -> list[dict[str, Any]]:
    """Build the symmetry-reduced c1/c2 grid and its derived CBF gains."""

    values = grid_values(min_c, max_c, step)
    candidates: list[dict[str, Any]] = []
    for c1_index, c1 in enumerate(values):
        for c2_index in range(c1_index, len(values)):
            c2 = values[c2_index]
            candidate_index = len(candidates) + 1
            candidates.append(
                {
                    "candidate_id": f"pair_{candidate_index:03d}",
                    "grid_index_c1": int(c1_index),
                    "grid_index_c2": int(c2_index),
                    "c1": _round_grid_value(c1),
                    "c2": _round_grid_value(c2),
                    "k1": _round_grid_value(c1 + c2),
                    "k0": _round_grid_value(c1 * c2),
                }
            )
    return candidates


def expected_episode_seeds(seed_start: int, episodes: int) -> list[int]:
    """Return one shared seed list used for every candidate."""

    seed_start = int(seed_start)
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    return [seed_start + offset for offset in range(episodes)]


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing required JSON file: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return protocol.canonical_config_hash({"payload": payload})


def _manifest_protocol_hash(manifest: dict[str, Any]) -> str:
    """Hash the effective evaluation protocol, excluding raw-source bookkeeping."""

    protocol_view = copy.deepcopy(manifest)
    protocol_view.pop("config_hash", None)
    # The raw run_config can acquire an orthogonal runtime override after a
    # sweep starts.  The embedded effective env/reward/CBF protocol is what
    # determines whether existing episode rows are safe to resume.
    protocol_view.pop("source_run_config_hash", None)
    return _json_hash(protocol_view)


def load_source_run(
    run_dir: Path,
    model_path: Path | None = None,
) -> tuple[dict[str, Any], Path, str]:
    """Load and verify the completed 1M nominal PPO source run."""

    run_config_path = run_dir / "run_config.json"
    completion_path = run_dir / "training_complete.json"
    run_config = _read_json(run_config_path)
    completion = _read_json(completion_path)
    if str(run_config.get("variant")) != VARIANT:
        raise ValueError(
            f"Stage 1 expects variant={VARIANT!r}; found {run_config.get('variant')!r}"
        )
    if int(run_config.get("timesteps", -1)) != 1_000_000:
        raise ValueError(
            "Stage 1 is wired for the completed 1M policy; "
            f"run_config reports {run_config.get('timesteps')!r} timesteps"
        )
    if int(run_config.get("training_seed", -1)) != TRAINING_SEED:
        raise ValueError(
            f"Stage 1 expects training seed {TRAINING_SEED}; "
            f"run_config reports {run_config.get('training_seed')!r}"
        )
    if int(completion.get("num_timesteps", -1)) != 1_000_000:
        raise ValueError(
            "training_complete.json does not confirm exactly 1,000,000 timesteps"
        )

    model_path = (model_path or (run_dir / "model_final.zip")).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing source policy: {model_path}")
    expected_model_name = str(completion.get("model_file", "model_final.zip"))
    if model_path.name != expected_model_name:
        raise ValueError(
            f"Completion record names {expected_model_name!r}, but requested {model_path.name!r}"
        )
    model_sha256 = protocol.file_sha256(model_path)
    expected_sha256 = str(completion.get("model_sha256", ""))
    if expected_sha256 and model_sha256 != expected_sha256:
        raise RuntimeError(
            f"Source policy checksum mismatch for {model_path}: "
            f"expected {expected_sha256}, observed {model_sha256}"
        )
    return run_config, model_path, model_sha256


def _source_cbf_snapshot(run_config: dict[str, Any]) -> dict[str, Any]:
    training_signature = run_config.get("training_signature", {})
    snapshot = training_signature.get("cbf")
    if not isinstance(snapshot, dict):
        raise ValueError("Source run_config is missing training_signature.cbf")
    required = (
        "CBF_AX_BOUNDS",
        "CBF_AY_BOUNDS",
        "CBF_EPS_SIDE",
        "CBF_K0",
        "CBF_K1",
        "CBF_MAX_NEIGHBOR_CONSTRAINTS",
        "CBF_NEIGHBOR_RANGE",
        "CBF_QP_FEASIBILITY_TOL",
        "CBF_TARGET_PAIR_DY",
    )
    missing = [key for key in required if key not in snapshot]
    if missing:
        raise ValueError(f"Source CBF snapshot is missing: {missing}")
    return {key: copy.deepcopy(snapshot[key]) for key in required}


def evaluation_env_config(run_config: dict[str, Any]) -> dict[str, Any]:
    """Keep source spawning but allow candidate-specific psi1 at reset.

    ``traffic_safety.spawn_cbf_safe_set`` remains unchanged, so the simulator
    samples the same source-compatible initial states for every candidate.  The
    explicit top-level override only prevents the wrapper from rejecting a
    state because the *new deployment gain* makes psi1 negative at reset.
    """

    env_config = copy.deepcopy(run_config["env_config"])
    env_config["cbf_require_initial_safe_set"] = False
    return env_config


def apply_deployment_gains(
    namespace: dict[str, Any],
    source_snapshot: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    """Install the source CBF settings and replace only deployment k0/k1."""

    namespace.update(copy.deepcopy(source_snapshot))
    namespace["CBF_K0"] = float(candidate["k0"])
    namespace["CBF_K1"] = float(candidate["k1"])


def evaluation_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the small argument object consumed by the canonical evaluator."""

    return argparse.Namespace(
        correction_epsilon=float(args.correction_epsilon),
        ttc_cap=float(args.ttc_cap),
        task_distance_m=float(args.task_distance_m),
        task_max_policy_steps=int(args.task_max_policy_steps),
    )


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _mean(values: Iterable[Any], default: float = np.nan) -> float:
    array = _finite_values(values)
    return float(np.mean(array)) if array.size else float(default)


def _std(values: Iterable[Any], default: float = np.nan) -> float:
    array = _finite_values(values)
    return float(np.std(array, ddof=1)) if array.size > 1 else float(default)


def _min(values: Iterable[Any], default: float = np.nan) -> float:
    array = _finite_values(values)
    return float(np.min(array)) if array.size else float(default)


def _weighted_mean(
    values: Iterable[Any], weights: Iterable[Any], default: float = np.nan
) -> float:
    value_array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(
        dtype=float
    )
    weight_array = pd.to_numeric(pd.Series(list(weights)), errors="coerce").to_numpy(
        dtype=float
    )
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0.0)
    if not np.any(valid):
        return float(default)
    return float(np.average(value_array[valid], weights=weight_array[valid]))


def summarize_episode_metrics(
    episodes: pd.DataFrame,
    candidates: list[dict[str, Any]],
) -> pd.DataFrame:
    """Create one candidate-level table without ranking by a single KPI."""

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        group = episodes[episodes["candidate_id"].eq(candidate_id)] if not episodes.empty else episodes
        distances = pd.to_numeric(group.get("total_distance_m", pd.Series(dtype=float)), errors="coerce")
        collisions = pd.to_numeric(
            group.get("distinct_ego_collision_events", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        timesteps = pd.to_numeric(
            group.get("timesteps", pd.Series(dtype=float)), errors="coerce"
        )
        total_distance = float(distances.sum()) if len(group) else 0.0
        total_collisions = float(collisions.sum()) if len(group) else 0.0
        row: dict[str, Any] = {
            **candidate,
            "episodes_completed": int(len(group)),
            "return_mean": _mean(group.get("episode_return", [])),
            "return_std": _std(group.get("episode_return", [])),
            "distance_mean_m": _mean(group.get("total_distance_m", [])),
            "distance_std_m": _std(group.get("total_distance_m", [])),
            "distance_completion_rate_mean": _mean(
                group.get("distance_completion_rate", [])
            ),
            "collision_episode_rate": _mean(
                collisions.to_numpy(dtype=float) > 0.0, default=np.nan
            ),
            "collision_events_per_km_pooled": (
                1000.0 * total_collisions / total_distance
                if total_distance > 1e-9
                else np.nan
            ),
            "mean_abs_speed_deviation_weighted": _weighted_mean(
                group.get("mean_abs_speed_deviation", []), timesteps
            ),
            "mean_lat_y_error_m_weighted": _weighted_mean(
                group.get("mean_lat_y_error_m", []), timesteps
            ),
            "event_intervention_rate_weighted": _weighted_mean(
                group.get("event_intervention_rate", []), timesteps
            ),
            "mean_correction_norm_weighted": _weighted_mean(
                group.get("mean_correction_norm", []), timesteps
            ),
            "qp_failure_rate_weighted": _weighted_mean(
                group.get("qp_failure_rate", []), timesteps
            ),
            "mean_jerk_norm_weighted": _weighted_mean(
                group.get("mean_jerk_norm", []), timesteps
            ),
            "h_min_min": _min(group.get("h_min", [])),
            "shadow_intervention_rate_weighted": _weighted_mean(
                group.get("shadow_event_intervention_rate", []), timesteps
            ),
            "total_distance_m": total_distance,
            "total_collision_events": int(total_collisions),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, default=str))


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if not str(key).startswith("_")}
        for row in rows
    ]


def _status(
    *,
    state: str,
    output_dir: Path,
    candidates: list[dict[str, Any]],
    episodes: int,
    completed_keys: set[tuple[str, int]],
    started_at: float,
    current_candidate: dict[str, Any] | None = None,
    last_row: dict[str, Any] | None = None,
    error: str | None = None,
    worker_count: int | None = None,
) -> dict[str, Any]:
    completed_candidate_ids = {
        candidate_id
        for candidate_id, _episode_index in completed_keys
        if sum(1 for key in completed_keys if key[0] == candidate_id) >= episodes
    }
    value: dict[str, Any] = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "state": str(state),
        "output_dir": str(output_dir.resolve()),
        "expected_candidates": int(len(candidates)),
        "expected_episodes_per_candidate": int(episodes),
        "expected_total_episodes": int(len(candidates) * episodes),
        "completed_episodes": int(len(completed_keys)),
        "completed_candidates": int(len(completed_candidate_ids)),
        "current_candidate_id": (
            current_candidate.get("candidate_id") if current_candidate else None
        ),
        "current_c1": current_candidate.get("c1") if current_candidate else None,
        "current_c2": current_candidate.get("c2") if current_candidate else None,
        "last_episode_index": last_row.get("episode_index") if last_row else None,
        "last_episode_seed": last_row.get("episode_seed") if last_row else None,
        "last_distance_m": last_row.get("total_distance_m") if last_row else None,
        "last_collision_events": (
            last_row.get("distinct_ego_collision_events") if last_row else None
        ),
        "elapsed_s": float(max(0.0, time.perf_counter() - started_at)),
    }
    if error is not None:
        value["error"] = error
    if worker_count is not None:
        value["worker_count"] = int(worker_count)
    return value


def _plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    c1_values = sorted(pd.to_numeric(summary["c1"], errors="coerce").dropna().unique())
    c2_values = sorted(pd.to_numeric(summary["c2"], errors="coerce").dropna().unique())
    if not c1_values or not c2_values:
        return
    metrics = (
        ("distance_completion_rate_mean", "Distance completion rate", "viridis"),
        ("collision_episode_rate", "Collision episode rate", "magma"),
        ("return_mean", "Mean episode return", "plasma"),
        ("event_intervention_rate_weighted", "CBF intervention rate", "cividis"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, (column, title, cmap) in zip(axes.flat, metrics):
        matrix = (
            summary.pivot(index="c2", columns="c1", values=column)
            .reindex(index=c2_values, columns=c1_values)
            .to_numpy(dtype=float)
        )
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            interpolation="none",
        )
        axis.set_title(title)
        axis.set_xlabel("c1")
        axis.set_ylabel("c2")
        axis.set_xticks(np.arange(len(c1_values)))
        axis.set_xticklabels([f"{value:g}" for value in c1_values])
        axis.set_yticks(np.arange(len(c2_values)))
        axis.set_yticklabels([f"{value:g}" for value in c2_values])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Stage 1 external-CBF gain grid")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _manifest(
    *,
    project_root: Path,
    run_dir: Path,
    run_config: dict[str, Any],
    model_path: Path,
    model_sha256: str,
    output_dir: Path,
    candidates: list[dict[str, Any]],
    episode_seeds: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_snapshot = _source_cbf_snapshot(run_config)
    env_config = evaluation_env_config(run_config)
    reward_config = copy.deepcopy(run_config["reward_config"])
    spawn_gain = (
        env_config.get("traffic_safety", {}).get("spawn_cbf_k1")
        if isinstance(env_config.get("traffic_safety"), dict)
        else None
    )
    manifest: dict[str, Any] = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "stage": "stage1_external_cbf_c_gain_grid",
        "project_root": str(project_root.resolve()),
        "source_run_dir": str(run_dir.resolve()),
        "source_run_config": str((run_dir / "run_config.json").resolve()),
        "source_run_config_hash": _json_hash(run_config),
        "source_model_path": str(model_path.resolve()),
        "source_model_sha256": model_sha256,
        "variant": VARIANT,
        "training_seed": TRAINING_SEED,
        "training_timesteps": 1_000_000,
        "grid": {
            "min_c": float(args.min_c),
            "max_c": float(args.max_c),
            "step": float(args.step),
            "values": sorted({float(candidate["c1"]) for candidate in candidates}),
            "symmetry_reduction": "evaluate only c1 <= c2",
            "candidate_count": int(len(candidates)),
        },
        "evaluation": {
            "episodes_per_candidate": int(args.episodes),
            "episode_seeds": list(map(int, episode_seeds)),
            "shared_episode_seeds_across_candidates": True,
            "external_cbf": "ON",
            "device": str(args.device),
            "correction_epsilon": float(args.correction_epsilon),
            "ttc_cap": float(args.ttc_cap),
            "task_distance_m": float(args.task_distance_m),
            "task_max_policy_steps": int(args.task_max_policy_steps),
        },
        "source_cbf_snapshot": source_snapshot,
        "deployment_cbf_parameters": "CBF_K0 and CBF_K1 vary per candidate; all other CBF settings remain from the 1M source run",
        "safe_spawn_protocol": {
            "spawn_cbf_k1": spawn_gain,
            "policy": "keep the source env_config traffic_safety spawn settings fixed so all candidates use the same seeded initial-state protocol",
        },
        "evaluation_env_overrides": {
            "cbf_require_initial_safe_set": False,
            "reason": "allow candidate-specific deployment gains to be evaluated from the paired source-safe spawn states",
        },
        "env_config": env_config,
        "reward_config": reward_config,
        "candidates": candidates,
    }
    manifest["config_hash"] = _json_hash(manifest)
    return manifest


def _write_run_readme(output_dir: Path, manifest: dict[str, Any], command: str) -> None:
    text = f"""# Stage 1 external-CBF gain grid

This directory is the evaluation-only sweep for the completed 1M nominal PPO
policy.  The source actor is fixed.  Each candidate uses

`k1 = c1 + c2`, `k0 = c1 * c2`

with `c1 <= c2` to remove the exact swap symmetry.  The external CBF is ON for
every episode.  The same episode seeds are reused for every candidate, and the
source safe-spawn configuration is kept fixed so the gain comparison is paired.

- candidates: `{manifest['grid']['candidate_count']}`
- episodes per candidate: `{manifest['evaluation']['episodes_per_candidate']}`
- total episodes: `{manifest['grid']['candidate_count'] * manifest['evaluation']['episodes_per_candidate']}`
- grid values: `{manifest['grid']['values']}`
- source model: `{manifest['source_model_path']}`

The run is resumable.  `status.json` and `episode_metrics.csv` are updated after
each completed episode.  The main outputs are:

- `grid_manifest.json`: immutable source and protocol metadata
- `episode_metrics.csv`: one row per candidate and episode
- `collision_events.csv`: event-level CBF/collision attribution when present
- `summary.csv`: candidate-level aggregate metrics
- `summary.png`: candidate heatmaps

Re-run the same command after an interruption to continue from the last saved
episode:

```text
{command}
```
"""
    _atomic_write_text(output_dir / "README.md", text)


_PARALLEL_WORKER_STATE: dict[str, Any] | None = None


def _initialize_parallel_worker(
    project_root: str,
    model_path: str,
    device: str,
    source_snapshot: dict[str, Any],
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    eval_args: argparse.Namespace,
) -> None:
    """Load one independent evaluator state in a spawned CPU worker."""

    protocol.set_stable_native_defaults()
    try:
        th.set_num_threads(1)
        th.set_num_interop_threads(1)
    except RuntimeError:
        pass

    root = Path(project_root).resolve()
    namespace = protocol.bootstrap_notebook_namespace(root)
    protocol.exec_required_notebook_cells(
        root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    namespace["DEVICE"] = str(device)
    model = progression.load_model(VARIANT, Path(model_path), str(device))

    global _PARALLEL_WORKER_STATE
    _PARALLEL_WORKER_STATE = {
        "namespace": namespace,
        "model": model,
        "source_snapshot": source_snapshot,
        "env_config": env_config,
        "reward_config": reward_config,
        "eval_args": eval_args,
    }


def _evaluate_parallel_episode(
    task: tuple[dict[str, Any], int, int]
) -> tuple[str, int, dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one candidate/seed pair inside an isolated worker."""

    state = _PARALLEL_WORKER_STATE
    if state is None:
        raise RuntimeError("parallel worker was not initialized")

    candidate, episode_index, episode_seed = task
    apply_deployment_gains(
        state["namespace"], state["source_snapshot"], candidate
    )
    row = progression.evaluate_completed_episode(
        state["namespace"],
        model=state["model"],
        variant=VARIANT,
        mode="cbf",
        training_seed=TRAINING_SEED,
        episode_index=int(episode_index),
        episode_seed=int(episode_seed),
        env_config=state["env_config"],
        reward_config=state["reward_config"],
        args=state["eval_args"],
    )
    event_rows = row.pop("_collision_event_records", [])
    return str(candidate["candidate_id"]), int(episode_index), row, event_rows


def _run_parallel_sweep(
    *,
    args: argparse.Namespace,
    project_root: Path,
    model_path: Path,
    model_sha256: str,
    output_dir: Path,
    candidates: list[dict[str, Any]],
    episode_seeds: list[int],
    rows: list[dict[str, Any]],
    completed_keys: set[tuple[str, int]],
    collision_events: list[dict[str, Any]],
    source_snapshot: dict[str, Any],
    env_config: dict[str, Any],
    reward_config: dict[str, float],
    eval_args: argparse.Namespace,
    started_at: float,
) -> int:
    """Finish a sweep with independent, single-threaded CPU workers."""

    candidate_map = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    pending_tasks = [
        (candidate, int(episode_index), int(episode_seed))
        for candidate in candidates
        for episode_index, episode_seed in enumerate(episode_seeds, start=1)
        if (str(candidate["candidate_id"]), int(episode_index)) not in completed_keys
    ]
    print(
        f"[stage1] parallel_workers={int(args.workers)} "
        f"pending_episodes={len(pending_tasks)}",
        flush=True,
    )

    if pending_tasks:
        executor = ProcessPoolExecutor(
            max_workers=int(args.workers),
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_parallel_worker,
            initargs=(
                str(project_root),
                str(model_path),
                str(args.device),
                source_snapshot,
                env_config,
                reward_config,
                eval_args,
            ),
        )
        try:
            future_map = {
                executor.submit(_evaluate_parallel_episode, task): task
                for task in pending_tasks
            }
            for completed_count, future in enumerate(
                as_completed(future_map), start=1
            ):
                candidate_id, episode_index, row, event_rows = future.result()
                key = (candidate_id, int(episode_index))
                if key in completed_keys:
                    raise RuntimeError(
                        f"parallel result duplicated existing episode {candidate_id} "
                        f"episode {episode_index}"
                    )
                candidate = candidate_map[candidate_id]
                row.update(
                    {
                        "candidate_id": candidate_id,
                        "grid_index_c1": int(candidate["grid_index_c1"]),
                        "grid_index_c2": int(candidate["grid_index_c2"]),
                        "c1": float(candidate["c1"]),
                        "c2": float(candidate["c2"]),
                        "k1": float(candidate["k1"]),
                        "k0": float(candidate["k0"]),
                        "source_model_sha256": model_sha256,
                    }
                )
                for event_row in event_rows:
                    event_row.update(
                        {
                            "candidate_id": candidate_id,
                            "c1": float(candidate["c1"]),
                            "c2": float(candidate["c2"]),
                            "k1": float(candidate["k1"]),
                            "k0": float(candidate["k0"]),
                        }
                    )
                rows.append(row)
                collision_events.extend(event_rows)
                completed_keys.add(key)
                _write_outputs(
                    output_dir=output_dir,
                    rows=rows,
                    collision_events=collision_events,
                    candidates=candidates,
                    plot=False,
                )
                _atomic_write_json(
                    output_dir / "status.json",
                    _status(
                        state="running",
                        output_dir=output_dir,
                        candidates=candidates,
                        episodes=int(args.episodes),
                        completed_keys=completed_keys,
                        started_at=started_at,
                        current_candidate=candidate,
                        last_row=row,
                        worker_count=int(args.workers),
                    ),
                )
                print(
                    f"[stage1] parallel_completed={completed_count}/"
                    f"{len(pending_tasks)} total={len(completed_keys)}/"
                    f"{len(candidates) * int(args.episodes)} "
                    f"{candidate_id} episode={episode_index}/{args.episodes} "
                    f"distance={float(row.get('total_distance_m', 0.0)):.1f}m "
                    f"collisions={int(row.get('distinct_ego_collision_events', 0))}",
                    flush=True,
                )
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    summary = _write_outputs(
        output_dir=output_dir,
        rows=rows,
        collision_events=collision_events,
        candidates=candidates,
    )
    _atomic_write_json(
        output_dir / "status.json",
        _status(
            state="complete",
            output_dir=output_dir,
            candidates=candidates,
            episodes=int(args.episodes),
            completed_keys=completed_keys,
            started_at=started_at,
            worker_count=int(args.workers),
        ),
    )
    print(f"[stage1] complete: {output_dir}", flush=True)
    print(summary.to_string(index=False), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Stage 1 external-CBF c1/c2 grid for the 1M nominal PPO policy."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-c", type=float, default=DEFAULT_MIN_C)
    parser.add_argument("--max-c", type=float, default=DEFAULT_MAX_C)
    parser.add_argument("--step", type=float, default=DEFAULT_GRID_STEP)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="number of independent single-threaded CPU evaluation workers",
    )
    parser.add_argument("--task-distance-m", type=float, default=DEFAULT_TASK_DISTANCE_M)
    parser.add_argument(
        "--task-max-policy-steps",
        type=int,
        default=DEFAULT_TASK_MAX_POLICY_STEPS,
    )
    parser.add_argument("--ttc-cap", type=float, default=DEFAULT_TTC_CAP)
    parser.add_argument(
        "--correction-epsilon",
        type=float,
        default=DEFAULT_CORRECTION_EPSILON,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the source policy/configuration and print the 66-pair plan without evaluating.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive")
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")
    if int(args.task_max_policy_steps) <= 0:
        raise ValueError("--task-max-policy-steps must be positive")
    for name in ("task_distance_m", "ttc_cap", "correction_epsilon"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if float(args.task_distance_m) <= 0.0:
        raise ValueError("--task-distance-m must be positive")


def _load_existing_rows(
    path: Path,
    candidates: list[dict[str, Any]],
    episode_seeds: list[int],
) -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    if not path.is_file():
        return [], set()
    frame = pd.read_csv(path)
    required = {"candidate_id", "episode_index", "episode_seed", "c1", "c2", "k0", "k1"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Existing episode_metrics.csv is missing columns: {missing}")
    candidate_map = {str(row["candidate_id"]): row for row in candidates}
    completed: set[tuple[str, int]] = set()
    rows = frame.to_dict("records")
    for row in rows:
        candidate_id = str(row["candidate_id"])
        episode_index = int(row["episode_index"])
        if candidate_id not in candidate_map:
            raise RuntimeError(f"Existing results contain unknown candidate {candidate_id}")
        if not 1 <= episode_index <= len(episode_seeds):
            raise RuntimeError(f"Existing result has invalid episode index {episode_index}")
        expected_seed = int(episode_seeds[episode_index - 1])
        if int(row["episode_seed"]) != expected_seed:
            raise RuntimeError(
                f"Existing result seed mismatch for {candidate_id} episode {episode_index}: "
                f"expected {expected_seed}, found {row['episode_seed']}"
            )
        expected = candidate_map[candidate_id]
        for key in ("c1", "c2", "k0", "k1"):
            if not np.isclose(float(row[key]), float(expected[key]), rtol=0.0, atol=1e-9):
                raise RuntimeError(
                    f"Existing result {candidate_id} disagrees with the requested grid in {key}"
                )
        key = (candidate_id, episode_index)
        if key in completed:
            raise RuntimeError(f"Duplicate existing result for {candidate_id} episode {episode_index}")
        completed.add(key)
    return rows, completed


def _load_existing_collision_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def _write_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    collision_events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    plot: bool = True,
) -> pd.DataFrame:
    episode_frame = pd.DataFrame(_public_rows(rows))
    _atomic_write_csv(output_dir / "episode_metrics.csv", episode_frame)
    event_frame = pd.DataFrame(collision_events)
    _atomic_write_csv(output_dir / "collision_events.csv", event_frame)
    summary = summarize_episode_metrics(episode_frame, candidates)
    _atomic_write_csv(output_dir / "summary.csv", summary)
    if plot:
        _plot_summary(summary, output_dir / "summary.png")
    return summary


def main(argv: list[str] | None = None) -> int:
    protocol.set_stable_native_defaults()
    try:
        th.set_num_threads(1)
        th.set_num_interop_threads(1)
    except RuntimeError:
        pass
    args = parse_args(argv)
    _validate_args(args)
    candidates = make_candidates(args.min_c, args.max_c, args.step)
    episode_seeds = expected_episode_seeds(args.seed_start, args.episodes)
    project_root = protocol.find_project_root(
        args.project_root or Path(__file__).resolve().parents[1]
    ).resolve()
    run_dir = _resolve_path(project_root, args.run_dir)
    model_path_arg = _resolve_path(project_root, args.model_path) if args.model_path else None
    run_config, model_path, model_sha256 = load_source_run(run_dir, model_path_arg)
    output_dir = _resolve_path(project_root, args.output_dir)
    manifest = _manifest(
        project_root=project_root,
        run_dir=run_dir,
        run_config=run_config,
        model_path=model_path,
        model_sha256=model_sha256,
        output_dir=output_dir,
        candidates=candidates,
        episode_seeds=episode_seeds,
        args=args,
    )
    print(
        f"[stage1] candidates={len(candidates)} episodes_per_candidate={args.episodes} "
        f"total_episodes={len(candidates) * int(args.episodes)}",
        flush=True,
    )
    print(f"[stage1] source_policy={model_path}", flush=True)
    print(f"[stage1] output_dir={output_dir}", flush=True)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "grid_values": manifest["grid"]["values"],
                    "candidate_count": len(candidates),
                    "episodes_per_candidate": int(args.episodes),
                    "total_episodes": len(candidates) * int(args.episodes),
                    "episode_seeds": episode_seeds,
                    "first_candidate": candidates[0],
                    "last_candidate": candidates[-1],
                    "source_model_sha256": model_sha256,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "grid_manifest.json"
    if manifest_path.is_file():
        observed_manifest = _read_json(manifest_path)
        if observed_manifest.get("config_hash") != manifest["config_hash"]:
            if _manifest_protocol_hash(observed_manifest) != _manifest_protocol_hash(
                manifest
            ):
                raise RuntimeError(
                    "Existing output directory belongs to a different Stage 1 configuration. "
                    "Choose a new --output-dir rather than mixing results."
                )
            print(
                "[stage1] raw source run_config hash changed, but the effective "
                "evaluation protocol is unchanged; resuming existing results",
                flush=True,
            )
    else:
        _atomic_write_json(manifest_path, manifest)
    command = (
        "python scripts\\evaluate_ppo_cbf_gain_grid_stage1.py "
        f"--output-dir \"{args.output_dir}\" --workers {int(args.workers)}"
    )
    _write_run_readme(output_dir, manifest, command)

    episode_path = output_dir / "episode_metrics.csv"
    event_path = output_dir / "collision_events.csv"
    rows, completed_keys = _load_existing_rows(episode_path, candidates, episode_seeds)
    collision_events = _load_existing_collision_events(event_path)
    started_at = time.perf_counter()
    _atomic_write_json(
        output_dir / "status.json",
        _status(
            state="running",
            output_dir=output_dir,
            candidates=candidates,
            episodes=int(args.episodes),
            completed_keys=completed_keys,
            started_at=started_at,
            worker_count=int(args.workers),
        ),
    )

    if int(args.workers) > 1:
        try:
            source_snapshot = _source_cbf_snapshot(run_config)
            env_config = evaluation_env_config(run_config)
            reward_config = copy.deepcopy(run_config["reward_config"])
            eval_args = evaluation_args(args)
            return _run_parallel_sweep(
                args=args,
                project_root=project_root,
                model_path=model_path,
                model_sha256=model_sha256,
                output_dir=output_dir,
                candidates=candidates,
                episode_seeds=episode_seeds,
                rows=rows,
                completed_keys=completed_keys,
                collision_events=collision_events,
                source_snapshot=source_snapshot,
                env_config=env_config,
                reward_config=reward_config,
                eval_args=eval_args,
                started_at=started_at,
            )
        except Exception:
            _atomic_write_json(
                output_dir / "status.json",
                _status(
                    state="failed",
                    output_dir=output_dir,
                    candidates=candidates,
                    episodes=int(args.episodes),
                    completed_keys=completed_keys,
                    started_at=started_at,
                    worker_count=int(args.workers),
                    error=traceback.format_exc(),
                ),
            )
            raise

    try:
        namespace = protocol.bootstrap_notebook_namespace(project_root)
        protocol.exec_required_notebook_cells(
            project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
        )
        namespace["DEVICE"] = str(args.device)
        source_snapshot = _source_cbf_snapshot(run_config)
        env_config = evaluation_env_config(run_config)
        reward_config = copy.deepcopy(run_config["reward_config"])
        eval_args = evaluation_args(args)
        print(f"[stage1] loading policy on {args.device}: {model_path}", flush=True)
        model = progression.load_model(
            VARIANT,
            model_path,
            str(args.device),
        )
        try:
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                if all(
                    (candidate_id, episode_index) in completed_keys
                    for episode_index in range(1, int(args.episodes) + 1)
                ):
                    print(f"[stage1] resume skip {candidate_id} (complete)", flush=True)
                    continue
                apply_deployment_gains(namespace, source_snapshot, candidate)
                print(
                    f"[stage1] {candidate_id} c1={candidate['c1']:g} c2={candidate['c2']:g} "
                    f"k1={candidate['k1']:g} k0={candidate['k0']:g}",
                    flush=True,
                )
                for episode_index, episode_seed in enumerate(episode_seeds, start=1):
                    key = (candidate_id, int(episode_index))
                    if key in completed_keys:
                        continue
                    row = progression.evaluate_completed_episode(
                        namespace,
                        model=model,
                        variant=VARIANT,
                        mode="cbf",
                        training_seed=TRAINING_SEED,
                        episode_index=int(episode_index),
                        episode_seed=int(episode_seed),
                        env_config=env_config,
                        reward_config=reward_config,
                        args=eval_args,
                    )
                    event_rows = row.pop("_collision_event_records", [])
                    row.update(
                        {
                            "candidate_id": candidate_id,
                            "grid_index_c1": int(candidate["grid_index_c1"]),
                            "grid_index_c2": int(candidate["grid_index_c2"]),
                            "c1": float(candidate["c1"]),
                            "c2": float(candidate["c2"]),
                            "k1": float(candidate["k1"]),
                            "k0": float(candidate["k0"]),
                            "source_model_sha256": model_sha256,
                        }
                    )
                    for event_row in event_rows:
                        event_row.update(
                            {
                                "candidate_id": candidate_id,
                                "c1": float(candidate["c1"]),
                                "c2": float(candidate["c2"]),
                                "k1": float(candidate["k1"]),
                                "k0": float(candidate["k0"]),
                            }
                        )
                    rows.append(row)
                    collision_events.extend(event_rows)
                    completed_keys.add(key)
                    _write_outputs(
                        output_dir=output_dir,
                        rows=rows,
                        collision_events=collision_events,
                        candidates=candidates,
                        plot=False,
                    )
                    _atomic_write_json(
                        output_dir / "status.json",
                        _status(
                            state="running",
                            output_dir=output_dir,
                            candidates=candidates,
                            episodes=int(args.episodes),
                            completed_keys=completed_keys,
                            started_at=started_at,
                            current_candidate=candidate,
                            last_row=row,
                        ),
                    )
                    print(
                        f"[stage1] completed={len(completed_keys)}/"
                        f"{len(candidates) * int(args.episodes)} "
                        f"{candidate_id} episode={episode_index}/{args.episodes} "
                        f"seed={episode_seed} distance={float(row.get('total_distance_m', 0.0)):.1f}m "
                        f"collisions={int(row.get('distinct_ego_collision_events', 0))}",
                        flush=True,
                    )
        finally:
            del model
            if th.cuda.is_available():
                th.cuda.empty_cache()

        summary = _write_outputs(
            output_dir=output_dir,
            rows=rows,
            collision_events=collision_events,
            candidates=candidates,
        )
        _atomic_write_json(
            output_dir / "status.json",
            _status(
                state="complete",
                output_dir=output_dir,
                candidates=candidates,
                episodes=int(args.episodes),
                completed_keys=completed_keys,
                started_at=started_at,
            ),
        )
        print(f"[stage1] complete: {output_dir}", flush=True)
        print(summary.to_string(index=False), flush=True)
        return 0
    except Exception:
        _atomic_write_json(
            output_dir / "status.json",
            _status(
                state="failed",
                output_dir=output_dir,
                candidates=candidates,
                episodes=int(args.episodes),
                completed_keys=completed_keys,
                started_at=started_at,
                error=traceback.format_exc(),
            ),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
