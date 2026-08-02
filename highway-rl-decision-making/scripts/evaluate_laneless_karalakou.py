from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from stable_baselines3 import DDPG, PPO

from guided_cbf_minimal import install_minimal_guided_cbf
from laneless_evaluation_registry import (
    build_evaluation_request,
    evaluation_cache_paths,
    latest_completed_training,
    load_matching_evaluation,
    sha256_file,
    stable_json_digest,
    sync_metrics_to_requested_output,
    write_evaluation_manifest,
)
from laneless_script_config import active_traffic_model, env_config_from_args


TEN_KPI_SPECS: tuple[tuple[str, str], ...] = (
    ("Episode return", "episode_return"),
    ("Episode length (steps)", "episode_length_steps"),
    ("Ego collisions / km", "ego_collisions_per_km"),
    ("Minimum h", "h_min"),
    ("QP failure rate", "qp_failure_rate"),
    ("Abs speed error (m/s)", "mean_abs_speed_deviation"),
    ("Mean lateral tracking error (m)", "mean_lat_y_error_m"),
    ("Intervention rate", "event_intervention_rate"),
    ("Correction norm", "mean_correction_norm"),
    ("Mean jerk norm", "mean_jerk_norm"),
)


def ten_kpi_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise the agreed final-evaluation KPIs as mean and sample SD."""

    rows: list[dict[str, float | str]] = []
    for label, column in TEN_KPI_SPECS:
        if column not in metrics:
            raise KeyError(f"Evaluation metrics are missing required KPI column {column!r}")
        values = pd.to_numeric(metrics[column], errors="coerce").dropna()
        rows.append(
            {
                "KPI": label,
                "Mean": float(values.mean()) if len(values) else float("nan"),
                "SD": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def print_ten_kpi_summary(metrics: pd.DataFrame, *, label: str) -> None:
    """Emit the standard KPI table directly in every preset final evaluation."""

    table = ten_kpi_summary(metrics)
    print(f"[eval-runner] 10-KPI final evaluation: {label} (mean +/- sample SD)", flush=True)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)


def set_stable_native_defaults() -> None:
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ]:
        os.environ.setdefault(key, "1")


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return candidate
        nested = candidate / "highway-rl-decision-making"
        if (nested / "notebooks" / "lanelessKaralakou.ipynb").exists():
            return nested
    raise RuntimeError("Could not find project root containing notebooks/lanelessKaralakou.ipynb")


def exec_notebook_cell(notebook: dict[str, Any], notebook_path: Path, cell_index: int, namespace: dict[str, Any]) -> None:
    source = "".join(notebook["cells"][cell_index].get("source", []))
    print(f"[eval-runner] executing notebook cell {cell_index}", flush=True)
    exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)


def apply_cbf_overrides(namespace: dict[str, Any], args: argparse.Namespace) -> None:
    if args.lambda_filter is not None:
        namespace["CBF_FILTER_REWARD_LAMBDA"] = float(args.lambda_filter)
    if args.k0 is not None:
        namespace["CBF_K0"] = float(args.k0)
    if args.k1 is not None:
        namespace["CBF_K1"] = float(args.k1)
    if args.eps_side is not None:
        namespace["CBF_EPS_SIDE"] = float(args.eps_side)


def exec_notebook_cells(
    notebook_path: Path,
    cell_indices: list[int],
    namespace: dict[str, Any],
    args: argparse.Namespace | None = None,
) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in cell_indices:
        cell = notebook["cells"][cell_index]
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        print(f"[evaluate_laneless_karalakou] executing notebook cell {cell_index}", flush=True)
        exec(compile(source, f"{notebook_path}:cell-{cell_index}", "exec"), namespace)
        if args is not None and cell_index == 33:
            apply_cbf_overrides(namespace, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run laneless Karalakou final evaluations out of process.")
    parser.add_argument("--variant", choices=["ppo", "ddpg", "ddpg-cbf", "guided-ddpg-cbf"], required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--traffic-model", choices=["force", "mtm"], default=None)
    parser.add_argument("--env-config-json", default=None)
    parser.add_argument("--env-config-file", type=Path, default=None)
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--lambda-filter", type=float, default=None)
    parser.add_argument("--k0", type=float, default=None)
    parser.add_argument("--k1", type=float, default=None)
    parser.add_argument("--eps-side", type=float, default=None)
    parser.add_argument(
        "--use-latest-training",
        action="store_true",
        help="Evaluate the latest completed immutable training run and reuse only its matching cached evaluation.",
    )
    parser.add_argument(
        "--force-reevaluate",
        action="store_true",
        help="Ignore a matching cached evaluation and run the requested evaluation again.",
    )
    return parser.parse_args()


def with_stem_suffix(path: Path, suffix: str) -> Path:
    suffixed = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    if os.name != "nt" or len(str(suffixed)) < 260:
        return suffixed

    digest = hashlib.sha1(suffixed.stem.encode("utf-8")).hexdigest()[:8]
    overflow = len(str(suffixed)) - 248
    keep = max(24, len(suffixed.stem) - overflow - len(digest) - 1)
    short_stem = suffixed.stem[:keep].rstrip("._-")
    return suffixed.with_name(f"{short_stem}_{digest}{suffixed.suffix}")


def normalize_artifact_suffix(value: str | None) -> str | None:
    if value is None:
        return None
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    suffix = suffix.strip("._-")
    if len(suffix) > 16:
        digest = hashlib.sha1(suffix.encode("utf-8")).hexdigest()[:8]
        suffix = f"{suffix[:7].strip('._-')}_{digest}"
    return suffix or None


def artifact_path(path: Path, traffic_model: str, artifact_suffix: str | None) -> Path:
    suffix = normalize_artifact_suffix(artifact_suffix)
    if suffix is None and traffic_model != "force":
        suffix = f"traffic_{traffic_model}"
    return with_stem_suffix(Path(path), suffix) if suffix else Path(path)


def notebook_code_sha256(notebook_path: Path) -> str:
    """Fingerprint notebook code only, so display-output changes do not invalidate an evaluation."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    return stable_json_digest(sources)


def default_model_path(
    namespace: dict[str, Any],
    args: argparse.Namespace,
    traffic_model: str,
) -> Path:
    model_key_by_variant = {
        "ppo": "MODEL_PATH",
        "ddpg": "DDPG_MODEL_PATH",
        "ddpg-cbf": "DDPG_CBF_MODEL_PATH",
        "guided-ddpg-cbf": "GUIDED_DDPG_CBF_MODEL_PATH",
    }
    return artifact_path(
        Path(namespace[model_key_by_variant[args.variant]]),
        traffic_model,
        args.artifact_suffix,
    )


def atomic_write_metrics(metrics: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    metrics.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def main() -> int:
    faulthandler.enable(all_threads=True)
    set_stable_native_defaults()
    args = parse_args()

    project_root = find_project_root(args.project_root or Path.cwd())
    notebook_path = project_root / "notebooks" / "lanelessKaralakou.ipynb"
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "PPO": PPO,
        "DDPG": DDPG,
    }

    base_cells = [2, 3, 5, 6, 8]
    # Cell 43 is the DDPG-CBF training cell.  Evaluation only needs the CBF
    # definitions and metric helper cells; executing 43 would honor its
    # notebook default and launch a fresh training subprocess before loading
    # the saved model.
    cbf_cells = [33, 35, 37, 39, 41]
    needs_cbf = args.variant in {"ddpg-cbf", "guided-ddpg-cbf"}
    exec_notebook_cells(
        notebook_path,
        base_cells + (cbf_cells if needs_cbf else []),
        namespace,
        args,
    )
    apply_cbf_overrides(namespace, args)

    namespace["DEVICE"] = args.device
    namespace["ENV_CONFIG"] = env_config_from_args(args, namespace["ENV_CONFIG"])
    if args.variant == "guided-ddpg-cbf":
        install_minimal_guided_cbf(namespace)
    traffic_model = active_traffic_model(namespace["ENV_CONFIG"])
    output_path = args.output
    training_run = None
    model_path = Path(args.model_path) if args.model_path is not None else default_model_path(namespace, args, traffic_model)

    if args.use_latest_training:
        if args.model_path is not None:
            raise ValueError("--model-path cannot be combined with --use-latest-training")
        training_run = latest_completed_training(Path(namespace["ARTIFACT_DIR"]), args.variant)
        if training_run is None:
            raise RuntimeError(
                f"No completed archived training run is registered for {args.variant!r}. "
                "Train through the notebook task runner first, or evaluate an explicit --model-path without cache mode."
            )
        model_path = training_run.model_path
        print(
            f"[eval-runner] latest saved training: {training_run.label} | "
            f"run={training_run.run_dir.name} | model={model_path}",
            flush=True,
        )

    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model does not exist: {model_path}")

    cache_metrics_path: Path | None = None
    cache_manifest_path: Path | None = None
    evaluation_request: dict[str, Any] | None = None
    evaluation_fingerprint: str | None = None
    if args.use_latest_training:
        cbf_config = (
            {
                "lambda_filter": float(namespace["CBF_FILTER_REWARD_LAMBDA"]),
                "k0": float(namespace["CBF_K0"]),
                "k1": float(namespace["CBF_K1"]),
                "eps_side": float(namespace["CBF_EPS_SIDE"]),
            }
            if needs_cbf
            else {}
        )
        evaluation_request = build_evaluation_request(
            variant=args.variant,
            model_path=model_path,
            training_run=training_run,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            traffic_model=traffic_model,
            env_config=namespace["ENV_CONFIG"],
            cbf_config=cbf_config,
            evaluator_code_sha256=sha256_file(Path(__file__).resolve()),
            notebook_code_sha256=notebook_code_sha256(notebook_path),
        )
        evaluation_fingerprint = stable_json_digest(evaluation_request)
        cache_metrics_path, cache_manifest_path = evaluation_cache_paths(
            artifact_dir=Path(namespace["ARTIFACT_DIR"]),
            model_path=model_path,
            training_run=training_run,
            evaluation_fingerprint=evaluation_fingerprint,
        )
        cached = None if args.force_reevaluate else load_matching_evaluation(
            metrics_path=cache_metrics_path,
            manifest_path=cache_manifest_path,
            evaluation_fingerprint=evaluation_fingerprint,
            model_sha256=str(evaluation_request["model_sha256"]),
        )
        if cached is not None:
            sync_metrics_to_requested_output(cache_metrics_path, output_path)
            metrics = pd.read_csv(cache_metrics_path)
            print(
                f"[eval-runner] matching evaluation cache hit for model {evaluation_request['model_sha256'][:12]}; "
                f"reused {cache_metrics_path}",
                flush=True,
            )
            print(f"[eval-runner] refreshed {output_path}", flush=True)
            print_ten_kpi_summary(metrics, label=f"{args.variant} (cached)")
            return 0
        print(
            "[eval-runner] no matching evaluation manifest; evaluating the latest saved training now",
            flush=True,
        )

    if args.variant == "ppo":
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["PPO"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating PPO", flush=True)
        metrics = namespace["evaluate_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            env_config=namespace["ENV_CONFIG"],
        )
    elif args.variant == "ddpg":
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["DDPG"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating DDPG", flush=True)
        metrics = namespace["evaluate_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            env_config=namespace["ENV_CONFIG"],
        )
    elif args.variant == "ddpg-cbf":
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["DDPG"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating DDPG-CBF", flush=True)
        metrics = namespace["evaluate_cbf_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=namespace["ENV_CONFIG"],
        )
    else:
        print(f"[eval-runner] loading {model_path}", flush=True)
        model = namespace["GuidedCBFDDPG"].load(str(model_path), device=args.device)
        print("[eval-runner] evaluating guided DDPG-CBF", flush=True)
        metrics = namespace["evaluate_cbf_policy_with_metrics"](
            model,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=True,
            lambda_filter=namespace["CBF_FILTER_REWARD_LAMBDA"],
            eps_side=namespace["CBF_EPS_SIDE"],
            env_config=namespace["ENV_CONFIG"],
        )

    if cache_metrics_path is not None and cache_manifest_path is not None:
        atomic_write_metrics(metrics, cache_metrics_path)
        write_evaluation_manifest(
            manifest_path=cache_manifest_path,
            metrics_path=cache_metrics_path,
            request=evaluation_request or {},
            evaluation_fingerprint=evaluation_fingerprint or "",
        )
        sync_metrics_to_requested_output(cache_metrics_path, output_path)
        print(f"[eval-runner] cached evaluation at {cache_metrics_path}", flush=True)
    else:
        atomic_write_metrics(metrics, output_path)
    print(f"[eval-runner] wrote {output_path}", flush=True)
    print_ten_kpi_summary(metrics, label=args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
