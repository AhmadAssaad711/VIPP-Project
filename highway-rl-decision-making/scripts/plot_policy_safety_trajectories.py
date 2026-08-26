"""Plot raw-policy safety traces from identical initial traffic scenarios.

The trajectories intentionally execute each actor's deterministic *raw neural
mean* with the external CBF action filter disabled.  This reveals learned
policy behavior; it is not a shielded-deployment evaluation.  The simulator's
ordinary traffic guard remains configured exactly as in the B3.2 run.

For every policy state, the script logs two continuous-time CBF quantities:

* ``min_h``: minimum geometric barrier value over neighbors and road bounds.
  ``min_h >= 0`` is the state-safe side of the barrier.
* ``raw_hocbf_margin``: minimum action-dependent CBF margin
  ``b_i(s) - A_i(s) u_raw`` across non-box HOCBF rows.  A negative value
  means the raw actor command violates at least one instantaneous HOCBF
  inequality at the current state.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

for _thread_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(_thread_key, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_cbf_filter_ablation as protocol
import run_ppo_cbf_progression as progression
from ppo_reward_safety import install_cbf_violation_reward


PAIRS: tuple[dict[str, str], ...] = (
    {
        "pair_id": "reward_only",
        "non_differentiable": "B2_1",
        "differentiable": "B3_1",
        "label": "Reward feedback only",
    },
    {
        "pair_id": "reward_actor_feedback",
        "non_differentiable": "B2_2",
        "differentiable": "B3_2",
        "label": "Reward + actor feedback",
    },
    {
        "pair_id": "actor_only",
        "non_differentiable": "B2_3",
        "differentiable": "B3_3",
        "label": "Actor feedback only",
    },
)
VARIANT_LABELS = {
    "B2_1": "B2.1 non-diff",
    "B2_2": "B2.2 detached",
    "B2_3": "B2.3 detached",
    "B3_1": "B3.1 differentiable",
    "B3_2": "B3.2 differentiable",
    "B3_3": "B3.3 differentiable",
}
COLORS = {"non_differentiable": "#e45756", "differentiable": "#54a24b"}
VARIANT_IDS = tuple(
    dict.fromkeys(
        variant
        for pair in PAIRS
        for variant in (pair["non_differentiable"], pair["differentiable"])
    )
)
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "final_Results" / "policy_analysis" / "traces"


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _run_dir(variant_id: str) -> Path:
    return PROJECT_ROOT / "artifacts" / "final_Results" / "models" / variant_id / "seed_307"


def load_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for variant_id in VARIANT_IDS:
        run_dir = _run_dir(variant_id)
        config_path = run_dir / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_path = run_dir / "model_final.zip"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        specs[variant_id] = {
            "variant_id": variant_id,
            "model_path": model_path,
            "variant": str(config["variant"]),
            "config": config,
        }
    return specs


def _find_context_wrapper(env: Any) -> Any:
    current = env
    for _ in range(20):
        if hasattr(current, "current_constraint_system"):
            return current
        if not hasattr(current, "env"):
            break
        current = current.env
    raise RuntimeError("Could not find the CBF context wrapper")


def _raw_actor_mean(model: Any, observation: np.ndarray) -> np.ndarray:
    """Return the raw neural mean, then apply only physical box bounds."""

    policy = model.policy
    obs_tensor, _vectorized = policy.obs_to_tensor(observation)
    with th.no_grad():
        if hasattr(policy, "_latents"):
            latent_pi, _latent_vf = policy._latents(obs_tensor)
        else:
            features = policy.extract_features(obs_tensor)
            if bool(getattr(policy, "share_features_extractor", True)):
                latent_pi, _latent_vf = policy.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = policy.mlp_extractor.forward_actor(pi_features)
                _latent_vf = policy.mlp_extractor.forward_critic(vf_features)
        action = policy.action_net(latent_pi)
    raw = np.asarray(action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)[:2]
    low = np.asarray(model.action_space.low, dtype=np.float32).reshape(-1)[:2]
    high = np.asarray(model.action_space.high, dtype=np.float32).reshape(-1)[:2]
    return np.clip(raw, low, high).astype(np.float32)


def _raw_hocbf_margin(system: dict[str, Any], action: np.ndarray) -> float:
    """Evaluate b - A u for non-box CBF rows; safe if the value is nonnegative."""

    rows = np.asarray(system.get("cbf_rows", ()), dtype=float)
    bounds = np.asarray(system.get("cbf_bounds", ()), dtype=float).reshape(-1)
    if rows.size == 0 or bounds.size == 0:
        return np.inf
    rows = rows.reshape(-1, 2)
    margin = bounds - rows @ np.asarray(action, dtype=float).reshape(-1)[:2]
    return float(np.min(margin))


def _make_raw_trace_env(namespace: dict[str, Any], config: dict[str, Any]) -> Any:
    """Expose CBF state/context while disabling policy-rate and physics-rate filtering."""

    env_config = copy.deepcopy(config["env_config"])
    env_config["cbf_substep_filtering"] = False
    env_config["cbf_require_initial_safe_set"] = True
    return progression.make_ppo_cbf_env(
        namespace,
        env_config=env_config,
        reward_config=copy.deepcopy(config["reward_config"]),
        project_inputs=False,
        lambda_delta=0.0,
        lambda_intervention=0.0,
        correction_epsilon=0.03,
        action_rate_penalty_lambda=0.0,
    )


def run_trace(
    env: Any,
    model: Any,
    *,
    variant_id: str,
    scenario_seed: int,
    max_steps: int,
    policy_frequency_hz: float,
) -> list[dict[str, Any]]:
    """Run one raw actor from one deterministic initial traffic scenario."""

    context_wrapper = _find_context_wrapper(env)
    observation, _info = env.reset(seed=int(scenario_seed))
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    output: list[dict[str, Any]] = []
    terminal_outcome = "horizon"
    for step in range(int(max_steps)):
        system = context_wrapper.current_constraint_system()
        action = _raw_actor_mean(model, observation)
        min_h = _finite(system.get("min_h"))
        center_distance = _finite(system.get("min_center_distance"))
        required_distance = _finite(system.get("min_required_distance"))
        raw_margin = _raw_hocbf_margin(system, action)
        output.append(
            {
                "variant_id": variant_id,
                "variant_label": VARIANT_LABELS[variant_id],
                "scenario_seed": int(scenario_seed),
                "policy_step": int(step),
                "time_s": float(step) / float(policy_frequency_hz),
                "min_h": min_h,
                "min_center_distance_m": center_distance,
                "min_required_distance_m": required_distance,
                "geometric_clearance_m": center_distance - required_distance
                if np.isfinite(center_distance) and np.isfinite(required_distance)
                else np.nan,
                "min_boundary_h": _finite(system.get("min_boundary_h")),
                "raw_ax": float(action[0]),
                "raw_ay": float(action[1]),
                "raw_hocbf_margin": raw_margin,
                "raw_hocbf_violated": bool(np.isfinite(raw_margin) and raw_margin < 0.0),
                "terminated": False,
                "truncated": False,
                "collision": False,
                "outcome": "running",
            }
        )
        observation, _reward, terminated, truncated, info = env.step(action)
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        collision = bool(
            info.get("ego_collision", False)
            or int(info.get("ego_collision_events", 0)) > 0
        )
        if bool(terminated) or bool(truncated):
            terminal_outcome = "collision" if collision else ("terminated" if terminated else "truncated")
            output[-1].update(
                {
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "collision": collision,
                    "outcome": terminal_outcome,
                }
            )
            break
    if output:
        output[-1]["outcome"] = terminal_outcome
    return output


def summarize_episodes(trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant_id, scenario_seed), data in trace.groupby(["variant_id", "scenario_seed"], sort=False):
        final = data.iloc[-1]
        finite_margin = pd.to_numeric(data["raw_hocbf_margin"], errors="coerce")
        finite_margin = finite_margin[np.isfinite(finite_margin)]
        rows.append(
            {
                "variant_id": variant_id,
                "variant_label": str(final["variant_label"]),
                "scenario_seed": int(scenario_seed),
                "policy_steps": int(len(data)),
                "final_time_s": float(final["time_s"]),
                "episode_min_h": float(pd.to_numeric(data["min_h"], errors="coerce").min()),
                "episode_min_clearance_m": float(
                    pd.to_numeric(data["geometric_clearance_m"], errors="coerce").min()
                ),
                "episode_min_raw_hocbf_margin": float(finite_margin.min())
                if len(finite_margin)
                else np.nan,
                "raw_hocbf_violation_rate": float(data["raw_hocbf_violated"].mean()),
                "collision": bool(final["collision"]),
                "outcome": str(final["outcome"]),
            }
        )
    return pd.DataFrame(rows)


def _pair_line_axis(
    axis: Any,
    trace: pd.DataFrame,
    *,
    scenario_seed: int,
    pair: dict[str, str],
    metric: str,
    ylabel: str,
) -> None:
    for role, variant_id in (
        ("non_differentiable", pair["non_differentiable"]),
        ("differentiable", pair["differentiable"]),
    ):
        data = trace.loc[
            (trace["scenario_seed"] == int(scenario_seed))
            & (trace["variant_id"] == variant_id)
        ].sort_values("policy_step")
        if data.empty:
            continue
        axis.plot(
            data["time_s"],
            data[metric],
            color=COLORS[role],
            linewidth=2.0,
            label=VARIANT_LABELS[variant_id],
        )
        terminal = data.loc[data["outcome"] != "running"]
        if not terminal.empty:
            row = terminal.iloc[-1]
            axis.scatter(
                [row["time_s"]], [row[metric]], color=COLORS[role], s=28, zorder=4)
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axis.set_title(f"{pair['label']} | seed {scenario_seed}", fontsize=10)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def plot_trajectories(trace: pd.DataFrame, *, metric: str, ylabel: str, output_path: Path) -> None:
    seeds = sorted(int(value) for value in trace["scenario_seed"].unique())
    figure, axes = plt.subplots(
        len(seeds), len(PAIRS), figsize=(17, max(4.0, 3.8 * len(seeds))), constrained_layout=True
    )
    axes_array = np.asarray(axes, dtype=object).reshape(len(seeds), len(PAIRS))
    for row, scenario_seed in enumerate(seeds):
        for column, pair in enumerate(PAIRS):
            axis = axes_array[row, column]
            _pair_line_axis(
                axis,
                trace,
                scenario_seed=scenario_seed,
                pair=pair,
                metric=metric,
                ylabel=ylabel,
            )
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    figure.suptitle(
        "Same initial scenario, raw actor executed with external CBF disabled",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _paired_summary_axis(
    axis: Any,
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    positive_favors_differentiable: bool,
) -> None:
    group_centers = np.arange(len(PAIRS), dtype=float)
    offsets = {"non_differentiable": -0.18, "differentiable": 0.18}
    for index, pair in enumerate(PAIRS):
        values: dict[str, pd.DataFrame] = {}
        for role, variant_id in (
            ("non_differentiable", pair["non_differentiable"]),
            ("differentiable", pair["differentiable"]),
        ):
            values[role] = summary.loc[summary["variant_id"] == variant_id].sort_values("scenario_seed")
            axis.scatter(
                np.full(len(values[role]), group_centers[index] + offsets[role]),
                values[role][metric],
                color=COLORS[role],
                s=40,
                alpha=0.9,
                label=("non-diff" if index == 0 and role == "non_differentiable" else "differentiable" if index == 0 else None),
                zorder=3,
            )
        left = values["non_differentiable"].set_index("scenario_seed")
        right = values["differentiable"].set_index("scenario_seed")
        for seed in sorted(set(left.index) & set(right.index)):
            axis.plot(
                [group_centers[index] + offsets["non_differentiable"], group_centers[index] + offsets["differentiable"]],
                [left.loc[seed, metric], right.loc[seed, metric]],
                color="#777777",
                alpha=0.5,
                linewidth=1.0,
                zorder=2,
            )
        non_mean = float(values["non_differentiable"][metric].mean())
        diff_mean = float(values["differentiable"][metric].mean())
        axis.plot(
            [group_centers[index] + offsets["non_differentiable"], group_centers[index] + offsets["differentiable"]],
            [non_mean, diff_mean],
            color="#111111",
            linewidth=2.4,
            zorder=4,
        )
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axis.set_xticks(group_centers, [pair["label"] for pair in PAIRS], rotation=18, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(
        "Higher favors differentiable" if positive_favors_differentiable else "Lower favors differentiable"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)


def plot_episode_summary(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    _paired_summary_axis(
        axes[0],
        summary,
        metric="episode_min_h",
        ylabel="Episode minimum h",
        positive_favors_differentiable=True,
    )
    _paired_summary_axis(
        axes[1],
        summary,
        metric="episode_min_clearance_m",
        ylabel="Episode minimum clearance (m)",
        positive_favors_differentiable=True,
    )
    _paired_summary_axis(
        axes[2],
        summary,
        metric="raw_hocbf_violation_rate",
        ylabel="Raw HOCBF violation fraction",
        positive_favors_differentiable=False,
    )
    figure.suptitle("Episode-level raw-policy safety summaries (paired seeds)", fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_readme(output_path: Path, *, trace: pd.DataFrame, summary: pd.DataFrame) -> None:
    seeds = sorted(int(value) for value in trace["scenario_seed"].unique())
    text = f"""# Raw-policy CBF safety trajectories

These plots compare matching B2 non-differentiable and B3 differentiable policies from the same initial traffic scenario seeds: {", ".join(map(str, seeds))}. After reset, trajectories diverge naturally because each ego policy chooses different actions and the social traffic evolves in response.

## What is plotted

- `min_h.png`: minimum state barrier h across neighbors and road boundaries. h >= 0 is the geometric safe side of the barrier.
- `margin.png`: minimum raw-action HOCBF margin b(s) - A(s) u_raw across the non-box constraint rows. A value below zero means the raw command violates the instantaneous CBF inequality.
- `summary.png`: paired episode minima and raw HOCBF violation fractions.

The external CBF is deliberately disabled at both policy and physics rate. These are raw learned-policy traces, not shielded deployment traces or KPI evaluation episodes. The simulator's ordinary social traffic guard is preserved.

`trace.csv` contains every policy-step state/action safety measurement; `summary.csv` contains one row per policy and scenario.
"""
    output_path.write_text(text, encoding="ascii", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--scenario-seeds",
        type=int,
        nargs="+",
        default=(1_300_000, 1_300_001, 1_300_002),
        help="Identical initial simulator seeds used for every policy",
    )
    parser.add_argument("--max-policy-steps", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(args.max_policy_steps) <= 0:
        raise ValueError("--max-policy-steps must be positive")
    scenario_seeds = tuple(dict.fromkeys(int(seed) for seed in args.scenario_seeds))
    if not scenario_seeds:
        raise ValueError("At least one scenario seed is required")

    protocol.set_stable_native_defaults()
    try:
        th.set_num_threads(1)
        th.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print("[safety-traces] loading notebook environment", flush=True)
    namespace = protocol.bootstrap_notebook_namespace(PROJECT_ROOT)
    protocol.exec_required_notebook_cells(
        PROJECT_ROOT / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    install_cbf_violation_reward(namespace)
    namespace["DEVICE"] = "cpu"
    specs = load_specs()
    print("[safety-traces] loading policies", flush=True)
    models = {
        variant_id: progression.load_model(spec["variant"], spec["model_path"], "cpu")
        for variant_id, spec in specs.items()
    }
    for model in models.values():
        model.policy.set_training_mode(False)

    # All B2/B3 variants share the 100 Hz/10 Hz training environment and CBF gains.
    trace_config = specs["B3_2"]["config"]
    policy_frequency_hz = float(trace_config["env_config"]["policy_frequency"])
    records: list[dict[str, Any]] = []
    for variant_id in VARIANT_IDS:
        print(f"[safety-traces] raw trace {variant_id}", flush=True)
        env = _make_raw_trace_env(namespace, trace_config)
        try:
            for scenario_seed in scenario_seeds:
                records.extend(
                    run_trace(
                        env,
                        models[variant_id],
                        variant_id=variant_id,
                        scenario_seed=scenario_seed,
                        max_steps=int(args.max_policy_steps),
                        policy_frequency_hz=policy_frequency_hz,
                    )
                )
        finally:
            env.close()
    trace = pd.DataFrame(records)
    if trace.empty:
        raise RuntimeError("No safety trace rows were collected")
    summary = summarize_episodes(trace)
    trace.to_csv(output_dir / "trace.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    plot_trajectories(
        trace,
        metric="min_h",
        ylabel="Minimum barrier h (safe >= 0)",
        output_path=output_dir / "min_h.png",
    )
    plot_trajectories(
        trace,
        metric="raw_hocbf_margin",
        ylabel="Raw HOCBF margin (safe >= 0)",
        output_path=output_dir / "margin.png",
    )
    plot_episode_summary(summary, output_dir / "summary.png")
    write_readme(output_dir / "README.md", trace=trace, summary=summary)
    print(
        f"[safety-traces] complete rows={len(trace)} episodes={len(summary)} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
