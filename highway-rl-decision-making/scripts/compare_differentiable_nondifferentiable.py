"""Directly compare the saved differentiable and non-differentiable PPO methods.

This report consumes the existing common-state policy probe.  It does not
train models, advance the simulator, or rerun any evaluation episode.

The three matched pairs retain the same reward-feedback condition:

* B2.1 (non-differentiable reward-only) vs B3.1 (differentiable projection)
* B2.2 (detached hard-target + reward) vs B3.2 (differentiable + mean loss)
* B2.3 (detached hard-target only) vs B3.3 (differentiable actor only)

These are method-level comparisons.  The differentiable method necessarily
changes the actor's projection/distribution path, so they should not be
presented as the effect of one scalar hyperparameter alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "artifacts" / "final_Results" / "policy_analysis"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "dvnd"

PAIRS: tuple[dict[str, str], ...] = (
    {
        "pair_id": "reward_only",
        "non_differentiable": "B2_1",
        "differentiable": "B3_1",
        "label": "Reward feedback only",
        "method_change": "external hard CBF -> PPO gradient through differentiable projected mean",
    },
    {
        "pair_id": "reward_actor_feedback",
        "non_differentiable": "B2_2",
        "differentiable": "B3_2",
        "label": "Reward + actor feedback",
        "method_change": "stopped-gradient hard target -> differentiable projection + mean-alignment loss",
    },
    {
        "pair_id": "actor_only",
        "non_differentiable": "B2_3",
        "differentiable": "B3_3",
        "label": "Actor feedback only",
        "method_change": "stopped-gradient hard target -> differentiable projection + mean-alignment loss",
    },
)

ACTION_METRICS = {
    "projection_correction_physical": "External correction (m/s^2)",
    "projection_correction_normalized": "Normalized external correction",
    "raw_feasible": "Raw CBF-feasible rate",
    "external_intervention": "External intervention rate",
    "raw_max_constraint_violation": "Raw max constraint violation",
}
TRAINING_METRICS = (
    "tail10_ep_rew_mean",
    "tail10_ep_len_mean",
    "tail10_return_per_timestep",
    "tail10_cbf_mean_correction",
    "tail10_cbf_mean_loss",
    "mean_g_cbf_to_g_ppo_ratio",
)


def _bootstrap_paired_difference(
    nondiff: np.ndarray, diff: np.ndarray, *, samples: int = 4000
) -> tuple[float, float, float]:
    """Return differentiable-minus-non-differentiable paired mean and CI."""

    delta = np.asarray(diff, dtype=float) - np.asarray(nondiff, dtype=float)
    delta = delta[np.isfinite(delta)]
    if not len(delta):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20260826)
    resampled_means = np.empty(int(samples), dtype=float)
    for index in range(int(samples)):
        indices = rng.integers(0, len(delta), size=len(delta))
        resampled_means[index] = float(delta[indices].mean())
    return (
        float(delta.mean()),
        float(np.quantile(resampled_means, 0.025)),
        float(np.quantile(resampled_means, 0.975)),
    )


def _shared_pair_actions(
    actions: pd.DataFrame, non_diff: str, diff: str
) -> pd.DataFrame:
    """Return exactly shared states whose CBF set is feasible."""

    usable = actions.loc[
        actions["action_set_feasible"].astype(bool)
        & actions["variant_id"].isin((non_diff, diff))
    ].copy()
    left = usable.loc[usable["variant_id"] == non_diff].set_index("state_id")
    right = usable.loc[usable["variant_id"] == diff].set_index("state_id")
    result = left.join(right, how="inner", lsuffix="_non", rsuffix="_diff")
    if result.empty:
        raise RuntimeError(f"No shared feasible states for {non_diff} vs {diff}")
    return result


def build_action_summary(actions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build one paired row per action metric for all gradient-path pairs."""

    rows: list[dict[str, Any]] = []
    shared_by_pair: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        non_diff = pair["non_differentiable"]
        diff = pair["differentiable"]
        shared = _shared_pair_actions(actions, non_diff, diff)
        shared_by_pair[pair["pair_id"]] = shared
        for metric, metric_label in ACTION_METRICS.items():
            non_values = shared[f"{metric}_non"].astype(float).to_numpy()
            diff_values = shared[f"{metric}_diff"].astype(float).to_numpy()
            delta, ci_low, ci_high = _bootstrap_paired_difference(non_values, diff_values)
            relative = (
                100.0 * delta / float(np.mean(non_values))
                if np.isfinite(np.mean(non_values)) and abs(float(np.mean(non_values))) > 1e-12
                else np.nan
            )
            rows.append(
                {
                    **pair,
                    "metric": metric,
                    "metric_label": metric_label,
                    "n_paired_feasible_states": int(len(shared)),
                    "non_differentiable_mean": float(np.mean(non_values)),
                    "differentiable_mean": float(np.mean(diff_values)),
                    "differentiable_minus_non_differentiable": delta,
                    "relative_change_percent": relative,
                    "bootstrap_95_ci_low": ci_low,
                    "bootstrap_95_ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows), shared_by_pair


def build_training_summary(training: pd.DataFrame) -> pd.DataFrame:
    """Join tail-training scalars to the same three method pairs."""

    indexed = training.set_index("variant_id")
    rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        non_diff = pair["non_differentiable"]
        diff = pair["differentiable"]
        row: dict[str, Any] = {**pair}
        for metric in TRAINING_METRICS:
            non_value = float(indexed.loc[non_diff, metric])
            diff_value = float(indexed.loc[diff, metric])
            row[f"non_differentiable_{metric}"] = non_value
            row[f"differentiable_{metric}"] = diff_value
            row[f"differentiable_minus_non_differentiable_{metric}"] = (
                diff_value - non_value
                if np.isfinite(non_value) and np.isfinite(diff_value)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_action_pairs(shared_by_pair: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Plot paired correction scatters and safety-rate bars for every pair."""

    figure, axes = plt.subplots(len(PAIRS), 2, figsize=(14, 14), constrained_layout=True)
    colors = {"non": "#e45756", "diff": "#54a24b"}
    for row_index, pair in enumerate(PAIRS):
        shared = shared_by_pair[pair["pair_id"]]
        left_axis, right_axis = axes[row_index]
        correction_non = shared["projection_correction_physical_non"].astype(float).to_numpy()
        correction_diff = shared["projection_correction_physical_diff"].astype(float).to_numpy()
        limit = max(float(np.nanmax(correction_non)), float(np.nanmax(correction_diff)), 1e-3) * 1.04
        left_axis.scatter(
            correction_non,
            correction_diff,
            color=colors["diff"],
            alpha=0.36,
            edgecolor="none",
            s=18,
        )
        left_axis.plot([0, limit], [0, limit], color="#333333", linestyle="--", linewidth=1.2)
        left_axis.set_xlim(0, limit)
        left_axis.set_ylim(0, limit)
        left_axis.set_aspect("equal", adjustable="box")
        left_axis.set_title(f"{pair['label']}: one dot = one identical state")
        left_axis.set_xlabel(f"{pair['non_differentiable']} non-diff correction (m/s^2)")
        left_axis.set_ylabel(f"{pair['differentiable']} diff correction (m/s^2)")
        left_axis.grid(alpha=0.25)

        indicators = ("raw_feasible", "external_intervention")
        non_rates = [100.0 * float(shared[f"{indicator}_non"].astype(float).mean()) for indicator in indicators]
        diff_rates = [100.0 * float(shared[f"{indicator}_diff"].astype(float).mean()) for indicator in indicators]
        x = np.arange(len(indicators))
        width = 0.35
        right_axis.bar(x - width / 2, non_rates, width, color=colors["non"], label="non-diff")
        right_axis.bar(x + width / 2, diff_rates, width, color=colors["diff"], label="differentiable")
        right_axis.set_xticks(x, ["raw\nCBF-feasible", "external\nintervention"])
        right_axis.set_ylim(0, 105)
        right_axis.set_ylabel("Shared feasible states (%)")
        right_axis.set_title(f"{pair['label']}: safety-response rates")
        right_axis.grid(axis="y", alpha=0.25)
        right_axis.legend(fontsize=8)
    figure.suptitle(
        "Differentiable vs non-differentiable CBF policy methods on common states",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}f}" if np.isfinite(number) else "N/A"


def write_readme(
    output_path: Path, action_summary: pd.DataFrame, training_summary: pd.DataFrame
) -> None:
    """State the comparison contract and concise findings next to the data."""

    paragraphs: list[str] = [
        "# Differentiable vs non-differentiable policy methods",
        "",
        "All comparisons use the same 87 probe states with feasible no-slack CBF sets. "
        "The policies were not stepped, trained, or evaluated here; their raw actor means "
        "were projected against the same CBF polytope.",
        "",
        "These are method-level comparisons. A differentiable projected policy changes the "
        "actor distribution and execution path, so it is not a one-scalar causal ablation. "
        "The clean scalar ablation remains B3.1 -> B3.2.",
        "",
        "## Shared-state action result",
        "",
    ]
    for pair in PAIRS:
        data = action_summary.loc[action_summary["pair_id"] == pair["pair_id"]]
        correction = data.loc[data["metric"] == "projection_correction_physical"].iloc[0]
        feasible = data.loc[data["metric"] == "raw_feasible"].iloc[0]
        intervention = data.loc[data["metric"] == "external_intervention"].iloc[0]
        paragraphs.extend(
            [
                f"### {pair['label']} ({pair['non_differentiable']} vs {pair['differentiable']})",
                "",
                f"- Method change: {pair['method_change']}.",
                f"- External correction: {_number(correction['non_differentiable_mean'])} -> "
                f"{_number(correction['differentiable_mean'])} m/s^2; differentiable minus "
                f"non-differentiable = {_number(correction['differentiable_minus_non_differentiable'])} "
                f"(95% paired bootstrap CI {_number(correction['bootstrap_95_ci_low'])} to "
                f"{_number(correction['bootstrap_95_ci_high'])}).",
                f"- Raw feasibility: {_number(100 * feasible['non_differentiable_mean'], 1)}% -> "
                f"{_number(100 * feasible['differentiable_mean'], 1)}%; intervention: "
                f"{_number(100 * intervention['non_differentiable_mean'], 1)}% -> "
                f"{_number(100 * intervention['differentiable_mean'], 1)}%.",
                "",
            ]
        )

    paragraphs.extend(
        [
            "## Training context",
            "",
            "The training table uses the final ten logged points. Episode return and episode "
            "length are not by themselves proof of safer raw actions; the common-state CBF "
            "alignment metrics above answer that separate question.",
            "",
            "## Files",
            "",
            "- `actions.csv`: paired raw-action/CBF alignment effects and confidence intervals.",
            "- `training.csv`: tail training metrics for the same pairs.",
            "- `plot.png`: paired response figure.",
        ]
    )
    output_path.write_text("\n".join(paragraphs) + "\n", encoding="ascii", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    actions_path = input_dir / "probe_actions.csv"
    training_path = input_dir / "training_effect_summary.csv"
    if not actions_path.is_file() or not training_path.is_file():
        raise FileNotFoundError(
            "Expected probe_actions.csv and training_effect_summary.csv under "
            f"{input_dir}; run analyze_final_results_policies.py first."
        )
    actions = pd.read_csv(actions_path)
    training = pd.read_csv(training_path)
    action_summary, shared_by_pair = build_action_summary(actions)
    training_pairs = build_training_summary(training)
    action_summary.to_csv(output_dir / "actions.csv", index=False)
    training_pairs.to_csv(output_dir / "training.csv", index=False)
    plot_action_pairs(shared_by_pair, output_dir / "plot.png")
    write_readme(output_dir / "README.md", action_summary, training_pairs)
    print(f"[diff-vs-nondiff] output: {output_dir}", flush=True)
    for pair in PAIRS:
        effect = action_summary.loc[
            (action_summary["pair_id"] == pair["pair_id"])
            & (action_summary["metric"] == "projection_correction_physical")
        ].iloc[0]
        print(
            f"[diff-vs-nondiff] {pair['non_differentiable']} -> {pair['differentiable']}: "
            f"correction delta={float(effect['differentiable_minus_non_differentiable']):.4f} m/s^2",
            flush=True,
        )


if __name__ == "__main__":
    main()
