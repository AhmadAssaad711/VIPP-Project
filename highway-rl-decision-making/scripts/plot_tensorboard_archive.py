"""Create readable comparison plots from an archived TensorBoard export.

The input directory is expected to contain ``scalars_long.csv`` and, when
available, ``training_episode_kpis.csv`` as produced by the TensorBoard
archive exporter.  The script writes PNGs plus a small index README; it does
not modify the source CSVs or TensorBoard event files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANT_ORDER = ["B1", "B2_1", "B2_2", "B2_3", "B3_1", "B3_2", "B3_3"]
VARIANT_LABELS = {
    "B1": "B1 nominal",
    "B2_1": "B2.1 CBF reward",
    "B2_2": "B2.2 CBF reward + actor",
    "B2_3": "B2.3 CBF actor only",
    "B3_1": "B3.1 CBF diff reward",
    "B3_2": "B3.2 integrated actor",
    "B3_3": "B3.3 projected reward off",
}
COLORS = dict(zip(VARIANT_ORDER, plt.get_cmap("tab10").colors[: len(VARIANT_ORDER)]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/final_Results/tensorboard"),
        help="Directory containing the exported TensorBoard CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input-dir>/graphs.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.20,
        help="EMA smoothing factor for TensorBoard scalar figures (0 < alpha <= 1).",
    )
    return parser.parse_args()


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 15,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def _ordered_variants(data: pd.DataFrame) -> list[str]:
    present = set(data["variant_id"].dropna().astype(str))
    return [variant for variant in VARIANT_ORDER if variant in present]


def _plot_grid(
    data: pd.DataFrame,
    specs: list[tuple[str, str, str]],
    title: str,
    output_path: Path,
    dpi: int,
    ncols: int = 3,
    sharex: bool = True,
    smooth_alpha: float = 0.20,
) -> None:
    nrows = int(np.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.25 * ncols, 3.25 * nrows),
        sharex=sharex,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    variants = _ordered_variants(data)
    handles = []
    labels = []

    for index, (tag, panel_title, ylabel) in enumerate(specs):
        ax = axes_flat[index]
        subset = data[data["tag"] == tag]
        for variant_id in variants:
            series = subset[subset["variant_id"] == variant_id].sort_values("step")
            if series.empty:
                continue
            smoothed_value = series["value"].ewm(
                alpha=smooth_alpha, adjust=False, min_periods=1
            ).mean()
            line, = ax.plot(
                series["step"] / 1000.0,
                smoothed_value,
                color=COLORS[variant_id],
                linewidth=1.55,
                alpha=0.92,
                label=VARIANT_LABELS[variant_id],
            )
            if variant_id not in labels:
                handles.append(line)
                labels.append(variant_id)
        ax.set_title(panel_title)
        ax.set_xlabel("Training timestep (×1,000)")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.02)
        ax.grid(True, alpha=0.25)
        if subset.empty:
            ax.text(
                0.5,
                0.5,
                "No logged values",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="0.4",
            )

    for ax in axes_flat[len(specs) :]:
        ax.set_visible(False)

    fig.suptitle(title, y=1.02, fontweight="bold")
    fig.legend(
        handles,
        [VARIANT_LABELS[variant_id] for variant_id in labels],
        loc="lower center",
        ncol=min(4, max(1, len(labels))),
        bbox_to_anchor=(0.5, -0.015),
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.985))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _rolling_episode_plot(
    episodes: pd.DataFrame,
    specs: list[tuple[str, str, str]],
    title: str,
    output_path: Path,
    dpi: int,
    window: int = 25,
) -> None:
    ncols = 3
    nrows = int(np.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.25 * ncols, 3.25 * nrows),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    variants = _ordered_variants(episodes)
    handles = []
    labels = []

    for index, (column, panel_title, ylabel) in enumerate(specs):
        ax = axes_flat[index]
        for variant_id in variants:
            series = episodes[episodes["variant_id"] == variant_id].sort_values(
                "episode_index"
            )
            if column not in series or series.empty:
                continue
            rolling = series[column].rolling(window=window, min_periods=1).mean()
            line, = ax.plot(
                series["episode_index"],
                rolling,
                color=COLORS[variant_id],
                linewidth=1.55,
                label=VARIANT_LABELS[variant_id],
            )
            if variant_id not in labels:
                handles.append(line)
                labels.append(variant_id)
        ax.set_title(f"{panel_title} ({window}-episode mean)")
        ax.set_xlabel("Training episode")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.02)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(specs) :]:
        ax.set_visible(False)
    fig.suptitle(title, y=1.02, fontweight="bold")
    fig.legend(
        handles,
        [VARIANT_LABELS[variant_id] for variant_id in labels],
        loc="lower center",
        ncol=min(4, max(1, len(labels))),
        bbox_to_anchor=(0.5, -0.015),
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.985))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _time_binned_episode_plot(
    episodes: pd.DataFrame,
    specs: list[tuple[str, str, str]],
    title: str,
    output_path: Path,
    csv_path: Path,
    dpi: int,
    bin_steps: int = 2500,
    horizon_steps: int = 50000,
    smooth_alpha: float = 0.20,
) -> None:
    """Plot episode KPIs on a common training-time axis.

    Each bin contains completed episodes whose end timestep falls in that
    interval.  The line is the bin median and the translucent band is the
    within-bin interquartile range.  This avoids treating an episode as a
    fixed amount of training for policies whose episode lengths differ.
    """
    if bin_steps <= 0:
        raise ValueError("bin_steps must be positive")

    work = episodes.copy()
    work["global_timestep"] = pd.to_numeric(work["global_timestep"], errors="coerce")
    work = work.dropna(subset=["global_timestep"])
    work["time_bin"] = ((work["global_timestep"].clip(lower=1) - 1) // bin_steps).astype(int)
    n_bins = int(np.ceil(horizon_steps / bin_steps))
    work["time_bin"] = work["time_bin"].clip(lower=0, upper=max(0, n_bins - 1))

    metric_columns = [column for column, _, _ in specs if column in work.columns]
    records: list[dict[str, float | int | str]] = []
    for (variant_id, time_bin), group in work.groupby(["variant_id", "time_bin"], sort=True):
        record: dict[str, float | int | str] = {
            "variant_id": variant_id,
            "time_bin": int(time_bin),
            "timestep_start": int(time_bin) * bin_steps + 1,
            "timestep_end": min((int(time_bin) + 1) * bin_steps, horizon_steps),
            "timestep_center": (int(time_bin) + 0.5) * bin_steps,
            "completed_episodes": int(len(group)),
        }
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            record[f"{column}_median"] = float(values.median())
            record[f"{column}_q25"] = float(values.quantile(0.25))
            record[f"{column}_q75"] = float(values.quantile(0.75))
        records.append(record)
    binned = pd.DataFrame(records)
    binned.to_csv(csv_path, index=False)

    ncols = 3
    nrows = int(np.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.25 * ncols, 3.25 * nrows),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    variants = _ordered_variants(episodes)
    handles = []
    labels = []

    for index, (column, panel_title, ylabel) in enumerate(specs):
        ax = axes_flat[index]
        for variant_id in variants:
            subset = binned[binned["variant_id"] == variant_id].sort_values("time_bin")
            median_column = f"{column}_median"
            if subset.empty or median_column not in subset.columns:
                continue
            valid = subset.dropna(subset=[median_column])
            if valid.empty:
                continue
            x = valid["timestep_center"] / 1000.0
            median = valid[median_column].ewm(
                alpha=smooth_alpha, adjust=False, min_periods=1
            ).mean()
            q25 = valid.get(f"{column}_q25", median).ewm(
                alpha=smooth_alpha, adjust=False, min_periods=1
            ).mean()
            q75 = valid.get(f"{column}_q75", median).ewm(
                alpha=smooth_alpha, adjust=False, min_periods=1
            ).mean()
            line, = ax.plot(
                x,
                median,
                color=COLORS[variant_id],
                linewidth=1.7,
                label=VARIANT_LABELS[variant_id],
            )
            ax.fill_between(
                x.to_numpy(),
                q25.to_numpy(),
                q75.to_numpy(),
                color=COLORS[variant_id],
                alpha=0.12,
                linewidth=0,
            )
            if variant_id not in labels:
                handles.append(line)
                labels.append(variant_id)
        ax.set_title(panel_title)
        ax.set_xlabel("Training timestep (×1,000)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, horizon_steps / 1000.0)
        ax.margins(x=0.02)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(specs) :]:
        ax.set_visible(False)
    fig.suptitle(title, y=1.02, fontweight="bold")
    fig.legend(
        handles,
        [VARIANT_LABELS[variant_id] for variant_id in labels],
        loc="lower center",
        ncol=min(4, max(1, len(labels))),
        bbox_to_anchor=(0.5, -0.015),
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.985))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _write_readme(
    output_dir: Path,
    data: pd.DataFrame,
    episodes: pd.DataFrame | None,
    smooth_alpha: float,
) -> None:
    variant_text = ", ".join(
        f"{variant} ({VARIANT_LABELS[variant]})" for variant in _ordered_variants(data)
    )
    episode_note = (
        "The episode KPI figure uses the archived per-episode CSV and a 25-episode rolling mean."
        if episodes is not None and not episodes.empty
        else "The per-episode KPI CSV was not available, so no episode-level figure was generated."
    )
    readme = f"""# TensorBoard training graphs

Generated from `../scalars_long.csv` for: {variant_text}.

The x-axis in TensorBoard scalar figures is training timestep in thousands.
The TensorBoard figures use an exponential moving average (EMA) with
alpha={smooth_alpha:.2f}; the source CSVs retain the raw logged values.
Missing series are left blank because a tag was not
logged for that variant. The CBF figures therefore only show diagnostics for
variants that actually emitted those tags.

## Figures

- [RL rollout learning curves](rl_rollout_learning_curves.png): rewards, episode lengths, return per timestep, and FPS.
- [PPO optimization](ppo_optimization.png): KL, clipping, entropy, losses, explained variance, standard deviation, and optimizer settings.
- [Rollout safety and behavior KPIs](rollout_kpis.png): distance, collision counters/rates, resets, action saturation, and raw-action clipping.
- [CBF training diagnostics](cbf_training_diagnostics.png): CBF correction/loss/infeasibility and actor-gradient diagnostics.
- [Per-episode KPI trends](training_episode_kpis.png): archived episode-level KPIs with a 25-episode rolling mean.
- [Time-aligned per-episode KPIs](training_episode_kpis_aligned.png): episode KPIs aligned by global training timestep, with 2,500-step-bin medians, IQR bands, and the same EMA smoothing.

{episode_note}

The numerical source files remain in the parent directory, including
`rl_training_scalars.csv`, `kpi_specific_scalars.csv`, and
`cbf_training_gradients.csv`. The time-aligned post-processing table is
`training_episode_kpis_time_binned.csv`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "graphs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scalar_path = input_dir / "scalars_long.csv"
    if not scalar_path.exists():
        raise FileNotFoundError(f"Missing TensorBoard scalar export: {scalar_path}")

    data = pd.read_csv(scalar_path)
    data["variant_id"] = data["variant_id"].astype(str)
    data["tag"] = data["tag"].astype(str)

    _style()
    _plot_grid(
        data,
        [
            ("rollout/ep_rew_mean", "Mean rollout reward", "Reward"),
            ("rollout/ep_len_mean", "Mean rollout episode length", "Timesteps"),
            ("rollout/return_per_timestep", "Return per timestep", "Return / timestep"),
            ("rollout/episode_return", "Completed episode return", "Return"),
            ("rollout/episode_length", "Completed episode length", "Timesteps"),
            ("time/fps", "Training throughput", "Environment steps / second"),
        ],
        "RL rollout learning curves",
        output_dir / "rl_rollout_learning_curves.png",
        args.dpi,
        smooth_alpha=args.smooth_alpha,
    )
    _plot_grid(
        data,
        [
            ("train/approx_kl", "Approximate KL", "KL"),
            ("train/clip_fraction", "PPO clip fraction", "Fraction"),
            ("train/entropy_loss", "Entropy loss", "Loss"),
            ("train/explained_variance", "Value explained variance", "Explained variance"),
            ("train/policy_gradient_loss", "Policy-gradient loss", "Loss"),
            ("train/value_loss", "Value loss", "Loss"),
            ("train/loss", "Total training loss", "Loss"),
            ("train/std", "Policy action standard deviation", "Std. dev."),
            ("train/learning_rate", "Learning rate", "Learning rate"),
        ],
        "PPO optimization diagnostics",
        output_dir / "ppo_optimization.png",
        args.dpi,
        smooth_alpha=args.smooth_alpha,
    )
    _plot_grid(
        data,
        [
            ("rollout/distance_m", "Rollout distance", "Distance (m)"),
            ("rollout/collisions_per_km", "Collisions per kilometer", "Collisions / km"),
            ("rollout/distinct_collision_events", "Distinct collision events", "Events"),
            ("rollout/collision_active_timestep", "Collision-active timesteps", "Timesteps"),
            ("rollout/action_saturation_mean", "Action saturation", "Fraction"),
            (
                "rollout/actor_raw_action_clip_rate_cumulative",
                "Cumulative raw-action clipping",
                "Fraction",
            ),
            ("rollout/reset_calls_total", "Cumulative reset calls", "Reset calls"),
        ],
        "Rollout safety and behavior KPIs",
        output_dir / "rollout_kpis.png",
        args.dpi,
        smooth_alpha=args.smooth_alpha,
    )
    _plot_grid(
        data,
        [
            ("train/cbf_mean_correction", "Mean CBF correction", "Correction magnitude"),
            ("train/cbf_mean_infeasible_rate", "Mean CBF infeasible rate", "Fraction"),
            ("train/cbf_mean_loss", "Mean CBF loss", "Loss"),
            ("train/cbf_sample_correction", "Sample CBF correction", "Correction magnitude"),
            ("train/cbf_sample_infeasible_rate", "Sample CBF infeasible rate", "Fraction"),
            ("train/cbf_sample_loss", "Sample CBF loss", "Loss"),
            ("train/cbf_lambda_mean", "Mean CBF lambda", "Lambda"),
            ("train/cbf_lambda_sample", "Sample CBF lambda", "Lambda"),
            ("train/cbf_detached_actor_lambda", "Detached actor lambda", "Lambda"),
            ("train/actor_g_ppo_norm", "PPO actor-gradient norm", "Norm"),
            ("train/actor_g_cbf_norm", "CBF actor-gradient norm", "Norm"),
            ("train/actor_g_cbf_to_g_ppo_ratio", "CBF/PPO gradient-norm ratio", "Ratio"),
            ("train/actor_g_ppo_g_cbf_cosine", "PPO/CBF gradient cosine", "Cosine similarity"),
        ],
        "CBF training diagnostics",
        output_dir / "cbf_training_diagnostics.png",
        args.dpi,
        ncols=3,
        smooth_alpha=args.smooth_alpha,
    )

    episode_path = input_dir / "training_episode_kpis.csv"
    episodes: pd.DataFrame | None = None
    if episode_path.exists():
        episodes = pd.read_csv(episode_path)
        episodes["variant_id"] = episodes["variant_id"].astype(str)
        _rolling_episode_plot(
            episodes,
            [
                ("episode_return", "Episode return", "Return"),
                ("episode_length", "Episode length", "Timesteps"),
                ("total_distance_m", "Episode distance", "Distance (m)"),
                ("ego_collision_active_timesteps", "Collision-active timesteps", "Timesteps"),
                ("action_saturation_mean", "Action saturation", "Fraction"),
                ("ego_collisions_per_km", "Ego collisions per kilometer", "Collisions / km"),
            ],
            "Per-episode KPI trends during training",
            output_dir / "training_episode_kpis.png",
            args.dpi,
        )
        _time_binned_episode_plot(
            episodes,
            [
                ("episode_return", "Episode return", "Return"),
                ("episode_length", "Episode length", "Timesteps"),
                ("total_distance_m", "Episode distance", "Distance (m)"),
                ("ego_collision_active_timesteps", "Collision-active timesteps", "Timesteps"),
                ("action_saturation_mean", "Action saturation", "Fraction"),
                ("ego_collisions_per_km", "Ego collisions per kilometer", "Collisions / km"),
            ],
            "Time-aligned per-episode KPI trends",
            output_dir / "training_episode_kpis_aligned.png",
            output_dir / "training_episode_kpis_time_binned.csv",
            args.dpi,
            smooth_alpha=args.smooth_alpha,
        )

    _write_readme(output_dir, data, episodes, args.smooth_alpha)
    generated = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    print(f"Generated {len(generated)} files in {output_dir}")
    for filename in generated:
        print(filename)


if __name__ == "__main__":
    main()
