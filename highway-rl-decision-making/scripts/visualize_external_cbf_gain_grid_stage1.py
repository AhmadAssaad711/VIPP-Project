"""Build comparison tables and heatmaps for the Stage 1 CBF gain sweep.

The Stage 1 grid is evaluated with the fixed 1M nominal PPO actor and
``k1 = c1 + c2``, ``k0 = c1 * c2``.  The older no-CBF/old-CBF comparison is
loaded from the paired post-training KPI export when it is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_GRID_DIR = (
    Path("artifacts")
    / "1MRun"
    / "nom"
    / "external_cbf_gain_grid_stage1"
)


def _as_float(value: object) -> float:
    return float(value) if value is not None and str(value) != "" else float("nan")


def _stage_row(row: pd.Series, *, label: str) -> dict[str, object]:
    return {
        "label": label,
        "protocol": "Stage 1 shared-seed grid (20 episodes)",
        "episodes": int(row["episodes_completed"]),
        "c1": _as_float(row["c1"]),
        "c2": _as_float(row["c2"]),
        "k1": _as_float(row["k1"]),
        "k0": _as_float(row["k0"]),
        "mean_return": _as_float(row["return_mean"]),
        "completion_rate": _as_float(row["distance_completion_rate_mean"]),
        "collision_events_per_km": _as_float(
            row["collision_events_per_km_pooled"]
        ),
        "collision_episode_rate": _as_float(row["collision_episode_rate"]),
        "intervention_rate": _as_float(row["event_intervention_rate_weighted"]),
        "qp_failure_rate": _as_float(row["qp_failure_rate_weighted"]),
        "mean_correction_norm": _as_float(row["mean_correction_norm_weighted"]),
        "mean_abs_speed_error": _as_float(
            row["mean_abs_speed_deviation_weighted"]
        ),
        "mean_lateral_error_m": _as_float(row["mean_lat_y_error_m_weighted"]),
        "mean_jerk_norm": _as_float(row["mean_jerk_norm_weighted"]),
        "mean_distance_m": _as_float(row["distance_mean_m"]),
    }


def _kpi_lookup(kpis: pd.DataFrame, *, external_cbf: str, mode: str) -> dict[str, float]:
    selected = kpis[
        (kpis["external_cbf"].astype(str) == external_cbf)
        & (kpis["mode"].astype(str) == mode)
    ]
    return {
        str(row["KPI"]): _as_float(row["Mean"])
        for _, row in selected.iterrows()
    }


def _legacy_row(
    kpis: pd.DataFrame,
    *,
    label: str,
    external_cbf: str,
    mode: str,
    k0: float | None,
    k1: float | None,
) -> dict[str, object]:
    values = _kpi_lookup(kpis, external_cbf=external_cbf, mode=mode)
    selected = kpis[
        (kpis["external_cbf"].astype(str) == external_cbf)
        & (kpis["mode"].astype(str) == mode)
    ]
    episodes = int(float(selected["episodes_per_mode"].iloc[0])) if not selected.empty else 0
    return {
        "label": label,
        "protocol": "Paired post-training KPI export (200 episodes)",
        "episodes": episodes,
        "c1": float("nan"),
        "c2": float("nan"),
        "k1": float(k1) if k1 is not None else float("nan"),
        "k0": float(k0) if k0 is not None else float("nan"),
        "mean_return": values.get("Episode return", float("nan")),
        "completion_rate": values.get("Distance-based completion rate", float("nan")),
        "collision_events_per_km": values.get("Ego collisions / km", float("nan")),
        "collision_episode_rate": float("nan"),
        "intervention_rate": values.get("Intervention rate", float("nan")),
        "qp_failure_rate": values.get("QP failure rate", float("nan")),
        "mean_correction_norm": values.get("Correction norm", float("nan")),
        "mean_abs_speed_error": values.get("Abs speed error (m/s)", float("nan")),
        "mean_lateral_error_m": values.get(
            "Mean lateral tracking error (m)", float("nan")
        ),
        "mean_jerk_norm": values.get("Mean jerk norm", float("nan")),
        "mean_distance_m": float("nan"),
    }


def build_comparison_table(grid_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(grid_dir / "summary.csv")
    baseline_path = grid_dir.parent / "post_train_200ep_kpis.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing legacy KPI export: {baseline_path}")
    kpis = pd.read_csv(baseline_path)

    rows = [
        _legacy_row(
            kpis,
            label="Nominal PPO — no CBF",
            external_cbf="OFF",
            mode="raw",
            k0=None,
            k1=None,
        ),
        _legacy_row(
            kpis,
            label="Nominal PPO — old CBF",
            external_cbf="ON",
            mode="cbf",
            k0=5.29,
            k1=3.68,
        ),
    ]

    rows.append(
        _stage_row(
            summary.loc[summary["candidate_id"] == "pair_001"].iloc[0],
            label="Stage 1 — zero-gain CBF (c1=0, c2=0)",
        )
    )

    # These are the two tied best-completion cells, selected by the existing
    # Stage 1 result rather than by a new evaluation.
    for candidate_id in ("pair_013", "pair_017"):
        match = summary.loc[summary["candidate_id"] == candidate_id]
        if not match.empty:
            row = match.iloc[0]
            rows.append(
                _stage_row(
                    row,
                    label=(
                        f"Stage 1 — candidate {candidate_id} "
                        f"(c1={float(row['c1']):g}, c2={float(row['c2']):g})"
                    ),
                )
            )

    return pd.DataFrame(rows)


def _format_value(column: str, value: object) -> str:
    numeric = _as_float(value)
    if not np.isfinite(numeric):
        return "—"
    if column in {
        "completion_rate",
        "collision_episode_rate",
        "intervention_rate",
        "qp_failure_rate",
    }:
        return f"{100.0 * numeric:.1f}%"
    if column in {"c1", "c2", "k0", "k1"}:
        return f"{numeric:g}"
    if column in {"mean_return", "mean_distance_m", "mean_abs_speed_error", "mean_lateral_error_m", "mean_jerk_norm"}:
        return f"{numeric:.2f}"
    return f"{numeric:.3f}"


def write_comparison_outputs(table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "comparison_table.csv", index=False)
    columns = [
        "label",
        "episodes",
        "c1",
        "c2",
        "k1",
        "k0",
        "mean_return",
        "completion_rate",
        "collision_events_per_km",
        "collision_episode_rate",
        "intervention_rate",
        "qp_failure_rate",
        "mean_jerk_norm",
        "mean_distance_m",
    ]
    headers = [
        "Setup",
        "Episodes",
        "c1",
        "c2",
        "k1",
        "k0",
        "Return",
        "Completion",
        "Collision events/km",
        "Collision episodes",
        "Intervention",
        "QP failure",
        "Jerk norm",
        "Distance (m)",
    ]
    lines = [
        "# Stage 1 CBF gain comparison",
        "",
        "The first two rows use the existing paired 200-episode KPI export. "
        "The Stage 1 rows use the shared-seed 20-episode grid evaluation.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in table.iterrows():
        cells = []
        for column in columns:
            if column == "label":
                cells.append(str(row[column]))
            elif column == "episodes":
                cells.append(str(int(row[column])))
            else:
                cells.append(_format_value(column, row[column]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "The old raw gains `(k0=5.29, k1=3.68)` do not correspond to a real "
            "factor pair `(c1, c2)` because the quadratic discriminant is negative.",
        ]
    )
    (output_dir / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _heatmap(
    summary: pd.DataFrame,
    axis: plt.Axes,
    *,
    column: str,
    title: str,
    cmap_name: str,
    transform=lambda value: value,
    annotation_format: str = ".2f",
    stars: list[tuple[float, float]] | None = None,
) -> None:
    c1_values = sorted(pd.to_numeric(summary["c1"], errors="coerce").dropna().unique())
    c2_values = sorted(pd.to_numeric(summary["c2"], errors="coerce").dropna().unique())
    matrix = (
        summary.pivot(index="c2", columns="c1", values=column)
        .reindex(index=c2_values, columns=c1_values)
        .to_numpy(dtype=float)
    )
    matrix = np.asarray(transform(matrix), dtype=float)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        extent=(
            min(c1_values) - 0.25,
            max(c1_values) + 0.25,
            min(c2_values) - 0.25,
            max(c2_values) + 0.25,
        ),
    )
    axis.set_title(title)
    axis.set_xlabel("c1")
    axis.set_ylabel("c2")
    axis.set_xticks(c1_values)
    axis.set_yticks(c2_values)
    axis.set_xticklabels([f"{value:g}" for value in c1_values])
    axis.set_yticklabels([f"{value:g}" for value in c2_values])
    for row_index, c2 in enumerate(c2_values):
        for column_index, c1 in enumerate(c1_values):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    c1,
                    c2,
                    format(value, annotation_format),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value < np.nanmedian(matrix) else "black",
                )
    for c1, c2 in stars or []:
        axis.plot(
            c1,
            c2,
            marker="*",
            markersize=13,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.4,
        )
    axis.figure.colorbar(image, ax=axis, shrink=0.86)


def write_heatmaps(grid_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(grid_dir / "summary.csv")
    stars = [(0.5, 1.0), (0.5, 3.0)]
    primary = [
        (
            "distance_completion_rate_mean",
            "Completion rate (%)",
            "viridis",
            lambda value: 100.0 * value,
            ".0f",
        ),
        (
            "collision_episode_rate",
            "Collision episode rate (%)",
            "magma",
            lambda value: 100.0 * value,
            ".0f",
        ),
        ("return_mean", "Mean episode return", "plasma", lambda value: value, ".0f"),
        (
            "collision_events_per_km_pooled",
            "Collision events / km",
            "inferno_r",
            lambda value: value,
            ".2f",
        ),
        (
            "event_intervention_rate_weighted",
            "Intervention rate (%)",
            "cividis",
            lambda value: 100.0 * value,
            ".0f",
        ),
        (
            "qp_failure_rate_weighted",
            "QP failure rate (%)",
            "coolwarm",
            lambda value: 100.0 * value,
            ".0f",
        ),
    ]
    safety = [
        (
            "mean_abs_speed_deviation_weighted",
            "Absolute speed error (m/s)",
            "viridis_r",
            lambda value: value,
            ".1f",
        ),
        (
            "mean_lat_y_error_m_weighted",
            "Lateral tracking error (m)",
            "viridis_r",
            lambda value: value,
            ".1f",
        ),
        (
            "mean_correction_norm_weighted",
            "Mean CBF correction norm",
            "cividis",
            lambda value: value,
            ".2f",
        ),
        (
            "mean_jerk_norm_weighted",
            "Mean jerk norm",
            "plasma",
            lambda value: value,
            ".1f",
        ),
        ("h_min_min", "Minimum h", "RdYlGn", lambda value: value, ".1f"),
        (
            "shadow_intervention_rate_weighted",
            "Shadow intervention rate (%)",
            "cividis",
            lambda value: 100.0 * value,
            ".0f",
        ),
    ]
    for filename, title, metrics in (
        ("heatmaps_primary.png", "Stage 1 CBF gain grid — outcome metrics", primary),
        ("heatmaps_safety.png", "Stage 1 CBF gain grid — control and safety metrics", safety),
    ):
        figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
        figure.suptitle(title, fontsize=18)
        for axis, (column, metric_title, cmap, transform, annotation_format) in zip(
            axes.flat, metrics
        ):
            _heatmap(
                summary,
                axis,
                column=column,
                title=metric_title,
                cmap_name=cmap,
                transform=transform,
                annotation_format=annotation_format,
                stars=stars,
            )
        figure.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    grid_dir = args.grid_dir.resolve()
    output_dir = (args.output_dir or grid_dir / "visual_report").resolve()
    table = build_comparison_table(grid_dir)
    write_comparison_outputs(table, output_dir)
    write_heatmaps(grid_dir, output_dir)
    print(f"comparison table: {output_dir / 'comparison_table.md'}")
    print(f"primary heatmaps: {output_dir / 'heatmaps_primary.png'}")
    print(f"safety heatmaps: {output_dir / 'heatmaps_safety.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
