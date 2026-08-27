"""Render matched-seed, time-series comparisons for the saved policy rollouts.

This renderer operates on the closed-loop CSV traces produced by
``visualize_policy_changes.py``.  It does not run the simulator again.  The
outputs are deliberately complementary:

* world and ego-frame trajectory "photographs";
* event-aligned and progress-aligned state/action timelines;
* normalized trajectory fingerprints and CBF/traffic-guard barcodes;
* event contact sheets showing the same traffic scene at key moments;
* paired-policy deviation ribbons; and
* synchronized MP4 dashboards for matched scenario seeds.

The rollout data use the policy-step clock (the saved traces are sampled at
the policy rate).  Raw CBF signals are shown separately from operational
actions so that a plot does not accidentally imply that a shadow projection
was applied to the vehicle.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

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
from matplotlib.colors import BoundaryNorm, Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "final_Results" / "policy_analysis" / "visualizations"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "time_series"

VARIANT_ORDER = ("B1", "B2_1", "B2_2", "B2_3", "B3_1", "B3_2", "B3_3")
VARIANT_LABELS = {
    "B1": "B1 nominal",
    "B2_1": "B2.1 reward only",
    "B2_2": "B2.2 reward + actor feedback",
    "B2_3": "B2.3 actor only",
    "B3_1": "B3.1 differentiable reward",
    "B3_2": "B3.2 differentiable reward + actor feedback",
    "B3_3": "B3.3 differentiable actor",
}
VARIANT_COLORS = {
    "B1": "#4c78a8",
    "B2_1": "#f58518",
    "B2_2": "#e45756",
    "B2_3": "#72b7b2",
    "B3_1": "#54a24b",
    "B3_2": "#b279a2",
    "B3_3": "#ff9da6",
}
SEEDS = (1_300_000, 1_300_001, 1_300_002)
PAIR_ORDER = (
    ("B2_1", "B3_1", "reward only -> differentiable reward"),
    ("B2_2", "B3_2", "reward + feedback -> differentiable reward + feedback"),
    ("B2_3", "B3_3", "actor only -> differentiable actor"),
    ("B3_1", "B3_2", "differentiable reward -> + actor feedback"),
)
ROAD_Y_MIN = 0.0
ROAD_Y_MAX = 10.2
ROAD_LENGTH_M = 380.0
DEFAULT_NORM_BINS = 120
VIDEO_FPS = 10


def _bool_column(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    if np.isfinite(default):
        result = result.fillna(default)
    return result


def _episode_key(frame: pd.DataFrame) -> pd.Series:
    return frame["variant_id"].astype(str) + "|" + frame["scenario_seed"].astype(str)


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _unwrap_periodic(values: pd.Series, period: float = ROAD_LENGTH_M) -> np.ndarray:
    """Unwrap a periodic road coordinate so a lap does not draw a false jump."""

    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if array.size == 0:
        return array
    finite = np.isfinite(array)
    if not finite.all():
        output = array.copy()
        if finite.any():
            output[finite] = _unwrap_periodic(pd.Series(array[finite]), period)
        return output
    phase = array / float(period) * 2.0 * np.pi
    return np.unwrap(phase) / (2.0 * np.pi) * float(period)


def _load_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trace_path = input_dir / "closed_loop_trace.csv"
    vehicle_path = input_dir / "closed_loop_vehicle_trace.csv"
    summary_path = input_dir / "closed_loop_summary.csv"
    missing = [str(path) for path in (trace_path, vehicle_path, summary_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing rollout input(s): " + ", ".join(missing))

    trace = pd.read_csv(trace_path)
    vehicles = pd.read_csv(vehicle_path)
    summary = pd.read_csv(summary_path)
    for frame in (trace, vehicles, summary):
        frame["scenario_seed"] = pd.to_numeric(frame["scenario_seed"], errors="coerce").astype(int)
    for name in ("raw_feasible", "operational_feasible", "shadow_external_intervention", "collision"):
        if name in trace:
            trace[name] = _bool_column(trace[name])
    if "collision" in summary:
        summary["collision"] = _bool_column(summary["collision"])
    if "is_ego" in vehicles:
        vehicles["is_ego"] = _bool_column(vehicles["is_ego"])

    trace["episode_key"] = _episode_key(trace)
    vehicles["episode_key"] = _episode_key(vehicles)
    summary["episode_key"] = _episode_key(summary)
    trace = trace.sort_values(["variant_id", "scenario_seed", "policy_step"]).reset_index(drop=True)
    vehicles = vehicles.sort_values(
        ["variant_id", "scenario_seed", "policy_step", "vehicle_index"]
    ).reset_index(drop=True)

    # The simulator uses a periodic 380 m road.  Keep the original coordinates
    # for literal simulator values, but add unwrapped coordinates for plots and
    # videos so crossing x=0 is not mistaken for a large backwards motion.
    trace["ego_x_unwrapped_m"] = np.nan
    for _, group in trace.groupby("episode_key", sort=False):
        trace.loc[group.index, "ego_x_unwrapped_m"] = _unwrap_periodic(group["ego_x_m"])
    vehicles["absolute_x_unwrapped_m"] = np.nan
    for _, group in vehicles.groupby(["episode_key", "vehicle_token"], sort=False):
        vehicles.loc[group.index, "absolute_x_unwrapped_m"] = _unwrap_periodic(group["absolute_x_m"])

    trace["progress_m"] = np.nan
    trace["event_time_s"] = np.nan
    trace["event_relative_time_s"] = np.nan
    trace["normalized_time"] = np.nan
    for _, group in trace.groupby("episode_key", sort=False):
        indices = group.index
        trace.loc[indices, "progress_m"] = group["ego_x_unwrapped_m"].to_numpy() - float(
            group["ego_x_unwrapped_m"].iloc[0]
        )
        danger = (
            (pd.to_numeric(group["operational_hocbf_margin"], errors="coerce") < 0.0)
            | (pd.to_numeric(group["critical_ttc_s"], errors="coerce") <= 1.0)
            | (pd.to_numeric(group["critical_clearance_m"], errors="coerce") <= 0.0)
        )
        if danger.any():
            event_time = float(group.loc[danger, "time_s"].iloc[0])
        else:
            min_index = pd.to_numeric(group["critical_clearance_m"], errors="coerce").idxmin()
            event_time = float(group.loc[min_index, "time_s"])
        times = pd.to_numeric(group["time_s"], errors="coerce").to_numpy(float)
        duration = max(float(times[-1] - times[0]), 1e-9)
        trace.loc[indices, "event_time_s"] = event_time
        trace.loc[indices, "event_relative_time_s"] = times - event_time
        trace.loc[indices, "normalized_time"] = (times - times[0]) / duration

    for column in (
        "time_s",
        "policy_step",
        "ego_x_m",
        "ego_x_unwrapped_m",
        "ego_y_m",
        "ego_vx_mps",
        "ego_vy_mps",
        "critical_clearance_m",
        "critical_ttc_s",
        "raw_hocbf_margin",
        "operational_hocbf_margin",
        "raw_ax",
        "raw_ay",
        "operational_ax",
        "operational_ay",
        "shadow_external_correction_norm",
        "traffic_guard_brakes_step",
    ):
        if column in trace:
            trace[column] = _numeric(trace[column])
    return trace, vehicles, summary


def _episode_groups(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {key: group.sort_values("policy_step") for key, group in frame.groupby("episode_key")}


def _variant_seed_key(variant: str, seed: int) -> str:
    return f"{variant}|{int(seed)}"


def _label(variant: str, seed: int | None = None) -> str:
    if seed is None:
        return VARIANT_LABELS.get(variant, variant)
    return f"{variant} / {int(seed)}"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _event_rows(group: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    group = group.sort_values("policy_step")
    rows: list[tuple[str, pd.Series]] = [("start", group.iloc[0])]
    danger = (
        (group["operational_hocbf_margin"] < 0.0)
        | (group["critical_ttc_s"] <= 1.0)
        | (group["critical_clearance_m"] <= 0.0)
    )
    if danger.any():
        rows.append(("first danger", group.loc[danger].iloc[0]))
    rows.append(("min clearance", group.loc[group["critical_clearance_m"].idxmin()]))
    correction = group["shadow_external_correction_norm"].fillna(0.0)
    rows.append(("max shadow correction", group.loc[correction.idxmax()]))
    rows.append(("end", group.iloc[-1]))
    return rows


def _row_at_or_before(group: pd.DataFrame, step: int) -> pd.Series:
    eligible = group[group["policy_step"] <= int(step)]
    if eligible.empty:
        return group.iloc[0]
    return eligible.iloc[-1]


def _vehicle_snapshot(vehicle_group: pd.DataFrame, step: int) -> pd.DataFrame:
    eligible = vehicle_group[vehicle_group["policy_step"] <= int(step)]
    if eligible.empty:
        selected_step = vehicle_group["policy_step"].min()
    else:
        selected_step = eligible["policy_step"].max()
    return vehicle_group[vehicle_group["policy_step"] == selected_step]


def _critical_index(token: Any) -> int:
    match = re.search(r"(\d+)", str(token))
    return int(match.group(1)) if match else -1


def _finite_bounds(values: Iterable[float], low_pad: float = 1.0, high_pad: float | None = None) -> tuple[float, float]:
    values_array = np.asarray(list(values), dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size == 0:
        return -1.0, 1.0
    lo = float(np.nanpercentile(values_array, 1.0))
    hi = float(np.nanpercentile(values_array, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    pad = float(low_pad if high_pad is None else high_pad)
    return lo - pad, hi + pad


def _trajectory_overlay_world(trace: pd.DataFrame, vehicles: pd.DataFrame, output_dir: Path) -> Path:
    ego = vehicles[vehicles["is_ego"]].copy()
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(19, 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    all_x = ego["absolute_x_unwrapped_m"].to_numpy(float)
    x_min, x_max = _finite_bounds(all_x, low_pad=8.0, high_pad=8.0)
    for ax, seed in zip(axes, SEEDS):
        seed_ego = ego[ego["scenario_seed"] == seed]
        for variant in VARIANT_ORDER:
            group = seed_ego[seed_ego["variant_id"] == variant].sort_values("policy_step")
            if group.empty:
                continue
            ax.plot(
                group["absolute_x_unwrapped_m"],
                group["absolute_y_m"],
                color=VARIANT_COLORS[variant],
                linewidth=2.0,
                alpha=0.88,
                label=variant,
            )
            ax.scatter(
                group["absolute_x_unwrapped_m"].iloc[0],
                group["absolute_y_m"].iloc[0],
                color=VARIANT_COLORS[variant],
                s=20,
                marker="o",
                edgecolor="white",
                linewidth=0.4,
                zorder=4,
            )
            final = group.iloc[-1]
            marker = "X" if bool(trace[(trace.variant_id == variant) & (trace.scenario_seed == seed)]["collision"].any()) else "o"
            ax.scatter(
                final["absolute_x_unwrapped_m"],
                final["absolute_y_m"],
                color=VARIANT_COLORS[variant],
                s=45 if marker == "X" else 24,
                marker=marker,
                edgecolor="black" if marker == "X" else "none",
                linewidth=0.6,
                zorder=5,
            )
        ax.set_title(f"Matched traffic seed {seed}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(ROAD_Y_MIN, ROAD_Y_MAX)
        ax.set_xlabel("unwrapped longitudinal position x (m)")
        ax.set_yticks(np.linspace(0, 10.2, 5))
        _style_axes(ax)
        for y in (0.0, ROAD_Y_MAX):
            ax.axhline(y, color="black", linewidth=0.8, alpha=0.45)
    axes[0].set_ylabel("world lateral position y (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle("World-fixed ego trajectory photographs", y=1.10, fontsize=14)
    fig.text(
        0.5,
        -0.01,
        "Lines are closed-loop ego paths; circles mark starts/ends and X marks an episode collision.",
        ha="center",
        fontsize=9,
    )
    path = output_dir / "trajectory_overlay_world.png"
    _save(fig, path)
    return path


def _initial_tokens_for_panel(vehicle_group: pd.DataFrame, count: int = 10) -> list[str]:
    start = vehicle_group[vehicle_group["policy_step"] == vehicle_group["policy_step"].min()]
    start = start[~start["is_ego"]].copy()
    if start.empty:
        return []
    start["distance"] = np.hypot(start["relative_x_m"], start["relative_y_m"])
    return start.sort_values("distance")["vehicle_token"].head(count).astype(str).tolist()


def _trajectory_overlay_ego_frame(vehicles: pd.DataFrame, trace: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(len(VARIANT_ORDER), len(SEEDS), figsize=(15.5, 25), sharex=True, sharey=True)
    axes = np.asarray(axes)
    for row, variant in enumerate(VARIANT_ORDER):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            key = _variant_seed_key(variant, seed)
            vehicle_group = vehicles[vehicles["episode_key"] == key].copy()
            trace_group = trace[trace["episode_key"] == key].sort_values("policy_step")
            if vehicle_group.empty:
                ax.set_visible(False)
                continue
            tokens = _initial_tokens_for_panel(vehicle_group, count=10)
            for token in tokens:
                token_group = vehicle_group[vehicle_group["vehicle_token"].astype(str) == token].sort_values("policy_step")
                ax.plot(
                    token_group["relative_x_m"],
                    token_group["relative_y_m"],
                    color="#9aa0a6",
                    alpha=0.42,
                    linewidth=0.8,
                )
            critical_token = str(trace_group["critical_vehicle_token"].iloc[0]) if not trace_group.empty else ""
            if critical_token:
                critical = vehicle_group[vehicle_group["vehicle_token"].astype(str) == critical_token].sort_values("policy_step")
                if not critical.empty:
                    ax.plot(
                        critical["relative_x_m"],
                        critical["relative_y_m"],
                        color="#d62728",
                        linewidth=1.8,
                        alpha=0.85,
                        label="critical vehicle",
                    )
                    final_critical = critical.iloc[-1]
                    ax.scatter(final_critical["relative_x_m"], final_critical["relative_y_m"], color="#d62728", marker="*", s=35, zorder=5)
            non_ego = vehicle_group[~vehicle_group["is_ego"]]
            latest = _vehicle_snapshot(vehicle_group, int(trace_group["policy_step"].iloc[-1])) if not trace_group.empty else vehicle_group
            latest = latest[~latest["is_ego"]]
            ax.scatter(non_ego["relative_x_m"].iloc[:: max(1, len(non_ego) // 350)], non_ego["relative_y_m"].iloc[:: max(1, len(non_ego) // 350)], s=2, color="#c5c9ce", alpha=0.10)
            ax.scatter(latest["relative_x_m"], latest["relative_y_m"], s=8, color="#636a73", alpha=0.45, zorder=3)
            ax.scatter(0.0, 0.0, color=VARIANT_COLORS[variant], marker="D", s=28, edgecolor="black", linewidth=0.4, zorder=6)
            outcome = str(trace_group["outcome"].iloc[-1]) if not trace_group.empty else ""
            ax.set_title(f"{variant} | {seed}\n{outcome}", fontsize=8)
            ax.set_xlim(-65, 65)
            ax.set_ylim(-12, 12)
            _style_axes(ax)
            ax.axhline(0, color="#b0b5ba", linewidth=0.5, alpha=0.45)
            ax.axvline(0, color="#b0b5ba", linewidth=0.5, alpha=0.45)
            if row == len(VARIANT_ORDER) - 1:
                ax.set_xlabel("other vehicle x relative to ego (m)")
            if col == 0:
                ax.set_ylabel(f"{variant}\nrelative y (m)")
    fig.suptitle("Ego-frame traffic worldlines for the same initial scenario seeds", y=0.995, fontsize=14)
    fig.text(0.5, 0.005, "Gray paths show initially-near traffic; red identifies the initially critical vehicle when it remains present.", ha="center", fontsize=9)
    path = output_dir / "trajectory_overlay_ego_frame.png"
    _save(fig, path)
    return path


EVENT_METRICS = (
    ("critical_clearance_m", "critical clearance (m)", None),
    ("critical_ttc_s", "critical TTC (s; clipped)", (0.0, 8.0)),
    ("raw_hocbf_margin", "raw HOCBF margin", None),
    ("operational_hocbf_margin", "operational HOCBF margin", None),
    ("raw_ax", "longitudinal action ax", None),
    ("operational_ax", "operational ax", None),
    ("raw_ay", "lateral action ay", None),
    ("operational_ay", "operational ay", None),
)


def _plot_group_metric(ax: plt.Axes, group: pd.DataFrame, x: str, column: str, clip: tuple[float, float] | None, color: str, label: str, alpha: float = 0.62, linewidth: float = 1.2) -> None:
    x_values = pd.to_numeric(group[x], errors="coerce").to_numpy(float)
    y_values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
    if clip is not None:
        y_values = np.clip(y_values, clip[0], clip[1])
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if finite.any():
        ax.plot(x_values[finite], y_values[finite], color=color, alpha=alpha, linewidth=linewidth, label=label)


def _event_aligned_timeseries(trace: pd.DataFrame, output_dir: Path) -> Path:
    groups = _episode_groups(trace)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharex=False)
    axes = axes.ravel()
    for ax, (column, title, clip) in zip(axes, EVENT_METRICS):
        for key, group in groups.items():
            variant = str(group["variant_id"].iloc[0])
            seed = int(group["scenario_seed"].iloc[0])
            _plot_group_metric(ax, group, "event_relative_time_s", column, clip, VARIANT_COLORS[variant], variant if seed == SEEDS[0] else "_nolegend_", alpha=0.50, linewidth=1.0)
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("time relative to first danger (s)")
        _style_axes(ax)
    axes[0].set_ylabel("value")
    axes[4].set_ylabel("value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.025), frameon=False)
    fig.suptitle("Event-aligned state, action, and safety traces", y=1.08, fontsize=14)
    fig.text(0.5, -0.015, "Each thin line is one matched-seed rollout; t=0 is the first operational-margin/clearance/TTC danger event.", ha="center", fontsize=9)
    path = output_dir / "event_aligned_state_action.png"
    _save(fig, path)
    return path


PROGRESS_METRICS = (
    ("ego_y_m", "ego lateral position y (m)", None),
    ("ego_vx_mps", "ego longitudinal speed (m/s)", None),
    ("front_gap_m", "front gap (m)", None),
    ("critical_clearance_m", "critical clearance (m)", None),
    ("operational_hocbf_margin", "operational HOCBF margin", None),
    ("raw_ax", "raw ax", None),
    ("operational_ax", "operational ax", None),
    ("raw_ay", "raw ay", None),
    ("operational_ay", "operational ay", None),
)


def _progress_aligned_timeseries(trace: pd.DataFrame, output_dir: Path) -> Path:
    groups = _episode_groups(trace)
    fig, axes = plt.subplots(3, 3, figsize=(17, 13), sharex=False)
    axes = axes.ravel()
    for ax, (column, title, clip) in zip(axes, PROGRESS_METRICS):
        for key, group in groups.items():
            variant = str(group["variant_id"].iloc[0])
            seed = int(group["scenario_seed"].iloc[0])
            _plot_group_metric(ax, group, "progress_m", column, clip, VARIANT_COLORS[variant], variant if seed == SEEDS[0] else "_nolegend_", alpha=0.50, linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("ego progress from rollout start (m)")
        _style_axes(ax)
    axes[0].set_ylabel("value")
    axes[3].set_ylabel("value")
    axes[6].set_ylabel("value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.015), frameon=False)
    fig.suptitle("Progress-aligned policy behavior across matched seeds", y=1.065, fontsize=14)
    fig.text(0.5, -0.01, "This view separates spatial progress from clock time; it is useful when policies terminate at different times.", ha="center", fontsize=9)
    path = output_dir / "progress_aligned_state_action.png"
    _save(fig, path)
    return path


def _interpolate_normalized(group: pd.DataFrame, column: str, grid: np.ndarray) -> np.ndarray:
    x = pd.to_numeric(group["normalized_time"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)
    x = x[finite]
    y = y[finite]
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique, indices = np.unique(x, return_index=True)
    y = y[indices]
    if unique.size == 1:
        return np.full_like(grid, y[0], dtype=float)
    return np.interp(grid, unique, y)


FINGERPRINT_METRICS = (
    ("progress_m", "progress (m)", "viridis", None),
    ("ego_y_m", "ego y (m)", "coolwarm", None),
    ("ego_vx_mps", "ego speed (m/s)", "magma", None),
    ("critical_clearance_m", "critical clearance (m)", "RdYlGn", (-2.0, 10.0)),
    ("critical_ttc_s", "critical TTC (s)", "YlGnBu", (0.0, 8.0)),
    ("operational_hocbf_margin", "operational HOCBF margin", "RdBu", (-10.0, 5.0)),
    ("shadow_external_correction_norm", "shadow correction norm", "plasma", (0.0, 2.0)),
    ("traffic_guard_brakes_step", "traffic-guard brake events", "cividis", (0.0, 40.0)),
)


def _episode_order(groups: dict[str, pd.DataFrame]) -> list[str]:
    def order_key(key: str) -> tuple[int, int]:
        variant, seed = key.split("|", 1)
        try:
            variant_index = VARIANT_ORDER.index(variant)
        except ValueError:
            variant_index = len(VARIANT_ORDER)
        return variant_index, int(seed)
    return sorted(groups, key=order_key)


def _fingerprint_heatmaps(trace: pd.DataFrame, output_dir: Path, bins: int) -> Path:
    groups = _episode_groups(trace)
    keys = _episode_order(groups)
    grid = np.linspace(0.0, 1.0, bins)
    fig, axes = plt.subplots(4, 2, figsize=(16, 15), constrained_layout=True)
    axes = axes.ravel()
    for ax, (column, title, cmap, clip) in zip(axes, FINGERPRINT_METRICS):
        matrix = np.vstack([_interpolate_normalized(groups[key], column, grid) for key in keys])
        if clip is not None:
            matrix = np.clip(matrix, clip[0], clip[1])
            norm: Normalize = Normalize(vmin=clip[0], vmax=clip[1])
        else:
            finite = matrix[np.isfinite(matrix)]
            if finite.size:
                lo, hi = np.nanpercentile(finite, [2, 98])
                if hi <= lo:
                    hi = lo + 1.0
                norm = Normalize(vmin=float(lo), vmax=float(hi))
            else:
                norm = Normalize(vmin=0.0, vmax=1.0)
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, extent=(0.0, 1.0, len(keys) - 0.5, -0.5))
        ax.set_title(title)
        ax.set_xlabel("normalized episode time")
        ax.set_yticks(np.arange(len(keys)))
        ax.set_yticklabels([f"{key.split('|')[0]} / {key.split('|')[1]}" for key in keys], fontsize=7)
        fig.colorbar(image, ax=ax, pad=0.01, fraction=0.035)
    fig.suptitle("Trajectory fingerprints: one row per rollout, one column per normalized time", fontsize=14)
    fig.text(0.5, 0.002, "Heatmaps preserve termination differences while making policy-specific temporal signatures easy to scan.", ha="center", fontsize=9)
    path = output_dir / "trajectory_fingerprint_heatmaps.png"
    _save(fig, path)
    return path


def _barcode(trace: pd.DataFrame, output_dir: Path, bins: int) -> Path:
    groups = _episode_groups(trace)
    keys = _episode_order(groups)
    grid = np.linspace(0.0, 1.0, bins)
    barcode_metrics = (
        ("operational_hocbf_margin", "operational HOCBF margin", "RdBu_r", (-10.0, 5.0)),
        ("critical_clearance_m", "critical clearance (m)", "RdYlGn", (-2.0, 10.0)),
        ("traffic_guard_brakes_step", "guard brake events / step", "cividis", (0.0, 40.0)),
        ("critical_vehicle_index", "critical vehicle index", "tab20", None),
    )
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True, constrained_layout=True)
    for ax, (column, title, cmap, clip) in zip(axes, barcode_metrics):
        matrices: list[np.ndarray] = []
        for key in keys:
            group = groups[key].copy()
            if column == "critical_vehicle_index":
                group[column] = group["critical_vehicle_token"].map(_critical_index).replace(-1, np.nan)
            else:
                group[column] = pd.to_numeric(group[column], errors="coerce")
                if column == "traffic_guard_brakes_step":
                    group[column] = np.log1p(np.maximum(group[column].fillna(0.0), 0.0))
            values = _interpolate_normalized(group, column, grid)
            if clip is not None:
                values = np.clip(values, clip[0], clip[1])
            matrices.append(values)
        matrix = np.vstack(matrices)
        if clip is not None:
            norm: Normalize = Normalize(vmin=clip[0], vmax=clip[1])
        else:
            finite = matrix[np.isfinite(matrix)]
            norm = Normalize(vmin=float(np.nanmin(finite)) if finite.size else 0.0, vmax=float(np.nanmax(finite)) if finite.size else 1.0)
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, extent=(0.0, 1.0, len(keys) - 0.5, -0.5))
        ax.set_title(title)
        ax.set_yticks(np.arange(len(keys)))
        ax.set_yticklabels([f"{key.split('|')[0]} / {key.split('|')[1]}" for key in keys], fontsize=7)
        fig.colorbar(image, ax=ax, pad=0.01, fraction=0.018)
    axes[-1].set_xlabel("normalized episode time")
    fig.suptitle("Safety and traffic-guard activity barcodes", fontsize=14)
    fig.text(0.5, 0.002, "The brake barcode uses log(1 + events/step); critical-vehicle identity shows which actor drives the active constraint.", ha="center", fontsize=9)
    path = output_dir / "cbf_guard_activity_barcode.png"
    _save(fig, path)
    return path


def _pair_deviation_ribbons(trace: pd.DataFrame, output_dir: Path) -> Path:
    metrics = (
        ("ego_y_m", "delta ego y (m)"),
        ("ego_vx_mps", "delta ego speed (m/s)"),
        ("critical_clearance_m", "delta critical clearance (m)"),
    )
    fig, axes = plt.subplots(len(PAIR_ORDER), len(metrics), figsize=(16, 14), sharex=False)
    axes = np.asarray(axes)
    for row, (left, right, pair_label) in enumerate(PAIR_ORDER):
        for col, (column, title) in enumerate(metrics):
            ax = axes[row, col]
            for seed in SEEDS:
                left_group = trace[(trace.variant_id == left) & (trace.scenario_seed == seed)].sort_values("time_s")
                right_group = trace[(trace.variant_id == right) & (trace.scenario_seed == seed)].sort_values("time_s")
                if left_group.empty or right_group.empty:
                    continue
                max_time = min(float(left_group.time_s.max()), float(right_group.time_s.max()))
                grid = np.linspace(0.0, max_time, 100)
                left_values = np.interp(grid, left_group.time_s, left_group[column])
                right_values = np.interp(grid, right_group.time_s, right_group[column])
                ax.plot(grid, right_values - left_values, color=VARIANT_COLORS[right], alpha=0.72, linewidth=1.4, label=str(seed))
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("time (s)")
            _style_axes(ax)
            if col == 0:
                ax.set_ylabel(f"{right} - {left}\n{pair_label}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False, title="scenario seed")
    fig.suptitle("Matched-seed policy deviation ribbons", y=1.045, fontsize=14)
    fig.text(0.5, 0.002, "Positive values mean the right-hand policy is larger; ribbons are interpolated only over the common horizon.", ha="center", fontsize=9)
    path = output_dir / "paired_policy_deviation_ribbons.png"
    _save(fig, path)
    return path


def _scene_axes(ax: plt.Axes, snapshot: pd.DataFrame, ego_row: pd.Series, trace_group: pd.DataFrame, step: int, title: str, color: str, x_half_width: float = 65.0) -> None:
    center_x = float(ego_row["ego_x_unwrapped_m"])
    ax.set_xlim(center_x - x_half_width, center_x + x_half_width)
    ax.set_ylim(ROAD_Y_MIN, ROAD_Y_MAX)
    ax.axhline(ROAD_Y_MIN, color="black", linewidth=0.8, alpha=0.5)
    ax.axhline(ROAD_Y_MAX, color="black", linewidth=0.8, alpha=0.5)
    others = snapshot[~snapshot["is_ego"]].copy()
    others = others[np.abs(others["absolute_x_unwrapped_m"] - center_x) <= x_half_width + 8.0]
    ax.scatter(others["absolute_x_unwrapped_m"], others["absolute_y_m"], s=15, color="#8d959e", alpha=0.60, zorder=2)
    ego_history = trace_group[trace_group["policy_step"] <= int(step)]
    ax.plot(ego_history["ego_x_unwrapped_m"], ego_history["ego_y_m"], color=color, linewidth=2.0, alpha=0.8, zorder=3)
    ax.scatter(ego_row["ego_x_unwrapped_m"], ego_row["ego_y_m"], color=color, s=42, marker="D", edgecolor="black", linewidth=0.5, zorder=5)
    token = str(ego_row.get("critical_vehicle_token", ""))
    critical = snapshot[snapshot["vehicle_token"].astype(str) == token]
    if not critical.empty:
        ax.scatter(critical["absolute_x_unwrapped_m"], critical["absolute_y_m"], color="#d62728", s=55, marker="*", edgecolor="black", linewidth=0.4, zorder=6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("unwrapped world x (m)")
    ax.set_ylabel("world y (m)")
    _style_axes(ax)


def _event_contact_sheets(trace: pd.DataFrame, vehicles: pd.DataFrame, output_dir: Path) -> list[Path]:
    trace_groups = _episode_groups(trace)
    vehicle_groups = _episode_groups(vehicles)
    outputs: list[Path] = []
    for seed in SEEDS:
        fig, axes = plt.subplots(len(VARIANT_ORDER), 5, figsize=(20, 22), sharex=False, sharey=True)
        axes = np.asarray(axes)
        for row, variant in enumerate(VARIANT_ORDER):
            key = _variant_seed_key(variant, seed)
            trace_group = trace_groups.get(key)
            vehicle_group = vehicle_groups.get(key)
            if trace_group is None or vehicle_group is None:
                continue
            for col, (event_name, event_row) in enumerate(_event_rows(trace_group)):
                ax = axes[row, col]
                step = int(event_row["policy_step"])
                snapshot = _vehicle_snapshot(vehicle_group, step)
                _scene_axes(ax, snapshot, event_row, trace_group, step, f"{event_name}\nstep {step}, t={float(event_row['time_s']):.1f}s", VARIANT_COLORS[variant], x_half_width=58.0)
                if row < len(VARIANT_ORDER) - 1:
                    ax.set_xlabel("")
                if col > 0:
                    ax.set_ylabel("")
                if col == 0:
                    ax.text(0.02, 0.98, variant, transform=ax.transAxes, va="top", fontsize=9, fontweight="bold")
        fig.suptitle(f"Event contact sheet: seed {seed}", y=0.998, fontsize=15)
        fig.text(0.5, 0.003, "Each row is one policy on the same initial traffic seed; red star is the active critical vehicle and the diamond is ego.", ha="center", fontsize=9)
        path = output_dir / f"event_contact_sheet_seed_{seed}.png"
        _save(fig, path)
        outputs.append(path)
    return outputs


def _prepare_video_lookup(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return _episode_groups(frame)


def _history_values(group: pd.DataFrame, step: int, column: str) -> tuple[np.ndarray, np.ndarray]:
    history = group[group["policy_step"] <= int(step)]
    return history["time_s"].to_numpy(float), pd.to_numeric(history[column], errors="coerce").to_numpy(float)


def _action_limits(trace: pd.DataFrame) -> tuple[float, float]:
    values = np.concatenate([
        pd.to_numeric(trace["raw_ax"], errors="coerce").to_numpy(float),
        pd.to_numeric(trace["operational_ax"], errors="coerce").to_numpy(float),
        pd.to_numeric(trace["raw_ay"], errors="coerce").to_numpy(float),
        pd.to_numeric(trace["operational_ay"], errors="coerce").to_numpy(float),
    ])
    values = values[np.isfinite(values)]
    maximum = max(float(np.nanmax(np.abs(values))) if values.size else 1.0, 0.5)
    return -1.15 * maximum, 1.15 * maximum


def _draw_dashboard_frame(
    fig: plt.Figure,
    axes: tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes],
    policy_keys: list[str],
    trace_groups: dict[str, pd.DataFrame],
    vehicle_groups: dict[str, pd.DataFrame],
    step: int,
    action_limits: tuple[float, float],
    main_title: str,
    show_pair_labels: bool = True,
) -> None:
    scene_axes, safety_ax, ax_ax, ay_ax = axes
    scene_axes.clear()
    safety_ax.clear()
    ax_ax.clear()
    ay_ax.clear()
    active_rows: list[pd.Series] = []
    for key in policy_keys:
        group = trace_groups.get(key)
        if group is not None and not group.empty:
            active_rows.append(_row_at_or_before(group, step))
    if not active_rows:
        return
    center_x = float(np.mean([row["ego_x_unwrapped_m"] for row in active_rows]))
    scene_axes.set_xlim(center_x - 65.0, center_x + 65.0)
    scene_axes.set_ylim(ROAD_Y_MIN, ROAD_Y_MAX)
    scene_axes.axhline(ROAD_Y_MIN, color="black", linewidth=0.8, alpha=0.5)
    scene_axes.axhline(ROAD_Y_MAX, color="black", linewidth=0.8, alpha=0.5)
    for key in policy_keys:
        group = trace_groups.get(key)
        if group is None or group.empty:
            continue
        variant = key.split("|", 1)[0]
        row = _row_at_or_before(group, step)
        vehicle_group = vehicle_groups.get(key)
        if vehicle_group is not None:
            snapshot = _vehicle_snapshot(vehicle_group, step)
            others = snapshot[~snapshot["is_ego"]].copy()
            others = others[np.abs(others["absolute_x_unwrapped_m"] - center_x) <= 73.0]
            scene_axes.scatter(others["absolute_x_unwrapped_m"], others["absolute_y_m"], s=7, color="#9aa0a6", alpha=0.18, zorder=1)
            token = str(row.get("critical_vehicle_token", ""))
            critical = snapshot[snapshot["vehicle_token"].astype(str) == token]
            if not critical.empty:
                scene_axes.scatter(critical["absolute_x_m"], critical["absolute_y_m"], color="#d62728", s=34, marker="*", alpha=0.8, zorder=5)
        history = group[group["policy_step"] <= int(step)]
        scene_axes.plot(history["ego_x_unwrapped_m"], history["ego_y_m"], color=VARIANT_COLORS[variant], linewidth=1.6, alpha=0.75, label=variant)
        scene_axes.scatter(row["ego_x_unwrapped_m"], row["ego_y_m"], color=VARIANT_COLORS[variant], s=27, marker="D", edgecolor="black", linewidth=0.35, zorder=6)
        for metric_ax, column, linestyle in (
            (safety_ax, "critical_clearance_m", "-"),
            (safety_ax, "operational_hocbf_margin", "--"),
        ):
            times, values = _history_values(group, step, column)
            values = np.clip(values, -12.0, 12.0)
            metric_ax.plot(times, values, color=VARIANT_COLORS[variant], linestyle=linestyle, linewidth=1.3, alpha=0.78, label=f"{variant} {column}")
        times, values = _history_values(group, step, "critical_ttc_s")
        values = np.clip(values, 0.0, 8.0)
        safety_ax.plot(times, values, color=VARIANT_COLORS[variant], linestyle=":", linewidth=1.0, alpha=0.55)
        for metric_ax, raw_column, operational_column in ((ax_ax, "raw_ax", "operational_ax"), (ay_ax, "raw_ay", "operational_ay")):
            times, raw_values = _history_values(group, step, raw_column)
            _times, operational_values = _history_values(group, step, operational_column)
            metric_ax.plot(times, raw_values, color=VARIANT_COLORS[variant], linestyle="--", linewidth=1.0, alpha=0.55)
            metric_ax.plot(_times, operational_values, color=VARIANT_COLORS[variant], linestyle="-", linewidth=1.5, alpha=0.85)
    scene_axes.set_title("world-fixed branch trajectories\nred star: critical vehicle")
    scene_axes.set_xlabel("unwrapped world x (m)")
    scene_axes.set_ylabel("world y (m)")
    safety_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.65)
    safety_ax.set_ylim(-12.0, 12.0)
    safety_ax.set_title("clearance / HOCBF margin / TTC")
    safety_ax.set_xlabel("time (s)")
    safety_ax.set_ylabel("clearance, margin, clipped TTC")
    ax_ax.set_ylim(*action_limits)
    ay_ax.set_ylim(*action_limits)
    ax_ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ay_ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax_ax.set_title("longitudinal action ax\nraw dashed / operational solid")
    ay_ax.set_title("lateral action ay\nraw dashed / operational solid")
    ax_ax.set_xlabel("time (s)")
    ay_ax.set_xlabel("time (s)")
    ax_ax.set_ylabel("action")
    for axis in (scene_axes, safety_ax, ax_ax, ay_ax):
        _style_axes(axis)
    max_time = max(float(group["time_s"].max()) for group in trace_groups.values() if group is not None and not group.empty)
    current_time = max(float(row["time_s"]) for row in active_rows)
    for axis in (safety_ax, ax_ax, ay_ax):
        axis.set_xlim(0.0, max_time + 0.05)
        axis.axvline(current_time, color="black", linewidth=0.8, alpha=0.4)
    handles, labels = scene_axes.get_legend_handles_labels()
    if handles:
        scene_axes.legend(handles, labels, loc="upper left", fontsize=7, ncol=2, frameon=True, framealpha=0.75)
    fig.suptitle(main_title, fontsize=13, y=0.995)


def _write_matplotlib_video(fig: plt.Figure, path: Path, draw_frame: Any, steps: int, fps: int = VIDEO_FPS) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for MP4 rendering; install cv2 or use --skip-videos") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for step in range(int(steps)):
            draw_frame(step)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def _render_pair_videos(trace: pd.DataFrame, vehicles: pd.DataFrame, output_dir: Path, seeds: tuple[int, ...], max_steps: int | None) -> list[Path]:
    trace_groups = _prepare_video_lookup(trace)
    vehicle_groups = _prepare_video_lookup(vehicles)
    action_limits = _action_limits(trace)
    video_dir = output_dir / "videos"
    outputs: list[Path] = []
    for left, right, pair_label in PAIR_ORDER:
        for seed in seeds:
            keys = [_variant_seed_key(left, seed), _variant_seed_key(right, seed)]
            groups = [trace_groups.get(key) for key in keys]
            if any(group is None or group.empty for group in groups):
                continue
            steps = max(int(group["policy_step"].max()) for group in groups if group is not None)
            steps = steps + 1
            if max_steps is not None:
                steps = min(steps, int(max_steps))
            fig = plt.figure(figsize=(16, 9), dpi=80)
            grid = GridSpec(2, 4, figure=fig, height_ratios=(1.18, 1.0), hspace=0.28, wspace=0.25)
            axes = (
                fig.add_subplot(grid[0, :2]),
                fig.add_subplot(grid[1, :2]),
                fig.add_subplot(grid[1, 2]),
                fig.add_subplot(grid[1, 3]),
            )
            path = video_dir / f"pair_{left}_vs_{right}_seed_{seed}.mp4"
            def draw(step: int, keys: list[str] = keys, pair_label: str = pair_label, seed: int = seed) -> None:
                left_group = trace_groups[keys[0]]
                right_group = trace_groups[keys[1]]
                left_row = _row_at_or_before(left_group, step)
                right_row = _row_at_or_before(right_group, step)
                current_time = max(float(left_row["time_s"]), float(right_row["time_s"]))
                title = f"{pair_label} | matched seed {seed} | t={current_time:.1f}s | policy step {step}"
                _draw_dashboard_frame(fig, axes, keys, trace_groups, vehicle_groups, step, action_limits, title)
            _write_matplotlib_video(fig, path, draw, steps)
            plt.close(fig)
            outputs.append(path)
    return outputs


def _render_all_policy_videos(trace: pd.DataFrame, vehicles: pd.DataFrame, output_dir: Path, seeds: tuple[int, ...], max_steps: int | None) -> list[Path]:
    trace_groups = _prepare_video_lookup(trace)
    vehicle_groups = _prepare_video_lookup(vehicles)
    action_limits = _action_limits(trace)
    video_dir = output_dir / "videos"
    outputs: list[Path] = []
    for seed in seeds:
        keys = [_variant_seed_key(variant, seed) for variant in VARIANT_ORDER]
        present = [key for key in keys if key in trace_groups and not trace_groups[key].empty]
        if not present:
            continue
        steps = max(int(trace_groups[key]["policy_step"].max()) for key in present) + 1
        if max_steps is not None:
            steps = min(steps, int(max_steps))
        fig = plt.figure(figsize=(16, 9), dpi=80)
        grid = GridSpec(2, 4, figure=fig, height_ratios=(1.18, 1.0), hspace=0.28, wspace=0.25)
        axes = (
            fig.add_subplot(grid[0, :2]),
            fig.add_subplot(grid[1, :2]),
            fig.add_subplot(grid[1, 2]),
            fig.add_subplot(grid[1, 3]),
        )
        path = video_dir / f"all_policies_seed_{seed}.mp4"
        def draw(step: int, keys: list[str] = present, seed: int = seed) -> None:
            rows = [_row_at_or_before(trace_groups[key], step) for key in keys]
            current_time = max(float(row["time_s"]) for row in rows)
            title = f"All policy branches | matched seed {seed} | t={current_time:.1f}s | policy step {step}"
            _draw_dashboard_frame(fig, axes, keys, trace_groups, vehicle_groups, step, action_limits, title)
        _write_matplotlib_video(fig, path, draw, steps)
        plt.close(fig)
        outputs.append(path)
    return outputs


def _write_manifest(output_dir: Path, paths: list[Path], trace: pd.DataFrame, args: argparse.Namespace) -> Path:
    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "rollout_rows": int(len(trace)),
        "episodes": sorted(trace["episode_key"].unique().tolist()),
        "policy_step_clock": True,
        "videos": not bool(args.skip_videos),
        "video_fps": VIDEO_FPS,
        "max_video_steps": args.max_video_steps,
        "files": [str(path.relative_to(output_dir)) for path in paths if path.exists()],
    }
    import json
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _write_readme(output_dir: Path, args: argparse.Namespace) -> Path:
    text = f"""# Time-series policy comparison

This folder was rendered from the existing matched-seed closed-loop traces;
the simulator was not rerun.  The comparison uses the policy-step clock and
keeps each saved rollout's natural termination time.

## Static views

- `trajectory_overlay_world.png`: world-fixed ego paths for the three matched
  traffic seeds.  Longitudinal position is unwrapped across the 380 m periodic
  road.
- `trajectory_overlay_ego_frame.png`: ego-centered traffic worldlines for the
  initially-nearest vehicles, with the active critical vehicle highlighted.
- `event_aligned_state_action.png`: state, action, and safety traces centered
  on the first danger event.
- `progress_aligned_state_action.png`: the same quantities indexed by ego
  progress rather than elapsed time.
- `trajectory_fingerprint_heatmaps.png`: one normalized-time row per episode.
- `cbf_guard_activity_barcode.png`: operational margin, clearance, traffic
  guard activity, and critical-vehicle identity as compact temporal barcodes.
- `paired_policy_deviation_ribbons.png`: matched-seed differences for the
  differentiable/non-differentiable pairs.
- `event_contact_sheet_seed_*.png`: five scene snapshots per policy (start,
  first danger, minimum clearance, maximum shadow correction, and end).

## Videos

The `videos/` folder contains four pairwise dashboards and one all-policy
dashboard per matched seed.  Every frame shows the branch trajectories, the
critical safety signals, and raw-vs-operational actions.  Videos are written
at {VIDEO_FPS} frames/s, matching the saved policy-step sampling rate.

## Interpretation guardrails

The source traces were generated with the external CBF disabled during the
closed-loop rollouts while the ordinary simulator traffic guard remained
enabled.  Therefore the videos compare learned closed-loop behavior under the
same simulator guard; a shadow external-CBF signal is diagnostic and is not
the action sent to the vehicle.  Raw and operational actions are intentionally
shown separately.  Event alignment is defined as the earliest saved step with
operational HOCBF margin < 0, critical clearance <= 0 m, or critical TTC <= 1 s;
if none occurs, the minimum-clearance step is used.

Input: `{args.input_dir}`
"""
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--norm-bins", type=int, default=DEFAULT_NORM_BINS)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--max-video-steps", type=int, default=None)
    parser.add_argument("--video-seeds", type=int, nargs="*", default=list(SEEDS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.norm_bins < 2:
        raise ValueError("--norm-bins must be at least 2")
    trace, vehicles, summary = _load_data(args.input_dir)
    del summary
    paths: list[Path] = []
    paths.extend([
        _trajectory_overlay_world(trace, vehicles, args.output_dir),
        _trajectory_overlay_ego_frame(vehicles, trace, args.output_dir),
        _event_aligned_timeseries(trace, args.output_dir),
        _progress_aligned_timeseries(trace, args.output_dir),
        _fingerprint_heatmaps(trace, args.output_dir, args.norm_bins),
        _barcode(trace, args.output_dir, args.norm_bins),
        _pair_deviation_ribbons(trace, args.output_dir),
    ])
    paths.extend(_event_contact_sheets(trace, vehicles, args.output_dir))
    if not args.skip_videos:
        paths.extend(_render_pair_videos(trace, vehicles, args.output_dir, tuple(args.video_seeds), args.max_video_steps))
        paths.extend(_render_all_policy_videos(trace, vehicles, args.output_dir, tuple(args.video_seeds), args.max_video_steps))
    paths.append(_write_readme(args.output_dir, args))
    paths.append(_write_manifest(args.output_dir, paths, trace, args))
    print(f"Rendered {len(paths)} outputs to {args.output_dir}")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
