#!/usr/bin/env python3
"""Build a complete illustrated HTML dossier for the saved 50k PPO/CBF pilots.

The generated HTML is intentionally self-contained in structure but references
the existing local figures and generated video poster frames. Chromium/Edge is
then used by the calling workflow to print it to PDF.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
FINAL = REPO / "artifacts" / "final_Results"
OUT = FINAL / "report_50k"
POSTERS = OUT / "video_posters"

VARIANT_ORDER = ["B1", "B2_1", "B2_2", "B2_3", "B3_1", "B3_2", "B3_3"]
DISPLAY_LABEL = {
    "B1": "B1 nominal",
    "B2_1": "B2.1 non-diff. reward",
    "B2_2": "B2.2 non-diff. reward + detached actor",
    "B2_3": "B2.3 detached actor only",
    "B3_1": "B3.1 differentiable reward only",
    "B3_2": "B3.2 differentiable reward + mean loss",
    "B3_3": "B3.3 differentiable actor only",
}
COMPARISON_TO_ID = {
    "B1": "B1",
    "B2.1": "B2_1",
    "B2.2": "B2_2",
    "B2.3": "B2_3",
    "B3.1": "B3_1",
    "B3.2": "B3_2",
    "B3.3": "B3_3",
}
VARIANT_CODE_TO_ID = {
    "ppo_nominal": "B1",
    "ppo_cbf_reward": "B2_1",
    "ppo_cbf_nd_reward_actor": "B2_2",
    "ppo_cbf_nd_actor_only": "B2_3",
    "ppo_cbf_diff_reward_only": "B3_1",
    "ppo_cbf_integrated_actor_only": "B3_2",
    "ppo_cbf_projected_reward_off": "B3_3",
}

EXCLUDED_CANONICAL_PARTS = {
    "report_50k",
    "ppo200k_cbf5",
    "ppo200k_nosafety5",
    "recovery_ppo200k_cbf5",
    "_runlogs_ppo200k",
    "health_monitor_ppo200k",
}
SMOKE_PART_PREFIXES = (
    "_smoke_visualizations",
    "time_series_smoke",
    "time_series_smoke2",
    "time_series_video_smoke",
)

RUNS = {
    "B1": REPO / "artifacts/p50n/ppo_nominal/seed_307",
    "B2_1": REPO / "artifacts/B2_50k_q1_stable_307/ppo_cbf_reward/seed_307",
    "B2_2": REPO / "artifacts/B2_50k_q1_stable_307/ppo_cbf_nd_reward_actor/seed_307",
    "B2_3": REPO / "artifacts/B2_50k_q1_stable_307/ppo_cbf_nd_actor_only/seed_307",
    "B3_1": REPO / "artifacts/B3_50k_v2/dfro/ppo_cbf_diff_reward_only/seed_307",
    "B3_2": REPO / "artifacts/B3_50k_v2/iao/ppo_cbf_integrated_actor_only/seed_307",
    "B3_3": REPO / "artifacts/B3_50k_v2/dfao/ppo_cbf_projected_reward_off/seed_307",
}

FIGURE_CAPTIONS = {
    "rl_rollout_learning_curves.png": "Rollout reward, episode length, return density, completed-episode behavior, and collection throughput.",
    "ppo_optimization.png": "PPO optimization diagnostics: KL, clipping, entropy, value fit, losses, exploration scale, and learning rate.",
    "rollout_kpis.png": "Training rollout distance, collision counters, saturation, raw-action clipping, and reset diagnostics.",
    "cbf_training_diagnostics.png": "CBF correction, infeasibility, loss, and actor-gradient diagnostics during training.",
    "training_episode_kpis.png": "Per-completed-episode training KPIs with rolling smoothing.",
    "training_episode_kpis_aligned.png": "Per-episode KPIs aligned by global timestep with binned uncertainty bands.",
    "b3_training_gradient_evidence.png": "Primary B3.1 versus B3.2 training evidence for the explicit differentiable mean-alignment term.",
    "b3_differentiable_mean_probe.png": "Risk-conditioned and common-state evidence for B3.1 versus B3.2.",
    "shared_state_action_alignment.png": "Raw-policy alignment to one common external CBF over identical probe states.",
    "plot.png": "Differentiable versus non-differentiable common-state response comparison.",
    "min_h.png": "Minimum geometric barrier h over matched raw-policy trajectories.",
    "margin.png": "Minimum raw-action HOCBF margin over matched raw-policy trajectories.",
    "summary.png": "Matched trajectory minima and raw HOCBF violation fractions.",
    "policy_response_atlas_front_gap_closing_speed.png": "Policy response atlas over front gap and closing speed.",
    "policy_response_atlas_critical_geometry.png": "Policy response atlas over critical-neighbor longitudinal/lateral geometry.",
    "policy_deformation_difference_atlas.png": "Action and correction changes between matched method pairs.",
    "policy_deformation_vector_fields.png": "Direction and magnitude of policy deformation over common traffic states.",
    "feature_sensitivity_heatmap.png": "Finite-difference sensitivity of raw actor outputs to observation features.",
    "action_space_projection_snapshots.png": "Raw proposals, internal projected means, and common external projections on selected states.",
    "frozen_state_action_matrix.png": "Frozen-state action comparison across all seven actors.",
    "frozen_state_action_proposals.png": "Time-indexed policy proposals on one frozen observation sequence.",
    "closed_loop_primary_B2_2_vs_B3_2_storyboard.png": "Matched-seed B2.2/B3.2 clearance, actions, margins, and correction storyboard.",
    "closed_loop_raw_action_proposals.png": "Raw action proposals during diverging closed-loop rollouts.",
    "closed_loop_safety_timeseries_all_policies.png": "Closed-loop clearance, TTC, operational margin, and shadow correction for all policies.",
    "closed_loop_summary_plot.png": "Short matched-scenario closed-loop outcome summary.",
    "closed_loop_traffic_guard_activity.png": "Ordinary social-traffic guard activity during policy rollouts.",
    "closed_loop_traffic_space_time.png": "Space-time view of ego and surrounding traffic during matched rollouts.",
    "trajectory_overlay_world.png": "World-fixed ego trajectories across policies and matched seeds.",
    "trajectory_overlay_ego_frame.png": "Ego-centered traffic worldlines and critical vehicles.",
    "event_aligned_state_action.png": "State, action, and safety traces aligned to first danger.",
    "progress_aligned_state_action.png": "State, action, and safety traces aligned by longitudinal progress.",
    "trajectory_fingerprint_heatmaps.png": "Normalized-time trajectory fingerprints for all policy/seed episodes.",
    "cbf_guard_activity_barcode.png": "Compact safety, guard, and active-constraint barcodes.",
    "paired_policy_deviation_ribbons.png": "Matched-seed deviation ribbons for differentiable/non-differentiable pairs.",
    "overtake_event_timing.png": "Overtake opportunity, intent, commitment, abeam, and clearance timing.",
    "overtake_intent_raster.png": "Signed passing-side lateral action and detected overtake stages.",
    "overtake_relative_paths.png": "Ego path relative to the designated blocker in targeted overtake scenes.",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_smoke(path: Path) -> bool:
    rel = path.relative_to(FINAL)
    return any(part.startswith(SMOKE_PART_PREFIXES) for part in rel.parts)


def is_canonical(path: Path) -> bool:
    rel = path.relative_to(FINAL)
    if any(part in EXCLUDED_CANONICAL_PARTS for part in rel.parts):
        return False
    if is_smoke(path):
        return False
    return True


def safe_walk_files(root: Path, *, prune_excluded: bool = False) -> Iterable[Path]:
    """Yield files while tolerating stale OneDrive directory entries."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda _error: None):
        if prune_excluded:
            dirnames[:] = [
                name
                for name in dirnames
                if name not in EXCLUDED_CANONICAL_PARTS
                and not name.startswith(SMOKE_PART_PREFIXES)
            ]
        parent = Path(dirpath)
        for name in filenames:
            path = parent / name
            try:
                if path.is_file():
                    yield path
            except OSError:
                continue


def local_href(path: Path) -> str:
    return path.resolve().as_uri()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def table_html(df: pd.DataFrame, *, small: bool = False, max_rows: int | None = None) -> str:
    shown = df.copy()
    truncated = False
    if max_rows is not None and len(shown) > max_rows:
        shown = shown.head(max_rows)
        truncated = True
    shown = shown.replace({np.nan: "—", np.inf: "∞", -np.inf: "−∞"})
    klass = "data compact" if small else "data"
    result = shown.to_html(index=False, border=0, classes=klass, escape=True)
    if truncated:
        result += f'<p class="micro">Showing {len(shown):,} of {len(df):,} rows; the complete source is cataloged in the appendix.</p>'
    return result


def format_df(df: pd.DataFrame, specs: dict[str, tuple[str, float]] | None = None) -> pd.DataFrame:
    out = df.copy()
    if not specs:
        return out
    for col, (kind, scale) in specs.items():
        if col not in out:
            continue
        vals = pd.to_numeric(out[col], errors="coerce") * scale
        if kind == "pct":
            out[col] = vals.map(lambda x: "—" if pd.isna(x) else f"{x:.1f}%")
        elif kind == "f1":
            out[col] = vals.map(lambda x: "—" if pd.isna(x) else f"{x:.1f}")
        elif kind == "f2":
            out[col] = vals.map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
        elif kind == "f3":
            out[col] = vals.map(lambda x: "—" if pd.isna(x) else f"{x:.3f}")
        elif kind == "int":
            out[col] = vals.map(lambda x: "—" if pd.isna(x) else f"{int(round(x))}")
    return out


def section(title: str, anchor: str, kicker: str | None = None) -> str:
    k = f'<div class="kicker">{esc(kicker)}</div>' if kicker else ""
    return f'<section class="section pagebreak" id="{esc(anchor)}">{k}<h1>{esc(title)}</h1></section>'


def callout(title: str, body: str, kind: str = "finding") -> str:
    return f'<div class="callout {esc(kind)}"><strong>{esc(title)}</strong><div>{body}</div></div>'


def figure_html(path: Path, caption: str | None = None, *, page: bool = True) -> str:
    rel = path.relative_to(FINAL).as_posix()
    cap = caption or FIGURE_CAPTIONS.get(path.name, path.stem.replace("_", " ").title())
    cls = "figure-page" if page else "figure-card"
    return (
        f'<figure class="{cls}">'
        f'<img src="{esc(local_href(path))}" alt="{esc(cap)}">'
        f'<figcaption><strong>{esc(cap)}</strong><br><span>{esc(rel)}</span></figcaption>'
        "</figure>"
    )


def gallery_html(paths: Iterable[Path], *, columns: int = 2, page_each: bool = False) -> str:
    if page_each:
        return "".join(figure_html(p, page=True) for p in paths)
    cards = "".join(figure_html(p, page=False) for p in paths)
    return f'<div class="gallery cols-{columns}">{cards}</div>'


def metric_value(frame: pd.DataFrame, kpi: str) -> float:
    rows = frame.loc[frame["KPI"] == kpi, "Mean"]
    if rows.empty or pd.isna(rows.iloc[0]) or str(rows.iloc[0]).strip() == "":
        return float("nan")
    return float(rows.iloc[0])


def extract_video_poster(video: Path, target: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    target_index = max(0, int(frames * 0.55) - 1)
    if frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
    ok, image = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, image = cap.read()
    cap.release()
    if ok and image is not None:
        max_width = 720
        if image.shape[1] > max_width:
            scale = max_width / image.shape[1]
            image = cv2.resize(image, (max_width, max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 84])
    duration = frames / fps if fps > 0 else float("nan")
    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration_s": duration,
        "poster_ok": bool(ok),
    }


def build_video_catalog() -> tuple[pd.DataFrame, list[dict[str, Any]], list[list[str]]]:
    videos = sorted(p for p in safe_walk_files(FINAL, prune_excluded=True) if p.suffix.lower() == ".mp4" and is_canonical(p))
    groups: dict[str, list[Path]] = defaultdict(list)
    for video in videos:
        groups[sha256(video)].append(video)

    def preference(path: Path) -> tuple[int, str]:
        rel = path.relative_to(FINAL).as_posix()
        if rel.startswith("overtake_more/"):
            return (0, rel)
        if rel.startswith("videos/tcf/"):
            return (1, rel)
        if rel.startswith("policy_analysis/time_series/"):
            return (2, rel)
        return (3, rel)

    records: list[dict[str, Any]] = []
    card_records: list[dict[str, Any]] = []
    duplicate_paths: list[list[str]] = []
    for digest, paths in sorted(groups.items(), key=lambda item: preference(sorted(item[1], key=preference)[0])):
        paths = sorted(paths, key=preference)
        primary = paths[0]
        poster = POSTERS / f"{digest[:20]}.jpg"
        meta = extract_video_poster(primary, poster)
        rels = [p.relative_to(FINAL).as_posix() for p in paths]
        if len(rels) > 1:
            duplicate_paths.append(rels)
        category = rels[0].split("/")[0]
        record = {
            "category": category,
            "path": rels[0],
            "duplicates": " | ".join(rels[1:]) if len(rels) > 1 else "—",
            "duration_s": meta["duration_s"],
            "fps": meta["fps"],
            "resolution": f'{meta["width"]}×{meta["height"]}',
            "frames": meta["frames"],
            "size_mb": primary.stat().st_size / (1024 * 1024),
            "sha256": digest,
            "poster": poster,
            "video": primary,
        }
        records.append(record)
        card_records.append(record)
    table = pd.DataFrame(records)
    return table, card_records, duplicate_paths


def video_cards_html(records: list[dict[str, Any]], *, chunk_size: int = 4) -> str:
    chunks: list[str] = []
    for start in range(0, len(records), chunk_size):
        cards: list[str] = []
        for r in records[start : start + chunk_size]:
            poster = Path(r["poster"])
            img = f'<img src="{esc(local_href(poster))}" alt="Video poster">' if poster.exists() else '<div class="poster-missing">Poster unavailable</div>'
            duplicate = "" if r["duplicates"] == "—" else f'<div class="micro">Duplicate path: {esc(r["duplicates"])}</div>'
            cards.append(
                '<div class="video-card">'
                f'<a href="{esc(local_href(Path(r["video"])))}">{img}</a>'
                f'<div class="video-title">{esc(r["path"])}</div>'
                f'<div class="micro">{r["duration_s"]:.1f} s · {r["fps"]:.1f} fps · {esc(r["resolution"])} · {r["size_mb"]:.2f} MB</div>'
                f'{duplicate}</div>'
            )
        chunks.append(f'<div class="video-grid pagebreak">{"".join(cards)}</div>')
    return "".join(chunks)


def csv_shape(path: Path) -> tuple[int, int, str]:
    rows = 0
    header: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            for _ in reader:
                rows += 1
    except OSError:
        return (0, 0, "unreadable")
    preview = ", ".join(header[:6]) + (" …" if len(header) > 6 else "")
    return rows, len(header), preview


def artifact_catalog() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical = sorted(p for p in safe_walk_files(FINAL, prune_excluded=True) if is_canonical(p))
    records: list[dict[str, Any]] = []
    csv_records: list[dict[str, Any]] = []
    for p in canonical:
        rel = p.relative_to(FINAL).as_posix()
        records.append({
            "section": rel.split("/")[0],
            "extension": p.suffix.lower() or "[none]",
            "path": rel,
            "size_mb": p.stat().st_size / (1024 * 1024),
        })
        if p.suffix.lower() == ".csv":
            rows, columns, preview = csv_shape(p)
            csv_records.append({
                "section": rel.split("/")[0],
                "path": rel,
                "rows": rows,
                "columns": columns,
                "first_columns": preview,
                "size_mb": p.stat().st_size / (1024 * 1024),
            })
    full = pd.DataFrame(records)
    summary = (
        full.groupby(["section", "extension"], as_index=False)
        .agg(files=("path", "count"), size_mb=("size_mb", "sum"))
        .sort_values(["section", "extension"])
    )
    return full, summary, pd.DataFrame(csv_records)


def model_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for vid in VARIANT_ORDER:
        run = RUNS[vid]
        cfg = read_json(run / "run_config.json")
        spec = cfg.get("variant_spec", {})
        env = cfg.get("env_config", {})
        topology = cfg.get("collection_topology", {})
        model = run / "model_final.zip"
        train_csv = run / "training_episodes.csv"
        episodes = max(0, sum(1 for _ in train_csv.open("r", encoding="utf-8", errors="replace")) - 1)
        rows.append({
            "ID": vid.replace("_", "."),
            "method": DISPLAY_LABEL[vid],
            "execution during training": spec.get("execution_mode", "box"),
            "reward penalty": bool(spec.get("reward_penalty", False)),
            "projected mean": bool(spec.get("projected_mean", False)),
            "diff. actor term": bool(spec.get("differentiable_actor_loss", False)),
            "detached actor term": bool(spec.get("detached_actor_loss", False)),
            "λ intervention": cfg.get("lambda_intervention", 0.0),
            "λ mean": cfg.get("lambda_mean", 0.0),
            "λ detached": cfg.get("lambda_detached_actor", 0.0),
            "physics/policy Hz": f'{env.get("simulation_frequency", "—")}/{env.get("policy_frequency", "—")}',
            "obs. base dim": 30 if vid == "B1" else 32,
            "train envs": topology.get("n_envs", "—"),
            "completed train episodes": episodes,
            "model SHA-256": sha256(model)[:16],
        })
    return pd.DataFrame(rows)


def true_cbf_free_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = FINAL / "eval/postprocessed_true_cbf_free/summary_agent_relevant.csv"
    summary = pd.read_csv(summary_path)
    summary["variant_id"] = summary["comparison_label"].map(COMPARISON_TO_ID)
    summary["_order"] = summary["variant_id"].map({k: i for i, k in enumerate(VARIANT_ORDER)})
    summary = summary.sort_values("_order")
    primary = summary[[
        "variant_id", "episodes", "task_distance_m", "physics_frequency_hz", "policy_frequency_hz",
        "episode_return_mean", "distance_m_mean", "ego_collision_episode_rate",
        "ego_collision_events_per_km_pooled", "task_completion_rate",
        "abs_speed_error_mps_weighted", "lateral_error_m_weighted", "jerk_norm_weighted",
    ]].copy()
    primary["variant_id"] = primary["variant_id"].map(DISPLAY_LABEL)
    primary.columns = [
        "policy", "N", "task m", "physics Hz", "policy Hz", "return", "distance m",
        "collision episodes", "collisions/km", "completion", "speed error m/s",
        "lateral error m", "jerk norm",
    ]

    episodes = pd.read_csv(FINAL / "eval/true_cbf_free/episodes_true_cbf_free.csv")

    def paired(left: str, right: str) -> dict[str, Any]:
        l = episodes[episodes["comparison_label"] == left].set_index("scenario_seed")
        r = episodes[episodes["comparison_label"] == right].set_index("scenario_seed")
        common = l.index.intersection(r.index)
        ld, rd = l.loc[common], r.loc[common]
        ret = rd["episode_return"].astype(float) - ld["episode_return"].astype(float)
        dist = rd["total_distance_m"].astype(float) - ld["total_distance_m"].astype(float)
        comp = as_bool(rd["task_completed"]).astype(int) - as_bool(ld["task_completed"]).astype(int)
        coll = (rd["distinct_ego_collision_events"].astype(float) > 0).astype(int) - (ld["distinct_ego_collision_events"].astype(float) > 0).astype(int)
        return {
            "comparison": f"{left} → {right}",
            "paired N": len(common),
            "Δ return": ret.mean(),
            "return wins": int((ret > 0).sum()),
            "Δ distance m": dist.mean(),
            "distance wins": int((dist > 0).sum()),
            "Δ completion": comp.mean(),
            "Δ collision episode rate": coll.mean(),
        }

    paired_df = pd.DataFrame([
        paired("B2.1", "B3.1"),
        paired("B2.2", "B3.2"),
        paired("B2.3", "B3.3"),
        paired("B3.1", "B3.2"),
    ])
    return primary, paired_df, episodes


def external_cbf_table() -> pd.DataFrame:
    paths = [
        FINAL / "eval/cbf_on_off/B1_external_on_off.csv",
        FINAL / "eval/cbf_on_off/B2_external_on_off.csv",
        FINAL / "eval/cbf_on_off/B3_1_external_on_off.csv",
        FINAL / "eval/cbf_on_off/B3_2_external_on_off.csv",
        FINAL / "eval/cbf_on_off/B3_3_external_on_off.csv",
    ]
    data = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    data["variant_id"] = data["variant"].map(VARIANT_CODE_TO_ID)
    rows: list[dict[str, Any]] = []
    for vid in VARIANT_ORDER:
        v = data[data["variant_id"] == vid]
        raw = v[v["mode"] == "raw"]
        cbf = v[v["mode"] == "cbf"]
        rows.append({
            "policy": DISPLAY_LABEL[vid],
            "episodes/mode": int(float(raw["episodes_per_mode"].iloc[0])),
            "raw coll./km": metric_value(raw, "Ego collisions / km"),
            "CBF coll./km": metric_value(cbf, "Ego collisions / km"),
            "raw completion": metric_value(raw, "Distance-based completion rate"),
            "CBF completion": metric_value(cbf, "Distance-based completion rate"),
            "CBF intervention": metric_value(cbf, "Intervention rate"),
            "CBF QP fail": metric_value(cbf, "QP failure rate"),
            "raw jerk": metric_value(raw, "Mean jerk norm"),
            "CBF jerk": metric_value(cbf, "Mean jerk norm"),
            "CBF minimum h": metric_value(cbf, "Minimum h"),
        })
    return pd.DataFrame(rows)


def training_table() -> pd.DataFrame:
    source = pd.read_csv(FINAL / "policy_analysis/training_effect_summary.csv")
    cols = [
        "variant_id", "tail10_ep_rew_mean", "tail10_ep_len_mean", "tail10_return_per_timestep",
        "tail10_cbf_mean_correction", "tail10_cbf_mean_loss", "tail10_cbf_mean_infeasible_rate",
        "mean_g_cbf_to_g_ppo_ratio", "mean_g_ppo_g_cbf_cosine",
    ]
    out = source[cols].copy()
    out["variant_id"] = out["variant_id"].map(DISPLAY_LABEL)
    out.columns = [
        "policy", "tail reward", "tail length", "tail return/step", "tail CBF correction",
        "tail CBF loss", "tail infeasible", "mean |gCBF|/|gPPO|", "mean gradient cosine",
    ]
    return out


def probe_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probe = pd.read_csv(FINAL / "policy_analysis/probe_alignment_summary.csv")
    out = probe[[
        "variant_id", "state_count", "feasible_set_state_count", "raw_feasible_rate",
        "mean_projection_correction_physical", "p90_projection_correction_physical",
        "external_intervention_rate", "mean_raw_max_constraint_violation",
    ]].copy()
    out["variant_id"] = out["variant_id"].map(DISPLAY_LABEL)
    out.columns = [
        "policy", "states", "feasible-set states", "raw feasible", "mean correction m/s²",
        "P90 correction m/s²", "external intervention", "mean max constraint value",
    ]

    paired = pd.read_csv(FINAL / "policy_analysis/b3_1_vs_b3_2_paired_effect.csv")
    paired = paired[[
        "metric", "n_paired_states", "B3_1_mean", "B3_2_mean", "B3_2_minus_B3_1",
        "bootstrap_95_ci_low", "bootstrap_95_ci_high",
    ]].copy()
    paired.columns = ["metric", "N", "B3.1", "B3.2", "difference", "95% CI low", "95% CI high"]

    dvnd = pd.read_csv(FINAL / "policy_analysis/dvnd/actions.csv")
    physical = dvnd[dvnd["metric"] == "projection_correction_physical"][[
        "label", "non_differentiable", "differentiable", "n_paired_feasible_states",
        "non_differentiable_mean", "differentiable_mean", "differentiable_minus_non_differentiable",
        "relative_change_percent", "bootstrap_95_ci_low", "bootstrap_95_ci_high",
    ]].copy()
    physical.columns = [
        "comparison", "non-diff.", "diff.", "N", "non-diff correction", "diff correction",
        "difference", "relative change %", "95% CI low", "95% CI high",
    ]
    return out, paired, physical


def closed_loop_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    closed = pd.read_csv(FINAL / "policy_analysis/visualizations/closed_loop_summary.csv")
    closed["collision_bool"] = as_bool(closed["collision"])
    agg = closed.groupby("variant_id", as_index=False).agg(
        scenarios=("scenario_seed", "count"),
        collisions=("collision_bool", "sum"),
        mean_steps=("policy_steps", "mean"),
        worst_clearance_m=("episode_min_critical_clearance_m", "min"),
        mean_shadow_correction=("mean_shadow_external_correction", "mean"),
        mean_shadow_intervention=("shadow_intervention_rate", "mean"),
        mean_internal_shift=("mean_internal_mean_correction", "mean"),
        mean_guard_events=("traffic_guard_step_interventions", "mean"),
    )
    agg["variant_id"] = agg["variant_id"].map(DISPLAY_LABEL)
    agg.columns = [
        "policy", "scenarios", "collisions", "mean steps", "worst clearance m",
        "mean shadow correction", "shadow intervention", "mean internal shift", "mean guard events",
    ]

    raw = pd.read_csv(FINAL / "policy_analysis/traces/summary.csv")
    raw["collision_bool"] = as_bool(raw["collision"])
    raw_agg = raw.groupby("variant_id", as_index=False).agg(
        scenarios=("scenario_seed", "count"),
        collisions=("collision_bool", "sum"),
        mean_steps=("policy_steps", "mean"),
        worst_h=("episode_min_h", "min"),
        worst_clearance_m=("episode_min_clearance_m", "min"),
        mean_raw_violation_rate=("raw_hocbf_violation_rate", "mean"),
    )
    raw_agg["variant_id"] = raw_agg["variant_id"].map(DISPLAY_LABEL)
    raw_agg.columns = [
        "policy", "scenarios", "collisions", "mean steps", "worst h", "worst clearance m",
        "mean raw HOCBF violation",
    ]
    return agg, raw_agg


def overtake_table() -> pd.DataFrame:
    data = pd.read_csv(FINAL / "overtake_more/overtake_summary.csv")
    for col in ["attempted_overtake", "completed_overtake", "aborted_overtake"]:
        data[col] = as_bool(data[col])
    agg = data.groupby("variant_id", as_index=False).agg(
        scenes=("scenario_seed", "count"),
        attempts=("attempted_overtake", "sum"),
        completions=("completed_overtake", "sum"),
        aborts=("aborted_overtake", "sum"),
        median_intent_delay_s=("raw_intent_delay_s", "median"),
        median_clear_delay_s=("clear_delay_s", "median"),
        mean_distance_m=("distance_m", "mean"),
        worst_operational_margin=("min_operational_hocbf_margin", "min"),
        mean_guard_brakes=("traffic_guard_brakes", "mean"),
    )
    agg["completion rate"] = agg["completions"] / agg["scenes"]
    agg["variant_id"] = agg["variant_id"].map(DISPLAY_LABEL)
    agg.columns = [
        "policy", "scenes", "attempts", "completions", "aborts", "median intent delay s",
        "median clear delay s", "mean distance m", "worst operational margin", "mean guard brakes",
        "completion rate",
    ]
    return agg


def rendered_example_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((FINAL / "videos/tcf").glob("*_summary.json")):
        data = read_json(path)
        vid = VARIANT_CODE_TO_ID.get(data.get("variant", ""), data.get("variant", ""))
        rows.append({
            "policy": DISPLAY_LABEL.get(vid, vid),
            "seed": data.get("seed"),
            "steps": data.get("steps_rendered"),
            "return": data.get("total_return"),
            "distance m": data.get("total_distance_m"),
            "collisions": data.get("distinct_ego_collision_events"),
            "terminated": data.get("terminated"),
            "task m": data.get("task_distance_m"),
        })
    out = pd.DataFrame(rows)
    out["_order"] = out["policy"].map({DISPLAY_LABEL[k]: i for i, k in enumerate(VARIANT_ORDER)})
    return out.sort_values(["seed", "_order"]).drop(columns="_order")


def ancillary_50k_table() -> pd.DataFrame:
    files = [
        REPO / "artifacts/C_hocbf_margin_50k_q1_stable_307/post_train_200ep_kpis.csv",
        REPO / "artifacts/ppo_no_safety_reward_50k_q1_stable_307/post_train_200ep_kpis.csv",
        REPO / "artifacts/ppo_cbf_progression_parallel_v3/post_train_200ep_kpis.csv",
        REPO / "artifacts/ppo_nominal_lateral_y_wy2_50k_seed307/post_train_200ep_kpis.csv",
        REPO / "artifacts/ppo_nominal_pilot_rollout2000_seed307_v3/post_train_200ep_kpis.csv",
        REPO / "artifacts/ppo_nominal_target_y_no_dims_wy2_50k_seed307/post_train_200ep_kpis.csv",
        REPO / "artifacts/ppo_nominal_target_y_wy2_50k_seed307/post_train_200ep_kpis.csv",
        REPO / "artifacts/tynd50k_s307/post_train_200ep_kpis.csv",
    ]
    rows: list[dict[str, Any]] = []
    for path in files:
        if not path.exists():
            continue
        data = pd.read_csv(path)
        for (variant, mode), frame in data.groupby(["variant", "mode"]):
            rows.append({
                "study": path.parent.name,
                "variant": variant,
                "mode": mode,
                "episodes": int(float(frame["episodes_per_mode"].iloc[0])),
                "return": metric_value(frame, "Episode return"),
                "collisions/km": metric_value(frame, "Ego collisions / km"),
                "completion": metric_value(frame, "Distance-based completion rate"),
                "minimum h": metric_value(frame, "Minimum h"),
                "intervention": metric_value(frame, "Intervention rate"),
                "QP fail": metric_value(frame, "QP failure rate"),
            })
    return pd.DataFrame(rows)


def unique_png_groups() -> tuple[list[Path], list[list[str]]]:
    paths = sorted(p for p in safe_walk_files(FINAL, prune_excluded=True) if p.suffix.lower() == ".png" and is_canonical(p))
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        groups[sha256(p)].append(p)
    unique: list[Path] = []
    duplicates: list[list[str]] = []
    for _, group in groups.items():
        group = sorted(group, key=lambda p: (0 if "overtake_more" in p.parts else 1, p.as_posix()))
        unique.append(group[0])
        if len(group) > 1:
            duplicates.append([p.relative_to(FINAL).as_posix() for p in group])
    return sorted(unique), duplicates


def figure_groups(unique: list[Path]) -> list[tuple[str, str, list[Path], int, bool]]:
    groups: list[tuple[str, str, list[Path], int, bool]] = []

    def take(prefix: str, direct_only: bool = False) -> list[Path]:
        selected: list[Path] = []
        for p in unique:
            rel = p.relative_to(FINAL).as_posix()
            if rel.startswith(prefix):
                rest = rel[len(prefix):].strip("/")
                if not direct_only or "/" not in rest:
                    selected.append(p)
        return selected

    groups.append(("Training and TensorBoard figures", "All six archived training-curve products are shown at full width.", take("tensorboard/graphs/"), 1, True))
    root_policy = [p for p in take("policy_analysis/", direct_only=True)]
    groups.append(("Core policy-alignment figures", "The direct shared-state and differentiable-gradient evidence.", root_policy, 1, True))
    groups.append(("Differentiable versus non-differentiable figure", "Method-pair comparison on the same feasible probe states.", take("policy_analysis/dvnd/"), 1, True))
    groups.append(("Raw-policy safety trace figures", "Matched-seed trajectories with policy and physics CBF disabled.", take("policy_analysis/traces/"), 1, True))
    groups.append(("Policy-function heatmaps, atlases, and frozen-state analyses", "Every canonical policy-function visualization generated from the common state bank.", take("policy_analysis/visualizations/"), 1, True))
    groups.append(("Time-series and event-aligned figures", "All trajectory, event, progress, barcode, and contact-sheet outputs.", take("policy_analysis/time_series/", direct_only=True), 1, True))
    groups.append(("Original overtake diagnostic figures", "The first two-scene, three-seed overtake analysis, including seed 4244.", [p for p in take("policy_analysis/overtake_diagnostics/") if "/videos/" not in p.relative_to(FINAL).as_posix()], 1, True))
    groups.append(("Expanded overtake figures and freeze frames", "The expanded three-scene, two-seed, all-policy HUD analysis.", [p for p in take("overtake_more/") if "/videos/" not in p.relative_to(FINAL).as_posix()], 1, True))
    groups.append(("True-CBF-free rendered episode previews", "Final-frame previews for all 21 policy/seed videos, plus the separately saved nominal render preview.", take("videos/tcf/") + take("eval/true_cbf_free/renders/"), 2, False))
    groups.append(("Simulator-only context", "The 50-vehicle MTM traffic-guard simulator render used as context.", take("simulator/"), 1, False))
    return groups


def build_report() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    POSTERS.mkdir(parents=True, exist_ok=True)

    video_table, video_records, duplicate_videos = build_video_catalog()
    all_files, artifact_summary, csv_catalog = artifact_catalog()
    models = model_table()
    true_primary, paired_eval, true_episodes = true_cbf_free_tables()
    external = external_cbf_table()
    training = training_table()
    probe, b3_paired, dvnd = probe_tables()
    closed, raw_closed = closed_loop_tables()
    overtake = overtake_table()
    rendered_examples = rendered_example_table()
    ancillary = ancillary_50k_table()
    unique_pngs, duplicate_pngs = unique_png_groups()

    metadata = read_json(FINAL / "eval/true_cbf_free/metadata.json")
    stale_manifest = read_json(FINAL / "manifest.json")
    actual_rows = len(true_episodes)
    actual_per_variant = true_episodes.groupby("comparison_label").size().to_dict()

    counts = Counter(all_files["extension"])
    ext_count = pd.DataFrame([
        {"extension": ext, "files": int(count), "size MB": all_files.loc[all_files["extension"] == ext, "size_mb"].sum()}
        for ext, count in sorted(counts.items())
    ])

    smoke_files = sorted(p for p in safe_walk_files(FINAL) if is_smoke(p))
    smoke_count = Counter((p.suffix.lower() or "[none]") for p in smoke_files)
    smoke_df = pd.DataFrame([{"extension": k, "files": v} for k, v in sorted(smoke_count.items())])

    parts: list[str] = []
    cover_images = [
        FINAL / "policy_analysis/shared_state_action_alignment.png",
        FINAL / "policy_analysis/time_series/event_contact_sheet_seed_1300000.png",
        FINAL / "overtake_more/freeze_frames/freeze_frames_tight_upper_seed_4242.png",
    ]
    parts.append('<div class="cover">')
    parts.append('<div class="kicker light">COMPLETE ILLUSTRATED RESULTS DOSSIER</div>')
    parts.append('<h1>50k PPO–CBF Pilot Results</h1>')
    parts.append('<h2>Training · Evaluation · Policy Functions · Closed Loop · Time Series · Freeze Frames · Videos</h2>')
    parts.append('<p class="cover-summary">All seven seed-307 policies and every canonical generated result currently saved in <code>artifacts/final_Results</code>, with protocol auditing and interpretation.</p>')
    parts.append('<div class="cover-grid">')
    for p in cover_images:
        parts.append(f'<img src="{esc(local_href(p))}" alt="Cover figure">')
    parts.append('</div>')
    parts.append(f'<div class="cover-date">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} · Source workspace snapshot</div>')
    parts.append('</div>')

    parts.append('<section class="toc pagebreak"><h1>Contents</h1><ol>')
    toc = [
        ("scope", "Scope, inventory, and protocol audit"),
        ("executive", "Executive conclusions"),
        ("formulation", "Policy formulations and saved checkpoints"),
        ("training", "Training and optimization results"),
        ("evaluation", "True CBF-free and external-CBF evaluation"),
        ("policy", "Learned-policy and differentiability analyses"),
        ("closedloop", "Closed-loop, raw traces, and time-series behavior"),
        ("overtake", "Overtake timing, freeze frames, and tactical behavior"),
        ("figures", "Complete canonical figure atlas"),
        ("videos", "Complete video poster atlas and catalog"),
        ("ancillary", "Other 50k pilot tables outside the canonical seven"),
        ("catalog", "Complete data-artifact catalog"),
        ("judgment", "Overall assessment and recommended next experiments"),
    ]
    for anchor, title in toc:
        parts.append(f'<li><a href="#{anchor}">{esc(title)}</a></li>')
    parts.append('</ol></section>')

    parts.append(section("Scope, inventory, and protocol audit", "scope", "WHAT IS ACTUALLY PRESENT"))
    parts.append('<p>This dossier treats the seven consolidated 50k PPO pilots as the canonical comparison set and includes every non-smoke result derived from them: raw tables, aggregate tables, archived TensorBoard data, policy probes, heatmaps, closed-loop traces, time-series panels, freeze frames, and videos. In-progress 200k runs are explicitly excluded so they cannot contaminate the 50k conclusions.</p>')
    parts.append(callout("Canonical result inventory", f'<p><strong>{len(unique_pngs)} unique PNG figures</strong> ({len(duplicate_pngs)} duplicate groups), <strong>{len(video_records)} unique MP4 videos</strong> ({len(duplicate_videos)} duplicate groups), <strong>{len(csv_catalog)} CSV files</strong>, seven archived TensorBoard event files, and eight model ZIPs/checkpoints are represented.</p>'))
    parts.append('<h2>File counts by extension</h2>')
    parts.append(table_html(format_df(ext_count, {"size MB": ("f2", 1)}), small=True))
    parts.append('<h2>File counts by package section and extension</h2>')
    parts.append(table_html(format_df(artifact_summary, {"size_mb": ("f2", 1)}), small=True))

    parts.append('<h2>Protocol matrix</h2>')
    protocol = pd.DataFrame([
        {"result family": "B1 training/source evaluation", "task": "380 m", "physics/policy": "20/10 Hz", "observation": "30D", "episodes": "50 per ON/OFF mode", "CBF status": "external ON/OFF", "traffic guard": "on"},
        {"result family": "B2/B3 source evaluation", "task": "1,000 m", "physics/policy": "100/10 Hz", "observation": "32D", "episodes": "30 per ON/OFF mode", "CBF status": "external ON/OFF; B3 internal projection retained", "traffic guard": "on"},
        {"result family": "true CBF-free reevaluation", "task": "B1 380 m; B2/B3 1,000 m", "physics/policy": "B1 20/10; B2/B3 100/10 Hz", "observation": "model contract", "episodes": "30 per policy", "CBF status": "removed from env and policy", "traffic guard": "on"},
        {"result family": "common-state probe", "task": "not a rollout", "physics/policy": "state query", "observation": "adapted model contract", "episodes": "120 states; 87 feasible", "CBF status": "one common shadow projection", "traffic guard": "n/a"},
        {"result family": "visualization closed loop/time series", "task": "≤120 policy steps", "physics/policy": "100/10 Hz", "observation": "shared B3.2 environment", "episodes": "3 matched seeds/policy", "CBF status": "external off; B3 internal operational mean retained", "traffic guard": "on"},
        {"result family": "targeted overtakes", "task": "3 scripted scenes", "physics/policy": "100/10 Hz", "observation": "shared environment", "episodes": "2 seeds × 3 scenes × 7 policies", "CBF status": "external off; shadow only", "traffic guard": "on"},
    ])
    parts.append(table_html(protocol, small=True))
    parts.append(callout("Critical package-label error", f'<p>The top-level manifest says <strong>{stale_manifest.get("evaluation", {}).get("true_cbf_free", {}).get("episodes_per_variant")} episodes per variant</strong>, but the authoritative metadata says <strong>{metadata.get("episodes_per_variant")}</strong> and the episode CSV contains <strong>{actual_rows} total rows</strong>: {esc(actual_per_variant)}. This report uses the actual CSV/metadata and does not call the result a 200-episode-per-policy evaluation.</p>', "warning"))
    parts.append(callout("Historical settings, not current globals", '<p>All seven saved 50k runs trained with <strong>8 vector environments</strong>. Their saved policy clock is <strong>10 Hz</strong>. Later global changes to 20 workers and a 20 Hz policy do not retroactively apply to these models.</p>', "note"))
    parts.append('<h2>Smoke outputs and duplicate handling</h2>')
    parts.append('<p>Smoke directories are inventoried but not repeatedly rendered. Hash-identical images/videos are displayed once and every duplicate path remains in the catalogs.</p>')
    parts.append(table_html(smoke_df, small=True))

    parts.append(section("Executive conclusions", "executive", "THE RESULTS IN ONE VIEW"))
    parts.append('<div class="conclusion-grid">')
    parts.append(callout("B3.2 is the best aggregate 50k policy", '<p>With all policy/environment CBF execution removed, B3.2 has the highest return (90.39), greatest mean distance (433 m), and lowest pooled collision rate (2.156/km) among the matched B2/B3 policies.</p>'))
    parts.append(callout("B3.2 is still unsafe", '<p>It collides in 27/30 true-CBF-free episodes and completes only 3/30. The 50k run is promising evidence of learning, not a deployable controller.</p>', "warning"))
    parts.append(callout("The differentiable mean term has a measurable effect", '<p>On the clean B3.1→B3.2 probe, required external correction falls by 0.101 m/s² (95% bootstrap CI −0.158 to −0.049). Paired evaluation adds 43.54 return and 84 m distance on average.</p>'))
    parts.append(callout("Alignment is not tactical competence", '<p>B3.2 completes 0/6 targeted overtakes. B2.3 and B3.3 complete 4/6. A policy can better internalize a CBF projection while learning a conservative, stalled, or poorly timed passing strategy.</p>', "warning"))
    parts.append(callout("The external CBF carries substantial load", '<p>B2 intervention rates are about 31–47%; B3 extra-external intervention is about 20%. QP failures remain 13–21%, jerk rises sharply for B2, and minimum h remains negative.</p>'))
    parts.append(callout("The social traffic guard remains a major confound", '<p>All CBF-free and closed-loop results preserve the ordinary traffic guard. The short traces log hundreds to more than a thousand guard brake/yield events per episode across social vehicles, so traffic still adapts to avoid contact.</p>', "warning"))
    parts.append('</div>')
    parts.append('<h2>Interpretive hierarchy</h2>')
    hierarchy = pd.DataFrame([
        {"question": "Which saved actor aligns best to the CBF on identical states?", "best evidence": "120-state/87-feasible common probe", "answer": "B3.2 improves over B3.1 and B2.2; effect is modest and state-distribution dependent."},
        {"question": "Which matched policy performs best without any CBF execution?", "best evidence": "30 true-CBF-free episodes per policy", "answer": "B3.2, but with 90% collision episodes."},
        {"question": "Does the external CBF ensure safety?", "best evidence": "paired ON/OFF tables", "answer": "No; it improves many B2 outcomes but retains failures, collisions, negative h, and high intervention."},
        {"question": "Which policy overtakes best in targeted scenes?", "best evidence": "42 scripted overtake branches", "answer": "B2.3 and B3.3 complete 4/6; B3.2 completes 0/6."},
        {"question": "Can these results support a seed-level claim?", "best evidence": "training provenance", "answer": "No; all seven models use one training seed (307)."},
    ])
    parts.append(table_html(hierarchy, small=True))

    parts.append(section("Policy formulations and saved checkpoints", "formulation", "WHAT EACH LABEL MEANS"))
    parts.append('<p>The seven policies form nominal, non-differentiable, and differentiable branches. B3.1→B3.2 is the cleanest one-scalar causal comparison: both share the projected-policy architecture and reward, while B3.2 alone enables the explicit mean-alignment actor term.</p>')
    parts.append(table_html(models, small=True))
    parts.append(callout("Checkpoint completeness", '<p>Every policy has a final 50k model and complete training log. Only B1 has an archived intermediate <code>rollout_50000_steps.zip</code>; the six CBF-variant checkpoint directories were empty in the source studies.</p>', "note"))

    parts.append(section("Training and optimization results", "training", "WHAT WAS LEARNED BY 50K"))
    parts.append('<h2>Tail training summary</h2>')
    parts.append(table_html(format_df(training, {
        "tail reward": ("f2", 1), "tail length": ("f1", 1), "tail return/step": ("f3", 1),
        "tail CBF correction": ("f3", 1), "tail CBF loss": ("f3", 1), "tail infeasible": ("pct", 100),
        "mean |gCBF|/|gPPO|": ("f3", 1), "mean gradient cosine": ("f3", 1),
    }), small=True))
    parts.append(callout("Training winner", '<p>B3.2 ends with the strongest smoothed rollout reward and longest episodes. Relative to B3.1, its tail CBF correction is 0.113 versus 0.233 and its tail raw-to-projected loss is 0.169 versus 0.510.</p>'))
    parts.append(callout("Not fully converged", '<p>Several curves are still moving at 50k. B3.3 rises late, B2.2 changes late, and completed-episode curves remain noisy because shielded runs finish only 37–59 episodes. A 50k ranking should be treated as pilot selection, not final convergence.</p>', "warning"))
    parts.append(callout("Instrumentation caveats", '<p>The archived rollout plots contain zero-valued “distinct collision events” and “collision-active timesteps” series even though per-episode logs contain collisions. B1 also reports zero action saturation because its older logger did not emit the same fields. The per-episode CSV is the safer source for collision interpretation.</p>', "warning"))

    parts.append(section("True CBF-free and external-CBF evaluation", "evaluation", "WHAT THE POLICIES DO AT EXECUTION"))
    parts.append('<h2>True CBF-free primary outcomes</h2>')
    true_display = format_df(true_primary, {
        "task m": ("int", 1), "physics Hz": ("int", 1), "policy Hz": ("int", 1), "return": ("f2", 1),
        "distance m": ("f1", 1), "collision episodes": ("pct", 100), "collisions/km": ("f3", 1),
        "completion": ("pct", 100), "speed error m/s": ("f2", 1), "lateral error m": ("f2", 1), "jerk norm": ("f3", 1),
    })
    parts.append(table_html(true_display, small=True))
    parts.append('<p class="note-line"><strong>B1 warning:</strong> its 380 m/20 Hz/30D protocol makes its apparent completion and collision-episode rate incomparable to the 1,000 m/100 Hz/32D B2/B3 set.</p>')
    parts.append(callout("B3.2 leads aggregate task performance", '<p>Within the matched B2/B3 protocol, B3.2 has the highest return and distance and the lowest collisions/km. B3.1 is the smoothest and has the best lateral tracking, while B2.2 has the smallest speed error—but both collide in nearly every episode.</p>'))
    parts.append(callout("No standalone policy is safe", '<p>B2.1 and B2.2 collide in all 30 episodes. B2.3 and B3.2 collide in 27/30. B3.1 and B3.3 collide in 29/30. This dominates any modest ranking differences.</p>', "warning"))

    parts.append('<h2>Paired same-scenario differences</h2>')
    parts.append(table_html(format_df(paired_eval, {
        "Δ return": ("f2", 1), "Δ distance m": ("f1", 1), "Δ completion": ("pct", 100),
        "Δ collision episode rate": ("pct", 100),
    }), small=True))
    parts.append('<p>The B3.1→B3.2 row is the strongest evaluation evidence for the explicit differentiable mean term: +43.54 return, +84 m, +6.7 percentage points completion, and −6.7 points collision-episode rate. With N=30 and one training seed, this is encouraging but not definitive.</p>')

    parts.append('<h2>External CBF ON/OFF reference</h2>')
    parts.append(table_html(format_df(external, {
        "raw coll./km": ("f3", 1), "CBF coll./km": ("f3", 1), "raw completion": ("pct", 100),
        "CBF completion": ("pct", 100), "CBF intervention": ("pct", 100), "CBF QP fail": ("pct", 100),
        "raw jerk": ("f3", 1), "CBF jerk": ("f3", 1), "CBF minimum h": ("f3", 1),
    }), small=True))
    parts.append(callout("B2 gains are real but expensive", '<p>For B2.1/B2.2, external CBF cuts collisions/km by roughly three quarters and raises completion from 0–3% to 33–37%. It also intervenes around one third of the time, fails its QP in 13–15% of steps, and increases jerk from below 0.9 to about 3.6–3.7.</p>'))
    parts.append(callout("B3 “raw” is not truly raw in the old table", '<p>The old external-OFF B3 mode retains the differentiable policy’s internal projected mean. That is why B3.2 reports return 180.6 there but only 90.4 when projection is actually removed. Use the true-CBF-free table to judge the learned neural mean.</p>', "warning"))
    parts.append(callout("Negative h and QP failures invalidate a blanket guarantee claim", '<p>Every external-CBF row still has negative minimum h, and QP failure is nonzero. These saved results demonstrate a practical safety filter with residual failure modes—not an empirically maintained hard invariant.</p>', "warning"))

    secondary = pd.read_csv(FINAL / "eval/postprocessed_true_cbf_free/secondary_cbf_diagnostics.csv")
    secondary = secondary[["comparison_label", "minimum_h_mean", "minimum_h_sd", "h_dot_mean", "neighbor_count_mean"]]
    secondary.columns = ["policy", "mean minimum h", "SD minimum h", "mean h-dot", "mean neighbors"]
    parts.append('<h2>Secondary geometric diagnostics</h2>')
    parts.append(table_html(format_df(secondary, {
        "mean minimum h": ("f3", 1), "SD minimum h": ("f3", 1), "mean h-dot": ("f2", 1), "mean neighbors": ("f1", 1),
    }), small=True))
    parts.append('<p>These are geometric proximity diagnostics in a CBF-free run, not intervention metrics. Collision causality was not logged; no collision episode was deleted during post-processing.</p>')

    parts.append('<h2>Twenty-one rendered true-CBF-free examples</h2>')
    parts.append(table_html(format_df(rendered_examples, {
        "return": ("f2", 1), "distance m": ("f1", 1), "task m": ("int", 1),
    }), small=True))
    parts.append(callout("Video examples expose scenario variance", '<p>All 18 rendered B2/B3 examples collide. Yet B3.2 travels 986 m in seed 1100003 before collision, while B3.3 travels 848 m in seed 1100002. A single showcase video can therefore reverse the apparent ranking; the full evaluation table must remain primary.</p>', "warning"))

    parts.append(section("Learned-policy and differentiability analyses", "policy", "HOW THE ACTOR ITSELF CHANGED"))
    parts.append('<h2>Common 120-state probe</h2>')
    parts.append(table_html(format_df(probe, {
        "raw feasible": ("pct", 100), "mean correction m/s²": ("f3", 1), "P90 correction m/s²": ("f3", 1),
        "external intervention": ("pct", 100), "mean max constraint value": ("f3", 1),
    }), small=True))
    parts.append('<h2>Clean B3.1 → B3.2 paired effect</h2>')
    parts.append(table_html(format_df(b3_paired, {
        "B3.1": ("f3", 1), "B3.2": ("f3", 1), "difference": ("f3", 1), "95% CI low": ("f3", 1), "95% CI high": ("f3", 1),
    }), small=True))
    parts.append(callout("Internalization is measurable, but not complete", '<p>B3.2 reduces mean required correction from 0.698 to 0.598 m/s² and external intervention from 34.5% to 33.3%. Raw feasibility rises only 1.1 percentage points, so the term moves unsafe actions closer to the feasible set more than it eliminates unsafe proposals.</p>'))

    parts.append('<h2>Differentiable versus non-differentiable method pairs</h2>')
    parts.append(table_html(format_df(dvnd, {
        "non-diff correction": ("f3", 1), "diff correction": ("f3", 1), "difference": ("f3", 1),
        "relative change %": ("f1", 1), "95% CI low": ("f3", 1), "95% CI high": ("f3", 1),
    }), small=True))
    parts.append(callout("Differentiability alone is not uniformly beneficial", '<p>Reward-only B3.1 needs 8.8% more correction than B2.1 despite a slightly higher feasible rate. The strongest method-level gain is B2.2→B3.2: correction falls 24.6% and raw feasibility rises 5.7 points. Actor-only B3.3 improves correction by 10% but leaves feasibility unchanged.</p>'))

    risk = pd.read_csv(FINAL / "policy_analysis/b3_risk_conditioned_summary.csv")
    risk = risk[["risk_bin", "n_states", "mean_min_h", "B3_1_mean_correction", "B3_2_mean_correction", "B3_1_raw_feasible_rate", "B3_2_raw_feasible_rate"]]
    risk.columns = ["risk bin", "N", "mean min h", "B3.1 correction", "B3.2 correction", "B3.1 feasible", "B3.2 feasible"]
    parts.append('<h2>Risk-conditioned B3 comparison</h2>')
    parts.append(table_html(format_df(risk, {
        "mean min h": ("f3", 1), "B3.1 correction": ("f3", 1), "B3.2 correction": ("f3", 1),
        "B3.1 feasible": ("pct", 100), "B3.2 feasible": ("pct", 100),
    }), small=True))
    parts.append(callout("The dangerous tail remains difficult", '<p>In the lowest-h quartile, both B3 policies are feasible on only 45.5% of states and correction remains above 1.0 m/s². Most of B3.2’s feasibility gain appears in a middle-risk quartile, not the worst-risk tail.</p>', "warning"))
    parts.append(callout("State-bank choice changes the apparent effect", '<p>On the broader 3,697-state visualization bank—dominated by easier states—B3.2 and B3.1 have nearly identical normalized correction (0.0600 vs 0.0625) and raw feasibility (89.4% vs 89.8%). The targeted 87-feasible-state bank exposes a clearer correction benefit. Both are valid answers to different state distributions.</p>', "note"))
    parts.append('<h2>Policy-map interpretation</h2>')
    parts.append('<ul><li>B3.2 is visibly less longitudinally aggressive than B2.2 over substantial closing-traffic regions, while its lateral deformation is state dependent rather than a simple left/right bias.</li><li>B3 operational actions differ from their raw means exactly where the internal projection activates; this must not be mistaken for raw neural safety.</li><li>The feature-sensitivity maps show B3.2 strongly uses ego lateral position, previous action, and several neighbor channels. B1’s older 30D observation and missing previous-action input make its sensitivity pattern structurally incomparable.</li><li>Large deformation magnitude does not imply improved task behavior: the overtaking section shows B3.2 can deform into hesitation or stalling.</li></ul>')

    parts.append(section("Closed-loop, raw traces, and time-series behavior", "closedloop", "HOW SMALL POLICY DIFFERENCES COMPOUND"))
    parts.append('<h2>Visualization closed-loop protocol</h2>')
    parts.append(table_html(format_df(closed, {
        "mean steps": ("f1", 1), "worst clearance m": ("f3", 1), "mean shadow correction": ("f3", 1),
        "shadow intervention": ("pct", 100), "mean internal shift": ("f3", 1), "mean guard events": ("f1", 1),
    }), small=True))
    parts.append(callout("B2.2 versus B3.2 short-scenario contrast", '<p>B2.2 collides in all three matched branches; B3.2 collides in one. B3.2’s internal operational projection also removes the need for the extra shadow external projection in these traces, but its raw-to-operational shift remains substantial.</p>'))
    parts.append(callout("All policies enter dangerous geometry", '<p>Except for one B2.3 branch, every policy’s worst clearance is negative in the three-seed set. Operational HOCBF margins repeatedly cross below zero, and TTC frequently approaches zero.</p>', "warning"))
    parts.append(callout("Traffic guard activity is pervasive", '<p>The saved counter aggregates interventions across many social vehicles, so values can exceed policy steps. Means range around 600–1,200 events per branch. This is direct evidence that the ordinary traffic guard materially shapes the interaction even when the ego CBF is removed.</p>', "warning"))

    parts.append('<h2>Strict raw-policy trace protocol</h2>')
    parts.append(table_html(format_df(raw_closed, {
        "mean steps": ("f1", 1), "worst h": ("f3", 1), "worst clearance m": ("f3", 1),
        "mean raw HOCBF violation": ("pct", 100),
    }), small=True))
    parts.append('<p>The raw-trace package is a different branch from the visualization closed loop: it disables policy/physics CBF and measures the raw learned command. It therefore should not be numerically merged with the B3 operational-mean visualization traces.</p>')
    parts.append(callout("Raw HOCBF violations remain frequent", '<p>Mean violation fractions range from roughly 20% to 44%, with negative minimum h for nearly every branch. B3.2 survives all three strict raw trace scenarios, but the formal 30-episode evaluation shows that this small scenario set is not representative enough for a safety claim.</p>'))
    parts.append('<h2>Time-series interpretation</h2>')
    parts.append('<ul><li>Event alignment shows that danger is followed by large, sometimes saturated operational actions; internal projection often clips to ±3 m/s².</li><li>Critical vehicle identity changes repeatedly, so “the obstacle” is not a single persistent lead vehicle. Safety logic must handle active-constraint switching.</li><li>Progress-aligned and world/ego-frame views separate timing from distance: policies that survive longer do not necessarily make faster progress.</li><li>Barcodes make the coupling visible: negative margins, low clearance, guard activity, and active-neighbor switches occur in overlapping bursts.</li></ul>')

    parts.append(section("Overtake timing, freeze frames, and tactical behavior", "overtake", "WHEN—AND WHETHER—THE POLICY COMMITS"))
    parts.append('<h2>Expanded targeted-scene summary</h2>')
    parts.append(table_html(format_df(overtake, {
        "median intent delay s": ("f1", 1), "median clear delay s": ("f1", 1), "mean distance m": ("f1", 1),
        "worst operational margin": ("f3", 1), "mean guard brakes": ("f1", 1), "completion rate": ("pct", 100),
    }), small=True))
    parts.append(callout("The aggregate winner does not overtake", '<p>B3.2 attempts maneuvers but completes 0/6. It travels only about 104–126 m in the targeted branches, often displacing laterally without clearing the blocker. This looks like hesitation/slowdown rather than a confident pass.</p>', "warning"))
    parts.append(callout("B2.3 and B3.3 are the strongest targeted overtakers", '<p>Both complete 4/6. B2.3 often commits immediately; B3.3 also completes quickly in successful scenes. Their better overtake completion does not translate to better formal collision performance.</p>'))
    parts.append(callout("CBF alignment and overtake success are orthogonal", '<p>B2.2 sometimes has a positive raw HOCBF margin yet fails to clear the blocker. B3.2’s internal operational margin stays near zero while it still fails tactically. The CBF constrains feasible motion; it does not teach when, where, or how to overtake.</p>'))
    parts.append('<h2>How to read the freeze frames and HUD videos</h2>')
    parts.append('<p>Columns mark opportunity, raw intent, operational intent, lateral commitment, abeam, and clear. The red star is the designated blocker; each colored diamond is the ego for one policy. The videos add a live action-space panel: dots are raw proposals, diamonds are operational actions, crosses are shadow external-CBF projections, and the star is the executed action.</p>')

    parts.append(section("Complete canonical figure atlas", "figures", "EVERY UNIQUE GENERATED PNG"))
    parts.append(f'<p>This atlas displays all {len(unique_pngs)} hash-unique non-smoke PNG results. Hash-identical duplicates are listed once below and retained in the artifact catalog.</p>')
    for title, description, paths, cols, page_each in figure_groups(unique_pngs):
        if not paths:
            continue
        parts.append(f'<div class="subsection-title pagebreak"><h2>{esc(title)}</h2><p>{esc(description)}</p></div>')
        parts.append(gallery_html(paths, columns=cols, page_each=page_each))

    parts.append('<div class="pagebreak"><h2>Duplicate PNG paths</h2>')
    dup_png_df = pd.DataFrame([{"canonical": g[0], "duplicate": " | ".join(g[1:])} for g in duplicate_pngs])
    parts.append(table_html(dup_png_df, small=True) if not dup_png_df.empty else '<p>None.</p>')
    parts.append('</div>')

    parts.append(section("Complete video poster atlas and catalog", "videos", "EVERY UNIQUE GENERATED MP4"))
    parts.append(f'<p>The PDF cannot play MP4s, so every one of the {len(video_records)} unique videos is represented by a frame sampled at approximately 55% of its duration. Clicking a poster in the HTML/PDF targets the local MP4. The four hash-identical duplicate overtake videos are listed on their canonical cards and in the catalog.</p>')
    category_counts = video_table.groupby("category", as_index=False).agg(videos=("path", "count"), total_minutes=("duration_s", lambda s: s.sum() / 60), size_mb=("size_mb", "sum"))
    parts.append(table_html(format_df(category_counts, {"total_minutes": ("f1", 1), "size_mb": ("f2", 1)}), small=True))
    parts.append(video_cards_html(video_records, chunk_size=4))
    parts.append('<div class="pagebreak"><h2>Full video metadata catalog</h2>')
    video_display = video_table[["category", "path", "duplicates", "duration_s", "fps", "resolution", "frames", "size_mb", "sha256"]].copy()
    video_display["sha256"] = video_display["sha256"].str.slice(0, 16)
    parts.append(table_html(format_df(video_display, {"duration_s": ("f1", 1), "fps": ("f1", 1), "size_mb": ("f2", 1)}), small=True))
    parts.append('</div>')

    parts.append(section("Other 50k pilot tables outside the canonical seven", "ancillary", "LEGACY AND FOLLOW-UP CONTEXT"))
    parts.append('<p>The repository also contains formal KPI tables for earlier progression runs, nominal observation/reward pilots, a HOCBF-margin follow-up, and a no-safety-reward follow-up. They are displayed here for completeness but are not merged with the canonical comparison because episode counts, task definitions, clocks, observations, traffic mix, policy projection, and logging schemas differ.</p>')
    parts.append(table_html(format_df(ancillary, {
        "return": ("f2", 1), "collisions/km": ("f3", 1), "completion": ("pct", 100),
        "minimum h": ("f3", 1), "intervention": ("pct", 100), "QP fail": ("pct", 100),
    }), small=True))
    parts.append(callout("Legacy tables can look safer for the wrong reason", '<p>The old projected variants report identical “raw” and CBF results near 0.6 collisions/km because their internal projection remains active. This is exactly why the later true-CBF-free reevaluation was necessary.</p>', "warning"))
    parts.append(callout("No-safety-reward follow-up is interesting but separate", '<p>Its raw row reports 1.139 collisions/km and 46.7% completion, while adding the external CBF reports 1.731 collisions/km, 26.7% completion, and 77.9% intervention. That reversal suggests filter/policy conflict or a protocol-specific effect and deserves a controlled rerun, not cross-table ranking.</p>'))
    parts.append(callout("Duplicate nominal study", '<p><code>tynd50k_s307</code> and <code>ppo_nominal_target_y_no_dims_wy2_50k_seed307</code> have identical displayed KPIs and appear to be duplicate/copy outputs. They should not be counted as independent evidence.</p>', "note"))

    parts.append(section("Complete data-artifact catalog", "catalog", "EVERY CANONICAL CSV AND FILE FAMILY"))
    parts.append('<p>Summary tables are printed in the analytical sections above. Large raw traces—millions of state/action/proposal rows—are not pasted row-by-row into the PDF; every file is nevertheless listed with row count, column count, schema preview, and size so nothing is hidden.</p>')
    parts.append('<h2>All canonical CSV files</h2>')
    parts.append(table_html(format_df(csv_catalog, {"size_mb": ("f2", 1)}), small=True))
    parts.append('<h2>All canonical non-CSV files</h2>')
    non_csv = all_files[all_files["extension"] != ".csv"].copy()
    parts.append(table_html(format_df(non_csv, {"size_mb": ("f2", 1)}), small=True))

    parts.append(section("Overall assessment and recommended next experiments", "judgment", "WHAT I THINK THE RESULTS MEAN"))
    parts.append('<h2>Scientific assessment</h2>')
    parts.append('<p><strong>The differentiable gradient term is doing something real.</strong> B3.2 is not merely shielded better at runtime: its raw actor requires less correction on identical feasible states, wins the clean paired evaluation against B3.1, and changes its action map in systematic regions of the traffic state space. That is the strongest positive conclusion supported by the current evidence.</p>')
    parts.append('<p><strong>But the learned policies remain fundamentally unsafe at 50k.</strong> Ninety to one hundred percent collision-episode rates across the matched B2/B3 CBF-free evaluation are too large to explain away as noise or a few bad resets. The agent has not learned a robust collision-avoidance/tactical-driving policy.</p>')
    parts.append('<p><strong>The external CBF is not yet a hard empirical safety guarantee.</strong> High intervention, nonzero QP failure, negative minimum h, persistent collisions, and large jerk indicate feasibility/model/discretization issues. The filter improves outcomes, particularly for B2, but does not close the safety case.</p>')
    parts.append('<p><strong>The traffic guard is still part of the behavior.</strong> Because social vehicles brake/yield extensively, even the “CBF-free” ego is not being tested against fully ego-independent traffic. The current results answer “how does the ego behave with its CBF removed while guarded traffic remains?”—not “is the ego safe in open-loop independent traffic?”</p>')
    parts.append('<p><strong>Tactical competence is the missing dimension.</strong> B3.2’s poor targeted overtaking shows that CBF internalization can trade into hesitation, slowdown, or bad maneuver timing. Safety alignment, progress, comfort, and tactical success must be evaluated as separate axes.</p>')

    parts.append('<h2>Recommended next experiment stack</h2>')
    recommendations = pd.DataFrame([
        {"priority": "1", "experiment": "Protocol repair and provenance", "design": "Fix the stale 200-episode label; log collision partner, relative state, initiating actor, QP status, active constraint, and guard intervention by vehicle.", "decision unlocked": "Trustworthy safety attribution and publication-ready tables."},
        {"priority": "2", "experiment": "Matched multi-seed rerun", "design": "At least 5 training seeds for B3.1/B3.2 and B2.2; one shared 100 Hz physics / 20 Hz policy / 20-worker protocol; same task distance and observation.", "decision unlocked": "Whether the differentiable effect generalizes beyond seed 307."},
        {"priority": "3", "experiment": "Three execution layers", "design": "Evaluate raw neural mean, internal projected mean, and external substep CBF separately on identical scenario seeds.", "decision unlocked": "Where safety actually comes from."},
        {"priority": "4", "experiment": "Traffic-guard causal test", "design": "Keep social-social protection; remove social reaction to ego; add paired safe-controller counterfactual and collision provenance.", "decision unlocked": "How much behavior depends on traffic yielding to ego."},
        {"priority": "5", "experiment": "Targeted maneuver suite", "design": "Scale overtakes to ≥50 randomized blockers per scene family; score opportunity, intent, commit, clear, abort, collision, progress, and comfort.", "decision unlocked": "Whether policy improvements transfer to tactical competence."},
        {"priority": "6", "experiment": "CBF feasibility/discretization audit", "design": "Log continuous-time residual at every 100 Hz substep, initial-safe-set status, slack, active rows, and post-step h; sweep control frequency and buffer.", "decision unlocked": "Why negative h and QP failures persist."},
    ])
    parts.append(table_html(recommendations, small=True))
    parts.append(callout("Bottom line", '<p><strong>B3.2 is the correct 50k candidate to carry forward, not the correct policy to declare safe.</strong> The next claim to test is: “the explicit differentiable mean loss improves raw-action safety alignment and sample efficiency across seeds under one repaired protocol.”</p>'))

    style = """
    <style>
      @page { size: A4 landscape; margin: 12mm 12mm 15mm 12mm; }
      :root { --ink:#12202c; --muted:#596a78; --navy:#173a5e; --teal:#117c80; --gold:#d39a2c; --red:#b43b3b; --paper:#ffffff; --pale:#eef4f7; }
      * { box-sizing: border-box; }
      body { margin:0; color:var(--ink); background:var(--paper); font-family: "Segoe UI", Arial, sans-serif; font-size:10.2pt; line-height:1.42; }
      a { color:#075a9c; text-decoration:none; }
      code { font-family:Consolas, monospace; background:#edf1f3; padding:1px 4px; border-radius:3px; }
      h1 { color:var(--navy); font-size:27pt; line-height:1.08; margin:0 0 8mm; letter-spacing:-0.4px; }
      h2 { color:var(--navy); font-size:16pt; margin:8mm 0 3mm; }
      h3 { color:var(--teal); font-size:12pt; margin:5mm 0 2mm; }
      p { margin:0 0 3mm; }
      ul { margin:2mm 0 4mm 6mm; }
      li { margin-bottom:1.4mm; }
      .pagebreak { break-before: page; page-break-before: always; }
      .section { padding-top:2mm; }
      .kicker { color:var(--teal); font-weight:700; font-size:8.5pt; letter-spacing:1.3px; margin-bottom:2mm; }
      .kicker.light { color:#99d5d7; }
      .cover { width:100%; min-height:180mm; padding:12mm 14mm; background:linear-gradient(128deg,#0f2538 0%,#173a5e 58%,#117c80 100%); color:white; break-after:page; }
      .cover h1 { color:white; font-size:38pt; margin:4mm 0 3mm; }
      .cover h2 { color:#d7ecee; font-weight:400; font-size:16pt; margin:0 0 5mm; }
      .cover-summary { width:85%; font-size:12pt; color:#e6f2f4; }
      .cover-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:5mm; margin-top:8mm; }
      .cover-grid img { width:100%; height:78mm; object-fit:cover; object-position:center; border:1px solid rgba(255,255,255,.35); border-radius:5px; background:white; }
      .cover-date { margin-top:6mm; color:#c7dce2; font-size:9pt; }
      .toc { padding:4mm 10mm; }
      .toc ol { columns:2; column-gap:18mm; padding-left:8mm; }
      .toc li { font-size:12pt; margin:0 0 5mm; break-inside:avoid; }
      .callout { border-left:5px solid var(--teal); background:#eaf5f5; padding:4mm 5mm; margin:4mm 0; break-inside:avoid; border-radius:0 5px 5px 0; }
      .callout strong { display:block; color:var(--navy); font-size:11pt; margin-bottom:1mm; }
      .callout.warning { border-left-color:var(--red); background:#faeded; }
      .callout.note { border-left-color:var(--gold); background:#fbf5e7; }
      .conclusion-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:4mm; }
      .conclusion-grid .callout { margin:0; }
      .note-line { background:#fbf5e7; border:1px solid #e9d69f; padding:3mm; border-radius:4px; }
      table.data { width:100%; border-collapse:collapse; margin:3mm 0 5mm; font-size:8.1pt; break-inside:auto; }
      table.data.compact { font-size:7.2pt; }
      table.data thead { display:table-header-group; }
      table.data tr { break-inside:avoid; page-break-inside:avoid; }
      table.data th { color:white; background:var(--navy); text-align:left; padding:2mm 1.7mm; font-weight:600; }
      table.data td { padding:1.6mm 1.7mm; border-bottom:1px solid #d8e0e5; vertical-align:top; }
      table.data tr:nth-child(even) td { background:#f3f7f9; }
      .figure-page { margin:0; padding:2mm 0; break-before:page; page-break-before:always; text-align:center; }
      .figure-page img { max-width:100%; max-height:162mm; object-fit:contain; }
      figure figcaption { color:var(--muted); font-size:8.2pt; margin-top:2mm; text-align:left; }
      figure figcaption strong { color:var(--ink); }
      .gallery { display:grid; gap:5mm; align-items:start; }
      .gallery.cols-1 { grid-template-columns:1fr; }
      .gallery.cols-2 { grid-template-columns:repeat(2,1fr); }
      .figure-card { margin:0; padding:3mm; border:1px solid #d8e0e5; border-radius:5px; break-inside:avoid; background:#fff; }
      .figure-card img { width:100%; max-height:74mm; object-fit:contain; }
      .subsection-title { padding-top:2mm; }
      .video-grid { display:grid; grid-template-columns:repeat(2,1fr); grid-template-rows:repeat(2,1fr); gap:5mm; min-height:170mm; }
      .video-card { border:1px solid #cad5db; border-radius:5px; padding:3mm; break-inside:avoid; overflow:hidden; }
      .video-card img { width:100%; height:62mm; object-fit:contain; background:#111; }
      .video-title { font-family:Consolas, monospace; font-size:7.6pt; margin-top:2mm; overflow-wrap:anywhere; }
      .micro { color:var(--muted); font-size:7.2pt; }
      .poster-missing { height:62mm; display:flex; align-items:center; justify-content:center; background:#eee; }
    </style>
    """

    document = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>50k PPO–CBF Pilot Results</title>'
        + style
        + '</head><body>'
        + "".join(parts)
        + '</body></html>'
    )
    html_path = OUT / "50k_PPO_CBF_complete_results_report.html"
    html_path.write_text(document, encoding="utf-8")

    csv_catalog.to_csv(OUT / "canonical_csv_catalog.csv", index=False)
    video_table.drop(columns=["poster", "video"]).to_csv(OUT / "video_catalog.csv", index=False)
    all_files.to_csv(OUT / "canonical_file_catalog.csv", index=False)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "seven canonical 50k PPO pilots and every non-smoke result under artifacts/final_Results; 200k work excluded",
        "html": str(html_path),
        "expected_pdf": str(OUT / "50k_PPO_CBF_complete_results_report.pdf"),
        "counts": {
            "canonical_files": len(all_files),
            "canonical_csv_files": len(csv_catalog),
            "unique_png_figures": len(unique_pngs),
            "duplicate_png_groups": len(duplicate_pngs),
            "unique_videos": len(video_records),
            "duplicate_video_groups": len(duplicate_videos),
            "smoke_files_inventoried_not_repeated": len(smoke_files),
            "true_cbf_free_episode_rows": actual_rows,
            "true_cbf_free_episodes_per_variant": actual_per_variant,
        },
        "warnings": [
            "Top-level final_Results manifest/README says 200 CBF-free episodes per variant; actual metadata/CSV contains 30 per variant.",
            "B1 protocol differs from B2/B3.",
            "All policies use one training seed (307).",
            "Ordinary social-traffic guard remains enabled in CBF-free evaluations.",
            "Old B3 external-OFF mode retains internal policy projection and is not a true raw-actor evaluation.",
        ],
    }
    (OUT / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="Print the generated manifest JSON")
    args = parser.parse_args()
    manifest = build_report()
    if args.summary:
        print(json.dumps(manifest, indent=2))
    else:
        print(manifest["html"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
