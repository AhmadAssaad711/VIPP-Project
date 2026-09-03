"""Post-hoc congestion and safety diagnostics for structured DQN policies.

The functions in this module label lane-based rollouts using front/rear TTC,
lane gaps, traffic-flow context, action quality, and final-episode behavior.
They aggregate evidence for analysis and reporting; they do not alter the
training reward and must not be confused with adaptive_longitudinal.py reward
wrappers.

Inputs are normally an evaluated HighwayEnv environment plus action/episode
records. Outputs are scalar labels, row-level diagnostic fields, and
DataFrame/JSON summaries suitable for the structured congested-traffic
notebooks. Thresholds are kept in DEFAULT_DIAGNOSTIC_CONFIG so a report can
record exactly which diagnostic definition was used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pandas as pd


DEFAULT_DIAGNOSTIC_CONFIG: dict[str, float] = {
    "ttc_cap": 10.0,
    "front_ttc_safe": 4.0,
    "front_ttc_critical": 1.5,
    "rear_ttc_safe": 4.0,
    "rear_ttc_critical": 1.5,
    "lane_gap_safe": 12.0,
    "flow_radius": 90.0,
    "bad_action_margin": 0.35,
    "no_good_action_risk": 0.85,
    "wrong_lane_quality_margin": 0.18,
    "wrong_lane_lookback_steps": 6,
    "final_lookback_steps": 4,
}


def merge_diagnostic_config(config: dict[str, Any] | None = None) -> dict[str, float]:
    merged = dict(DEFAULT_DIAGNOSTIC_CONFIG)
    if config:
        merged.update(config)
    return {key: float(value) for key, value in merged.items()}


def action_label(env: gym.Env, action: Any) -> str:
    action_index = int(np.asarray(action).item())
    actions = getattr(env.unwrapped.action_type, "actions", {})
    if isinstance(actions, dict):
        return str(actions.get(action_index, action_index))
    if 0 <= action_index < len(actions):
        return str(actions[action_index])
    return str(action_index)


def available_actions(env: gym.Env) -> dict[str, int]:
    actions = getattr(env.unwrapped.action_type, "actions", {})
    if isinstance(actions, dict):
        return {str(label): int(index) for index, label in actions.items()}
    return {str(label): int(index) for index, label in enumerate(actions)}


def _forward_speed(vehicle) -> float:
    return float(vehicle.speed * np.cos(getattr(vehicle, "heading", 0.0)))


def _lane_longitudinal(vehicle, lane) -> float:
    longitudinal, _ = lane.local_coordinates(vehicle.position)
    return float(longitudinal)


def _vehicle_clearance(ego_vehicle, other_vehicle, ego_s: float, other_s: float) -> float:
    return max(
        0.0,
        abs(float(other_s) - float(ego_s))
        - 0.5 * float(getattr(ego_vehicle, "LENGTH", 0.0) + getattr(other_vehicle, "LENGTH", 0.0)),
    )


def _risk_from_ttc(ttc: float, safe_ttc: float, critical_ttc: float) -> float:
    if ttc <= critical_ttc:
        return 1.0
    if ttc >= safe_ttc:
        return 0.0
    return float((safe_ttc - ttc) / max(1e-6, safe_ttc - critical_ttc))


def _gap_risk(gap: float, safe_gap: float) -> float:
    if gap >= safe_gap:
        return 0.0
    return float((safe_gap - max(0.0, gap)) / max(1e-6, safe_gap))


def lane_indices(env: gym.Env) -> dict[str, tuple | None]:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if vehicle is None or road is None or getattr(vehicle, "lane_index", None) is None:
        return {"current": None, "left": None, "right": None}

    lane_from, lane_to, lane_id = vehicle.lane_index
    lane_count = len(road.network.graph[lane_from][lane_to])
    lane_id = int(lane_id)
    return {
        "current": vehicle.lane_index,
        "left": (lane_from, lane_to, lane_id - 1) if lane_id - 1 >= 0 else None,
        "right": (lane_from, lane_to, lane_id + 1) if lane_id + 1 < lane_count else None,
    }


def lane_metrics(env: gym.Env, lane_index: tuple | None, config: dict[str, float]) -> dict[str, float | bool]:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if vehicle is None or road is None or lane_index is None:
        return {
            "available": False,
            "front_ttc": 0.0,
            "rear_ttc": 0.0,
            "front_gap": 0.0,
            "rear_gap": 0.0,
            "front_risk": 1.0,
            "rear_risk": 1.0,
            "flow_speed": 0.0,
            "occupancy": 1.0,
            "lane_change_safe": False,
            "quality": -1.0,
        }

    lane = road.network.get_lane(lane_index)
    ego_s = _lane_longitudinal(vehicle, lane)
    front_vehicle, rear_vehicle = road.neighbour_vehicles(vehicle, lane_index)
    front_gap = float(config["ttc_cap"] * max(_forward_speed(vehicle), 1.0))
    rear_gap = front_gap
    front_ttc = float(config["ttc_cap"])
    rear_ttc = float(config["ttc_cap"])

    if front_vehicle is not None:
        front_s = _lane_longitudinal(front_vehicle, lane)
        front_gap = _vehicle_clearance(vehicle, front_vehicle, ego_s, front_s)
        closing_speed = _forward_speed(vehicle) - _forward_speed(front_vehicle)
        if closing_speed > 1e-6:
            front_ttc = 0.0 if front_gap <= 0.0 else front_gap / closing_speed
        front_ttc = float(np.clip(front_ttc, 0.0, config["ttc_cap"]))

    if rear_vehicle is not None:
        rear_s = _lane_longitudinal(rear_vehicle, lane)
        rear_gap = _vehicle_clearance(vehicle, rear_vehicle, ego_s, rear_s)
        rear_closing_speed = _forward_speed(rear_vehicle) - _forward_speed(vehicle)
        if rear_closing_speed > 1e-6:
            rear_ttc = 0.0 if rear_gap <= 0.0 else rear_gap / rear_closing_speed
        rear_ttc = float(np.clip(rear_ttc, 0.0, config["ttc_cap"]))

    nearby_speeds = []
    occupancy = 0
    for other in getattr(road, "vehicles", []):
        if other is vehicle or getattr(other, "lane_index", None) != lane_index:
            continue
        other_s = _lane_longitudinal(other, lane)
        if abs(other_s - ego_s) <= float(config["flow_radius"]):
            nearby_speeds.append(_forward_speed(other))
            occupancy += 1

    speed_limit = float(env.unwrapped.config.get("speed_limit", 30.0))
    flow_speed = float(np.mean(nearby_speeds)) if nearby_speeds else speed_limit
    front_risk = _risk_from_ttc(front_ttc, config["front_ttc_safe"], config["front_ttc_critical"])
    rear_risk = _risk_from_ttc(rear_ttc, config["rear_ttc_safe"], config["rear_ttc_critical"])
    front_gap_risk = _gap_risk(front_gap, config["lane_gap_safe"])
    rear_gap_risk = _gap_risk(rear_gap, config["lane_gap_safe"])
    lane_change_safe = bool(
        front_risk < 0.45
        and rear_risk < 0.45
        and front_gap_risk < 0.5
        and rear_gap_risk < 0.5
    )
    quality = (
        0.35 * (1.0 - front_risk)
        + 0.25 * (1.0 - rear_risk)
        + 0.25 * float(np.clip(flow_speed / max(speed_limit, 1.0), 0.0, 1.2))
        + 0.15 * (1.0 - min(front_gap_risk, 1.0))
        - 0.04 * float(occupancy)
    )
    return {
        "available": True,
        "front_ttc": float(front_ttc),
        "rear_ttc": float(rear_ttc),
        "front_gap": float(front_gap),
        "rear_gap": float(rear_gap),
        "front_risk": float(front_risk),
        "rear_risk": float(rear_risk),
        "flow_speed": float(flow_speed),
        "occupancy": float(occupancy),
        "lane_change_safe": bool(lane_change_safe),
        "quality": float(quality),
    }


def _action_risks(metrics: dict[str, dict[str, float | bool]], labels: dict[str, int]) -> dict[str, float]:
    current = metrics["current"]
    base = float(current["front_risk"]) + 0.7 * float(current["rear_risk"])
    risks: dict[str, float] = {}
    if "IDLE" in labels:
        risks["IDLE"] = base
    if "FASTER" in labels:
        risks["FASTER"] = base + 0.65 * float(current["front_risk"])
    if "SLOWER" in labels:
        risks["SLOWER"] = base + 0.85 * float(current["rear_risk"])
    for action_name, lane_key in (("LANE_LEFT", "left"), ("LANE_RIGHT", "right")):
        if action_name not in labels:
            continue
        lane = metrics[lane_key]
        if not bool(lane["available"]):
            risks[action_name] = 2.0
        else:
            risks[action_name] = (
                0.9 * float(lane["front_risk"])
                + 1.1 * float(lane["rear_risk"])
                + (0.0 if bool(lane["lane_change_safe"]) else 0.45)
            )
    return risks


def record_decision(env: gym.Env, action: Any, step: int, config: dict[str, float]) -> dict[str, Any]:
    lanes = lane_indices(env)
    metrics = {
        key: lane_metrics(env, lane_index, config)
        for key, lane_index in lanes.items()
    }
    labels = available_actions(env)
    risks = _action_risks(metrics, labels)
    chosen_label = action_label(env, action)
    chosen_risk = float(risks.get(chosen_label, np.nan))
    best_action = min(risks, key=risks.get) if risks else "UNKNOWN"
    best_risk = float(risks.get(best_action, np.nan))
    bad_action = bool(
        np.isfinite(chosen_risk)
        and np.isfinite(best_risk)
        and chosen_label != best_action
        and chosen_risk > best_risk + config["bad_action_margin"]
    )
    no_good_action = bool(np.isfinite(best_risk) and best_risk >= config["no_good_action_risk"])
    current_quality = float(metrics["current"]["quality"])
    adjacent_quality = {
        key: float(metrics[key]["quality"])
        for key in ("left", "right")
        if bool(metrics[key]["available"]) and bool(metrics[key]["lane_change_safe"])
    }
    best_lane_key = max(adjacent_quality, key=adjacent_quality.get) if adjacent_quality else "current"
    best_adjacent_quality = float(adjacent_quality.get(best_lane_key, current_quality))
    missed_better_lane = bool(
        best_lane_key != "current"
        and best_adjacent_quality > current_quality + config["wrong_lane_quality_margin"]
        and chosen_label != {"left": "LANE_LEFT", "right": "LANE_RIGHT"}.get(best_lane_key)
    )

    return {
        "step": int(step),
        "action": chosen_label,
        "best_action": best_action,
        "chosen_risk": chosen_risk,
        "best_risk": best_risk,
        "bad_action": bad_action,
        "no_good_discrete_action": no_good_action,
        "missed_better_lane": missed_better_lane,
        "best_lane": best_lane_key,
        "current_lane_quality": current_quality,
        "best_adjacent_lane_quality": best_adjacent_quality,
        "current_front_ttc": float(metrics["current"]["front_ttc"]),
        "current_rear_ttc": float(metrics["current"]["rear_ttc"]),
        "left_front_ttc": float(metrics["left"]["front_ttc"]),
        "left_rear_ttc": float(metrics["left"]["rear_ttc"]),
        "left_lane_change_safe": bool(metrics["left"]["lane_change_safe"]),
        "right_front_ttc": float(metrics["right"]["front_ttc"]),
        "right_rear_ttc": float(metrics["right"]["rear_ttc"]),
        "right_lane_change_safe": bool(metrics["right"]["lane_change_safe"]),
    }


def collision_type(env: gym.Env) -> str:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if vehicle is None or road is None or not bool(getattr(vehicle, "crashed", False)):
        return "none"

    nearest = None
    nearest_distance = float("inf")
    for other in getattr(road, "vehicles", []):
        if other is vehicle:
            continue
        distance = float(np.linalg.norm(np.asarray(other.position) - np.asarray(vehicle.position)))
        if distance < nearest_distance:
            nearest = other
            nearest_distance = distance
    if nearest is None:
        return "unknown"

    lane_index = getattr(vehicle, "lane_index", None)
    if lane_index is None:
        return "unknown"
    lane = road.network.get_lane(lane_index)
    ego_s, ego_lateral = lane.local_coordinates(vehicle.position)
    other_s, other_lateral = lane.local_coordinates(nearest.position)
    lateral_delta = abs(float(other_lateral - ego_lateral))
    if lateral_delta > 0.6 * float(getattr(vehicle, "WIDTH", 2.0)):
        return "side"
    return "front" if float(other_s - ego_s) >= 0.0 else "rear"


def summarize_episode(records: list[dict[str, Any]], env: gym.Env, total_reward: float, episode: int, config: dict[str, float]) -> dict[str, Any]:
    final_window = records[-int(config["final_lookback_steps"]) :] if records else []
    lane_window = records[-int(config["wrong_lane_lookback_steps"]) :] if records else []
    ctype = collision_type(env)
    agent_chose_badly = any(bool(row["bad_action"]) for row in final_window)
    no_good_discrete_action = bool(final_window) and all(bool(row["no_good_discrete_action"]) for row in final_window)
    wrong_lane_earlier = sum(bool(row["missed_better_lane"]) for row in lane_window) >= max(2, len(lane_window) // 3)
    min_rear_ttc = min((float(row["current_rear_ttc"]) for row in final_window), default=float(config["ttc_cap"]))
    min_front_ttc = min((float(row["current_front_ttc"]) for row in final_window), default=float(config["ttc_cap"]))
    rear_pressure = min_rear_ttc <= float(config["rear_ttc_critical"])
    unavoidable_rear_pressure = bool(
        rear_pressure
        and no_good_discrete_action
        and not agent_chose_badly
        and min_front_ttc > float(config["front_ttc_critical"])
    )

    return {
        "episode": int(episode),
        "reward": float(total_reward),
        "collision": ctype != "none",
        "collision_type": ctype,
        "agent_chose_badly": bool(agent_chose_badly),
        "no_good_discrete_action": bool(no_good_discrete_action),
        "wrong_lane_earlier": bool(wrong_lane_earlier),
        "unavoidable_rear_pressure": bool(unavoidable_rear_pressure),
        "min_final_front_ttc": float(min_front_ttc),
        "min_final_rear_ttc": float(min_rear_ttc),
        "bad_action_count": int(sum(bool(row["bad_action"]) for row in records)),
        "no_good_action_count": int(sum(bool(row["no_good_discrete_action"]) for row in records)),
        "missed_better_lane_count": int(sum(bool(row["missed_better_lane"]) for row in records)),
        "steps": int(len(records)),
    }


def evaluate_congestion_diagnostics(
    model,
    make_env: Callable[..., gym.Env],
    *,
    env_config: dict[str, Any],
    episodes: int,
    seed: int,
    diagnostic_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[list[dict[str, Any]]]]:
    config = merge_diagnostic_config(diagnostic_config)
    summaries: list[dict[str, Any]] = []
    traces: list[list[dict[str, Any]]] = []

    for episode_idx in range(int(episodes)):
        env = make_env(render_mode=None, config=env_config)
        records: list[dict[str, Any]] = []
        try:
            obs, _ = env.reset(seed=int(seed) + episode_idx)
            terminated = False
            truncated = False
            total_reward = 0.0
            step = 0
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                records.append(record_decision(env, action, step, config))
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                step += 1
            summaries.append(summarize_episode(records, env, total_reward, episode_idx + 1, config))
            traces.append(records)
        finally:
            env.close()
    return pd.DataFrame(summaries), traces


def save_congestion_diagnostics(
    summary_df: pd.DataFrame,
    traces: list[list[dict[str, Any]]],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "congestion_diagnostic_summary.json"
    traces_path = output_dir / "congestion_diagnostic_traces.json"
    summary_df.to_json(summary_path, orient="records", indent=2)
    traces_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    return {
        "summary_path": str(summary_path),
        "traces_path": str(traces_path),
    }
