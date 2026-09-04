"""Optional structured HighwayEnv reward, observation, and control wrappers.

The laned experiments use native highway-v0 as their environment and compose
these wrappers around it from notebooks/_shared/dqn_notebook_utils. The
module keeps the structured research terms in one importable place:

* adaptive longitudinal target-speed control driven by front/rear TTC;
* rear-flow pressure and traffic-flow reward signals;
* TTC safety and potential-field reward extensions;
* driver-aggressiveness and TTC observation features; and
* lane-change safety penalties and diagnostics.

Every feature is disabled in the default configuration. The wrappers are
therefore intended for controlled ablations: enable one term or an explicitly
named combination, preserve the resulting configuration in the run manifest,
and compare only runs with the same environment and evaluation protocol.
This module belongs to the structured track. Its optional potential field
operates on HighwayEnv lane geometry and lane-based actions.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import gymnasium as gym
import highway_env  # noqa: F401 - register highway-v0
import numpy as np
from highway_env import utils


ADAPTIVE_LONGITUDINAL_CONFIG_KEY = "adaptive_longitudinal"
REAR_FLOW_CONFIG_KEY = "rear_flow"
TRAFFIC_FLOW_REWARD_CONFIG_KEY = "traffic_flow_reward"
SAFETY_TTC_FLOW_REWARD_CONFIG_KEY = "safety_ttc_flow_reward"
POTENTIAL_FIELD_REWARD_CONFIG_KEY = "potential_field_reward"
DRIVER_AGGRESSIVENESS_CONFIG_KEY = "driver_aggressiveness"
DRIVER_AGGRESSIVENESS_OBSERVATION_CONFIG_KEY = "driver_aggressiveness_observation"
TTC_OBSERVATION_CONFIG_KEY = "ttc_observation"
LANE_CHANGE_SAFETY_CONFIG_KEY = "lane_change_safety"

DEFAULT_ADAPTIVE_LONGITUDINAL_CONFIG: dict[str, Any] = {
    "enabled": False,
    "mode": "delta",
    "ttc_midpoint": 4.0,
    "ttc_temperature": 1.0,
    "ttc_cap": 10.0,
    "safety_ttc_threshold": 1.0,
    "unsafe_action_penalty": 1.0,
    "min_target_speed": 10.0,
    "max_target_speed": 35.0,
    "faster_max_delta": 1.25,
    "slower_min_delta": 1.25,
    "slower_max_delta": 2.5,
    "cruise_speed": 28.0,
    "action_speed_delta": 3.0,
}

DEFAULT_REAR_FLOW_CONFIG: dict[str, Any] = {
    "enabled": False,
    "spawn_on_reset": True,
    "spawn_during_episode": True,
    "vehicles_per_lane": 1,
    "lanes": "ego_and_adjacent",
    "distance_range": [25.0, 70.0],
    "speed_offset_range": [2.0, 6.0],
    "absolute_speed_range": [23.0, 34.0],
    "min_lane_gap": 18.0,
    "spawn_probability": 0.35,
    "cooldown_policy_steps": 3,
    "min_ego_progress": 80.0,
    "max_extra_vehicles": 12,
}

DEFAULT_TRAFFIC_FLOW_REWARD_CONFIG: dict[str, Any] = {
    "enabled": False,
    "penalty_weight": 0.12,
    "speed_tolerance": 2.0,
    "max_penalty": 0.8,
    "front_ttc_safe": 4.0,
    "rear_ttc_pressure": 5.0,
    "ttc_cap": 10.0,
    "rear_pressure_floor": 0.25,
    "flow_radius": 120.0,
    "lanes": "ego_and_adjacent",
}

DEFAULT_SAFETY_TTC_FLOW_REWARD_CONFIG: dict[str, Any] = {
    "enabled": False,
    "ttc_safe_threshold": 4.0,
    "ttc_target": 6.0,
    "ttc_cap": 10.0,
    "low_ttc_penalty_weight": 0.7,
    "max_low_ttc_penalty": 0.9,
    "safe_ttc_bonus_weight": 0.08,
    "max_safe_ttc_bonus": 0.12,
    "lag_penalty_weight": 0.16,
    "speed_tolerance": 2.0,
    "max_lag_penalty": 0.9,
    "rear_ttc_pressure": 5.0,
    "rear_pressure_floor": 0.25,
    "flow_radius": 120.0,
    "lanes": "ego_and_adjacent",
}

DEFAULT_POTENTIAL_FIELD_REWARD_CONFIG: dict[str, Any] = {
    "enabled": False,
    "weight": 0.25,
    "sensing_range": 120.0,
    "field_magnitude": 0.5,
    "field_px": 2.0,
    "field_py": 2.0,
    "field_pt": 1.0,
    "timegap": 0.7,
    "lateral_timegap": 0.7,
    "max_cost": 1.0,
    "min_longitudinal_scale": 1e-3,
    "min_lateral_scale": 1e-3,
    "lanes": "ego_and_adjacent",
}

DEFAULT_DRIVER_AGGRESSIVENESS_CONFIG: dict[str, Any] = {
    "enabled": False,
    "distribution": "uniform",
    "min_score": 0.0,
    "max_score": 100.0,
    "fixed_score": None,
    "normal_mean": 50.0,
    "normal_std": 20.0,
    "conservative": {
        "target_speed": 20.0,
        "acc_max": 4.0,
        "comfort_acc_max": 2.0,
        "comfort_acc_min": -4.0,
        "delta": 4.5,
        "time_wanted": 2.4,
        "distance_wanted": 14.0,
        "politeness": 0.8,
        "lane_change_min_acc_gain": 0.8,
        "lane_change_max_braking_imposed": 1.0,
        "lane_change_delay": 1.5,
    },
    "aggressive": {
        "target_speed": 30.0,
        "acc_max": 7.0,
        "comfort_acc_max": 5.5,
        "comfort_acc_min": -6.5,
        "delta": 3.5,
        "time_wanted": 0.6,
        "distance_wanted": 6.0,
        "politeness": 0.0,
        "lane_change_min_acc_gain": 0.05,
        "lane_change_max_braking_imposed": 3.5,
        "lane_change_delay": 0.5,
    },
}

DEFAULT_DRIVER_AGGRESSIVENESS_OBSERVATION_CONFIG: dict[str, Any] = {
    "enabled": False,
    "feature_name": "driver_aggressiveness",
    "normalize": True,
    "ego_value": 0.0,
    "missing_value": 0.0,
}

DEFAULT_TTC_OBSERVATION_CONFIG: dict[str, Any] = {
    "enabled": False,
    "feature_name": "ttc",
    "ttc_cap": 10.0,
    "gap_cap": 120.0,
    "lane_y_threshold": 0.35,
    "front_only": True,
    "normalize": True,
    "include_lane_context": False,
    "lane_context_sides": ["left", "right"],
}

DEFAULT_LANE_CHANGE_SAFETY_CONFIG: dict[str, Any] = {
    "enabled": False,
    "ttc_cap": 10.0,
    "gap_cap": 120.0,
    "front_ttc_safe": 4.0,
    "rear_ttc_safe": 4.0,
    "front_gap_safe": 12.0,
    "rear_gap_safe": 12.0,
    "penalty_weight": 1.25,
    "max_penalty": 1.5,
    "penalty_power": 1.0,
    "unavailable_lane_penalty": 1.0,
    "penalize_unavailable_lane": True,
    "use_ttc": True,
    "use_gap": True,
}

DRIVER_PROFILE_KEYS = {
    "target_speed": "target_speed",
    "acc_max": "ACC_MAX",
    "comfort_acc_max": "COMFORT_ACC_MAX",
    "comfort_acc_min": "COMFORT_ACC_MIN",
    "delta": "DELTA",
    "time_wanted": "TIME_WANTED",
    "distance_wanted": "DISTANCE_WANTED",
    "politeness": "POLITENESS",
    "lane_change_min_acc_gain": "LANE_CHANGE_MIN_ACC_GAIN",
    "lane_change_max_braking_imposed": "LANE_CHANGE_MAX_BRAKING_IMPOSED",
    "lane_change_delay": "LANE_CHANGE_DELAY",
}


def build_adaptive_longitudinal_config(
    adaptive_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_ADAPTIVE_LONGITUDINAL_CONFIG)
    if adaptive_config:
        merged.update(dict(adaptive_config))
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["mode"] = str(merged.get("mode", "delta")).lower()
    if merged["mode"] not in {"delta", "safe_speed_limiter", "ttc_safety_override"}:
        raise ValueError(
            f"Unsupported adaptive_longitudinal mode={merged['mode']!r}. "
            "Expected 'delta', 'safe_speed_limiter', or 'ttc_safety_override'."
        )
    merged["ttc_midpoint"] = float(merged["ttc_midpoint"])
    merged["ttc_temperature"] = float(max(1e-6, merged["ttc_temperature"]))
    merged["ttc_cap"] = float(max(1e-6, merged["ttc_cap"]))
    merged["safety_ttc_threshold"] = float(max(0.0, merged["safety_ttc_threshold"]))
    merged["unsafe_action_penalty"] = float(max(0.0, merged["unsafe_action_penalty"]))
    merged["min_target_speed"] = float(merged["min_target_speed"])
    merged["max_target_speed"] = float(max(merged["min_target_speed"], merged["max_target_speed"]))
    merged["faster_max_delta"] = float(max(0.0, merged["faster_max_delta"]))
    merged["slower_min_delta"] = float(max(0.0, merged["slower_min_delta"]))
    merged["slower_max_delta"] = float(max(merged["slower_min_delta"], merged["slower_max_delta"]))
    merged["cruise_speed"] = float(
        np.clip(merged["cruise_speed"], merged["min_target_speed"], merged["max_target_speed"])
    )
    merged["action_speed_delta"] = float(max(0.0, merged["action_speed_delta"]))
    return merged


def build_rear_flow_config(rear_flow_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_REAR_FLOW_CONFIG)
    if rear_flow_config:
        merged.update(dict(rear_flow_config))
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["spawn_on_reset"] = bool(merged.get("spawn_on_reset", True))
    merged["spawn_during_episode"] = bool(merged.get("spawn_during_episode", True))
    merged["vehicles_per_lane"] = int(max(0, merged["vehicles_per_lane"]))
    merged["lanes"] = str(merged.get("lanes", "ego_and_adjacent")).lower()
    merged["distance_range"] = _ordered_pair(merged["distance_range"], minimum=0.0)
    merged["speed_offset_range"] = _ordered_pair(merged["speed_offset_range"], minimum=0.0)
    merged["absolute_speed_range"] = _ordered_pair(merged["absolute_speed_range"], minimum=0.0)
    merged["min_lane_gap"] = float(max(0.0, merged["min_lane_gap"]))
    merged["spawn_probability"] = float(np.clip(merged["spawn_probability"], 0.0, 1.0))
    merged["cooldown_policy_steps"] = int(max(0, merged["cooldown_policy_steps"]))
    merged["min_ego_progress"] = float(max(0.0, merged["min_ego_progress"]))
    merged["max_extra_vehicles"] = int(max(0, merged["max_extra_vehicles"]))
    return merged


def build_traffic_flow_reward_config(
    traffic_flow_reward_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_TRAFFIC_FLOW_REWARD_CONFIG)
    if traffic_flow_reward_config:
        merged.update(dict(traffic_flow_reward_config))
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["penalty_weight"] = float(max(0.0, merged["penalty_weight"]))
    merged["speed_tolerance"] = float(max(0.0, merged["speed_tolerance"]))
    merged["max_penalty"] = float(max(0.0, merged["max_penalty"]))
    merged["front_ttc_safe"] = float(max(1e-6, merged["front_ttc_safe"]))
    merged["rear_ttc_pressure"] = float(max(1e-6, merged["rear_ttc_pressure"]))
    merged["ttc_cap"] = float(max(1e-6, merged["ttc_cap"]))
    merged["rear_pressure_floor"] = float(np.clip(merged["rear_pressure_floor"], 0.0, 1.0))
    merged["flow_radius"] = float(max(0.0, merged["flow_radius"]))
    merged["lanes"] = str(merged.get("lanes", "ego_and_adjacent")).lower()
    return merged


def build_safety_ttc_flow_reward_config(
    safety_ttc_flow_reward_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_SAFETY_TTC_FLOW_REWARD_CONFIG)
    if safety_ttc_flow_reward_config:
        merged.update(dict(safety_ttc_flow_reward_config))
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["ttc_safe_threshold"] = float(max(1e-6, merged["ttc_safe_threshold"]))
    merged["ttc_target"] = float(max(merged["ttc_safe_threshold"], merged["ttc_target"]))
    merged["ttc_cap"] = float(max(merged["ttc_target"], merged["ttc_cap"]))
    merged["low_ttc_penalty_weight"] = float(max(0.0, merged["low_ttc_penalty_weight"]))
    merged["max_low_ttc_penalty"] = float(max(0.0, merged["max_low_ttc_penalty"]))
    merged["safe_ttc_bonus_weight"] = float(max(0.0, merged["safe_ttc_bonus_weight"]))
    merged["max_safe_ttc_bonus"] = float(max(0.0, merged["max_safe_ttc_bonus"]))
    merged["lag_penalty_weight"] = float(max(0.0, merged["lag_penalty_weight"]))
    merged["speed_tolerance"] = float(max(0.0, merged["speed_tolerance"]))
    merged["max_lag_penalty"] = float(max(0.0, merged["max_lag_penalty"]))
    merged["rear_ttc_pressure"] = float(max(1e-6, merged["rear_ttc_pressure"]))
    merged["rear_pressure_floor"] = float(np.clip(merged["rear_pressure_floor"], 0.0, 1.0))
    merged["flow_radius"] = float(max(0.0, merged["flow_radius"]))
    merged["lanes"] = str(merged.get("lanes", "ego_and_adjacent")).lower()
    return merged


def build_potential_field_reward_config(
    potential_field_reward_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_POTENTIAL_FIELD_REWARD_CONFIG)
    if potential_field_reward_config:
        merged.update(dict(potential_field_reward_config))
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["weight"] = float(max(0.0, merged["weight"]))
    merged["sensing_range"] = float(max(0.0, merged["sensing_range"]))
    merged["field_magnitude"] = float(max(0.0, merged["field_magnitude"]))
    merged["field_px"] = float(max(1e-6, merged["field_px"]))
    merged["field_py"] = float(max(1e-6, merged["field_py"]))
    merged["field_pt"] = float(max(1e-6, merged["field_pt"]))
    merged["timegap"] = float(max(0.0, merged["timegap"]))
    merged["lateral_timegap"] = float(max(0.0, merged["lateral_timegap"]))
    merged["max_cost"] = float(max(0.0, merged["max_cost"]))
    merged["min_longitudinal_scale"] = float(max(1e-9, merged["min_longitudinal_scale"]))
    merged["min_lateral_scale"] = float(max(1e-9, merged["min_lateral_scale"]))
    merged["lanes"] = str(merged.get("lanes", "ego_and_adjacent")).lower()
    if merged["lanes"] not in {"current", "ego", "ego_and_adjacent", "all"}:
        raise ValueError(
            f"Unsupported potential_field_reward lanes={merged['lanes']!r}. "
            "Expected 'current', 'ego', 'ego_and_adjacent', or 'all'."
        )
    return merged


def build_driver_aggressiveness_config(
    driver_aggressiveness_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_DRIVER_AGGRESSIVENESS_CONFIG)
    if driver_aggressiveness_config:
        updates = dict(driver_aggressiveness_config)
        for profile_key in ("conservative", "aggressive"):
            if isinstance(updates.get(profile_key), Mapping):
                merged[profile_key].update(dict(updates.pop(profile_key)))
        merged.update(updates)

    merged["enabled"] = bool(merged.get("enabled", False))
    merged["distribution"] = str(merged.get("distribution", "uniform")).lower()
    merged["min_score"] = float(np.clip(merged.get("min_score", 0.0), 0.0, 100.0))
    merged["max_score"] = float(np.clip(merged.get("max_score", 100.0), merged["min_score"], 100.0))
    merged["normal_mean"] = float(np.clip(merged.get("normal_mean", 50.0), 0.0, 100.0))
    merged["normal_std"] = float(max(1e-6, merged.get("normal_std", 20.0)))
    if merged.get("fixed_score") is not None:
        merged["fixed_score"] = float(np.clip(float(merged["fixed_score"]), merged["min_score"], merged["max_score"]))

    for profile_name in ("conservative", "aggressive"):
        profile = merged[profile_name]
        for key in DRIVER_PROFILE_KEYS:
            profile[key] = float(profile[key])
    return merged


def build_driver_aggressiveness_observation_config(
    driver_aggressiveness_observation_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_DRIVER_AGGRESSIVENESS_OBSERVATION_CONFIG)
    if driver_aggressiveness_observation_config:
        merged.update(dict(driver_aggressiveness_observation_config))
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["feature_name"] = str(merged.get("feature_name", "driver_aggressiveness"))
    merged["normalize"] = bool(merged.get("normalize", True))
    if merged["normalize"]:
        merged["ego_value"] = float(np.clip(merged.get("ego_value", 0.0), 0.0, 1.0))
        merged["missing_value"] = float(np.clip(merged.get("missing_value", 0.0), 0.0, 1.0))
    else:
        merged["ego_value"] = float(np.clip(merged.get("ego_value", 0.0), 0.0, 100.0))
        merged["missing_value"] = float(np.clip(merged.get("missing_value", 0.0), 0.0, 100.0))
    return merged


def build_ttc_observation_config(
    ttc_observation_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_TTC_OBSERVATION_CONFIG)
    if ttc_observation_config:
        merged.update(dict(ttc_observation_config))
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["feature_name"] = str(merged.get("feature_name", "ttc"))
    merged["ttc_cap"] = float(max(1e-6, merged["ttc_cap"]))
    merged["gap_cap"] = float(max(1e-6, merged["gap_cap"]))
    merged["lane_y_threshold"] = float(max(0.0, merged["lane_y_threshold"]))
    merged["front_only"] = bool(merged.get("front_only", True))
    merged["normalize"] = bool(merged.get("normalize", True))
    merged["include_lane_context"] = bool(merged.get("include_lane_context", False))
    sides = [str(side).lower() for side in merged.get("lane_context_sides", ["left", "right"])]
    unsupported_sides = sorted(set(sides) - {"left", "right"})
    if unsupported_sides:
        raise ValueError(
            f"Unsupported TTC lane_context_sides={unsupported_sides!r}. "
            "Expected any subset of ['left', 'right']."
        )
    merged["lane_context_sides"] = sides
    return merged


def build_lane_change_safety_config(
    lane_change_safety_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_LANE_CHANGE_SAFETY_CONFIG)
    if lane_change_safety_config:
        merged.update(dict(lane_change_safety_config))
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["ttc_cap"] = float(max(1e-6, merged["ttc_cap"]))
    merged["gap_cap"] = float(max(1e-6, merged["gap_cap"]))
    merged["front_ttc_safe"] = float(max(1e-6, merged["front_ttc_safe"]))
    merged["rear_ttc_safe"] = float(max(1e-6, merged["rear_ttc_safe"]))
    merged["front_gap_safe"] = float(max(1e-6, merged["front_gap_safe"]))
    merged["rear_gap_safe"] = float(max(1e-6, merged["rear_gap_safe"]))
    merged["penalty_weight"] = float(max(0.0, merged["penalty_weight"]))
    merged["max_penalty"] = float(max(0.0, merged["max_penalty"]))
    merged["penalty_power"] = float(max(1e-6, merged["penalty_power"]))
    merged["unavailable_lane_penalty"] = float(max(0.0, merged["unavailable_lane_penalty"]))
    merged["penalize_unavailable_lane"] = bool(merged.get("penalize_unavailable_lane", True))
    merged["use_ttc"] = bool(merged.get("use_ttc", True))
    merged["use_gap"] = bool(merged.get("use_gap", True))
    return merged


def split_highway_env_and_custom_configs(
    config: Mapping[str, Any] | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    base_config = dict(config or {})
    adaptive_config = build_adaptive_longitudinal_config(
        base_config.pop(ADAPTIVE_LONGITUDINAL_CONFIG_KEY, None)
    )
    rear_flow_config = build_rear_flow_config(base_config.pop(REAR_FLOW_CONFIG_KEY, None))
    traffic_flow_reward_config = build_traffic_flow_reward_config(
        base_config.pop(TRAFFIC_FLOW_REWARD_CONFIG_KEY, None)
    )
    safety_ttc_flow_reward_config = build_safety_ttc_flow_reward_config(
        base_config.pop(SAFETY_TTC_FLOW_REWARD_CONFIG_KEY, None)
    )
    potential_field_reward_config = build_potential_field_reward_config(
        base_config.pop(POTENTIAL_FIELD_REWARD_CONFIG_KEY, None)
    )
    driver_aggressiveness_config = build_driver_aggressiveness_config(
        base_config.pop(DRIVER_AGGRESSIVENESS_CONFIG_KEY, None)
    )
    driver_aggressiveness_observation_config = build_driver_aggressiveness_observation_config(
        base_config.pop(DRIVER_AGGRESSIVENESS_OBSERVATION_CONFIG_KEY, None)
    )
    ttc_observation_config = build_ttc_observation_config(
        base_config.pop(TTC_OBSERVATION_CONFIG_KEY, None)
    )
    lane_change_safety_config = build_lane_change_safety_config(
        base_config.pop(LANE_CHANGE_SAFETY_CONFIG_KEY, None)
    )
    return (
        base_config,
        adaptive_config,
        rear_flow_config,
        traffic_flow_reward_config,
        safety_ttc_flow_reward_config,
        potential_field_reward_config,
        driver_aggressiveness_config,
        driver_aggressiveness_observation_config,
        ttc_observation_config,
        lane_change_safety_config,
    )


def _ordered_pair(values: Any, *, minimum: float) -> list[float]:
    low, high = list(values)
    low = float(max(minimum, low))
    high = float(max(low, high))
    return [low, high]


def _forward_speed(vehicle) -> float:
    return float(vehicle.speed * np.cos(getattr(vehicle, "heading", 0.0)))


def _lateral_speed(vehicle) -> float:
    return float(vehicle.speed * np.sin(getattr(vehicle, "heading", 0.0)))


def _lane_longitudinal(vehicle, lane) -> float:
    longitudinal, _ = lane.local_coordinates(vehicle.position)
    return float(longitudinal)


def _vehicle_clearance(ego_vehicle, other_vehicle, ego_s: float, other_s: float) -> float:
    return max(
        0.0,
        abs(float(other_s) - float(ego_s))
        - 0.5 * float(getattr(ego_vehicle, "LENGTH", 0.0) + getattr(other_vehicle, "LENGTH", 0.0)),
    )


def _vehicle_length(vehicle) -> float:
    return float(getattr(vehicle, "LENGTH", getattr(vehicle, "length", 5.0)))


def _vehicle_width(vehicle) -> float:
    return float(getattr(vehicle, "WIDTH", getattr(vehicle, "width", 2.0)))


def adjacent_lane_index(env: gym.Env, side: str):
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    lane_index = getattr(vehicle, "lane_index", None)
    if vehicle is None or road is None or lane_index is None:
        return None

    lane_from, lane_to, lane_id = lane_index
    lane_count = len(road.network.graph[lane_from][lane_to])
    delta = -1 if str(side).lower() == "left" else 1
    candidate_id = int(lane_id) + delta
    if candidate_id < 0 or candidate_id >= lane_count:
        return None
    return (lane_from, lane_to, candidate_id)


def lane_front_rear_metrics(
    env: gym.Env,
    lane_index,
    *,
    ttc_cap: float = 10.0,
    gap_cap: float = 120.0,
) -> dict[str, float | bool]:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if vehicle is None or road is None or lane_index is None:
        return {
            "available": False,
            "front_ttc": 0.0,
            "rear_ttc": 0.0,
            "front_gap": 0.0,
            "rear_gap": 0.0,
        }

    lane = road.network.get_lane(lane_index)
    ego_s = _lane_longitudinal(vehicle, lane)
    front_vehicle, rear_vehicle = road.neighbour_vehicles(vehicle, lane_index)
    front_gap = float(gap_cap)
    rear_gap = float(gap_cap)
    front_ttc = float(ttc_cap)
    rear_ttc = float(ttc_cap)

    if front_vehicle is not None:
        front_s = _lane_longitudinal(front_vehicle, lane)
        front_gap = min(float(gap_cap), _vehicle_clearance(vehicle, front_vehicle, ego_s, front_s))
        closing_speed = _forward_speed(vehicle) - _forward_speed(front_vehicle)
        if closing_speed > 1e-6:
            front_ttc = 0.0 if front_gap <= 0.0 else front_gap / closing_speed
        front_ttc = float(np.clip(front_ttc, 0.0, ttc_cap))

    if rear_vehicle is not None:
        rear_s = _lane_longitudinal(rear_vehicle, lane)
        rear_gap = min(float(gap_cap), _vehicle_clearance(vehicle, rear_vehicle, ego_s, rear_s))
        closing_speed = _forward_speed(rear_vehicle) - _forward_speed(vehicle)
        if closing_speed > 1e-6:
            rear_ttc = 0.0 if rear_gap <= 0.0 else rear_gap / closing_speed
        rear_ttc = float(np.clip(rear_ttc, 0.0, ttc_cap))

    return {
        "available": True,
        "front_ttc": float(front_ttc),
        "rear_ttc": float(rear_ttc),
        "front_gap": float(front_gap),
        "rear_gap": float(rear_gap),
    }


def compute_same_lane_ttc(env: gym.Env, ttc_cap: float = 10.0) -> float:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if vehicle is None or road is None:
        return float(ttc_cap)

    lane_index = getattr(vehicle, "lane_index", None)
    if lane_index is None:
        return float(ttc_cap)

    front_vehicle, _ = road.neighbour_vehicles(vehicle, lane_index)
    if front_vehicle is None:
        return float(ttc_cap)

    lane = road.network.get_lane(lane_index)
    ego_s, _ = lane.local_coordinates(vehicle.position)
    front_s, _ = lane.local_coordinates(front_vehicle.position)
    clearance = max(
        0.0,
        float(front_s - ego_s)
        - 0.5 * float(getattr(vehicle, "LENGTH", 0.0) + getattr(front_vehicle, "LENGTH", 0.0)),
    )

    ego_speed = float(vehicle.speed * np.cos(getattr(vehicle, "heading", 0.0)))
    front_speed = float(front_vehicle.speed * np.cos(getattr(front_vehicle, "heading", 0.0)))
    closing_speed = ego_speed - front_speed
    if closing_speed <= 1e-6:
        return float(ttc_cap)

    ttc = 0.0 if clearance <= 0.0 else clearance / closing_speed
    return float(np.clip(ttc, 0.0, ttc_cap))


def compute_same_lane_rear_ttc(env: gym.Env, ttc_cap: float = 10.0) -> float:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if vehicle is None or road is None:
        return float(ttc_cap)

    lane_index = getattr(vehicle, "lane_index", None)
    if lane_index is None:
        return float(ttc_cap)

    _, rear_vehicle = road.neighbour_vehicles(vehicle, lane_index)
    if rear_vehicle is None:
        return float(ttc_cap)

    lane = road.network.get_lane(lane_index)
    ego_s = _lane_longitudinal(vehicle, lane)
    rear_s = _lane_longitudinal(rear_vehicle, lane)
    clearance = _vehicle_clearance(vehicle, rear_vehicle, ego_s, rear_s)
    closing_speed = _forward_speed(rear_vehicle) - _forward_speed(vehicle)
    if closing_speed <= 1e-6:
        return float(ttc_cap)

    rear_ttc = 0.0 if clearance <= 0.0 else clearance / closing_speed
    return float(np.clip(rear_ttc, 0.0, ttc_cap))


def driver_color_from_aggressiveness(score: float) -> tuple[int, int, int]:
    t = float(np.clip(score, 0.0, 100.0)) / 100.0
    return int(round(255.0 * t)), 0, int(round(255.0 * (1.0 - t)))


def interpolate_driver_profile(score: float, config: Mapping[str, Any], speed_limit: float | None = None) -> dict[str, float]:
    t = float(np.clip(score, 0.0, 100.0)) / 100.0
    conservative = config["conservative"]
    aggressive = config["aggressive"]
    profile: dict[str, float] = {}
    for key in DRIVER_PROFILE_KEYS:
        low = float(conservative[key])
        high = float(aggressive[key])
        value = low + t * (high - low)
        if key == "target_speed" and speed_limit is not None:
            value = float(np.clip(value, 0.0, float(speed_limit)))
        profile[key] = float(value)
    return profile


class DriverAggressivenessWrapper(gym.Wrapper):
    """Assign continuous IDM/MOBIL driver personalities and blue-red render colors."""

    def __init__(self, env: gym.Env, driver_aggressiveness_config: Mapping[str, Any] | None = None) -> None:
        super().__init__(env)
        self.driver_aggressiveness_config = build_driver_aggressiveness_config(driver_aggressiveness_config)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._configure_unscored_traffic()
        info = dict(info)
        info.update(self._driver_aggressiveness_info())
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._configure_unscored_traffic()
        info = dict(info)
        info.update(self._driver_aggressiveness_info())
        return observation, reward, terminated, truncated, info

    def _configure_unscored_traffic(self) -> None:
        if not self.driver_aggressiveness_config["enabled"]:
            return

        road = getattr(self.unwrapped, "road", None)
        if road is None:
            return

        controlled_ids = {id(vehicle) for vehicle in getattr(self.unwrapped, "controlled_vehicles", [])}
        for vehicle in getattr(road, "vehicles", []):
            if id(vehicle) in controlled_ids or hasattr(vehicle, "driver_aggressiveness_score"):
                continue
            if not all(hasattr(vehicle, attr) for attr in ("TIME_WANTED", "POLITENESS")):
                continue
            self._configure_vehicle(vehicle)

    def _configure_vehicle(self, vehicle) -> None:
        score = self._sample_score()
        profile = interpolate_driver_profile(
            score,
            self.driver_aggressiveness_config,
            speed_limit=float(self.unwrapped.config.get("speed_limit", 30.0)),
        )
        for profile_key, vehicle_attribute in DRIVER_PROFILE_KEYS.items():
            setattr(vehicle, vehicle_attribute, float(profile[profile_key]))
        vehicle.driver_aggressiveness_score = float(score)
        vehicle.driver_profile = profile
        vehicle.color = driver_color_from_aggressiveness(score)

    def _sample_score(self) -> float:
        config = self.driver_aggressiveness_config
        min_score = float(config["min_score"])
        max_score = float(config["max_score"])
        fixed_score = config.get("fixed_score")
        if fixed_score is not None:
            return float(np.clip(float(fixed_score), min_score, max_score))

        distribution = str(config["distribution"]).lower()
        rng = getattr(getattr(self.unwrapped, "road", None), "np_random", np.random)
        if distribution == "normal":
            score = float(rng.normal(float(config["normal_mean"]), float(config["normal_std"])))
        else:
            score = float(rng.uniform(min_score, max_score))
        return float(np.clip(score, min_score, max_score))

    def _driver_aggressiveness_scores(self) -> list[float]:
        road = getattr(self.unwrapped, "road", None)
        if road is None:
            return []
        controlled_ids = {id(vehicle) for vehicle in getattr(self.unwrapped, "controlled_vehicles", [])}
        scores = []
        for vehicle in getattr(road, "vehicles", []):
            if id(vehicle) in controlled_ids or not hasattr(vehicle, "driver_aggressiveness_score"):
                continue
            scores.append(float(vehicle.driver_aggressiveness_score))
        return scores

    def _driver_aggressiveness_info(self) -> dict[str, float]:
        scores = self._driver_aggressiveness_scores()
        if not scores:
            return {}
        return {
            "driver_aggressiveness_mean": float(np.mean(scores)),
            "driver_aggressiveness_min": float(np.min(scores)),
            "driver_aggressiveness_max": float(np.max(scores)),
        }


class DriverAggressivenessObservationWrapper(gym.ObservationWrapper):
    """Append each observed vehicle's continuous driver-aggressiveness score."""

    def __init__(
        self,
        env: gym.Env,
        driver_aggressiveness_observation_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.driver_aggressiveness_observation_config = build_driver_aggressiveness_observation_config(
            driver_aggressiveness_observation_config
        )
        if not isinstance(env.observation_space, gym.spaces.Box) or len(env.observation_space.shape) != 2:
            raise TypeError(
                "DriverAggressivenessObservationWrapper expects a 2D Box observation from highway-env "
                f"Kinematics, got {env.observation_space!r}"
            )
        low = np.asarray(env.observation_space.low, dtype=np.float32)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        high_value = 1.0 if self.driver_aggressiveness_observation_config["normalize"] else 100.0
        feature_low = np.zeros((low.shape[0], 1), dtype=np.float32)
        feature_high = np.full((high.shape[0], 1), high_value, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, feature_low], axis=1),
            high=np.concatenate([high, feature_high], axis=1),
            dtype=np.float32,
        )

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim != 2:
            return observation
        feature = self._aggressiveness_feature(row_count=obs.shape[0])
        return np.concatenate([obs, feature[:, None]], axis=1).astype(np.float32)

    def _aggressiveness_feature(self, *, row_count: int) -> np.ndarray:
        cfg = self.driver_aggressiveness_observation_config
        values = np.full(row_count, float(cfg["missing_value"]), dtype=np.float32)
        if row_count <= 0:
            return values
        values[0] = float(cfg["ego_value"])

        road = getattr(self.unwrapped, "road", None)
        observer = getattr(self.unwrapped, "vehicle", None)
        observation_type = getattr(self.unwrapped, "observation_type", None)
        if road is None or observer is None or observation_type is None:
            return values

        vehicles_count = int(getattr(observation_type, "vehicles_count", row_count))
        observe_count = min(row_count - 1, max(0, vehicles_count - 1))
        if observe_count <= 0:
            return values

        close_objects = road.close_objects_to(
            observer,
            self.unwrapped.PERCEPTION_DISTANCE,
            count=observe_count,
            see_behind=bool(getattr(observation_type, "see_behind", False)),
            sort=getattr(observation_type, "order", "sorted") == "sorted",
            vehicles_only=not bool(getattr(observation_type, "include_obstacles", True)),
        )
        observed_objects = close_objects[-observe_count:]
        for row_idx, vehicle in enumerate(observed_objects, start=1):
            if row_idx >= row_count:
                break
            score = getattr(vehicle, "driver_aggressiveness_score", None)
            if score is None:
                continue
            values[row_idx] = self._score_to_feature(float(score))
        return values

    def _score_to_feature(self, score: float) -> float:
        score = float(np.clip(score, 0.0, 100.0))
        if self.driver_aggressiveness_observation_config["normalize"]:
            return score / 100.0
        return score


class TTCObservationWrapper(gym.ObservationWrapper):
    """Append a continuous TTC feature to each row of a Kinematics observation."""

    def __init__(self, env: gym.Env, ttc_observation_config: Mapping[str, Any] | None = None) -> None:
        super().__init__(env)
        self.ttc_observation_config = build_ttc_observation_config(ttc_observation_config)
        if not isinstance(env.observation_space, gym.spaces.Box) or len(env.observation_space.shape) != 2:
            raise TypeError(
                "TTCObservationWrapper expects a 2D Box observation from highway-env Kinematics, "
                f"got {env.observation_space!r}"
            )
        low = np.asarray(env.observation_space.low, dtype=np.float32)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        context_width = self._lane_context_width()
        added_width = 1 + context_width
        ttc_low = np.zeros((low.shape[0], added_width), dtype=np.float32)
        ttc_high_value = 1.0 if self.ttc_observation_config["normalize"] else self.ttc_observation_config["ttc_cap"]
        ttc_high = np.full((high.shape[0], added_width), float(ttc_high_value), dtype=np.float32)
        if context_width:
            ttc_high[:, 1:] = self._lane_context_high_values(row_count=high.shape[0])
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, ttc_low], axis=1),
            high=np.concatenate([high, ttc_high], axis=1),
            dtype=np.float32,
        )

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim != 2:
            return observation
        ttc_feature = self._ttc_feature_from_observation(obs)
        features = [obs, ttc_feature[:, None]]
        lane_context = self._lane_context_features(row_count=obs.shape[0])
        if lane_context.size:
            features.append(lane_context)
        return np.concatenate(features, axis=1).astype(np.float32)

    def _lane_context_width(self) -> int:
        if not bool(self.ttc_observation_config["include_lane_context"]):
            return 0
        return 5 * len(self.ttc_observation_config["lane_context_sides"])

    def _lane_context_high_values(self, *, row_count: int) -> np.ndarray:
        high_value = 1.0 if self.ttc_observation_config["normalize"] else self.ttc_observation_config["ttc_cap"]
        gap_high_value = 1.0 if self.ttc_observation_config["normalize"] else self.ttc_observation_config["gap_cap"]
        row = []
        for _ in self.ttc_observation_config["lane_context_sides"]:
            row.extend([1.0, high_value, high_value, gap_high_value, gap_high_value])
        return np.tile(np.asarray(row, dtype=np.float32), (row_count, 1))

    def _lane_context_features(self, *, row_count: int) -> np.ndarray:
        width = self._lane_context_width()
        values = np.zeros((row_count, width), dtype=np.float32)
        if width == 0 or row_count <= 0:
            return values

        ego_features = []
        for side in self.ttc_observation_config["lane_context_sides"]:
            metrics = lane_front_rear_metrics(
                self.env,
                adjacent_lane_index(self.env, side),
                ttc_cap=float(self.ttc_observation_config["ttc_cap"]),
                gap_cap=float(self.ttc_observation_config["gap_cap"]),
            )
            ego_features.extend(
                [
                    float(bool(metrics["available"])),
                    float(metrics["front_ttc"]),
                    float(metrics["rear_ttc"]),
                    float(metrics["front_gap"]),
                    float(metrics["rear_gap"]),
                ]
            )

        ego_features_array = np.asarray(ego_features, dtype=np.float32)
        if self.ttc_observation_config["normalize"]:
            for offset in range(0, len(ego_features), 5):
                ego_features_array[offset + 1] /= float(self.ttc_observation_config["ttc_cap"])
                ego_features_array[offset + 2] /= float(self.ttc_observation_config["ttc_cap"])
                ego_features_array[offset + 3] /= float(self.ttc_observation_config["gap_cap"])
                ego_features_array[offset + 4] /= float(self.ttc_observation_config["gap_cap"])
        values[0, :] = np.clip(ego_features_array, 0.0, None)
        return values

    def _ttc_feature_from_observation(self, obs: np.ndarray) -> np.ndarray:
        feature_names = list(self.unwrapped.config.get("observation", {}).get("features", []))
        try:
            presence_idx = feature_names.index("presence")
            x_idx = feature_names.index("x")
            y_idx = feature_names.index("y")
            vx_idx = feature_names.index("vx")
        except ValueError:
            return np.zeros(obs.shape[0], dtype=np.float32)

        presence = obs[:, presence_idx] > 0.5
        relative_x = self._raw_observation_feature(obs[:, x_idx], "x")
        relative_y = self._raw_observation_feature(obs[:, y_idx], "y")
        relative_vx = self._raw_observation_feature(obs[:, vx_idx], "vx")
        closing_speed = -relative_vx

        same_lane = np.abs(relative_y) <= float(self.ttc_observation_config["lane_y_threshold"])
        front = relative_x > 0.0 if self.ttc_observation_config["front_only"] else np.ones_like(relative_x, dtype=bool)
        closing = closing_speed > 1e-6
        valid = presence & same_lane & front & closing

        ttc = np.full(obs.shape[0], float(self.ttc_observation_config["ttc_cap"]), dtype=np.float32)
        ttc[valid] = np.clip(
            relative_x[valid] / np.maximum(closing_speed[valid], 1e-6),
            0.0,
            float(self.ttc_observation_config["ttc_cap"]),
        )
        ttc[~presence] = float(self.ttc_observation_config["ttc_cap"])
        if self.ttc_observation_config["normalize"]:
            ttc = ttc / float(self.ttc_observation_config["ttc_cap"])
        return ttc.astype(np.float32)

    def _raw_observation_feature(self, values: np.ndarray, feature_name: str) -> np.ndarray:
        observation_type = getattr(self.unwrapped, "observation_type", None)
        normalize = bool(getattr(observation_type, "normalize", False))
        features_range = getattr(observation_type, "features_range", None) or {}
        if not normalize or feature_name not in features_range:
            return values.astype(np.float32)
        low, high = features_range[feature_name]
        return (0.5 * (values + 1.0) * (float(high) - float(low)) + float(low)).astype(np.float32)


class RearFlowPressureWrapper(gym.Wrapper):
    """Add faster vehicles behind the ego vehicle to keep traffic pressure realistic."""

    def __init__(self, env: gym.Env, rear_flow_config: Mapping[str, Any] | None = None) -> None:
        super().__init__(env)
        self.rear_flow_config = build_rear_flow_config(rear_flow_config)
        self._policy_steps_since_spawn = int(self.rear_flow_config["cooldown_policy_steps"])
        self._spawned_extra_vehicles = 0

    def reset(self, **kwargs):
        self._policy_steps_since_spawn = int(self.rear_flow_config["cooldown_policy_steps"])
        self._spawned_extra_vehicles = 0
        observation, info = self.env.reset(**kwargs)
        spawned_count = 0
        if self.rear_flow_config["enabled"] and self.rear_flow_config["spawn_on_reset"]:
            spawned_count = self._spawn_initial_rear_flow()
            if spawned_count:
                observation = self.unwrapped.observation_type.observe()

        info = dict(info)
        info["rear_flow_spawned_count"] = int(spawned_count)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        spawned_count = 0
        if (
            self.rear_flow_config["enabled"]
            and self.rear_flow_config["spawn_during_episode"]
            and not (terminated or truncated)
        ):
            self._policy_steps_since_spawn += 1
            if self._should_spawn_during_episode():
                spawned_count = int(self._spawn_one_rear_vehicle())
                if spawned_count:
                    self._policy_steps_since_spawn = 0

        info = dict(info)
        info["rear_flow_spawned_count"] = int(spawned_count)
        return observation, reward, terminated, truncated, info

    def _spawn_initial_rear_flow(self) -> int:
        vehicle = getattr(self.unwrapped, "vehicle", None)
        if vehicle is None:
            return 0

        spawned_count = 0
        lane_indices = self._candidate_lane_indices(vehicle)
        for lane_index in lane_indices:
            for _ in range(int(self.rear_flow_config["vehicles_per_lane"])):
                spawned_count += int(self._spawn_one_rear_vehicle(lane_index=lane_index))
        return spawned_count

    def _should_spawn_during_episode(self) -> bool:
        config = self.rear_flow_config
        if self._spawned_extra_vehicles >= int(config["max_extra_vehicles"]):
            return False
        if self._policy_steps_since_spawn < int(config["cooldown_policy_steps"]):
            return False

        ego_vehicle = getattr(self.unwrapped, "vehicle", None)
        road = getattr(self.unwrapped, "road", None)
        if ego_vehicle is None or road is None:
            return False
        if float(ego_vehicle.position[0]) < float(config["min_ego_progress"]):
            return False
        return bool(road.np_random.uniform() <= float(config["spawn_probability"]))

    def _spawn_one_rear_vehicle(self, lane_index=None) -> bool:
        ego_vehicle = getattr(self.unwrapped, "vehicle", None)
        road = getattr(self.unwrapped, "road", None)
        if ego_vehicle is None or road is None:
            return False

        lane_indices = [lane_index] if lane_index is not None else self._candidate_lane_indices(ego_vehicle)
        road.np_random.shuffle(lane_indices)
        for candidate_lane_index in lane_indices:
            spawn_s = self._sample_spawn_position(ego_vehicle, candidate_lane_index)
            if spawn_s is None or not self._lane_has_clear_gap(candidate_lane_index, spawn_s):
                continue
            self._spawn_vehicle(candidate_lane_index, spawn_s, ego_vehicle)
            self._spawned_extra_vehicles += 1
            return True
        return False

    def _candidate_lane_indices(self, ego_vehicle) -> list[tuple]:
        lane_index = getattr(ego_vehicle, "lane_index", None)
        if lane_index is None:
            return []

        lane_scope = str(self.rear_flow_config["lanes"]).lower()
        road = self.unwrapped.road
        lane_from, lane_to, lane_id = lane_index
        lane_count = len(road.network.graph[lane_from][lane_to])

        if lane_scope in {"current", "ego"}:
            return [lane_index]
        if lane_scope == "all":
            return [(lane_from, lane_to, idx) for idx in range(lane_count)]

        candidates = [lane_index]
        for delta in (-1, 1):
            candidate_id = int(lane_id) + delta
            if 0 <= candidate_id < lane_count:
                candidates.append((lane_from, lane_to, candidate_id))
        return candidates

    def _sample_spawn_position(self, ego_vehicle, lane_index) -> float | None:
        lane = self.unwrapped.road.network.get_lane(lane_index)
        ego_s = _lane_longitudinal(ego_vehicle, lane)
        low, high = self.rear_flow_config["distance_range"]
        spawn_distance = self.unwrapped.road.np_random.uniform(float(low), float(high))
        spawn_s = float(ego_s - spawn_distance)
        if spawn_s <= 5.0:
            return None
        return spawn_s

    def _lane_has_clear_gap(self, lane_index, spawn_s: float) -> bool:
        min_lane_gap = float(self.rear_flow_config["min_lane_gap"])
        road = self.unwrapped.road
        lane = road.network.get_lane(lane_index)
        for other in road.vehicles:
            if getattr(other, "lane_index", None) != lane_index:
                continue
            other_s = _lane_longitudinal(other, lane)
            if abs(other_s - float(spawn_s)) < min_lane_gap:
                return False
        return True

    def _spawn_vehicle(self, lane_index, spawn_s: float, ego_vehicle) -> None:
        road = self.unwrapped.road
        lane = road.network.get_lane(lane_index)
        vehicle_cls = utils.class_from_path(self.unwrapped.config["other_vehicles_type"])
        offset_low, offset_high = self.rear_flow_config["speed_offset_range"]
        abs_low, abs_high = self.rear_flow_config["absolute_speed_range"]
        sampled_speed = max(
            float(_forward_speed(ego_vehicle)) + float(road.np_random.uniform(offset_low, offset_high)),
            float(abs_low),
        )
        sampled_speed = float(np.clip(sampled_speed, float(abs_low), float(abs_high)))
        vehicle = vehicle_cls(
            road,
            lane.position(float(spawn_s), 0),
            lane.heading_at(float(spawn_s)),
            sampled_speed,
        )
        if hasattr(vehicle, "randomize_behavior"):
            vehicle.randomize_behavior()
        road.vehicles.append(vehicle)


class TrafficFlowRewardWrapper(gym.Wrapper):
    """Penalize ego speeds that fall well below nearby rear traffic when the front is safe."""

    def __init__(
        self,
        env: gym.Env,
        traffic_flow_reward_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.traffic_flow_reward_config = build_traffic_flow_reward_config(
            traffic_flow_reward_config
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        info.update(self._traffic_flow_info(base_reward=0.0, penalty=0.0, shaped_reward=0.0))
        return observation, info

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        penalty, metrics = self._compute_penalty()
        shaped_reward = float(base_reward) - float(penalty)
        info = dict(info)
        info.update(metrics)
        info.update(
            {
                "traffic_flow_penalty": float(penalty),
                "base_reward": float(base_reward),
                "shaped_reward": float(shaped_reward),
            }
        )
        return observation, shaped_reward, terminated, truncated, info

    def _compute_penalty(self) -> tuple[float, dict[str, float]]:
        if not self.traffic_flow_reward_config["enabled"]:
            return 0.0, self._traffic_flow_info(base_reward=0.0, penalty=0.0, shaped_reward=0.0)

        cfg = self.traffic_flow_reward_config
        ego_vehicle = getattr(self.unwrapped, "vehicle", None)
        if ego_vehicle is None:
            return 0.0, self._traffic_flow_info(base_reward=0.0, penalty=0.0, shaped_reward=0.0)

        ego_speed = _forward_speed(ego_vehicle)
        front_ttc = compute_same_lane_ttc(self.env, ttc_cap=cfg["ttc_cap"])
        rear_ttc = compute_same_lane_rear_ttc(self.env, ttc_cap=cfg["ttc_cap"])
        flow_speed, rear_vehicle_count = self._rear_flow_speed(ego_vehicle)
        speed_deficit = max(0.0, flow_speed - ego_speed - float(cfg["speed_tolerance"]))
        front_safe_factor = 1.0 if front_ttc >= float(cfg["front_ttc_safe"]) else 0.0
        rear_pressure_factor = max(
            0.0,
            (float(cfg["rear_ttc_pressure"]) - rear_ttc) / float(cfg["rear_ttc_pressure"]),
        )
        if rear_vehicle_count > 0 and speed_deficit > 0.0:
            rear_pressure_factor = max(rear_pressure_factor, float(cfg["rear_pressure_floor"]))

        raw_penalty = (
            float(cfg["penalty_weight"])
            * speed_deficit
            * front_safe_factor
            * rear_pressure_factor
        )
        penalty = float(min(raw_penalty, float(cfg["max_penalty"])))
        metrics = {
            "traffic_front_ttc": float(front_ttc),
            "traffic_rear_ttc": float(rear_ttc),
            "traffic_flow_speed": float(flow_speed),
            "traffic_rear_vehicle_count": float(rear_vehicle_count),
            "traffic_speed_deficit": float(speed_deficit),
            "traffic_rear_pressure_factor": float(rear_pressure_factor),
        }
        return penalty, metrics

    def _rear_flow_speed(self, ego_vehicle) -> tuple[float, int]:
        road = getattr(self.unwrapped, "road", None)
        lane_index = getattr(ego_vehicle, "lane_index", None)
        if road is None or lane_index is None:
            return _forward_speed(ego_vehicle), 0

        candidate_lanes = self._candidate_lane_indices(lane_index)
        speeds: list[float] = []
        for candidate_lane_index in candidate_lanes:
            lane = road.network.get_lane(candidate_lane_index)
            ego_s = _lane_longitudinal(ego_vehicle, lane)
            for other in road.vehicles:
                if other is ego_vehicle or getattr(other, "lane_index", None) != candidate_lane_index:
                    continue
                other_s = _lane_longitudinal(other, lane)
                if other_s >= ego_s:
                    continue
                if ego_s - other_s > float(self.traffic_flow_reward_config["flow_radius"]):
                    continue
                speeds.append(_forward_speed(other))

        if not speeds:
            return _forward_speed(ego_vehicle), 0
        return float(np.mean(speeds)), len(speeds)

    def _candidate_lane_indices(self, lane_index) -> list[tuple]:
        lane_scope = str(self.traffic_flow_reward_config["lanes"]).lower()
        road = self.unwrapped.road
        lane_from, lane_to, lane_id = lane_index
        lane_count = len(road.network.graph[lane_from][lane_to])
        if lane_scope in {"current", "ego"}:
            return [lane_index]
        if lane_scope == "all":
            return [(lane_from, lane_to, idx) for idx in range(lane_count)]

        candidates = [lane_index]
        for delta in (-1, 1):
            candidate_id = int(lane_id) + delta
            if 0 <= candidate_id < lane_count:
                candidates.append((lane_from, lane_to, candidate_id))
        return candidates

    @staticmethod
    def _traffic_flow_info(base_reward: float, penalty: float, shaped_reward: float) -> dict[str, float]:
        return {
            "traffic_front_ttc": np.nan,
            "traffic_rear_ttc": np.nan,
            "traffic_flow_speed": np.nan,
            "traffic_rear_vehicle_count": 0.0,
            "traffic_speed_deficit": 0.0,
            "traffic_rear_pressure_factor": 0.0,
            "traffic_flow_penalty": float(penalty),
            "base_reward": float(base_reward),
            "shaped_reward": float(shaped_reward),
        }


class SafetyTTCFlowRewardWrapper(gym.Wrapper):
    """
    Reward safe front TTC, penalize short front TTC, and penalize lagging behind
    nearby rear traffic only when the front is already safe.
    """

    def __init__(
        self,
        env: gym.Env,
        safety_ttc_flow_reward_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.safety_ttc_flow_reward_config = build_safety_ttc_flow_reward_config(
            safety_ttc_flow_reward_config
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        info.update(self._safety_info(base_reward=0.0, shaped_reward=0.0))
        return observation, info

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        shaping, metrics = self._compute_shaping()
        shaped_reward = float(base_reward) + float(shaping)
        info = dict(info)
        info.update(metrics)
        info.update(
            {
                "safety_base_reward": float(base_reward),
                "safety_shaped_reward": float(shaped_reward),
                "safety_reward_shaping": float(shaping),
            }
        )
        return observation, shaped_reward, terminated, truncated, info

    def _compute_shaping(self) -> tuple[float, dict[str, float]]:
        if not self.safety_ttc_flow_reward_config["enabled"]:
            return 0.0, self._safety_info(base_reward=0.0, shaped_reward=0.0)

        cfg = self.safety_ttc_flow_reward_config
        ego_vehicle = getattr(self.unwrapped, "vehicle", None)
        if ego_vehicle is None:
            return 0.0, self._safety_info(base_reward=0.0, shaped_reward=0.0)

        ego_speed = _forward_speed(ego_vehicle)
        front_ttc = compute_same_lane_ttc(self.env, ttc_cap=cfg["ttc_cap"])
        rear_ttc = compute_same_lane_rear_ttc(self.env, ttc_cap=cfg["ttc_cap"])
        flow_speed, rear_vehicle_count = self._rear_flow_speed(ego_vehicle)

        safe_threshold = float(cfg["ttc_safe_threshold"])
        ttc_target = float(cfg["ttc_target"])
        low_ttc_shortfall = max(0.0, (safe_threshold - front_ttc) / safe_threshold)
        low_ttc_penalty = min(
            float(cfg["low_ttc_penalty_weight"]) * low_ttc_shortfall,
            float(cfg["max_low_ttc_penalty"]),
        )

        safe_ttc_progress = 0.0
        if ttc_target > safe_threshold:
            safe_ttc_progress = (front_ttc - safe_threshold) / (ttc_target - safe_threshold)
        safe_ttc_progress = float(np.clip(safe_ttc_progress, 0.0, 1.0))
        safe_ttc_bonus = min(
            float(cfg["safe_ttc_bonus_weight"]) * safe_ttc_progress,
            float(cfg["max_safe_ttc_bonus"]),
        )

        speed_deficit = max(0.0, flow_speed - ego_speed - float(cfg["speed_tolerance"]))
        front_safe_factor = 1.0 if front_ttc >= safe_threshold else 0.0
        rear_pressure_factor = max(
            0.0,
            (float(cfg["rear_ttc_pressure"]) - rear_ttc) / float(cfg["rear_ttc_pressure"]),
        )
        if rear_vehicle_count > 0 and speed_deficit > 0.0:
            rear_pressure_factor = max(rear_pressure_factor, float(cfg["rear_pressure_floor"]))

        lag_penalty = min(
            float(cfg["lag_penalty_weight"])
            * speed_deficit
            * front_safe_factor
            * rear_pressure_factor,
            float(cfg["max_lag_penalty"]),
        )
        shaping = float(safe_ttc_bonus - low_ttc_penalty - lag_penalty)
        metrics = {
            "safety_front_ttc": float(front_ttc),
            "safety_rear_ttc": float(rear_ttc),
            "safety_ttc_bonus": float(safe_ttc_bonus),
            "safety_low_ttc_penalty": float(low_ttc_penalty),
            "safety_lag_penalty": float(lag_penalty),
            "safety_flow_speed": float(flow_speed),
            "safety_rear_vehicle_count": float(rear_vehicle_count),
            "safety_speed_deficit": float(speed_deficit),
            "safety_rear_pressure_factor": float(rear_pressure_factor),
        }
        return shaping, metrics

    def _rear_flow_speed(self, ego_vehicle) -> tuple[float, int]:
        road = getattr(self.unwrapped, "road", None)
        lane_index = getattr(ego_vehicle, "lane_index", None)
        if road is None or lane_index is None:
            return _forward_speed(ego_vehicle), 0

        candidate_lanes = self._candidate_lane_indices(lane_index)
        speeds: list[float] = []
        for candidate_lane_index in candidate_lanes:
            lane = road.network.get_lane(candidate_lane_index)
            ego_s = _lane_longitudinal(ego_vehicle, lane)
            for other in road.vehicles:
                if other is ego_vehicle or getattr(other, "lane_index", None) != candidate_lane_index:
                    continue
                other_s = _lane_longitudinal(other, lane)
                if other_s >= ego_s:
                    continue
                if ego_s - other_s > float(self.safety_ttc_flow_reward_config["flow_radius"]):
                    continue
                speeds.append(_forward_speed(other))

        if not speeds:
            return _forward_speed(ego_vehicle), 0
        return float(np.mean(speeds)), len(speeds)

    def _candidate_lane_indices(self, lane_index) -> list[tuple]:
        lane_scope = str(self.safety_ttc_flow_reward_config["lanes"]).lower()
        road = self.unwrapped.road
        lane_from, lane_to, lane_id = lane_index
        lane_count = len(road.network.graph[lane_from][lane_to])
        if lane_scope in {"current", "ego"}:
            return [lane_index]
        if lane_scope == "all":
            return [(lane_from, lane_to, idx) for idx in range(lane_count)]

        candidates = [lane_index]
        for delta in (-1, 1):
            candidate_id = int(lane_id) + delta
            if 0 <= candidate_id < lane_count:
                candidates.append((lane_from, lane_to, candidate_id))
        return candidates

    @staticmethod
    def _safety_info(base_reward: float, shaped_reward: float) -> dict[str, float]:
        return {
            "safety_front_ttc": np.nan,
            "safety_rear_ttc": np.nan,
            "safety_ttc_bonus": 0.0,
            "safety_low_ttc_penalty": 0.0,
            "safety_lag_penalty": 0.0,
            "safety_flow_speed": np.nan,
            "safety_rear_vehicle_count": 0.0,
            "safety_speed_deficit": 0.0,
            "safety_rear_pressure_factor": 0.0,
            "safety_base_reward": float(base_reward),
            "safety_shaped_reward": float(shaped_reward),
            "safety_reward_shaping": 0.0,
        }


class PotentialFieldRewardWrapper(gym.Wrapper):
    """Penalize proximity to surrounding vehicles with an ellipsoidal potential field."""

    def __init__(
        self,
        env: gym.Env,
        potential_field_reward_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.potential_field_reward_config = build_potential_field_reward_config(
            potential_field_reward_config
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        info.update(self._potential_field_info(base_reward=0.0, shaped_reward=0.0))
        return observation, info

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        cost, metrics = self._compute_cost()
        penalty = float(self.potential_field_reward_config["weight"]) * float(cost)
        shaped_reward = float(base_reward) - penalty
        info = dict(info)
        info.update(metrics)
        info.update(
            {
                "potential_field_base_reward": float(base_reward),
                "potential_field_shaped_reward": float(shaped_reward),
                "potential_field_penalty": float(penalty),
            }
        )
        return observation, shaped_reward, terminated, truncated, info

    def _compute_cost(self) -> tuple[float, dict[str, float]]:
        if not self.potential_field_reward_config["enabled"]:
            return 0.0, self._potential_field_info(base_reward=0.0, shaped_reward=0.0)

        cfg = self.potential_field_reward_config
        ego_vehicle = getattr(self.unwrapped, "vehicle", None)
        road = getattr(self.unwrapped, "road", None)
        ego_lane_index = getattr(ego_vehicle, "lane_index", None)
        if ego_vehicle is None or road is None or ego_lane_index is None:
            return 0.0, self._potential_field_info(base_reward=0.0, shaped_reward=0.0)

        ego_lane = road.network.get_lane(ego_lane_index)
        ego_s, ego_lateral = ego_lane.local_coordinates(ego_vehicle.position)
        ego_speed = _forward_speed(ego_vehicle)
        ego_lateral_speed = _lateral_speed(ego_vehicle)
        candidate_lanes = set(self._candidate_lane_indices(ego_lane_index))

        vehicle_count = 0
        total_cost = 0.0
        max_vehicle_cost = 0.0
        closest_longitudinal_gap = float(cfg["sensing_range"])
        closest_lateral_gap = np.nan

        for other in getattr(road, "vehicles", []):
            if other is ego_vehicle:
                continue
            other_lane_index = getattr(other, "lane_index", None)
            if candidate_lanes and other_lane_index not in candidate_lanes:
                continue

            other_s, other_lateral = ego_lane.local_coordinates(other.position)
            dx = float(other_s - ego_s)
            if abs(dx) > float(cfg["sensing_range"]):
                continue

            dy = float(other_lateral - ego_lateral)
            longitudinal_gap = max(0.0, abs(dx) - 0.5 * (_vehicle_length(ego_vehicle) + _vehicle_length(other)))
            lateral_gap = max(0.0, abs(dy) - 0.5 * (_vehicle_width(ego_vehicle) + _vehicle_width(other)))
            closest_longitudinal_gap = min(closest_longitudinal_gap, longitudinal_gap)
            closest_lateral_gap = lateral_gap if np.isnan(closest_lateral_gap) else min(closest_lateral_gap, lateral_gap)

            if dx >= 0.0:
                closing_speed = ego_speed - _forward_speed(other)
            else:
                closing_speed = _forward_speed(other) - ego_speed
            lateral_closing_speed = abs(ego_lateral_speed - _lateral_speed(other))

            critical_a = max(
                0.5 * (_vehicle_length(ego_vehicle) + _vehicle_length(other)),
                float(cfg["min_longitudinal_scale"]),
            )
            critical_b = max(
                0.5 * (_vehicle_width(ego_vehicle) + _vehicle_width(other)),
                float(cfg["min_lateral_scale"]),
            )
            broad_a = critical_a + float(cfg["timegap"]) * max(closing_speed, 0.0)
            broad_b = critical_b + float(cfg["lateral_timegap"]) * lateral_closing_speed

            vehicle_cost = self._ellipsoid(dx, dy, critical_a, critical_b)
            vehicle_cost += self._ellipsoid(dx, dy, broad_a, broad_b)
            total_cost += vehicle_cost
            max_vehicle_cost = max(max_vehicle_cost, vehicle_cost)
            vehicle_count += 1

        cost = float(min(total_cost, float(cfg["max_cost"])))
        metrics = {
            "potential_field_cost": cost,
            "potential_field_vehicle_count": float(vehicle_count),
            "potential_field_max_vehicle_cost": float(max_vehicle_cost),
            "potential_field_closest_longitudinal_gap": float(closest_longitudinal_gap),
            "potential_field_closest_lateral_gap": float(closest_lateral_gap),
        }
        return cost, metrics

    def _candidate_lane_indices(self, lane_index) -> list[tuple]:
        lane_scope = str(self.potential_field_reward_config["lanes"]).lower()
        if lane_scope == "all":
            return []

        road = self.unwrapped.road
        lane_from, lane_to, lane_id = lane_index
        lane_count = len(road.network.graph[lane_from][lane_to])
        if lane_scope in {"current", "ego"}:
            return [lane_index]

        candidates = [lane_index]
        for delta in (-1, 1):
            candidate_id = int(lane_id) + delta
            if 0 <= candidate_id < lane_count:
                candidates.append((lane_from, lane_to, candidate_id))
        return candidates

    def _ellipsoid(self, dx: float, dy: float, a: float, b: float) -> float:
        cfg = self.potential_field_reward_config
        scaled = (
            (abs(float(dx)) / max(float(a), 1e-9)) ** float(cfg["field_px"])
            + (abs(float(dy)) / max(float(b), 1e-9)) ** float(cfg["field_py"])
            + 1.0
        )
        return float(float(cfg["field_magnitude"]) / (scaled ** float(cfg["field_pt"])))

    @staticmethod
    def _potential_field_info(base_reward: float, shaped_reward: float) -> dict[str, float]:
        return {
            "potential_field_cost": 0.0,
            "potential_field_penalty": 0.0,
            "potential_field_vehicle_count": 0.0,
            "potential_field_max_vehicle_cost": 0.0,
            "potential_field_closest_longitudinal_gap": np.nan,
            "potential_field_closest_lateral_gap": np.nan,
            "potential_field_base_reward": float(base_reward),
            "potential_field_shaped_reward": float(shaped_reward),
        }


class LaneChangeSafetyRewardWrapper(gym.Wrapper):
    """Penalize lane-change actions whose target lane has unsafe front/rear TTC or gap."""

    def __init__(
        self,
        env: gym.Env,
        lane_change_safety_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.lane_change_safety_config = build_lane_change_safety_config(lane_change_safety_config)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        info.update(self._lane_change_safety_info(side="none", penalty=0.0, risk=0.0))
        return observation, info

    def step(self, action):
        side = self._lane_change_side(action)
        penalty, risk, metrics = self._compute_penalty(side)
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = float(base_reward) - float(penalty)
        info = dict(info)
        info.update(metrics)
        info.update(self._lane_change_safety_info(side=side or "none", penalty=penalty, risk=risk))
        info["lane_change_safety_base_reward"] = float(base_reward)
        info["lane_change_safety_shaped_reward"] = float(shaped_reward)
        return observation, shaped_reward, terminated, truncated, info

    def _compute_penalty(self, side: str | None) -> tuple[float, float, dict[str, float]]:
        if not self.lane_change_safety_config["enabled"] or side is None:
            return 0.0, 0.0, self._target_metrics_info({})

        cfg = self.lane_change_safety_config
        target_lane_index = adjacent_lane_index(self.env, side)
        metrics = lane_front_rear_metrics(
            self.env,
            target_lane_index,
            ttc_cap=float(cfg["ttc_cap"]),
            gap_cap=float(cfg["gap_cap"]),
        )
        if not bool(metrics["available"]):
            risk = 1.0 if bool(cfg["penalize_unavailable_lane"]) else 0.0
            penalty = float(cfg["unavailable_lane_penalty"]) * risk
            return min(penalty, float(cfg["max_penalty"])), risk, self._target_metrics_info(metrics)

        risk_terms: list[float] = []
        if bool(cfg["use_ttc"]):
            risk_terms.extend(
                [
                    self._shortfall(float(metrics["front_ttc"]), float(cfg["front_ttc_safe"])),
                    self._shortfall(float(metrics["rear_ttc"]), float(cfg["rear_ttc_safe"])),
                ]
            )
        if bool(cfg["use_gap"]):
            risk_terms.extend(
                [
                    self._shortfall(float(metrics["front_gap"]), float(cfg["front_gap_safe"])),
                    self._shortfall(float(metrics["rear_gap"]), float(cfg["rear_gap_safe"])),
                ]
            )
        risk = max(risk_terms, default=0.0)
        raw_penalty = float(cfg["penalty_weight"]) * (risk ** float(cfg["penalty_power"]))
        penalty = float(min(raw_penalty, float(cfg["max_penalty"])))
        return penalty, risk, self._target_metrics_info(metrics)

    def _lane_change_side(self, action) -> str | None:
        action_label = self._action_label(int(np.asarray(action).item()))
        if action_label == "LANE_LEFT":
            return "left"
        if action_label == "LANE_RIGHT":
            return "right"
        return None

    def _action_label(self, action_index: int) -> str:
        actions = getattr(self.env.unwrapped.action_type, "actions", {})
        if isinstance(actions, dict):
            return str(actions.get(int(action_index), action_index))
        if 0 <= int(action_index) < len(actions):
            return str(actions[int(action_index)])
        return str(action_index)

    @staticmethod
    def _shortfall(value: float, safe_value: float) -> float:
        return float(np.clip((safe_value - value) / max(safe_value, 1e-6), 0.0, 1.0))

    @staticmethod
    def _target_metrics_info(metrics: Mapping[str, float | bool]) -> dict[str, float]:
        return {
            "lane_change_target_available": float(bool(metrics.get("available", False))),
            "lane_change_target_front_ttc": float(metrics.get("front_ttc", np.nan)),
            "lane_change_target_rear_ttc": float(metrics.get("rear_ttc", np.nan)),
            "lane_change_target_front_gap": float(metrics.get("front_gap", np.nan)),
            "lane_change_target_rear_gap": float(metrics.get("rear_gap", np.nan)),
        }

    @staticmethod
    def _lane_change_safety_info(side: str, penalty: float, risk: float) -> dict[str, float | str]:
        return {
            "lane_change_safety_side": str(side),
            "lane_change_safety_risk": float(risk),
            "lane_change_safety_penalty": float(penalty),
        }


class AdaptiveLongitudinalTTCWrapper(gym.Wrapper):
    """
    Reinterpret FASTER/SLOWER as TTC-adaptive target-speed updates while
    preserving the original 5-action discrete interface.
    """

    def __init__(
        self,
        env: gym.Env,
        adaptive_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.adaptive_config = build_adaptive_longitudinal_config(adaptive_config)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        action_index = self._to_action_index(action)
        requested_label = self._action_label(action_index)
        forwarded_action = action_index
        target_speed_before = self._current_target_speed()
        ttc_value = compute_same_lane_ttc(self.env, ttc_cap=self.adaptive_config["ttc_cap"])
        ttc_score = self._ttc_score(ttc_value)
        applied_delta = 0.0

        if self.adaptive_config["mode"] == "safe_speed_limiter":
            new_target_speed = self._safe_speed_limiter_target(requested_label, ttc_score)
            applied_delta = new_target_speed - target_speed_before
            self._set_target_speed(new_target_speed)
            if requested_label in {"FASTER", "SLOWER"}:
                forwarded_action = self._idle_action_index()
        elif self.adaptive_config["mode"] == "ttc_safety_override":
            new_target_speed = self._ttc_only_target_speed(ttc_score)
            applied_delta = new_target_speed - target_speed_before
            self._set_target_speed(new_target_speed)
            if requested_label in {"FASTER", "SLOWER"}:
                forwarded_action = self._idle_action_index()
        elif requested_label in {"FASTER", "SLOWER"}:
            applied_delta = self._adaptive_delta(requested_label, ttc_score)
            self._apply_target_speed_delta(applied_delta)
            forwarded_action = self._idle_action_index()

        obs, reward, terminated, truncated, info = self.env.step(forwarded_action)
        unsafe_request = (
            self.adaptive_config["mode"] == "ttc_safety_override"
            and requested_label == "FASTER"
            and ttc_value < float(self.adaptive_config["safety_ttc_threshold"])
        )
        safety_penalty = float(self.adaptive_config["unsafe_action_penalty"] if unsafe_request else 0.0)
        reward = float(reward) - safety_penalty
        info = dict(info)
        info.update(
            {
                "adaptive_enabled": True,
                "adaptive_ttc": float(ttc_value),
                "adaptive_ttc_score": float(ttc_score),
                "adaptive_requested_action": str(requested_label),
                "adaptive_forwarded_action": str(self._action_label(forwarded_action)),
                "adaptive_mode": str(self.adaptive_config["mode"]),
                "adaptive_speed_delta": float(applied_delta),
                "adaptive_target_speed_before": float(target_speed_before),
                "adaptive_target_speed_after": float(self._current_target_speed()),
                "adaptive_safety_override_penalty": float(safety_penalty),
                "adaptive_unsafe_speed_request": bool(unsafe_request),
            }
        )
        return obs, reward, terminated, truncated, info

    def _to_action_index(self, action) -> int:
        return int(np.asarray(action).item())

    def _action_label(self, action_index: int) -> str:
        actions = getattr(self.env.unwrapped.action_type, "actions", {})
        if isinstance(actions, dict):
            return str(actions.get(int(action_index), action_index))
        if 0 <= int(action_index) < len(actions):
            return str(actions[int(action_index)])
        return str(action_index)

    def _idle_action_index(self) -> int:
        actions_indexes = getattr(self.env.unwrapped.action_type, "actions_indexes", {})
        idle_index = actions_indexes.get("IDLE")
        if idle_index is None:
            raise KeyError("Expected highway-env DiscreteMetaAction to define an IDLE action")
        return int(idle_index)

    def _current_target_speed(self) -> float:
        vehicle = self.env.unwrapped.vehicle
        return float(getattr(vehicle, "target_speed", getattr(vehicle, "speed", 0.0)))

    def _ttc_score(self, ttc_value: float) -> float:
        midpoint = self.adaptive_config["ttc_midpoint"]
        temperature = self.adaptive_config["ttc_temperature"]
        normalized = (float(ttc_value) - midpoint) / temperature
        return float(1.0 / (1.0 + np.exp(-normalized)))

    def _adaptive_delta(self, requested_label: str, ttc_score: float) -> float:
        if requested_label == "FASTER":
            return float(self.adaptive_config["faster_max_delta"] * ttc_score)
        if requested_label == "SLOWER":
            slower_min = self.adaptive_config["slower_min_delta"]
            slower_max = self.adaptive_config["slower_max_delta"]
            return float(-(slower_min + (slower_max - slower_min) * (1.0 - ttc_score)))
        return 0.0

    def _safe_speed_limiter_target(self, requested_label: str, ttc_score: float) -> float:
        desired_speed = float(self.adaptive_config["cruise_speed"])
        action_delta = float(self.adaptive_config["action_speed_delta"])
        if requested_label == "FASTER":
            desired_speed += action_delta
        elif requested_label == "SLOWER":
            desired_speed -= action_delta
        desired_speed = float(
            np.clip(
                desired_speed,
                self.adaptive_config["min_target_speed"],
                self.adaptive_config["max_target_speed"],
            )
        )
        min_speed = float(self.adaptive_config["min_target_speed"])
        safety_limited_speed = min_speed + (desired_speed - min_speed) * float(ttc_score)
        return float(
            np.clip(
                safety_limited_speed,
                self.adaptive_config["min_target_speed"],
                self.adaptive_config["max_target_speed"],
            )
        )

    def _ttc_only_target_speed(self, ttc_score: float) -> float:
        min_speed = float(self.adaptive_config["min_target_speed"])
        max_speed = float(self.adaptive_config["max_target_speed"])
        target_speed = min_speed + (max_speed - min_speed) * float(ttc_score)
        return float(np.clip(target_speed, min_speed, max_speed))

    def _apply_target_speed_delta(self, delta: float) -> None:
        new_target_speed = float(
            np.clip(
                self._current_target_speed() + float(delta),
                self.adaptive_config["min_target_speed"],
                self.adaptive_config["max_target_speed"],
            )
        )
        self._set_target_speed(new_target_speed)

    def _set_target_speed(self, new_target_speed: float) -> None:
        vehicle = self.env.unwrapped.vehicle
        if hasattr(vehicle, "target_speed"):
            vehicle.target_speed = new_target_speed
        else:
            vehicle.speed = new_target_speed


def make_highway_env_with_adaptive_longitudinal(
    *,
    render_mode: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> gym.Env:
    (
        base_config,
        adaptive_config,
        rear_flow_config,
        traffic_flow_reward_config,
        safety_ttc_flow_reward_config,
        potential_field_reward_config,
        driver_aggressiveness_config,
        driver_aggressiveness_observation_config,
        ttc_observation_config,
        lane_change_safety_config,
    ) = split_highway_env_and_custom_configs(config)
    env = gym.make("highway-v0", render_mode=render_mode, config=base_config)
    if rear_flow_config["enabled"]:
        env = RearFlowPressureWrapper(env, rear_flow_config=rear_flow_config)
    if driver_aggressiveness_config["enabled"]:
        env = DriverAggressivenessWrapper(
            env,
            driver_aggressiveness_config=driver_aggressiveness_config,
        )
    if potential_field_reward_config["enabled"]:
        env = PotentialFieldRewardWrapper(
            env,
            potential_field_reward_config=potential_field_reward_config,
        )
    if safety_ttc_flow_reward_config["enabled"]:
        env = SafetyTTCFlowRewardWrapper(
            env,
            safety_ttc_flow_reward_config=safety_ttc_flow_reward_config,
        )
    if traffic_flow_reward_config["enabled"]:
        env = TrafficFlowRewardWrapper(
            env,
            traffic_flow_reward_config=traffic_flow_reward_config,
        )
    if lane_change_safety_config["enabled"]:
        env = LaneChangeSafetyRewardWrapper(
            env,
            lane_change_safety_config=lane_change_safety_config,
        )
    if adaptive_config["enabled"]:
        env = AdaptiveLongitudinalTTCWrapper(env, adaptive_config=adaptive_config)
    if ttc_observation_config["enabled"]:
        env = TTCObservationWrapper(env, ttc_observation_config=ttc_observation_config)
    if driver_aggressiveness_observation_config["enabled"]:
        env = DriverAggressivenessObservationWrapper(
            env,
            driver_aggressiveness_observation_config=driver_aggressiveness_observation_config,
        )
    return env
