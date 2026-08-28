"""Compare how the saved PPO policies decide when and how to overtake.

This is a targeted policy-diagnostic protocol rather than a benchmark score.
Each policy is run on the same hand-authored traffic scene, with a slow
designated blocker ahead of the ego and an open passing side.  The script
produces two complementary kinds of evidence:

* event/phase aligned freeze-frame storyboards, where every policy is shown at
  opportunity, raw intent, operational intent, abeam, and clear stages; and
* live MP4 dashboards, where the simulator render is shown beside all seven
  same-state action proposals, the active policy's raw/internal/operational
  stages, a shadow external-CBF projection, and rolling target/safety traces.

The external CBF is off for the actual branch by default.  Its projection is
computed as a shadow diagnostic so the learned policy's own behavior is not
confounded with a deployment shield.  Pass ``--apply-external-cbf`` to make
the shadow external projection the action actually sent to the simulator.
The simulator's ordinary traffic dynamics guard remains enabled.

The saved models use the policy-step clock in their metadata (10 Hz for the
current checkpoints), while their configured physics clock is 100 Hz.  The
video therefore writes one dashboard frame per policy step.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_final_results_policies as base_analysis
import visualize_policy_changes as policy_diagnostics
from cbf_projection import project_polytope_2d_numpy


VARIANT_ORDER = tuple(base_analysis.VARIANT_ORDER)
VARIANT_LABELS = dict(base_analysis.VARIANT_LABELS)
VARIANT_COLORS = dict(base_analysis.VARIANT_COLORS)

ROAD_LENGTH_M = 380.0
ROAD_WIDTH_M = 10.2
ACTION_LOW = np.asarray([-3.0, -3.0], dtype=np.float32)
ACTION_HIGH = np.asarray([3.0, 3.0], dtype=np.float32)
INTERVENTION_THRESHOLD = float(policy_diagnostics.INTERVENTION_THRESHOLD)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "final_Results"
    / "policy_analysis"
    / "overtake_diagnostics"
)
DEFAULT_SCENARIOS = ("open_upper", "open_lower")
DEFAULT_SEEDS = (4242, 4243, 4244)
DEFAULT_MAX_STEPS = 120
DEFAULT_VIDEO_FPS = 10


@dataclass(frozen=True)
class VehicleSpec:
    """One deterministic vehicle relative to the ego's initial x position."""

    label: str
    dx: float
    y: float
    vx: float
    vy: float = 0.0
    role: str = "traffic"
    length: float = 3.5
    width: float = 1.8


@dataclass(frozen=True)
class OvertakeScenario:
    scenario_id: str
    title: str
    pass_side: int
    vehicles: tuple[VehicleSpec, ...]
    target_index: int = 1
    ego_x: float = 100.0
    ego_y: float = 5.1


def scenario_catalog() -> dict[str, OvertakeScenario]:
    """Return deterministic scenes with a common center-lane blocker.

    y increases toward the upper side of the road.  The two open scenes are
    mirror images, which exposes directional asymmetry in the learned policy.
    The tight scene is useful for visualizing a late/abortable decision.
    """

    center = 5.1
    upper = 8.3
    lower = 2.9
    return {
        "open_upper": OvertakeScenario(
            "open_upper",
            "Slow center blocker; upper side open",
            +1,
            (
                VehicleSpec("ego", 0.0, center, 16.0, role="ego"),
                VehicleSpec("slow blocker", 26.0, center, 11.0, role="target"),
                VehicleSpec("upper lead", 48.0, upper, 19.5, role="pass_lead"),
                VehicleSpec("upper rear", -32.0, upper, 18.0, role="pass_rear"),
                VehicleSpec("lower lead", 35.0, lower, 18.5, role="traffic"),
                VehicleSpec("far center lead", 72.0, 5.4, 20.0, role="traffic"),
            ),
        ),
        "open_lower": OvertakeScenario(
            "open_lower",
            "Slow center blocker; lower side open",
            -1,
            (
                VehicleSpec("ego", 0.0, center, 16.0, role="ego"),
                VehicleSpec("slow blocker", 26.0, center, 11.0, role="target"),
                VehicleSpec("lower lead", 48.0, lower, 19.5, role="pass_lead"),
                VehicleSpec("lower rear", -32.0, lower, 18.0, role="pass_rear"),
                VehicleSpec("upper lead", 35.0, upper, 18.5, role="traffic"),
                VehicleSpec("far center lead", 72.0, 4.8, 20.0, role="traffic"),
            ),
        ),
        "tight_upper": OvertakeScenario(
            "tight_upper",
            "Slow blocker; upper side narrows late",
            +1,
            (
                VehicleSpec("ego", 0.0, center, 16.0, role="ego"),
                VehicleSpec("slow blocker", 24.0, center, 10.0, role="target"),
                VehicleSpec("upper lead", 32.0, upper, 12.5, role="pass_lead"),
                VehicleSpec("upper rear", -18.0, upper, 19.0, role="pass_rear"),
                VehicleSpec("lower lead", 38.0, lower, 18.5, role="traffic"),
                VehicleSpec("far center lead", 70.0, 5.4, 20.0, role="traffic"),
            ),
        ),
    }


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _as_action(value: Any) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32).reshape(-1)[:2]
    if action.size < 2:
        action = np.pad(action, (0, 2 - action.size))
    return np.nan_to_num(action, nan=0.0, posinf=3.0, neginf=-3.0).astype(
        np.float32
    )


def _episode_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["scenario_id"].astype(str)
        + "|"
        + frame["variant_id"].astype(str)
        + "|"
        + frame["scenario_seed"].astype(str)
    )


def _unwrap_periodic(values: Iterable[float], period: float = ROAD_LENGTH_M) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return array
    phase = array / float(period) * 2.0 * np.pi
    return np.unwrap(phase) / (2.0 * np.pi) * float(period)


def _find_wrapper(env: Any, predicate: Any) -> Any:
    current = env
    for _ in range(24):
        if predicate(current):
            return current
        current = getattr(current, "env", None)
        if current is None:
            break
    return None


def _scenario_config(
    specs: dict[str, dict[str, Any]], scenario: OvertakeScenario, max_steps: int
) -> dict[str, Any]:
    """Use the saved B3.2 simulator contract, changing only scene size/horizon."""

    config = copy.deepcopy(specs["B3_2"]["config"]["env_config"])
    simulation_frequency = 100
    policy_frequency = 10
    physics_steps = int(max_steps * simulation_frequency / policy_frequency)
    config.update(
        {
            "vehicles_count": len(scenario.vehicles),
            "neighbors_count": min(5, len(scenario.vehicles) - 1),
            # lane_free_env counts episode_steps in physics frames, whereas
            # max_steps here is expressed in policy frames.
            "episode_steps": physics_steps,
            "duration": physics_steps,
            "simulation_frequency": simulation_frequency,
            "policy_frequency": policy_frequency,
            "cbf_substep_filtering": False,
            "cbf_require_initial_safe_set": False,
            "real_time_rendering": False,
            "show_trajectories": False,
            "screen_width": 1000,
            "screen_height": 300,
            "offscreen_rendering": True,
        }
    )
    # Preserve the source ordinary traffic dynamics guard.  This is important
    # because the fixed-scene comparison should not allow social traffic to
    # become an artificial source of collisions.
    traffic_safety = copy.deepcopy(config.get("traffic_safety", {}))
    if isinstance(traffic_safety, dict):
        traffic_safety["dynamics_guard"] = True
        config["traffic_safety"] = traffic_safety
    return config


def _policy_observation_after_mutation(env: Any, context: Any) -> np.ndarray:
    """Rebuild the exact 104-D observation after editing vehicles in place.

    The base simulator exposes 30 features.  The saved PPO checkpoints use
    the notebook's previous-action variant (30 + 2), followed by the padded
    CBF context.  Calling ``base._observe()`` directly therefore needs the
    same two-feature insertion that a normal reset would perform.
    """

    base = env.unwrapped
    observation = np.asarray(base._observe(), dtype=np.float32).reshape(-1)
    augmenters: list[Any] = []
    current = env
    while current is not None:
        if hasattr(current, "_augment_observation"):
            augmenters.append(current)
        current = getattr(current, "env", None)

    for wrapper in reversed(augmenters):
        if wrapper is context:
            expected = int(context.layout.base_observation_dim)
            if observation.size == expected - 2:
                previous = getattr(context, "_previous_executed_action_normalized", None)
                if previous is None:
                    previous = np.zeros(2, dtype=np.float32)
                observation = np.concatenate(
                    (observation, np.asarray(previous, dtype=np.float32).reshape(-1)[:2])
                )
            if observation.size != expected:
                raise RuntimeError(
                    "Scenario observation did not match the saved PPO base width: "
                    f"{observation.size} != {expected}"
                )
        observation = wrapper._augment_observation(observation)
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    expected_total = int(np.prod(env.observation_space.shape))
    if observation.size != expected_total:
        raise RuntimeError(
            f"Scenario observation width {observation.size} != environment width {expected_total}"
        )
    return observation


def _apply_scenario(env: Any, scenario: OvertakeScenario) -> np.ndarray:
    """Overwrite the reset state while keeping all simulator bookkeeping valid."""

    base = env.unwrapped
    context = _find_wrapper(env, lambda item: hasattr(item, "current_constraint_system"))
    if context is None:
        raise RuntimeError("Could not locate the CBF context wrapper")
    vehicles = list(base.road.vehicles)
    if len(vehicles) != len(scenario.vehicles):
        raise RuntimeError(
            f"Scenario {scenario.scenario_id} needs {len(scenario.vehicles)} vehicles, "
            f"but the environment created {len(vehicles)}"
        )
    road_length = float(base.config.get("road_length", ROAD_LENGTH_M))
    for index, (vehicle, spec) in enumerate(zip(vehicles, scenario.vehicles)):
        vehicle.position[0] = float((scenario.ego_x + spec.dx) % road_length)
        vehicle.position[1] = float(spec.y)
        vehicle.vx = float(spec.vx)
        vehicle.vy = float(spec.vy)
        vehicle.heading = 0.0
        vehicle.length = float(spec.length)
        vehicle.width = float(spec.width)
        vehicle.LENGTH = float(spec.length)
        vehicle.WIDTH = float(spec.width)
        vehicle.desired_speed = float(max(spec.vx, 1.0))
        vehicle.is_ego = index == 0
        vehicle.crashed = False
        vehicle.hit = False
        if hasattr(vehicle, "_sync_graphics_fields"):
            vehicle._sync_graphics_fields()

    base.vehicle = vehicles[0]
    base.controlled_vehicles = [vehicles[0]]
    base._last_action = np.zeros(2, dtype=np.float32)
    base._last_accelerations = np.zeros((len(vehicles), 2), dtype=float)
    for name, value in {
        "_last_boundary_violations": 0,
        "_last_collision_count": 0,
        "_last_active_collision_count": 0,
        "_last_ego_collision_count": 0,
        "_last_ego_collision": False,
        "_cumulative_collision_count": 0,
        "_active_collision_pairs": set(),
        "_flow_count": 0,
    }.items():
        if hasattr(base, name):
            setattr(base, name, copy.deepcopy(value))
    if hasattr(context, "_previous_executed_action_normalized"):
        context._previous_executed_action_normalized = None

    ego_position = np.asarray(base.vehicle.position[:2], dtype=float).copy()
    for wrapper in (env, getattr(env, "env", None), context):
        if wrapper is not None and hasattr(wrapper, "_previous_position"):
            wrapper._previous_position = ego_position.copy()
    return _policy_observation_after_mutation(env, context)


def _vehicle_velocity(vehicle: Any) -> tuple[float, float]:
    velocity = getattr(vehicle, "velocity", np.zeros(2, dtype=float))
    return (
        _finite(getattr(vehicle, "vx", velocity[0]), 0.0),
        _finite(getattr(vehicle, "vy", velocity[1]), 0.0),
    )


def _vehicle_snapshot(
    env: Any,
    scenario: OvertakeScenario,
    variant_id: str,
    seed: int,
    step: int,
    time_s: float,
) -> list[dict[str, Any]]:
    base = env.unwrapped
    ego = base.vehicle
    rows: list[dict[str, Any]] = []
    for index, vehicle in enumerate(base.road.vehicles):
        vx, vy = _vehicle_velocity(vehicle)
        try:
            relative_x = float(base._signed_distance(ego.position[0], vehicle.position[0]))
        except (AttributeError, TypeError, ValueError):
            relative_x = float(vehicle.position[0] - ego.position[0])
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "variant_id": variant_id,
                "scenario_seed": int(seed),
                "policy_step": int(step),
                "time_s": float(time_s),
                "vehicle_index": int(index),
                "vehicle_token": f"v{index}",
                "vehicle_label": scenario.vehicles[index].label,
                "vehicle_role": scenario.vehicles[index].role,
                "is_ego": bool(vehicle is ego),
                "relative_x_m": relative_x,
                "relative_y_m": float(vehicle.position[1] - ego.position[1]),
                "absolute_x_m": float(vehicle.position[0]),
                "absolute_y_m": float(vehicle.position[1]),
                "vx_mps": vx,
                "vy_mps": vy,
                "crashed": bool(getattr(vehicle, "crashed", False)),
            }
        )
    return rows


def _query_proposals(
    models: dict[str, Any], observation: np.ndarray, system: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Query every actor on the same observation and CBF context."""

    proposals: dict[str, dict[str, Any]] = {}
    half_range = np.maximum(0.5 * (ACTION_HIGH - ACTION_LOW), 1e-6)
    rows = np.asarray(system.get("rows", ()), dtype=float).reshape(-1, 2)
    bounds = np.asarray(system.get("bounds", ()), dtype=float).reshape(-1)
    for variant_id in VARIANT_ORDER:
        stages = policy_diagnostics._query_stages(models[variant_id], observation)
        raw = _as_action(stages["mu_raw"])
        internal = _as_action(stages["mu_safe"])
        operational = internal if variant_id.startswith("B3") else raw
        raw_box = np.clip(raw, ACTION_LOW, ACTION_HIGH).astype(np.float32)
        operational_box = np.clip(operational, ACTION_LOW, ACTION_HIGH).astype(np.float32)
        external_raw = project_polytope_2d_numpy(
            raw, rows, bounds, action_low=ACTION_LOW, action_high=ACTION_HIGH
        )
        external_operational = project_polytope_2d_numpy(
            operational, rows, bounds, action_low=ACTION_LOW, action_high=ACTION_HIGH
        )
        external_raw_action = _as_action(external_raw.action)
        external_operational_action = _as_action(external_operational.action)
        raw_margin = policy_diagnostics._hocbf_margin(system, raw)
        internal_margin = policy_diagnostics._hocbf_margin(system, internal)
        operational_margin = policy_diagnostics._hocbf_margin(system, operational)
        external_margin = policy_diagnostics._hocbf_margin(
            system, external_operational_action
        )
        raw_delta = external_raw_action - raw_box
        internal_delta = internal - raw
        proposals[variant_id] = {
            "variant_id": variant_id,
            "label": VARIANT_LABELS[variant_id],
            "raw": raw,
            "raw_box": raw_box,
            "internal": internal,
            "operational": operational,
            "operational_box": operational_box,
            "external_raw": external_raw_action,
            "external_operational": external_operational_action,
            "raw_ax": float(raw[0]),
            "raw_ay": float(raw[1]),
            "internal_ax": float(internal[0]),
            "internal_ay": float(internal[1]),
            "operational_ax": float(operational[0]),
            "operational_ay": float(operational[1]),
            "raw_hocbf_margin": float(raw_margin),
            "internal_hocbf_margin": float(internal_margin),
            "operational_hocbf_margin": float(operational_margin),
            "external_hocbf_margin": float(external_margin),
            "internal_correction_norm": float(np.linalg.norm(internal_delta / half_range)),
            "external_raw_correction_norm": float(np.linalg.norm(raw_delta / half_range)),
            "external_operational_correction_norm": float(
                np.linalg.norm((external_operational_action - operational_box) / half_range)
            ),
            "raw_feasible": bool(
                external_raw.feasible
                and np.all(raw >= ACTION_LOW - 1e-6)
                and np.all(raw <= ACTION_HIGH + 1e-6)
                and raw_margin >= -1e-6
            ),
            "operational_feasible": bool(
                external_operational.feasible and operational_margin >= -1e-6
            ),
            "external_feasible": bool(external_raw.feasible),
            "external_fallback": bool(external_raw.fallback_used),
            "external_source": str(external_raw.source),
            "external_intervention": bool(
                np.linalg.norm(raw_delta / half_range) > INTERVENTION_THRESHOLD
            ),
        }
    return proposals


def _proposal_row(
    proposal: dict[str, Any],
    *,
    scenario: OvertakeScenario,
    active_variant: str,
    seed: int,
    step: int,
    time_s: float,
    target_dx_m: float,
    target_dy_m: float,
    ego_y_m: float,
    ego_vx_mps: float,
    phase: str,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_title": scenario.title,
        "scenario_seed": int(seed),
        "active_variant_id": active_variant,
        "variant_id": proposal["variant_id"],
        "variant_label": proposal["label"],
        "policy_step": int(step),
        "time_s": float(time_s),
        "target_dx_m": float(target_dx_m),
        "target_dy_m": float(target_dy_m),
        "ego_y_m": float(ego_y_m),
        "ego_vx_mps": float(ego_vx_mps),
        "pass_side": int(scenario.pass_side),
        "phase_online": phase,
        "raw_ax": proposal["raw_ax"],
        "raw_ay": proposal["raw_ay"],
        "internal_ax": proposal["internal_ax"],
        "internal_ay": proposal["internal_ay"],
        "operational_ax": proposal["operational_ax"],
        "operational_ay": proposal["operational_ay"],
        "raw_box_ax": float(proposal["raw_box"][0]),
        "raw_box_ay": float(proposal["raw_box"][1]),
        "operational_box_ax": float(proposal["operational_box"][0]),
        "operational_box_ay": float(proposal["operational_box"][1]),
        "external_raw_ax": float(proposal["external_raw"][0]),
        "external_raw_ay": float(proposal["external_raw"][1]),
        "external_operational_ax": float(proposal["external_operational"][0]),
        "external_operational_ay": float(proposal["external_operational"][1]),
        "raw_hocbf_margin": proposal["raw_hocbf_margin"],
        "internal_hocbf_margin": proposal["internal_hocbf_margin"],
        "operational_hocbf_margin": proposal["operational_hocbf_margin"],
        "external_hocbf_margin": proposal["external_hocbf_margin"],
        "internal_correction_norm": proposal["internal_correction_norm"],
        "external_raw_correction_norm": proposal["external_raw_correction_norm"],
        "external_operational_correction_norm": proposal[
            "external_operational_correction_norm"
        ],
        "raw_feasible": proposal["raw_feasible"],
        "operational_feasible": proposal["operational_feasible"],
        "external_intervention": proposal["external_intervention"],
        "external_projection_feasible": proposal["external_feasible"],
        "external_projection_fallback": proposal["external_fallback"],
        "external_projection_source": proposal["external_source"],
    }


def _online_phase(
    scenario: OvertakeScenario, target_dx_m: float, target_dy_m: float, raw_ay: float, ego_y: float
) -> str:
    signed_lateral = float(scenario.pass_side) * float(ego_y - scenario.ego_y)
    signed_action = float(scenario.pass_side) * float(raw_ay)
    if target_dx_m > 35.0:
        return "approach"
    if signed_action >= 0.08 or signed_lateral >= 0.25:
        return "intent / commit"
    if target_dx_m > 0.0:
        return "closing / abeam"
    if signed_lateral >= 1.7:
        return "clear"
    return "settle / follow"


def _transition_metrics(info: dict[str, Any]) -> dict[str, Any]:
    info = dict(info)
    collision_events = max(
        int(_finite(info.get("pipeline_distinct_ego_collision_events", info.get("ego_collision_events", 0)), 0.0)),
        0,
    )
    active_collision = _bool(
        info.get("ego_collision", info.get("pipeline_ego_collision_active_timestep", False))
    )
    guards = {
        key: _finite(info.get(key, 0.0), 0.0)
        for key in (
            "traffic_guard_brakes",
            "traffic_guard_traffic_only",
            "traffic_guard_lateral_yields",
            "traffic_guard_ego_emergency_interventions",
            "traffic_guard_traffic_constraints",
        )
    }
    return {
        **guards,
        "collision_events_step": collision_events,
        "ego_collision_active": bool(active_collision),
        "all_pair_collision_events": max(
            int(_finite(info.get("pipeline_distinct_all_pair_collision_events", info.get("collisions", 0)), 0.0)),
            0,
        ),
        "active_collision_pairs": max(
            int(_finite(info.get("pipeline_active_collision_pairs", info.get("active_collisions", 0)), 0.0)),
            0,
        ),
        "pipeline_distance_step_m": _finite(info.get("pipeline_distance_step_m"), 0.0),
        "terminated": bool(info.get("terminated", False)),
    }


def _trace_row(
    *,
    scenario: OvertakeScenario,
    variant_id: str,
    seed: int,
    step: int,
    time_s: float,
    ego: dict[str, Any],
    target: Any,
    geometry: dict[str, Any],
    proposal: dict[str, Any],
    transition: dict[str, Any],
    outcome: str,
    executed: np.ndarray,
    external_applied: bool,
) -> dict[str, Any]:
    target_vx, target_vy = _vehicle_velocity(target)
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_title": scenario.title,
        "variant_id": variant_id,
        "variant_label": VARIANT_LABELS[variant_id],
        "scenario_seed": int(seed),
        "policy_step": int(step),
        "time_s": float(time_s),
        "pass_side": int(scenario.pass_side),
        "target_token": f"v{scenario.target_index}",
        "target_vx_mps": target_vx,
        "target_vy_mps": target_vy,
        "raw_ax": proposal["raw_ax"],
        "raw_ay": proposal["raw_ay"],
        "internal_ax": proposal["internal_ax"],
        "internal_ay": proposal["internal_ay"],
        "operational_ax": proposal["operational_ax"],
        "operational_ay": proposal["operational_ay"],
        "executed_ax": float(executed[0]),
        "executed_ay": float(executed[1]),
        "executed_source": "external_cbf" if external_applied else "policy_operational",
        "raw_box_ax": float(proposal["raw_box"][0]),
        "raw_box_ay": float(proposal["raw_box"][1]),
        "operational_box_ax": float(proposal["operational_box"][0]),
        "operational_box_ay": float(proposal["operational_box"][1]),
        "shadow_external_raw_ax": float(proposal["external_raw"][0]),
        "shadow_external_raw_ay": float(proposal["external_raw"][1]),
        "shadow_external_operational_ax": float(proposal["external_operational"][0]),
        "shadow_external_operational_ay": float(proposal["external_operational"][1]),
        "raw_hocbf_margin": proposal["raw_hocbf_margin"],
        "internal_hocbf_margin": proposal["internal_hocbf_margin"],
        "operational_hocbf_margin": proposal["operational_hocbf_margin"],
        "shadow_external_hocbf_margin": proposal["external_hocbf_margin"],
        "internal_mean_correction_norm": proposal["internal_correction_norm"],
        "shadow_external_correction_norm": proposal[
            "external_operational_correction_norm"
        ],
        "shadow_external_raw_correction_norm": proposal[
            "external_raw_correction_norm"
        ],
        "raw_feasible": proposal["raw_feasible"],
        "operational_feasible": proposal["operational_feasible"],
        "shadow_external_intervention": proposal["external_intervention"],
        "overall_h": geometry.get("overall_h", np.nan),
        "neighbor_h_min": geometry.get("neighbor_h_min", np.nan),
        "critical_clearance_m": geometry.get("critical_clearance_m", np.nan),
        "critical_ttc_s": geometry.get("critical_ttc_s", np.nan),
        "front_gap_m": geometry.get("front_gap_m", np.nan),
        "left_boundary_h": geometry.get("left_boundary_h", np.nan),
        "right_boundary_h": geometry.get("right_boundary_h", np.nan),
        "neighbor_count": geometry.get("neighbor_count", 0),
        "ego_x_m": _finite(ego.get("x")),
        "ego_y_m": _finite(ego.get("y")),
        "ego_vx_mps": _finite(ego.get("vx")),
        "ego_vy_mps": _finite(ego.get("vy")),
        "collision_events_step": transition["collision_events_step"],
        "ego_collision_active": transition["ego_collision_active"],
        "all_pair_collision_events": transition["all_pair_collision_events"],
        "active_collision_pairs": transition["active_collision_pairs"],
        "traffic_guard_brakes": transition.get("traffic_guard_brakes", 0.0),
        "traffic_guard_traffic_only": transition.get("traffic_guard_traffic_only", 0.0),
        "traffic_guard_lateral_yields": transition.get("traffic_guard_lateral_yields", 0.0),
        "traffic_guard_ego_emergency_interventions": transition.get(
            "traffic_guard_ego_emergency_interventions", 0.0
        ),
        "traffic_guard_traffic_constraints": transition.get(
            "traffic_guard_traffic_constraints", 0.0
        ),
        "outcome": outcome,
    }


def _video_writer(path: Path, fps: int, width: int, height: int) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for live HUD MP4 output") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height))
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def _write_video_frame(writer: Any, frame_rgb: np.ndarray) -> None:
    import cv2

    writer.write(cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))


def _render_frame(env: Any) -> np.ndarray:
    frame = env.unwrapped.render()
    if frame is None:
        raise RuntimeError("Simulator returned no rgb_array frame")
    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(f"Unexpected simulator frame shape {array.shape}")
    return array


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.52,
    color: tuple[int, int, int] = (35, 35, 35),
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(
        image,
        str(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _panel(image: np.ndarray, x: int, y: int, width: int, height: int, title: str) -> None:
    import cv2

    cv2.rectangle(image, (x, y), (x + width, y + height), (214, 218, 224), 1)
    _put_text(image, title, (x + 10, y + 22), scale=0.52, thickness=1)


def _map_action(point: np.ndarray, x: int, y: int, width: int, height: int) -> tuple[int, int]:
    px = int(round(x + (float(point[0]) + 3.0) / 6.0 * width))
    py = int(round(y + (3.0 - float(point[1])) / 6.0 * height))
    return px, py


def _polytope_vertices(system: dict[str, Any]) -> np.ndarray:
    """Find the vertices of the active 2-D action polytope for the HUD."""

    rows = np.asarray(system.get("rows", ()), dtype=float).reshape(-1, 2)
    bounds = np.asarray(system.get("bounds", ()), dtype=float).reshape(-1)
    constraints = [(row, bound) for row, bound in zip(rows, bounds)]
    constraints.extend(
        [
            (np.asarray([1.0, 0.0]), 3.0),
            (np.asarray([-1.0, 0.0]), 3.0),
            (np.asarray([0.0, 1.0]), 3.0),
            (np.asarray([0.0, -1.0]), 3.0),
        ]
    )
    candidates: list[np.ndarray] = []
    for index, (a, b) in enumerate(constraints):
        for other_a, other_b in constraints[index + 1 :]:
            matrix = np.vstack((a, other_a))
            determinant = float(np.linalg.det(matrix))
            if abs(determinant) < 1e-9:
                continue
            try:
                point = np.linalg.solve(matrix, np.asarray([b, other_b], dtype=float))
            except np.linalg.LinAlgError:
                continue
            if np.all(rows @ point <= bounds + 1e-6) if rows.size else True:
                if np.all(point >= ACTION_LOW - 1e-6) and np.all(point <= ACTION_HIGH + 1e-6):
                    if not any(np.linalg.norm(point - old) <= 1e-5 for old in candidates):
                        candidates.append(point)
    if len(candidates) < 3:
        return np.empty((0, 2), dtype=float)
    vertices = np.asarray(candidates, dtype=float)
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    return vertices[np.argsort(angles)]


def _draw_series(
    image: np.ndarray,
    values: list[float],
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    ymin: float,
    ymax: float,
    color: tuple[int, int, int],
    zero: bool = True,
) -> None:
    import cv2

    if len(values) < 2:
        return
    finite = np.asarray(values, dtype=float)
    finite = np.nan_to_num(finite, nan=0.0, posinf=ymax, neginf=ymin)
    finite = np.clip(finite, ymin, ymax)
    points = []
    for index, value in enumerate(finite):
        px = x + int(round(index / max(len(finite) - 1, 1) * width))
        py = y + int(round((ymax - value) / max(ymax - ymin, 1e-9) * height))
        points.append((px, py))
    cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    if zero and ymin <= 0.0 <= ymax:
        py = y + int(round(ymax / max(ymax - ymin, 1e-9) * height))
        cv2.line(image, (x, py), (x + width, py), (190, 190, 190), 1)


def compose_live_hud(
    simulator_rgb: np.ndarray,
    *,
    scenario: OvertakeScenario,
    active_variant: str,
    seed: int,
    step: int,
    time_s: float,
    phase: str,
    target_dx_m: float,
    target_dy_m: float,
    ego_y_m: float,
    proposals: dict[str, dict[str, Any]],
    system: dict[str, Any],
    history: list[dict[str, float]],
) -> np.ndarray:
    """Compose one render-time policy dashboard frame."""

    import cv2

    canvas = np.full((960, 1680, 3), 250, dtype=np.uint8)
    _put_text(
        canvas,
        f"{scenario.title} | seed {seed} | active {active_variant} | step {step} | t={time_s:.1f}s",
        (20, 32),
        scale=0.72,
        color=(20, 20, 20),
        thickness=2,
    )
    _put_text(
        canvas,
        f"phase: {phase} | target dx={target_dx_m:+.1f} m | target dy={target_dy_m:+.1f} m | pass side={'upper' if scenario.pass_side > 0 else 'lower'}",
        (20, 57),
        scale=0.48,
        color=(70, 70, 70),
    )

    sim_x, sim_y, sim_w, sim_h = 20, 75, 1030, 300
    sim = cv2.resize(simulator_rgb, (sim_w, sim_h), interpolation=cv2.INTER_AREA)
    canvas[sim_y : sim_y + sim_h, sim_x : sim_x + sim_w] = sim
    cv2.rectangle(canvas, (sim_x, sim_y), (sim_x + sim_w, sim_y + sim_h), (35, 35, 35), 1)
    _panel(canvas, 1070, 75, 590, 300, "same-state action proposals (physical ax, ay)")

    plot_x, plot_y, plot_w, plot_h = 1125, 125, 440, 220
    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (180, 184, 190), 1)
    for value in (-3, 0, 3):
        px, _ = _map_action(np.asarray([value, 0]), plot_x, plot_y, plot_w, plot_h)
        cv2.line(canvas, (px, plot_y), (px, plot_y + plot_h), (228, 230, 234), 1)
        _put_text(canvas, f"{value:+d}", (px - 9, plot_y + plot_h + 18), scale=0.40, color=(100, 100, 100))
        _, py = _map_action(np.asarray([0, value]), plot_x, plot_y, plot_w, plot_h)
        cv2.line(canvas, (plot_x, py), (plot_x + plot_w, py), (228, 230, 234), 1)
        _put_text(canvas, f"{value:+d}", (plot_x - 29, py + 4), scale=0.40, color=(100, 100, 100))
    vertices = _polytope_vertices(system)
    if len(vertices) >= 3:
        polygon = np.asarray(
            [_map_action(point, plot_x, plot_y, plot_w, plot_h) for point in vertices],
            dtype=np.int32,
        )
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [polygon], (186, 226, 196))
        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0.0, canvas)
        cv2.polylines(canvas, [polygon], True, (55, 145, 75), 1, cv2.LINE_AA)
    for variant_id in VARIANT_ORDER:
        proposal = proposals[variant_id]
        color_hex = VARIANT_COLORS[variant_id].lstrip("#")
        rgb = tuple(int(color_hex[index : index + 2], 16) for index in (0, 2, 4))
        raw_point = _map_action(proposal["raw_box"], plot_x, plot_y, plot_w, plot_h)
        op_point = _map_action(proposal["operational_box"], plot_x, plot_y, plot_w, plot_h)
        shadow_point = _map_action(proposal["external_raw"], plot_x, plot_y, plot_w, plot_h)
        cv2.circle(canvas, raw_point, 5, rgb, -1, cv2.LINE_AA)
        cv2.drawMarker(canvas, op_point, rgb, cv2.MARKER_DIAMOND, 11, 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, shadow_point, rgb, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
    active = proposals[active_variant]
    executed_point = _map_action(active["operational_box"], plot_x, plot_y, plot_w, plot_h)
    if active_variant in proposals:
        executed_point = _map_action(
            proposals[active_variant].get("executed", proposals[active_variant]["operational_box"]),
            plot_x,
            plot_y,
            plot_w,
            plot_h,
        )
    cv2.drawMarker(canvas, executed_point, (15, 15, 15), cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)
    _put_text(canvas, "dot raw | diamond operational | x external shadow | star executed", (1090, 105), scale=0.40, color=(75, 75, 75))
    _put_text(canvas, "green polygon = current CBF-feasible action set", (1090, 352), scale=0.40, color=(50, 120, 65))

    # Live target-relative motion panel.
    _panel(canvas, 20, 405, 800, 240, "target-relative longitudinal motion")
    motion_x, motion_y, motion_w, motion_h = 70, 455, 700, 150
    cv2.rectangle(canvas, (motion_x, motion_y), (motion_x + motion_w, motion_y + motion_h), (180, 184, 190), 1)
    target_values = [item.get("target_dx_m", 0.0) for item in history]
    lateral_values = [scenario.pass_side * (item.get("ego_y_m", scenario.ego_y) - scenario.ego_y) for item in history]
    _draw_series(canvas, target_values, motion_x, motion_y, motion_w, motion_h, ymin=-20, ymax=45, color=(34, 93, 160))
    _draw_series(canvas, lateral_values, motion_x, motion_y, motion_w, motion_h, ymin=-20, ymax=45, color=(224, 112, 44))
    _put_text(canvas, "blue target dx (m)   orange pass-side lateral displacement (m)", (30, 632), scale=0.43, color=(65, 65, 65))
    _put_text(canvas, "45", (38, 464), scale=0.38, color=(100, 100, 100))
    _put_text(canvas, "-20", (31, 605), scale=0.38, color=(100, 100, 100))

    # Live active action and HOCBF panel.
    _panel(canvas, 850, 405, 810, 240, "active policy action and safety trajectory")
    action_x, action_y, action_w, action_h = 900, 455, 710, 150
    cv2.rectangle(canvas, (action_x, action_y), (action_x + action_w, action_y + action_h), (180, 184, 190), 1)
    raw_ay = [item.get("raw_ay", 0.0) for item in history]
    op_ay = [item.get("operational_ay", 0.0) for item in history]
    margins = [item.get("raw_hocbf_margin", 0.0) for item in history]
    _draw_series(canvas, raw_ay, action_x, action_y, action_w, action_h, ymin=-3, ymax=3, color=(128, 77, 153))
    _draw_series(canvas, op_ay, action_x, action_y, action_w, action_h, ymin=-3, ymax=3, color=(21, 137, 91))
    scaled_margin = [float(np.clip(value, -3.0, 3.0)) for value in margins]
    _draw_series(canvas, scaled_margin, action_x, action_y, action_w, action_h, ymin=-3, ymax=3, color=(210, 70, 70))
    _put_text(canvas, "purple raw ay   green operational ay   red clipped HOCBF margin", (860, 632), scale=0.43, color=(65, 65, 65))

    # Text table keeps all seven policy proposals readable in the video.
    _panel(canvas, 20, 680, 1640, 255, "policy proposal table: all rows are evaluated on the active branch's exact current observation")
    headers = [(35, "policy"), (235, "raw ay"), (330, "oper ay"), (435, "raw ax"), (530, "op ax"), (635, "shadow Δ"), (760, "raw margin"), (905, "external margin"), (1100, "interpretation")]
    for x, label in headers:
        _put_text(canvas, label, (x, 720), scale=0.43, color=(80, 80, 80), thickness=1)
    for row_index, variant_id in enumerate(VARIANT_ORDER):
        proposal = proposals[variant_id]
        y = 747 + row_index * 24
        color_hex = VARIANT_COLORS[variant_id].lstrip("#")
        rgb = tuple(int(color_hex[index : index + 2], 16) for index in (0, 2, 4))
        _put_text(canvas, variant_id, (35, y), scale=0.50, color=rgb, thickness=2 if variant_id == active_variant else 1)
        _put_text(canvas, f"{proposal['raw_ay']:+.2f}", (235, y), scale=0.46)
        _put_text(canvas, f"{proposal['operational_ay']:+.2f}", (330, y), scale=0.46)
        _put_text(canvas, f"{proposal['raw_ax']:+.2f}", (435, y), scale=0.46)
        _put_text(canvas, f"{proposal['operational_ax']:+.2f}", (530, y), scale=0.46)
        _put_text(canvas, f"{proposal['external_raw_correction_norm']:.2f}", (635, y), scale=0.46)
        _put_text(canvas, f"{proposal['raw_hocbf_margin']:+.2f}", (760, y), scale=0.46)
        _put_text(canvas, f"{proposal['external_hocbf_margin']:+.2f}", (905, y), scale=0.46)
        direction = scenario.pass_side * proposal["raw_ay"]
        text = "toward pass" if direction >= 0.08 else "away / neutral"
        if proposal["external_intervention"]:
            text += "; shadow would alter"
        _put_text(canvas, text, (1100, y), scale=0.44, color=(60, 60, 60))
    return canvas


def collect_targeted_rollouts(
    namespace: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    models: dict[str, Any],
    scenarios: list[OvertakeScenario],
    seeds: Iterable[int],
    *,
    max_steps: int,
    output_dir: Path,
    video_variants: set[str],
    video_fps: int,
    skip_videos: bool,
    apply_external_cbf: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trace_rows: list[dict[str, Any]] = []
    vehicle_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    video_paths: list[Path] = []
    for scenario in scenarios:
        config = _scenario_config(specs, scenario, max_steps)
        reward_config = copy.deepcopy(specs["B3_2"]["config"]["reward_config"])
        for variant_id in VARIANT_ORDER:
            print(f"[overtake] {scenario.scenario_id} | {variant_id}", flush=True)
            for seed in seeds:
                env = policy_diagnostics._make_raw_env(namespace, config, reward_config)
                writer = None
                try:
                    observation, reset_info = env.reset(seed=int(seed))
                    observation = _apply_scenario(env, scenario)
                    env.unwrapped.render_mode = "rgb_array"
                    context = _find_wrapper(env, lambda item: hasattr(item, "current_constraint_system"))
                    if context is None:
                        raise RuntimeError("Could not locate the CBF context wrapper")
                    history: list[dict[str, float]] = []
                    last_transition: dict[str, Any] = {}
                    outcome = "horizon"
                    path = output_dir / "videos" / (
                        f"live_hud_{scenario.scenario_id}_{variant_id}_seed_{int(seed)}.mp4"
                    )
                    for step in range(int(max_steps)):
                        base = env.unwrapped
                        ego_object = base.vehicle
                        ego = dict(namespace["get_ego_state"](env))
                        neighbors = [
                            dict(item)
                            for item in namespace["get_neighbor_states"](
                                env, neighbor_range=float(config["sensing_range"])
                            )
                        ]
                        system = context.current_constraint_system()
                        geometry = policy_diagnostics._neighbor_metrics(
                            namespace,
                            ego,
                            neighbors,
                            eps_side=float(context.eps_side),
                            k0=float(context.k0),
                            k1=float(context.k1),
                            road_width=float(config["road_width"]),
                        )
                        proposals = _query_proposals(models, observation, system)
                        active = proposals[variant_id]
                        target = base.road.vehicles[scenario.target_index]
                        try:
                            target_dx = float(base._signed_distance(ego_object.position[0], target.position[0]))
                        except (AttributeError, TypeError, ValueError):
                            target_dx = float(target.position[0] - ego_object.position[0])
                        target_dy = float(target.position[1] - ego_object.position[1])
                        time_s = float(step) / max(float(config["policy_frequency"]), 1e-9)
                        phase = _online_phase(
                            scenario, target_dx, target_dy, active["raw_ay"], float(ego.get("y", scenario.ego_y))
                        )
                        for proposal in proposals.values():
                            proposal_rows.append(
                                _proposal_row(
                                    proposal,
                                    scenario=scenario,
                                    active_variant=variant_id,
                                    seed=int(seed),
                                    step=step,
                                    time_s=time_s,
                                    target_dx_m=target_dx,
                                    target_dy_m=target_dy,
                                    ego_y_m=float(ego.get("y", np.nan)),
                                    ego_vx_mps=float(ego.get("vx", np.nan)),
                                    phase=phase,
                                )
                            )
                        executed = (
                            active["external_operational"]
                            if apply_external_cbf
                            else active["operational_box"]
                        )
                        executed = np.clip(_as_action(executed), ACTION_LOW, ACTION_HIGH)
                        active["executed"] = executed.copy()
                        history.append(
                            {
                                "target_dx_m": float(target_dx),
                                "ego_y_m": float(ego.get("y", np.nan)),
                                "raw_ay": float(active["raw_ay"]),
                                "operational_ay": float(active["operational_ay"]),
                                "raw_hocbf_margin": float(active["raw_hocbf_margin"]),
                            }
                        )
                        if (
                            not skip_videos
                            and variant_id in video_variants
                        ):
                            hud = compose_live_hud(
                                _render_frame(env),
                                scenario=scenario,
                                active_variant=variant_id,
                                seed=int(seed),
                                step=step,
                                time_s=time_s,
                                phase=phase,
                                target_dx_m=target_dx,
                                target_dy_m=target_dy,
                                ego_y_m=float(ego.get("y", np.nan)),
                                proposals=proposals,
                                system=system,
                                history=history,
                            )
                            if writer is None:
                                writer = _video_writer(path, video_fps, hud.shape[1], hud.shape[0])
                                video_paths.append(path)
                            _write_video_frame(writer, hud)
                        vehicle_rows.extend(
                            _vehicle_snapshot(
                                env,
                                scenario,
                                variant_id,
                                int(seed),
                                step,
                                time_s,
                            )
                        )
                        next_observation, reward, terminated, truncated, info = env.step(executed)
                        last_transition = _transition_metrics(info)
                        if last_transition["collision_events_step"] > 0 or last_transition["ego_collision_active"]:
                            outcome = "collision"
                        elif bool(terminated):
                            outcome = "terminated"
                        elif bool(truncated):
                            outcome = "truncated"
                        row = _trace_row(
                            scenario=scenario,
                            variant_id=variant_id,
                            seed=int(seed),
                            step=step,
                            time_s=time_s,
                            ego=ego,
                            target=target,
                            geometry=geometry,
                            proposal=active,
                            transition=last_transition,
                            outcome=outcome,
                            executed=executed,
                            external_applied=apply_external_cbf,
                        )
                        trace_rows.append(row)
                        observation = np.asarray(next_observation, dtype=np.float32).reshape(-1)
                        if bool(terminated) or bool(truncated):
                            break
                finally:
                    if writer is not None:
                        writer.release()
                    env.close()
    trace = pd.DataFrame(trace_rows)
    vehicles = pd.DataFrame(vehicle_rows)
    proposals = pd.DataFrame(proposal_rows)
    if trace.empty or vehicles.empty or proposals.empty:
        raise RuntimeError("Targeted rollout collection produced no data")
    print(f"[overtake] collected {len(trace)} policy rows, {len(vehicles)} vehicle rows, {len(proposals)} proposal rows", flush=True)
    return trace, vehicles, proposals


def add_unwrapped_geometry(trace: pd.DataFrame, vehicles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace = trace.copy()
    vehicles = vehicles.copy()
    trace["episode_key"] = _episode_key(trace)
    vehicles["episode_key"] = _episode_key(vehicles)
    vehicles["absolute_x_unwrapped_m"] = np.nan
    for (_episode, _token), group in vehicles.groupby(["episode_key", "vehicle_token"], sort=False):
        vehicles.loc[group.index, "absolute_x_unwrapped_m"] = _unwrap_periodic(
            group.sort_values("policy_step")["absolute_x_m"]
        )
    ego = vehicles.loc[vehicles["is_ego"].astype(bool), ["episode_key", "policy_step", "absolute_x_unwrapped_m", "absolute_y_m", "vx_mps", "vy_mps"]].rename(
        columns={
            "absolute_x_unwrapped_m": "ego_x_unwrapped_m",
            "absolute_y_m": "ego_snapshot_y_m",
            "vx_mps": "ego_snapshot_vx_mps",
            "vy_mps": "ego_snapshot_vy_mps",
        }
    )
    target = vehicles.loc[vehicles["vehicle_token"].astype(str).eq("v1"), ["episode_key", "policy_step", "absolute_x_unwrapped_m", "absolute_y_m", "vx_mps", "vy_mps"]].rename(
        columns={
            "absolute_x_unwrapped_m": "target_x_unwrapped_m",
            "absolute_y_m": "target_y_snapshot_m",
            "vx_mps": "target_snapshot_vx_mps",
            "vy_mps": "target_snapshot_vy_mps",
        }
    )
    trace = trace.drop(columns=["ego_x_unwrapped_m", "target_x_unwrapped_m"], errors="ignore")
    trace = trace.merge(ego, on=["episode_key", "policy_step"], how="left")
    trace = trace.merge(target, on=["episode_key", "policy_step"], how="left")
    trace["target_dx_m"] = trace["target_x_unwrapped_m"] - trace["ego_x_unwrapped_m"]
    trace["target_dy_m"] = trace["target_y_snapshot_m"] - trace["ego_snapshot_y_m"]
    trace["target_lateral_gap_m"] = trace["target_dy_m"].abs() - 1.8
    trace["target_longitudinal_gap_m"] = trace["target_dx_m"].abs() - 3.5
    trace["closing_speed_mps"] = trace["ego_snapshot_vx_mps"] - trace["target_snapshot_vx_mps"]
    trace = trace.sort_values(["scenario_id", "variant_id", "scenario_seed", "policy_step"]).reset_index(drop=True)
    vehicles = vehicles.sort_values(["scenario_id", "variant_id", "scenario_seed", "policy_step", "vehicle_index"]).reset_index(drop=True)
    return trace, vehicles


def _first_sustained(mask: np.ndarray, start: int = 0, length: int = 3) -> int | None:
    mask = np.asarray(mask, dtype=bool)
    for index in range(max(int(start), 0), max(len(mask) - int(length) + 1, 0)):
        if bool(np.all(mask[index : index + int(length)])):
            return int(index)
    return None


def _event_time(group: pd.DataFrame, step: int | None) -> float:
    if step is None or group.empty:
        return np.nan
    subset = group.loc[group["policy_step"].eq(int(step)), "time_s"]
    return float(subset.iloc[0]) if not subset.empty else np.nan


def detect_overtake_event(
    group: pd.DataFrame,
    scenario: OvertakeScenario,
    *,
    intent_threshold: float = 0.08,
) -> dict[str, Any]:
    """Detect an interpretable opportunity -> intent -> pass -> clear sequence."""

    group = group.sort_values("policy_step").reset_index(drop=True)
    if group.empty:
        return {"scenario_id": scenario.scenario_id, "pass_side": scenario.pass_side}
    steps = group["policy_step"].to_numpy(int)
    dx = group["target_dx_m"].to_numpy(float)
    dy = group["target_dy_m"].to_numpy(float)
    ego_y = group["ego_snapshot_y_m"].to_numpy(float)
    raw_ay = group["raw_ay"].to_numpy(float)
    operational_ay = group["operational_ay"].to_numpy(float)
    ego_vy = group["ego_snapshot_vy_mps"].to_numpy(float)
    signed_raw = float(scenario.pass_side) * raw_ay
    signed_operational = float(scenario.pass_side) * operational_ay
    signed_lateral = float(scenario.pass_side) * (ego_y - float(scenario.ego_y))
    opportunity_candidates = np.flatnonzero((dx <= 35.0) & (dx >= -2.0))
    opportunity_index = int(opportunity_candidates[0]) if len(opportunity_candidates) else 0
    raw_index = _first_sustained(signed_raw >= float(intent_threshold), opportunity_index, 3)
    operational_index = _first_sustained(
        signed_operational >= float(intent_threshold), opportunity_index, 3
    )
    motion_index = _first_sustained(
        (signed_lateral >= 0.25) | ((float(scenario.pass_side) * ego_vy) >= 0.08),
        opportunity_index,
        2,
    )
    after_intent = min(
        [index for index in (raw_index, operational_index, motion_index) if index is not None],
        default=opportunity_index,
    )
    abeam_candidates = np.flatnonzero((np.arange(len(group)) >= after_intent) & (dx <= 0.0))
    abeam_index = int(abeam_candidates[0]) if len(abeam_candidates) else None
    clear_start = after_intent if abeam_index is None else abeam_index
    clear_candidates = np.flatnonzero(
        (np.arange(len(group)) >= clear_start)
        & (dx <= -6.0)
        & (np.abs(dy) - 1.8 >= 0.25)
        & (signed_lateral >= 1.5)
    )
    clear_index = int(clear_candidates[0]) if len(clear_candidates) else None
    settle_index = _first_sustained(
        (np.abs(operational_ay) <= 0.15) & (np.abs(ego_vy) <= 0.20),
        (clear_index if clear_index is not None else len(group)),
        3,
    )
    attempted = bool(raw_index is not None or motion_index is not None)
    completed = bool(clear_index is not None)
    event = {
        "scenario_id": scenario.scenario_id,
        "scenario_title": scenario.title,
        "variant_id": str(group["variant_id"].iloc[0]),
        "variant_label": VARIANT_LABELS.get(str(group["variant_id"].iloc[0]), str(group["variant_id"].iloc[0])),
        "scenario_seed": int(group["scenario_seed"].iloc[0]),
        "pass_side": int(scenario.pass_side),
        "pass_side_label": "upper" if scenario.pass_side > 0 else "lower",
        "target_token": "v1",
        "opportunity_step": int(steps[opportunity_index]),
        "raw_intent_step": np.nan if raw_index is None else int(steps[raw_index]),
        "operational_intent_step": np.nan if operational_index is None else int(steps[operational_index]),
        "motion_onset_step": np.nan if motion_index is None else int(steps[motion_index]),
        "abeam_step": np.nan if abeam_index is None else int(steps[abeam_index]),
        "clear_step": np.nan if clear_index is None else int(steps[clear_index]),
        "settle_step": np.nan if settle_index is None else int(steps[settle_index]),
        "opportunity_time_s": _event_time(group, int(steps[opportunity_index])),
        "raw_intent_time_s": _event_time(group, None if raw_index is None else int(steps[raw_index])),
        "operational_intent_time_s": _event_time(group, None if operational_index is None else int(steps[operational_index])),
        "motion_onset_time_s": _event_time(group, None if motion_index is None else int(steps[motion_index])),
        "abeam_time_s": _event_time(group, None if abeam_index is None else int(steps[abeam_index])),
        "clear_time_s": _event_time(group, None if clear_index is None else int(steps[clear_index])),
        "settle_time_s": _event_time(group, None if settle_index is None else int(steps[settle_index])),
        "attempted_overtake": attempted,
        "completed_overtake": completed,
        "aborted_overtake": bool(attempted and not completed),
        "last_target_dx_m": float(dx[-1]),
        "max_pass_side_lateral_displacement_m": float(np.nanmax(signed_lateral)),
        "minimum_target_lateral_gap_m": float(np.nanmin(np.abs(dy) - 1.8)),
        "minimum_raw_hocbf_margin": float(np.nanmin(group["raw_hocbf_margin"].to_numpy(float))),
        "minimum_operational_hocbf_margin": float(np.nanmin(group["operational_hocbf_margin"].to_numpy(float))),
        "collision_events": int(pd.to_numeric(group["collision_events_step"], errors="coerce").fillna(0).sum()),
    }
    opportunity_time = float(event["opportunity_time_s"])
    for stage in ("raw_intent", "operational_intent", "motion_onset", "abeam", "clear", "settle"):
        value = event[f"{stage}_time_s"]
        event[f"{stage}_delay_s"] = float(value - opportunity_time) if np.isfinite(value) else np.nan
    if np.isfinite(event["raw_intent_time_s"]) and np.isfinite(event["operational_intent_time_s"]):
        event["operational_minus_raw_intent_delay_s"] = float(
            event["operational_intent_time_s"] - event["raw_intent_time_s"]
        )
    else:
        event["operational_minus_raw_intent_delay_s"] = np.nan
    return event


def detect_events(
    trace: pd.DataFrame, scenarios: dict[str, OvertakeScenario], intent_threshold: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario_id, variant_id, seed), group in trace.groupby(
        ["scenario_id", "variant_id", "scenario_seed"], sort=False
    ):
        scenario = scenarios[str(scenario_id)]
        rows.append(detect_overtake_event(group, scenario, intent_threshold=intent_threshold))
    return pd.DataFrame(rows).sort_values(["scenario_id", "scenario_seed", "variant_id"]).reset_index(drop=True)


def attach_phases(trace: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    trace = trace.copy()
    trace["phase"] = "approach"
    event_index = events.set_index(["scenario_id", "variant_id", "scenario_seed"])
    for key, group in trace.groupby(["scenario_id", "variant_id", "scenario_seed"], sort=False):
        if key not in event_index.index:
            continue
        event = event_index.loc[key]
        labels = np.full(len(group), "approach", dtype=object)
        steps = group["policy_step"].to_numpy(int)
        stages = [
            ("raw_intent_step", "raw intent"),
            ("operational_intent_step", "operational intent"),
            ("motion_onset_step", "lateral commit"),
            ("abeam_step", "abeam"),
            ("clear_step", "clear"),
            ("settle_step", "settle"),
        ]
        for stage, label in stages:
            value = _finite(event.get(stage))
            if np.isfinite(value):
                labels[steps >= int(value)] = label
        trace.loc[group.index, "phase"] = labels
    return trace


def _row_at_step(group: pd.DataFrame, step: int | None) -> pd.Series:
    if group.empty:
        return pd.Series(dtype=object)
    if step is None or not np.isfinite(float(step)):
        return group.iloc[-1]
    exact = group.loc[group["policy_step"].eq(int(step))]
    if not exact.empty:
        return exact.iloc[0]
    distances = (group["policy_step"].to_numpy(float) - float(step)) ** 2
    return group.iloc[int(np.argmin(distances))]


FREEZE_STAGES = (
    ("opportunity", "opportunity_step"),
    ("raw intent", "raw_intent_step"),
    ("operational intent", "operational_intent_step"),
    ("lateral commit", "motion_onset_step"),
    ("abeam", "abeam_step"),
    ("clear", "clear_step"),
)


def render_freeze_storyboards(
    trace: pd.DataFrame,
    vehicles: pd.DataFrame,
    events: pd.DataFrame,
    scenarios: dict[str, OvertakeScenario],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for (scenario_id, seed), event_group in events.groupby(["scenario_id", "scenario_seed"], sort=True):
        scenario = scenarios[str(scenario_id)]
        figure, axes = plt.subplots(
            len(VARIANT_ORDER),
            len(FREEZE_STAGES),
            figsize=(20, 16),
            squeeze=False,
            constrained_layout=True,
        )
        for row_index, variant_id in enumerate(VARIANT_ORDER):
            event_rows = event_group.loc[event_group["variant_id"].eq(variant_id)]
            event = event_rows.iloc[0] if not event_rows.empty else pd.Series(dtype=object)
            episode_key = f"{scenario_id}|{variant_id}|{int(seed)}"
            trace_group = trace.loc[trace["episode_key"].eq(episode_key)].sort_values("policy_step")
            vehicle_group = vehicles.loc[vehicles["episode_key"].eq(episode_key)].sort_values("policy_step")
            for column_index, (stage_label, event_column) in enumerate(FREEZE_STAGES):
                axis = axes[row_index, column_index]
                step = _finite(event.get(event_column)) if not event.empty else np.nan
                step_int = int(step) if np.isfinite(step) else None
                current = _row_at_step(trace_group, step_int)
                current_step = int(current.get("policy_step", 0)) if not current.empty else 0
                snapshot = vehicle_group.loc[vehicle_group["policy_step"].eq(current_step)].copy()
                if snapshot.empty:
                    snapshot = vehicle_group.loc[vehicle_group["policy_step"].le(current_step)].tail(len(scenario.vehicles))
                target_rows = snapshot.loc[snapshot["vehicle_token"].astype(str).eq("v1")]
                target_x = float(target_rows["absolute_x_unwrapped_m"].iloc[0]) if not target_rows.empty else 0.0
                target_y = float(target_rows["absolute_y_m"].iloc[0]) if not target_rows.empty else scenario.ego_y
                if not trace_group.empty:
                    history = vehicle_group.loc[
                        vehicle_group["is_ego"].astype(bool) & vehicle_group["policy_step"].le(current_step)
                    ].sort_values("policy_step")
                    if not history.empty:
                        target_history = vehicle_group.loc[
                            vehicle_group["vehicle_token"].astype(str).eq("v1")
                            & vehicle_group["policy_step"].le(current_step)
                        ].sort_values("policy_step")
                        if not target_history.empty:
                            shared = history.merge(
                                target_history[["policy_step", "absolute_x_unwrapped_m"]],
                                on="policy_step",
                                suffixes=("_ego", "_target"),
                            )
                            axis.plot(
                                shared["absolute_x_unwrapped_m_ego"] - shared["absolute_x_unwrapped_m_target"],
                                shared["absolute_y_m"].to_numpy(float) - target_y,
                                color=VARIANT_COLORS[variant_id],
                                linewidth=1.0,
                                alpha=0.35,
                            )
                others = snapshot.loc[~snapshot["is_ego"].astype(bool)]
                for _, item in others.iterrows():
                    x = float(item["absolute_x_unwrapped_m"] - target_x)
                    y = float(item["absolute_y_m"] - target_y)
                    color = "#d33f49" if str(item["vehicle_token"]) == "v1" else "#a6abb2"
                    marker = "*" if str(item["vehicle_token"]) == "v1" else "o"
                    size = 105 if marker == "*" else 24
                    axis.scatter([x], [y], color=color, marker=marker, s=size, alpha=0.88, zorder=5)
                    if marker == "*":
                        axis.text(x + 1.1, y + 0.22, "target", fontsize=5.5, color=color)
                ego_rows = snapshot.loc[snapshot["is_ego"].astype(bool)]
                if not ego_rows.empty:
                    ego_item = ego_rows.iloc[0]
                    axis.scatter(
                        [float(ego_item["absolute_x_unwrapped_m"] - target_x)],
                        [float(ego_item["absolute_y_m"] - target_y)],
                        color=VARIANT_COLORS[variant_id],
                        marker="D",
                        s=42,
                        edgecolor="black",
                        linewidth=0.4,
                        zorder=6,
                    )
                axis.axhline(-target_y, color="#c2c7cc", linewidth=0.6)
                axis.axhline(ROAD_WIDTH_M - target_y, color="#c2c7cc", linewidth=0.6)
                axis.axhline(0.0, color="#e1e3e6", linewidth=0.5, linestyle="--")
                axis.set_xlim(-45.0, 45.0)
                axis.set_ylim(-5.6, 5.6)
                axis.grid(alpha=0.14)
                if row_index == 0:
                    reached = "not reached" if not np.isfinite(step) else f"t={_finite(current.get('time_s'), 0.0):.1f}s"
                    axis.set_title(f"{stage_label}\n{reached}", fontsize=8)
                if column_index == 0:
                    axis.set_ylabel(f"{variant_id}\nrelative y (m)", fontsize=7)
                if row_index == len(VARIANT_ORDER) - 1:
                    axis.set_xlabel("ego x − target x (m)", fontsize=7)
        figure.suptitle(
            f"Overtake freeze-frame storyboard | {scenario.title} | seed {int(seed)}\n"
            "red star = designated blocker; colored diamond = ego; x-axis crosses zero when ego draws level",
            fontsize=14,
        )
        path = output_dir / f"freeze_frames_{scenario_id}_seed_{int(seed)}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def render_event_timing(events: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    frame = events.copy()
    frame["row_label"] = (
        frame["scenario_id"].astype(str)
        + " / "
        + frame["scenario_seed"].astype(str)
        + " / "
        + frame["variant_id"].astype(str)
    )
    frame = frame.sort_values(["scenario_id", "scenario_seed", "variant_id"]).reset_index(drop=True)
    figure, (timeline, bars) = plt.subplots(
        1, 2, figsize=(18, max(8, 0.28 * len(frame) + 3)), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )
    y_positions = np.arange(len(frame))
    for index, row in frame.iterrows():
        color = VARIANT_COLORS.get(str(row["variant_id"]), "#555555")
        opportunity = _finite(row.get("opportunity_time_s"), 0.0)
        timeline.axhline(index, color="#e4e6e9", linewidth=0.5, zorder=0)
        for column, marker, label, marker_color in (
            ("raw_intent_time_s", "o", "raw intent", "#7b3fb2"),
            ("operational_intent_time_s", "D", "operational intent", "#e07b25"),
            ("abeam_time_s", "|", "abeam", "#2e6da4"),
            ("clear_time_s", "*", "clear", "#25864b"),
        ):
            value = _finite(row.get(column))
            if np.isfinite(value):
                timeline.scatter(value - opportunity, index, color=marker_color, marker=marker, s=45, zorder=3)
        timeline.scatter(0.0, index, color="black", marker="|", s=45, zorder=3)
        timeline.text(
            -0.02,
            index,
            str(row["variant_id"]),
            transform=timeline.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.5,
            color=color,
        )
    timeline.axvline(0.0, color="black", linewidth=0.8)
    timeline.set_yticks(y_positions, frame["row_label"].tolist(), fontsize=6.5)
    timeline.set_xlabel("seconds after opportunity (policy clock)")
    timeline.set_title("When each policy commits to the overtake")
    timeline.grid(axis="x", alpha=0.2)
    timeline.legend(
        handles=[
            plt.Line2D([0], [0], marker="|", color="black", linestyle="none", label="opportunity"),
            plt.Line2D([0], [0], marker="o", color="#7b3fb2", linestyle="none", label="raw intent"),
            plt.Line2D([0], [0], marker="D", color="#e07b25", linestyle="none", label="operational intent"),
            plt.Line2D([0], [0], marker="|", color="#2e6da4", linestyle="none", label="abeam"),
            plt.Line2D([0], [0], marker="*", color="#25864b", linestyle="none", label="clear"),
        ],
        loc="lower right",
        fontsize=7,
    )
    summary = frame.groupby("variant_id", sort=False).agg(
        intent_delay_s=("raw_intent_delay_s", "median"),
        clear_delay_s=("clear_delay_s", "median"),
        completed=("completed_overtake", "mean"),
    ).reindex(VARIANT_ORDER)
    x = np.arange(len(summary))
    bars.bar(x - 0.18, summary["intent_delay_s"].fillna(0.0), width=0.36, color="#7b3fb2", label="raw intent delay")
    bars.bar(x + 0.18, summary["clear_delay_s"].fillna(0.0), width=0.36, color="#25864b", label="clear delay")
    bars.set_xticks(x, VARIANT_ORDER, rotation=35, ha="right")
    bars.set_ylabel("median seconds after opportunity")
    bars.set_title("Timing summary; labels show completion rate")
    bars.grid(axis="y", alpha=0.2)
    for index, (variant_id, row) in enumerate(summary.iterrows()):
        bars.text(index, max(float(row[["intent_delay_s", "clear_delay_s"]].fillna(0.0).max()), 0.0) + 0.1, f"{100.0 * float(row['completed']):.0f}%", ha="center", fontsize=7)
    bars.legend(fontsize=7)
    figure.suptitle("Overtake event clock", fontsize=15)
    path = output_dir / "overtake_event_timing.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    summary_path = output_dir / "overtake_timing_summary.csv"
    summary.reset_index().to_csv(summary_path, index=False)
    paths.append(summary_path)
    return paths


def render_intent_raster(
    trace: pd.DataFrame, events: pd.DataFrame, output_dir: Path
) -> list[Path]:
    frame = events.sort_values(["scenario_id", "scenario_seed", "variant_id"]).reset_index(drop=True)
    time_grid = np.linspace(-1.0, 12.0, 131)
    matrix = np.full((len(frame), len(time_grid)), np.nan, dtype=float)
    labels: list[str] = []
    for row_index, event in frame.iterrows():
        key = (
            (trace["scenario_id"].astype(str) == str(event["scenario_id"]))
            & (trace["variant_id"].astype(str) == str(event["variant_id"]))
            & (trace["scenario_seed"].astype(int) == int(event["scenario_seed"]))
        )
        group = trace.loc[key].sort_values("time_s")
        opportunity = _finite(event.get("opportunity_time_s"), 0.0)
        if not group.empty:
            relative_time = group["time_s"].to_numpy(float) - opportunity
            signal = float(event["pass_side"]) * group["raw_ay"].to_numpy(float)
            valid = np.isfinite(relative_time) & np.isfinite(signal)
            if np.sum(valid) >= 2:
                matrix[row_index] = np.interp(time_grid, relative_time[valid], signal[valid], left=np.nan, right=np.nan)
        labels.append(f"{event['scenario_id']} / {event['scenario_seed']} / {event['variant_id']}")
    figure, axis = plt.subplots(figsize=(14, max(7, 0.28 * len(frame) + 2.5)), constrained_layout=True)
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        extent=[time_grid[0], time_grid[-1], len(frame) - 0.5, -0.5],
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    axis.axvline(0.0, color="black", linewidth=0.9, label="opportunity")
    for row_index, event in frame.iterrows():
        opportunity = _finite(event.get("opportunity_time_s"), 0.0)
        for column, color, marker in (
            ("raw_intent_time_s", "#7b3fb2", "o"),
            ("operational_intent_time_s", "#e07b25", "D"),
            ("clear_time_s", "#25864b", "*"),
        ):
            value = _finite(event.get(column))
            if np.isfinite(value):
                axis.scatter(value - opportunity, row_index, color=color, marker=marker, s=28, edgecolor="white", linewidth=0.3)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=6.5)
    axis.set_xlabel("seconds after opportunity")
    axis.set_title("Intent raster: signed lateral action (positive = toward designated passing side)\nmarkers: raw intent, operational intent, clear")
    axis.grid(axis="y", alpha=0.08)
    figure.colorbar(image, ax=axis, label="pass-side signed raw ay")
    path = output_dir / "overtake_intent_raster.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [path]


def render_relative_paths(
    trace: pd.DataFrame, events: pd.DataFrame, scenarios: dict[str, OvertakeScenario], output_dir: Path
) -> list[Path]:
    paths: list[Path] = []
    figure, axes = plt.subplots(1, len(scenarios), figsize=(8 * len(scenarios), 6), squeeze=False, constrained_layout=True)
    for axis, (scenario_id, scenario) in zip(axes.flat, scenarios.items()):
        subset_events = events.loc[events["scenario_id"].eq(scenario_id)]
        for variant_id in VARIANT_ORDER:
            for seed in sorted(subset_events["scenario_seed"].unique()):
                key = (
                    trace["scenario_id"].eq(scenario_id)
                    & trace["variant_id"].eq(variant_id)
                    & trace["scenario_seed"].eq(int(seed))
                )
                group = trace.loc[key].sort_values("policy_step")
                if group.empty:
                    continue
                lateral = scenario.pass_side * (group["ego_snapshot_y_m"] - scenario.ego_y)
                relative_x = -group["target_dx_m"]
                axis.plot(relative_x, lateral, color=VARIANT_COLORS[variant_id], alpha=0.30, linewidth=1.0)
                axis.scatter(relative_x.iloc[0], lateral.iloc[0], color=VARIANT_COLORS[variant_id], s=10, alpha=0.6)
                axis.scatter(relative_x.iloc[-1], lateral.iloc[-1], color=VARIANT_COLORS[variant_id], s=18, marker="D", alpha=0.8)
        axis.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
        axis.axhline(0.0, color="#b8bdc3", linewidth=0.7)
        axis.axhline(1.5, color="#25864b", linewidth=0.7, linestyle=":")
        axis.set_xlim(-45, 45)
        axis.set_ylim(-2.5, 4.8)
        axis.set_xlabel("ego x − target x (m); positive = ego ahead")
        axis.set_ylabel("pass-side lateral displacement (m)")
        axis.set_title(f"{scenario.title}\nline color = policy; translucent lines = matched seeds")
        axis.grid(alpha=0.2)
        handles = [plt.Line2D([0], [0], color=VARIANT_COLORS[v], label=v) for v in VARIANT_ORDER]
        axis.legend(handles=handles, fontsize=7, ncol=2, loc="upper left")
    path = output_dir / "overtake_relative_paths.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)
    return paths


def build_summary(trace: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in trace.groupby(["scenario_id", "variant_id", "scenario_seed"], sort=False):
        scenario_id, variant_id, seed = key
        event = events.loc[
            events["scenario_id"].eq(scenario_id)
            & events["variant_id"].eq(variant_id)
            & events["scenario_seed"].eq(int(seed))
        ]
        event_row = event.iloc[0].to_dict() if not event.empty else {}
        rows.append(
            {
                "scenario_id": scenario_id,
                "variant_id": variant_id,
                "variant_label": VARIANT_LABELS.get(variant_id, variant_id),
                "scenario_seed": int(seed),
                "steps": int(len(group)),
                "duration_s": float(group["time_s"].max()),
                "distance_m": float(group["ego_x_unwrapped_m"].iloc[-1] - group["ego_x_unwrapped_m"].iloc[0]),
                "min_target_dx_m": float(group["target_dx_m"].min()),
                "max_pass_side_lateral_displacement_m": event_row.get("max_pass_side_lateral_displacement_m", np.nan),
                "attempted_overtake": event_row.get("attempted_overtake", False),
                "completed_overtake": event_row.get("completed_overtake", False),
                "aborted_overtake": event_row.get("aborted_overtake", False),
                "raw_intent_delay_s": event_row.get("raw_intent_delay_s", np.nan),
                "operational_intent_delay_s": event_row.get("operational_intent_delay_s", np.nan),
                "clear_delay_s": event_row.get("clear_delay_s", np.nan),
                "collision_events": int(group["collision_events_step"].sum()),
                "min_raw_hocbf_margin": float(group["raw_hocbf_margin"].min()),
                "min_operational_hocbf_margin": float(group["operational_hocbf_margin"].min()),
                "max_shadow_external_correction_norm": float(group["shadow_external_correction_norm"].max()),
                "traffic_guard_brakes": float(group.get("traffic_guard_brakes", pd.Series(dtype=float)).max()) if "traffic_guard_brakes" in group else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario_id", "scenario_seed", "variant_id"]).reset_index(drop=True)


def write_readme(
    output_dir: Path,
    scenarios: list[OvertakeScenario],
    seeds: list[int],
    *,
    max_steps: int,
    video_variants: Iterable[str],
    apply_external_cbf: bool,
    files: list[Path],
) -> Path:
    config_line = "external CBF projection was applied to the active branch" if apply_external_cbf else "external CBF was disabled on the active branch; external projection is shadow-only"
    text = f"""# Overtake policy diagnostics

This directory was produced by `scripts/render_overtake_policy_diagnostics.py`.

The protocol uses {len(scenarios)} deterministic targeted scenes, seeds {seeds}, and up to {max_steps} policy steps per branch.  Live HUD videos were rendered for: {list(video_variants)}.  The saved checkpoints use a 10 Hz policy clock and a 100 Hz physics clock.

{config_line}.  The ordinary simulator traffic dynamics guard remains enabled.

The event detector uses the following interpretable sequence:

`opportunity -> raw intent -> operational intent -> lateral commit -> abeam -> clear -> settle`

`raw intent` means three consecutive policy frames with lateral action toward the designated passing side.  `operational intent` applies the same test after the policy's internal action stage.  `clear` requires the ego to be at least 6 m longitudinally ahead, laterally separated from the blocker, and displaced at least 1.5 m toward the selected side.  A missing stage is retained as `not reached`; it is not silently imputed as a successful overtake.

The MP4 HUD compares all policy proposals on the exact current observation.  Dots are raw actions, diamonds are operational actions, crosses are shadow external-CBF actions, and the star is the action actually executed by the active branch.  The lower panels show target-relative motion, lateral action, and HOCBF margin over the live rollout.

Generated files:

"""
    for path in files:
        text += f"- `{path.name}`\n"
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenarios", nargs="*", default=list(DEFAULT_SCENARIOS), choices=sorted(scenario_catalog()))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--active-variant", choices=VARIANT_ORDER, default="B3_2")
    parser.add_argument(
        "--video-variants",
        nargs="*",
        choices=VARIANT_ORDER,
        default=None,
        help="Active branches for which to write live HUD videos; defaults to --active-variant.",
    )
    parser.add_argument("--intent-threshold", type=float, default=0.08)
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--apply-external-cbf", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_steps <= 0 or args.video_fps <= 0:
        raise ValueError("max-steps and video-fps must be positive")
    if not args.scenarios or not args.seeds:
        raise ValueError("at least one scenario and one seed are required")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = scenario_catalog()
    selected_scenarios = [catalog[name] for name in args.scenarios]
    selected_catalog = {scenario.scenario_id: scenario for scenario in selected_scenarios}
    selected_seeds = [int(seed) for seed in args.seeds]
    selected_video_variants = set(args.video_variants or [args.active_variant])

    print("[overtake] loading saved policy metadata", flush=True)
    specs = base_analysis.load_specs(PROJECT_ROOT)
    namespace = policy_diagnostics._bootstrap_namespace()
    models = base_analysis.load_models(specs)
    trace, vehicles, proposals = collect_targeted_rollouts(
        namespace,
        specs,
        models,
        selected_scenarios,
        selected_seeds,
        max_steps=int(args.max_steps),
        output_dir=output_dir,
        video_variants=selected_video_variants,
        video_fps=int(args.video_fps),
        skip_videos=bool(args.skip_videos),
        apply_external_cbf=bool(args.apply_external_cbf),
    )
    trace, vehicles = add_unwrapped_geometry(trace, vehicles)
    events = detect_events(trace, selected_catalog, float(args.intent_threshold))
    trace = attach_phases(trace, events)
    summary = build_summary(trace, events)

    trace_path = output_dir / "overtake_trace.csv"
    vehicle_path = output_dir / "overtake_vehicle_trace.csv"
    proposal_path = output_dir / "same_state_policy_proposals.csv"
    event_path = output_dir / "overtake_events.csv"
    summary_path = output_dir / "overtake_summary.csv"
    trace.to_csv(trace_path, index=False)
    vehicles.to_csv(vehicle_path, index=False)
    proposals.to_csv(proposal_path, index=False)
    events.to_csv(event_path, index=False)
    summary.to_csv(summary_path, index=False)
    files: list[Path] = [trace_path, vehicle_path, proposal_path, event_path, summary_path]

    if not args.skip_static:
        freeze_dir = output_dir / "freeze_frames"
        files.extend(render_freeze_storyboards(trace, vehicles, events, selected_catalog, freeze_dir))
        static_dir = output_dir / "plots"
        static_dir.mkdir(parents=True, exist_ok=True)
        files.extend(render_event_timing(events, static_dir))
        files.extend(render_intent_raster(trace, events, static_dir))
        files.extend(render_relative_paths(trace, events, selected_catalog, static_dir))

    video_dir = output_dir / "videos"
    if video_dir.exists():
        files.extend(sorted(video_dir.glob("*.mp4")))

    manifest = {
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "scenarios": [scenario.scenario_id for scenario in selected_scenarios],
        "seeds": selected_seeds,
        "max_steps": int(args.max_steps),
        "active_variant": str(args.active_variant),
        "video_variants": sorted(selected_video_variants),
        "intent_threshold": float(args.intent_threshold),
        "physics_frequency_hz": 100,
        "policy_frequency_hz": 10,
        "external_cbf_applied": bool(args.apply_external_cbf),
        "ordinary_traffic_dynamics_guard": True,
        "trace_rows": int(len(trace)),
        "vehicle_rows": int(len(vehicles)),
        "proposal_rows": int(len(proposals)),
        "event_rows": int(len(events)),
        "files": [str(path.relative_to(output_dir)) for path in files if path.exists()],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files.append(manifest_path)
    files.append(
        write_readme(
            output_dir,
            selected_scenarios,
            selected_seeds,
            max_steps=int(args.max_steps),
            video_variants=sorted(selected_video_variants),
            apply_external_cbf=bool(args.apply_external_cbf),
            files=files,
        )
    )
    print("[overtake] event summary", flush=True)
    print(
        summary.groupby("variant_id", sort=False)[
            ["attempted_overtake", "completed_overtake", "clear_delay_s", "collision_events"]
        ].agg({"attempted_overtake": "mean", "completed_overtake": "mean", "clear_delay_s": "median", "collision_events": "sum"}).to_string(),
        flush=True,
    )
    print(f"[overtake] outputs written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
