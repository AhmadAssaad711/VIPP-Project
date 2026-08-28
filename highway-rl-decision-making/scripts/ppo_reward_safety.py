"""Reward extensions used by the PPO safety-cost studies.

The active notebook contains the legacy reciprocal Karalakou reward.  The
PPO safety-potential studies additionally need the newer additive reward path
and the ``cbf_violation`` safety cost.  This module installs that wrapper in
the execution namespace used by both the learner and spawned evaluation
workers, without changing the notebook's interactive state.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np


def install_scalar_safe_cbf_geometry(namespace: dict[str, Any]) -> None:
    """Replace hot scalar CBF geometry ufuncs with equivalent ``math`` calls.

    The Windows worker failures captured by ``PYTHONFAULTHANDLER`` terminate
    inside the notebook's scalar ``np.sqrt`` call in
    ``ellipse_radius_along_line``.  These calculations do not require NumPy
    arrays: they operate on four Python floats.  Keeping them on Python's
    scalar math path removes that native ufunc from every finite-difference
    CBF constraint evaluation without changing the ellipse/clearance formula.

    Notebook functions resolve globals through their execution namespace, so
    replacing these four names also hardens downstream
    ``centerline_barrier_derivatives`` and ``pairwise_hocbf_constraint`` calls.
    """

    eps_side_default = float(namespace.get("CBF_EPS_SIDE", 0.10))
    sqrt_two = math.sqrt(2.0)

    def inflated_ellipse_axes(
        length: float,
        width: float,
        eps_side: float = eps_side_default,
    ) -> tuple[float, float]:
        side = eps_side_default if eps_side is None else float(eps_side)
        a = float(length) / sqrt_two + 2.0 * side
        b = float(width) / sqrt_two + 2.0 * side
        return max(float(a), 1e-6), max(float(b), 1e-6)

    def wrap_angle(angle: float) -> float:
        return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)

    def ellipse_radius_along_line(a: float, b: float, delta: float) -> float:
        cos_delta = math.cos(float(delta))
        sin_delta = math.sin(float(delta))
        denom = math.sqrt(
            (float(b) * cos_delta) ** 2 + (float(a) * sin_delta) ** 2
        )
        return float(float(a) * float(b) / max(float(denom), 1e-9))

    def pairwise_centerline_clearance(
        p: Any,
        ego: dict[str, float],
        other: dict[str, float],
        eps_side: float | None = None,
    ) -> tuple[float, float, float, float]:
        px = float(p[0])
        py = float(p[1])
        radius = math.hypot(px, py)
        phi = 0.0 if radius < 1e-9 else math.atan2(py, px)
        side = eps_side_default if eps_side is None else float(eps_side)
        a_ego, b_ego = inflated_ellipse_axes(
            ego["length"], ego["width"], side
        )
        a_other, b_other = inflated_ellipse_axes(
            other["length"], other["width"], side
        )
        l_ego = ellipse_radius_along_line(
            a_ego,
            b_ego,
            wrap_angle(phi - float(ego.get("heading", 0.0))),
        )
        l_other = ellipse_radius_along_line(
            a_other,
            b_other,
            wrap_angle(phi - float(other.get("heading", 0.0))),
        )
        required_distance = l_ego + l_other
        h = radius - required_distance
        return float(h), float(radius), float(l_ego), float(l_other)

    namespace["inflated_ellipse_axes"] = inflated_ellipse_axes
    namespace["_wrap_angle"] = wrap_angle
    namespace["ellipse_radius_along_line"] = ellipse_radius_along_line
    namespace["pairwise_centerline_clearance"] = pairwise_centerline_clearance
    namespace["CBF_SCALAR_GEOMETRY_BACKEND"] = "python_math_v1"


def install_cbf_violation_reward(namespace: dict[str, Any]) -> None:
    """Install the enhanced Karalakou reward wrapper in ``namespace``.

    The wrapper preserves the notebook's task, progress, jerk, collision, and
    overtake terms.  The CBF term is the repository's newest
    ``psi = h_dot + alpha*h`` violation formulation.  When that formulation
    is selected, it replaces the legacy reciprocal potential-field cost in
    the same reward slot; it is not an additional safety term.
    """

    install_scalar_safe_cbf_geometry(namespace)
    base_reward_wrapper = namespace["KaralakouRewardWrapper"]

    class PPOSafetyKaralakouRewardWrapper(base_reward_wrapper):  # type: ignore[misc, valid-type]
        """Karalakou reward with the optional direct CBF-violation cost."""

        def __init__(
            self,
            env: gym.Env,
            reward_config: dict[str, float] | None = None,
        ) -> None:
            super().__init__(env, reward_config=reward_config)
            defaults: dict[str, float | str | bool] = {
                "reward_mode": "reciprocal",
                # Match the notebook's default bounded comfort term.  A caller
                # can still disable it explicitly with a zero weight.
                "jerk_penalty_weight": 0.02,
                "jerk_scale": 10.0,
                "speed_reward_weight": 0.25,
                "lateral_reward_weight": 0.25,
                "risk_penalty_weight": 0.5,
                "risk_potential_shaping_weight": 0.0,
                "risk_potential_shaping_gamma": 0.99,
                "collision_reward_override": False,
                "safety_potential_formulation": "none",
                "safety_potential_weight": 0.0,
                "safety_ellipse_weight": 0.0,
                "safety_ttc_weight": 0.0,
                "safety_potential_warning_h": 4.0,
                "safety_potential_eps_side": 0.10,
                "safety_cbf_alpha": 1.0,
                "safety_cbf_psi_scale": 1.0,
                "speed_reward_sigma": 4.0,
                "lateral_reward_sigma": 1.0,
            }
            for key, value in defaults.items():
                self.reward_config.setdefault(key, value)  # type: ignore[arg-type]

        def _karalakou_reward(
            self,
            previous_dx: dict[int, float],
            previous_ego_x: float | None = None,
        ) -> tuple[float, dict[str, float]]:
            base = self.base_env
            ego = base.vehicle
            cfg = self.reward_config

            target_y, target_speed, zone_found = self._lateral_target_and_speed()
            ego_y = float(ego.position[1])
            ego_speed = float(ego.vx)
            desired_speed = float(ego.desired_speed)
            road_width = max(float(base.config["road_width"]), 1e-6)
            cx = abs(ego_speed - target_speed) / max(target_speed, 1e-6)
            cy = abs(ego_y - target_y) / road_width
            lat_y_error_m = abs(ego_y - target_y)
            lat_y_coherence = float(np.clip(1.0 - cy, 0.0, 1.0))

            safety_potential, safety_potential_min_h, safety_ellipse_cost, safety_ttc_cost = (
                self._safety_potential_cost()
            )
            safety_formulation = str(
                cfg.get("safety_potential_formulation", "none")
            ).strip().lower()
            if safety_formulation == "cbf_violation":
                # In the reciprocal reward, ``cf`` is the legacy safety
                # potential-field cost.  The CBF study replaces that exact
                # term; it is not an additional penalty and the legacy
                # potential field is not evaluated for this formulation.
                cf = float(np.clip(safety_potential, 0.0, 1.0))
            else:
                cf = float(np.clip(self._potential_field_cost(), 0.0, 1.0))
            previous_cf = cf
            cay = self._lateral_acceleration_cost()
            overtakes = self._overtake_count(previous_dx)
            progress_m = self._forward_progress(previous_ego_x)
            progress_normalized = self._normalized_forward_progress(
                progress_m, desired_speed
            )
            progress_clipped = float(
                np.clip(
                    progress_normalized,
                    0.0,
                    float(cfg.get("progress_clip", 1.25)),
                )
            )
            progress_reward = float(cfg.get("progress_reward_weight", 0.0)) * progress_clipped
            jerk_vector, jerk_norm, jerk_cost = self._jerk_metrics()
            jerk_penalty = float(cfg.get("jerk_penalty_weight", 0.0)) * jerk_cost

            reward_mode = str(cfg.get("reward_mode", "reciprocal")).strip().lower()
            if reward_mode not in {"reciprocal", "additive"}:
                raise ValueError(f"Unsupported reward_mode={reward_mode!r}")

            speed_sigma = max(float(cfg.get("speed_reward_sigma", 4.0)), 1e-6)
            lateral_sigma = max(float(cfg.get("lateral_reward_sigma", 1.0)), 1e-6)
            speed_tracking_reward = float(
                np.exp(-0.5 * ((ego_speed - target_speed) / speed_sigma) ** 2)
            )
            lateral_tracking_reward = float(
                np.exp(-0.5 * ((ego_y - target_y) / lateral_sigma) ** 2)
            )

            denom = (
                cfg["epsilon_r"]
                + cfg["wx"] * cx
                + cfg["wy"] * cy
                + cfg["wf"] * cf
                + cfg.get("way", 0.0) * cay
            )
            reciprocal_reward = cfg["epsilon_r"] / max(denom, 1e-9)
            additive_speed_reward = float(cfg.get("speed_reward_weight", 0.25)) * speed_tracking_reward
            additive_lateral_reward = float(cfg.get("lateral_reward_weight", 0.25)) * lateral_tracking_reward
            additive_risk_penalty = -float(cfg.get("risk_penalty_weight", 0.5)) * cf
            safety_potential_penalty = -float(cfg.get("safety_potential_weight", 0.0)) * float(safety_potential)
            safety_ellipse_penalty = -float(cfg.get("safety_ellipse_weight", 0.0)) * float(safety_ellipse_cost)
            safety_ttc_penalty = -float(cfg.get("safety_ttc_weight", 0.0)) * float(safety_ttc_cost)
            safety_total_penalty = (
                safety_potential_penalty
                + safety_ellipse_penalty
                + safety_ttc_penalty
            )
            potential_shaping_weight = float(cfg.get("risk_potential_shaping_weight", 0.0))
            potential_shaping_gamma = float(
                np.clip(cfg.get("risk_potential_shaping_gamma", 0.99), 0.0, 1.0)
            )
            potential_shaping_reward = potential_shaping_weight * (
                previous_cf - potential_shaping_gamma * cf
            )
            if bool(base._last_ego_collision):
                potential_shaping_reward = 0.0

            if reward_mode == "additive":
                reward = (
                    progress_reward
                    + additive_speed_reward
                    + additive_lateral_reward
                    + additive_risk_penalty
                    + safety_total_penalty
                    + potential_shaping_reward
                    - jerk_penalty
                )
            else:
                reward = (
                    reciprocal_reward
                    + progress_reward
                    + potential_shaping_reward
                    - jerk_penalty
                )

            if bool(base._last_ego_collision):
                if bool(cfg.get("collision_reward_override", False)):
                    reward = float(cfg["collision_penalty"])
                else:
                    reward += cfg["collision_penalty"]
            elif overtakes > 0:
                reward += cfg["overtake_bonus"] * min(overtakes, 1)

            components = {
                "reward": float(reward),
                "reward_mode_additive": float(reward_mode == "additive"),
                "reciprocal_reward": float(reciprocal_reward),
                "speed_tracking_reward": float(speed_tracking_reward),
                "lateral_tracking_reward": float(lateral_tracking_reward),
                "additive_speed_reward": float(additive_speed_reward),
                "additive_lateral_reward": float(additive_lateral_reward),
                "additive_risk_penalty": float(additive_risk_penalty),
                "safety_potential": float(safety_potential),
                "safety_potential_min_h": float(safety_potential_min_h),
                "safety_ellipse_cost": float(safety_ellipse_cost),
                "safety_ttc_cost": float(safety_ttc_cost),
                "safety_potential_penalty": float(safety_potential_penalty),
                "safety_ellipse_penalty": float(safety_ellipse_penalty),
                "safety_ttc_penalty": float(safety_ttc_penalty),
                "safety_total_penalty": float(safety_total_penalty),
                "potential_previous_cf": float(previous_cf),
                "potential_shaping_reward": float(potential_shaping_reward),
                "cx": float(cx),
                "cy": float(cy),
                "cf": float(cf),
                "cay": float(cay),
                "ay": float(self._ego_lateral_acceleration()),
                "ego_y": float(ego_y),
                "ego_speed": float(ego_speed),
                "desired_speed": float(desired_speed),
                "speed_deviation": float(ego_speed - desired_speed),
                "abs_speed_deviation": float(abs(ego_speed - desired_speed)),
                "target_speed_deviation": float(ego_speed - target_speed),
                "abs_target_speed_deviation": float(abs(ego_speed - target_speed)),
                "target_y": float(target_y),
                "lat_y_error_m": float(lat_y_error_m),
                "lat_y_coherence": float(lat_y_coherence),
                "target_speed": float(target_speed),
                "zone_found": float(zone_found),
                "overtakes": float(overtakes),
                "progress_m": float(progress_m),
                "progress_normalized": float(progress_normalized),
                "progress_clipped": float(progress_clipped),
                "progress_reward": float(progress_reward),
                "jerk_ax": float(jerk_vector[0]),
                "jerk_ay": float(jerk_vector[1]),
                "jerk_norm": float(jerk_norm),
                "jerk_cost": float(jerk_cost),
                "jerk_penalty": float(jerk_penalty),
                "ego_collision": float(base._last_ego_collision),
            }
            components["applied_ax"] = float(self._current_applied_acceleration()[0])
            components["applied_ay"] = float(self._current_applied_acceleration()[1])
            return float(reward), components

        def _safety_potential_cost(self) -> tuple[float, float, float, float]:
            """Return the direct CBF violation potential and diagnostics."""

            base = self.base_env
            cfg = self.reward_config
            formulation = str(
                cfg.get("safety_potential_formulation", "none")
            ).strip().lower()
            if formulation in {"none", "off", "legacy"}:
                return 0.0, float("inf"), 0.0, 0.0
            if formulation != "cbf_violation":
                raise ValueError(
                    "This PPO reward extension supports only "
                    f"safety_potential_formulation='cbf_violation' or 'none', got {formulation!r}"
                )

            eps_side = max(float(cfg.get("safety_potential_eps_side", 0.10)), 0.0)
            sensing_range = float(base.config["sensing_range"])
            ego = base.vehicle
            ego_a = max(float(ego.length) / np.sqrt(2.0) + 2.0 * eps_side, 1e-6)
            ego_b = max(float(ego.width) / np.sqrt(2.0) + 2.0 * eps_side, 1e-6)
            alpha = float(cfg.get("safety_cbf_alpha", 1.0))
            psi_scale = max(float(cfg.get("safety_cbf_psi_scale", 1.0)), 1e-6)

            risks: list[float] = []
            h_values: list[float] = []
            for vehicle in base.road.vehicles:
                if vehicle is ego:
                    continue
                dx = float(base._signed_distance(ego.position[0], vehicle.position[0]))
                if abs(dx) > sensing_range:
                    continue
                dy = float(vehicle.position[1] - ego.position[1])
                other_a = max(
                    float(vehicle.length) / np.sqrt(2.0) + 2.0 * eps_side,
                    1e-6,
                )
                other_b = max(
                    float(vehicle.width) / np.sqrt(2.0) + 2.0 * eps_side,
                    1e-6,
                )
                A = ego_a + other_a
                B = ego_b + other_b
                h_value = (dx / A) ** 2 + (dy / B) ** 2 - 1.0
                dvx = float(vehicle.vx - ego.vx)
                dvy = float(getattr(vehicle, "vy", 0.0) - getattr(ego, "vy", 0.0))
                h_dot = 2.0 * dx * dvx / (A**2) + 2.0 * dy * dvy / (B**2)
                psi_value = h_dot + alpha * h_value
                risk_value = float(
                    np.clip(max(0.0, -psi_value / psi_scale), 0.0, 1.0) ** 2
                )
                h_values.append(float(h_value))
                risks.append(risk_value)

            if not risks:
                return 0.0, float("inf"), 0.0, 0.0
            return (
                float(np.clip(max(risks), 0.0, 1.0)),
                float(min(h_values)),
                0.0,
                0.0,
            )

    namespace["KaralakouRewardWrapper"] = PPOSafetyKaralakouRewardWrapper
    namespace["PPOSafetyKaralakouRewardWrapper"] = PPOSafetyKaralakouRewardWrapper
