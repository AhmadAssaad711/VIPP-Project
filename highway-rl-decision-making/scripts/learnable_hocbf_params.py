"""Learnable HOCBF parameters for the projected PPO pilot.

This module deliberately keeps the existing :mod:`projected_ppo_cbf` policy and
its actor correction loss intact.  It adds a separate rate network and a thin
environment/algorithm adapter around that policy:

* the environment owns the augmented state ``[s, p1, p2]`` and advances ``p``
  once per policy action;
* the learner evaluates ``Gamma_psi`` centrally, then refreshes the worker's
  dynamic HOCBF context before the PPO actor is evaluated;
* PPO sees dynamic rows/bounds as detached data;
* the parameter learner rebuilds the bounds in Torch and differentiates through
  a truncated ``p`` recurrence, without differentiating through the simulator.

The implementation is intentionally self-contained so the notebook can launch
it in Windows subprocess workers without pickling notebook-defined classes.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from cbf_projection import project_polytope_2d_torch
from ppo_cbf_env import CBFContextPhysicalActionWrapper
from projected_ppo_cbf import ProjectedCBFPPO, ProjectedCBFActorCriticPolicy


@dataclass(frozen=True)
class LearnableCBFConfig:
    """Numerical contract for the first learnable-CBF pilot."""

    p_nominal: tuple[float, float] = (2.3, 2.3)
    p_min: tuple[float, float] = (1.15, 1.15)
    p_max: tuple[float, float] = (3.45, 3.45)
    nu_max: tuple[float, float] = (2.3, 2.3)
    gamma_lower: tuple[float, float] = (2.0, 2.0)
    gamma_upper: tuple[float, float] = (2.0, 2.0)
    dt_policy: float = 0.05
    unroll_horizon: int = 16
    lambda_feas: float = 10.0
    lambda_intervention: float = 1.0
    lambda_smooth: float = 0.01
    lambda_reg: float = 0.1
    feasibility_epsilon: float = 1e-6
    bound_hit_tolerance: float = 1e-3
    action_scale: tuple[float, float] = (6.0, 6.0)

    def validate(self) -> "LearnableCBFConfig":
        p0 = np.asarray(self.p_nominal, dtype=float)
        pmin = np.asarray(self.p_min, dtype=float)
        pmax = np.asarray(self.p_max, dtype=float)
        numax = np.asarray(self.nu_max, dtype=float)
        gl = np.asarray(self.gamma_lower, dtype=float)
        gu = np.asarray(self.gamma_upper, dtype=float)
        scale = np.asarray(self.action_scale, dtype=float)
        for name, value in (
            ("p_nominal", p0),
            ("p_min", pmin),
            ("p_max", pmax),
            ("nu_max", numax),
            ("gamma_lower", gl),
            ("gamma_upper", gu),
            ("action_scale", scale),
        ):
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain two finite values")
        if np.any(pmin <= 0.0) or np.any(pmin >= pmax):
            raise ValueError("p bounds must be positive and strictly ordered")
        if np.any(p0 < pmin) or np.any(p0 > pmax):
            raise ValueError("p_nominal must lie inside p bounds")
        if np.any(numax <= 0.0) or np.any(gl <= 0.0) or np.any(gu <= 0.0):
            raise ValueError("rate and auxiliary CBF gains must be positive")
        if not np.isfinite(self.dt_policy) or self.dt_policy <= 0.0:
            raise ValueError("dt_policy must be finite and positive")
        if float(self.dt_policy * max(*gl, *gu)) > 1.0 + 1e-8:
            raise ValueError("dt_policy * gamma must not exceed one for Euler invariance")
        if int(self.unroll_horizon) < 1:
            raise ValueError("unroll_horizon must be positive")
        for name in (
            "lambda_feas",
            "lambda_intervention",
            "lambda_smooth",
            "lambda_reg",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        return self

    def tensors(
        self, *, device: th.device, dtype: th.dtype = th.float32
    ) -> dict[str, th.Tensor]:
        self.validate()
        return {
            name: th.as_tensor(value, dtype=dtype, device=device)
            for name, value in (
                ("p_nominal", self.p_nominal),
                ("p_min", self.p_min),
                ("p_max", self.p_max),
                ("nu_max", self.nu_max),
                ("gamma_lower", self.gamma_lower),
                ("gamma_upper", self.gamma_upper),
                ("action_scale", self.action_scale),
            )
        }


def config_from_dict(value: Optional[dict[str, Any]]) -> LearnableCBFConfig:
    """Create and validate a configuration from JSON/notebook values."""

    if value is None:
        return LearnableCBFConfig().validate()
    fields = set(LearnableCBFConfig.__dataclass_fields__)
    kwargs = {key: value[key] for key in fields if key in value}
    for key in (
        "p_nominal",
        "p_min",
        "p_max",
        "nu_max",
        "gamma_lower",
        "gamma_upper",
        "action_scale",
    ):
        if key in kwargs:
            kwargs[key] = tuple(float(x) for x in kwargs[key])
    result = LearnableCBFConfig(**kwargs)
    return result.validate()


class ParameterRateNetwork(nn.Module):
    """Small Gamma network whose output is an unconstrained rate coordinate."""

    def __init__(
        self,
        input_dim: int = 34,
        hidden_dims: Sequence[int] = (64, 64),
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, int(width)), nn.Tanh()])
            previous = int(width)
        final = nn.Linear(previous, int(output_dim))
        # A zero rate is the exact continuation point of the fixed-pilot arm.
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, state_aug: th.Tensor) -> th.Tensor:
        return self.net(state_aug)


def parameter_rate_interval(
    p: th.Tensor, config: LearnableCBFConfig
) -> tuple[th.Tensor, th.Tensor]:
    """Return the auxiliary-CBF lower/upper rate limits."""

    tensors = config.tensors(device=p.device, dtype=p.dtype)
    lower = th.maximum(
        -tensors["nu_max"],
        -tensors["gamma_lower"] * (p - tensors["p_min"]),
    )
    upper = th.minimum(
        tensors["nu_max"],
        tensors["gamma_upper"] * (tensors["p_max"] - p),
    )
    return lower, upper


def smooth_project_parameter_rates(
    nu_raw: th.Tensor,
    p: th.Tensor,
    config: LearnableCBFConfig,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Map raw Gamma coordinates smoothly into the admissible rate interval."""

    lower, upper = parameter_rate_interval(p, config)
    nu_safe = 0.5 * (lower + upper) + 0.5 * (upper - lower) * th.tanh(nu_raw)
    return nu_safe, lower, upper


def _dynamic_bound(
    q: np.ndarray | float,
    h: np.ndarray | float,
    h_dot: np.ndarray | float,
    p1: float,
    p2: float,
    nu1: float,
) -> np.ndarray:
    return np.asarray(q, dtype=float) + (float(p1) + float(p2)) * np.asarray(
        h_dot, dtype=float
    ) + (float(p1) * float(p2) + float(nu1)) * np.asarray(h, dtype=float)


def build_dynamic_cbf_action_constraints(
    namespace: dict[str, Any],
    ego: dict[str, float],
    neighbors: list[dict[str, float]],
    road_width: float,
    ax_bounds: tuple[float, float],
    ay_bounds: tuple[float, float],
    eps_side: float,
    p1: float,
    p2: float,
    nu1: float,
    max_neighbor_constraints: Optional[int],
) -> dict[str, Any]:
    """Build ``A a <= b`` and gain-independent HOCBF primitives.

    ``q`` is the gain-independent part of ``h_ddot``.  The first
    ``num_dynamic`` rows are neighbor and boundary HOCBF rows; the final four
    rows are fixed actuator-box rows.
    """

    if max_neighbor_constraints is not None:
        neighbors = list(neighbors)[: int(max_neighbor_constraints)]
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    q_values: list[float] = []
    h_values: list[float] = []
    hdot_values: list[float] = []
    dynamic_mask: list[bool] = []
    min_h = np.inf
    min_center_distance = np.inf
    min_required_distance = np.inf

    pairwise_relative_state = namespace["pairwise_relative_state"]
    derivatives = namespace["centerline_barrier_derivatives"]
    for neighbor in neighbors:
        dx, dy, dvx, dvy = pairwise_relative_state(ego, neighbor)
        position = np.asarray([dx, dy], dtype=float)
        velocity = np.asarray([dvx, dvy], dtype=float)
        h, grad, hessian, center_distance, l_ego, l_other = derivatives(
            position, ego, neighbor, eps_side
        )
        other_acc = np.asarray(
            [float(neighbor.get("ax", 0.0)), float(neighbor.get("ay", 0.0))],
            dtype=float,
        )
        h_dot = float(np.asarray(grad, dtype=float) @ velocity)
        q = float(velocity.T @ np.asarray(hessian, dtype=float) @ velocity + np.asarray(grad, dtype=float) @ other_acc)
        rows.append(np.asarray(grad, dtype=float).reshape(2))
        q_values.append(q)
        h_values.append(float(h))
        hdot_values.append(h_dot)
        bounds.append(float(_dynamic_bound(q, h, h_dot, p1, p2, nu1)))
        dynamic_mask.append(True)
        min_h = min(min_h, float(h))
        min_center_distance = min(min_center_distance, float(center_distance))
        min_required_distance = min(min_required_distance, float(l_ego + l_other))

    ego_y = float(ego["y"])
    ego_vy = float(ego["vy"])
    ego_half_width = 0.5 * float(ego["width"])
    h_left = ego_y - ego_half_width
    h_right = float(road_width) - ego_half_width - ego_y
    boundary_terms = (
        (np.asarray([0.0, -1.0], dtype=float), h_left, ego_vy),
        (np.asarray([0.0, 1.0], dtype=float), h_right, -ego_vy),
    )
    for row, h_value, h_dot in boundary_terms:
        rows.append(row)
        q_values.append(0.0)
        h_values.append(float(h_value))
        hdot_values.append(float(h_dot))
        bounds.append(float(_dynamic_bound(0.0, h_value, h_dot, p1, p2, nu1)))
        dynamic_mask.append(True)

    lb = np.asarray([float(ax_bounds[0]), float(ay_bounds[0])], dtype=float)
    ub = np.asarray([float(ax_bounds[1]), float(ay_bounds[1])], dtype=float)
    box_rows = (
        (np.asarray([1.0, 0.0]), float(ub[0])),
        (np.asarray([-1.0, 0.0]), float(-lb[0])),
        (np.asarray([0.0, 1.0]), float(ub[1])),
        (np.asarray([0.0, -1.0]), float(-lb[1])),
    )
    for row, bound in box_rows:
        rows.append(row)
        bounds.append(bound)
        q_values.append(bound)
        h_values.append(0.0)
        hdot_values.append(0.0)
        dynamic_mask.append(False)

    row_array = np.asarray(rows, dtype=np.float32).reshape(-1, 2)
    bound_array = np.asarray(bounds, dtype=np.float32).reshape(-1)
    return {
        "rows": row_array,
        "bounds": bound_array,
        "cbf_rows": row_array[: len(dynamic_mask) - 4].copy(),
        "cbf_bounds": bound_array[: len(dynamic_mask) - 4].copy(),
        "lb": lb.astype(np.float32),
        "ub": ub.astype(np.float32),
        "min_h": np.nan if not np.isfinite(min_h) else float(min_h),
        "min_center_distance": np.nan if not np.isfinite(min_center_distance) else float(min_center_distance),
        "min_required_distance": np.nan if not np.isfinite(min_required_distance) else float(min_required_distance),
        "num_neighbor_constraints": int(len(neighbors)),
        "left_boundary_h": float(h_left),
        "right_boundary_h": float(h_right),
        "min_boundary_h": float(min(h_left, h_right)),
        "primitive_q": np.asarray(q_values, dtype=np.float32),
        "primitive_h": np.asarray(h_values, dtype=np.float32),
        "primitive_hdot": np.asarray(hdot_values, dtype=np.float32),
        "primitive_dynamic_mask": np.asarray(dynamic_mask, dtype=bool),
    }


def dynamic_bounds_from_primitives(
    q: th.Tensor,
    h: th.Tensor,
    h_dot: th.Tensor,
    static_bounds: th.Tensor,
    dynamic_mask: th.Tensor,
    p: th.Tensor,
    nu: th.Tensor,
) -> th.Tensor:
    """Construct dynamic bounds while preserving the computation graph."""

    p1, p2 = p[..., 0:1], p[..., 1:2]
    nu1 = nu[..., 0:1]
    dynamic = q + (p1 + p2) * h_dot + (p1 * p2 + nu1) * h
    return th.where(dynamic_mask, dynamic, static_bounds)


class AugmentedParameterStateWrapper(gym.Wrapper):
    """Append and advance ``p1,p2`` around an existing 32D state wrapper."""

    def __init__(
        self,
        env: gym.Env,
        *,
        p_nominal: tuple[float, float],
        p_min: tuple[float, float],
        p_max: tuple[float, float],
        dt_policy: float,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("learnable CBF requires a flat Box base observation")
        self.base_observation_dim = int(np.prod(env.observation_space.shape))
        self.p_nominal = np.asarray(p_nominal, dtype=np.float32).reshape(2)
        self.p_min = np.asarray(p_min, dtype=np.float32).reshape(2)
        self.p_max = np.asarray(p_max, dtype=np.float32).reshape(2)
        self.dt_policy = float(dt_policy)
        base_low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        base_high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([base_low, self.p_min]),
            high=np.concatenate([base_high, self.p_max]),
            dtype=np.float32,
        )
        self.p = self.p_nominal.copy()
        self.pending_nu = np.zeros(2, dtype=np.float32)
        self._last_observation: Optional[np.ndarray] = None

    def _augment(self, observation: Any) -> np.ndarray:
        base = np.asarray(observation, dtype=np.float32).reshape(-1)
        if base.size != self.base_observation_dim:
            raise ValueError(
                f"base observation width changed ({base.size} != {self.base_observation_dim})"
            )
        result = np.concatenate([base, self.p]).astype(np.float32)
        self._last_observation = result.copy()
        return result

    def reset(self, **kwargs):
        self.p = self.p_nominal.copy()
        self.pending_nu = np.zeros(2, dtype=np.float32)
        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        info.update(
            {
                "learnable_cbf_p1": float(self.p[0]),
                "learnable_cbf_p2": float(self.p[1]),
                "learnable_cbf_nu1": 0.0,
                "learnable_cbf_nu2": 0.0,
            }
        )
        return self._augment(observation), info

    def set_parameter_rate(self, nu_safe: Any) -> None:
        value = np.asarray(nu_safe, dtype=np.float32).reshape(-1)[:2]
        if value.size != 2 or not np.all(np.isfinite(value)):
            raise ValueError("nu_safe must contain two finite values")
        self.pending_nu = value.copy()

    def current_augmented_observation(self) -> np.ndarray:
        if self._last_observation is None:
            raise RuntimeError("environment has not been reset")
        return self._last_observation.copy()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        p_before = self.p.copy()
        requested = self.pending_nu.copy()
        proposed = p_before + self.dt_policy * requested
        clipped = np.clip(proposed, self.p_min, self.p_max).astype(np.float32)
        self.p = clipped
        info = dict(info)
        info.update(
            {
                "learnable_cbf_p1": float(p_before[0]),
                "learnable_cbf_p2": float(p_before[1]),
                "learnable_cbf_nu1": float(requested[0]),
                "learnable_cbf_nu2": float(requested[1]),
                "learnable_cbf_p1_next": float(clipped[0]),
                "learnable_cbf_p2_next": float(clipped[1]),
                "learnable_cbf_p_clipped": bool(np.any(np.abs(proposed - clipped) > 1e-6)),
            }
        )
        return self._augment(observation), reward, terminated, truncated, info

    def parameter_state(self) -> dict[str, Any]:
        return {
            "p": self.p.copy(),
            "pending_nu": self.pending_nu.copy(),
            "dt_policy": float(self.dt_policy),
        }


class LearnableCBFContextPhysicalActionWrapper(CBFContextPhysicalActionWrapper):
    """Dynamic HOCBF context with a latched policy-rate ``p,nu`` pair."""

    def __init__(self, env: gym.Env, *, namespace: dict[str, Any], config: LearnableCBFConfig, **kwargs) -> None:
        config.validate()
        self.learnable_config = config
        self.current_nu = np.zeros(2, dtype=np.float32)
        self.current_nu_raw = np.zeros(2, dtype=np.float32)
        super().__init__(
            env,
            namespace=namespace,
            k0=float(config.p_nominal[0] * config.p_nominal[1]),
            k1=float(config.p_nominal[0] + config.p_nominal[1]),
            base_observation_dim=int(np.prod(env.observation_space.shape)),
            **kwargs,
        )

    @property
    def parameter_state_wrapper(self) -> AugmentedParameterStateWrapper:
        if not isinstance(self.env, AugmentedParameterStateWrapper):
            raise TypeError("learnable CBF wrapper must directly wrap AugmentedParameterStateWrapper")
        return self.env

    @property
    def current_p(self) -> np.ndarray:
        return self.parameter_state_wrapper.p.copy()

    def _constraint_system(self) -> dict[str, Any]:
        ego = self.namespace["get_ego_state"](self)
        neighbors = self.namespace["get_neighbor_states"](
            self, neighbor_range=self.neighbor_range
        )
        road_width = float(self.namespace["_lane_free_base"](self).config["road_width"])
        p = self.current_p
        system = build_dynamic_cbf_action_constraints(
            self.namespace,
            ego,
            neighbors,
            road_width,
            self.ax_bounds,
            self.ay_bounds,
            self.eps_side,
            float(p[0]),
            float(p[1]),
            float(self.current_nu[0]),
            self.max_neighbor_constraints,
        )
        if int(system["rows"].shape[0]) > self.layout.max_constraints:
            raise RuntimeError(
                f"CBF produced {system['rows'].shape[0]} rows for a "
                f"{self.layout.max_constraints}-row context"
            )
        from ppo_cbf_env import constraint_system_hash

        system["hash"] = constraint_system_hash(system["rows"], system["bounds"])
        return system

    def _initial_safety_diagnostics(self) -> dict[str, Any]:
        """Check h >= 0 and ``dot(h)+p1*h >= 0`` at reset."""

        ego = self.namespace["get_ego_state"](self)
        neighbors = self.namespace["get_neighbor_states"](self, neighbor_range=self.neighbor_range)
        p1 = float(self.current_p[0])
        h_values: list[float] = []
        psi_values: list[float] = []
        geometry = self.namespace.get("pairwise_centerline_clearance")
        relative_state = self.namespace.get("pairwise_relative_state")
        derivatives = self.namespace.get("centerline_barrier_derivatives")
        if geometry is not None and relative_state is not None and derivatives is not None:
            for neighbor in neighbors:
                dx, dy, dvx, dvy = relative_state(ego, neighbor)
                h_value = float(geometry(np.asarray([dx, dy], dtype=float), ego, neighbor, eps_side=self.eps_side)[0])
                _, grad, _, *_ = derivatives(np.asarray([dx, dy], dtype=float), ego, neighbor, self.eps_side)
                h_dot = float(np.asarray(grad, dtype=float) @ np.asarray([dvx, dvy], dtype=float))
                h_values.append(h_value)
                psi_values.append(h_dot + p1 * h_value)
        base = self.namespace["_lane_free_base"](self)
        road_width = float(base.config["road_width"])
        half_width = 0.5 * float(ego["width"])
        left_h = float(ego["y"] - half_width)
        right_h = float(road_width - half_width - ego["y"])
        h_values.extend([left_h, right_h])
        psi_values.extend([
            float(ego["vy"] + p1 * left_h),
            float(-ego["vy"] + p1 * right_h),
        ])
        min_h = float(np.min(h_values)) if h_values else np.nan
        min_psi = float(np.min(psi_values)) if psi_values else np.nan
        tolerance = float(self.namespace.get("CBF_QP_FEASIBILITY_TOL", 1e-5))
        safe = bool(np.isfinite(min_h) and min_h >= -tolerance and np.isfinite(min_psi) and min_psi >= -tolerance)
        return {
            "cbf_initial_min_h": min_h,
            "cbf_initial_min_psi": min_psi,
            "cbf_initial_safe_set": safe,
        }

    def reset(self, **kwargs):
        self.current_nu = np.zeros(2, dtype=np.float32)
        self.current_nu_raw = np.zeros(2, dtype=np.float32)
        return super().reset(**kwargs)

    def refresh_parameter_rate(self, nu_safe: Any, nu_raw: Any | None = None) -> np.ndarray:
        safe = np.asarray(nu_safe, dtype=np.float32).reshape(-1)[:2]
        raw = safe if nu_raw is None else np.asarray(nu_raw, dtype=np.float32).reshape(-1)[:2]
        if safe.size != 2 or raw.size != 2 or not np.all(np.isfinite(safe)) or not np.all(np.isfinite(raw)):
            raise ValueError("nu_safe and nu_raw must contain two finite values")
        self.current_nu = safe.copy()
        self.current_nu_raw = raw.copy()
        self.parameter_state_wrapper.set_parameter_rate(self.current_nu)
        system = self._constraint_system()
        observation = self._augment_observation(
            self.parameter_state_wrapper.current_augmented_observation(), system
        )
        return observation.astype(np.float32)

    def parameter_snapshot(self) -> dict[str, Any]:
        system = self.current_constraint_system()
        max_constraints = int(self.layout.max_constraints)
        count = int(system["rows"].shape[0])
        rows = np.zeros((max_constraints, 2), dtype=np.float32)
        q = np.zeros(max_constraints, dtype=np.float32)
        h = np.zeros(max_constraints, dtype=np.float32)
        hdot = np.zeros(max_constraints, dtype=np.float32)
        static_bounds = np.zeros(max_constraints, dtype=np.float32)
        dynamic_mask = np.zeros(max_constraints, dtype=bool)
        mask = np.zeros(max_constraints, dtype=bool)
        rows[:count] = np.asarray(system["rows"], dtype=np.float32)
        q[:count] = np.asarray(system["primitive_q"], dtype=np.float32)
        h[:count] = np.asarray(system["primitive_h"], dtype=np.float32)
        hdot[:count] = np.asarray(system["primitive_hdot"], dtype=np.float32)
        static_bounds[:count] = np.asarray(system["bounds"], dtype=np.float32)
        dynamic_mask[:count] = np.asarray(system["primitive_dynamic_mask"], dtype=bool)
        mask[:count] = True
        return {
            "p": self.current_p,
            "nu_raw": self.current_nu_raw.copy(),
            "nu_safe": self.current_nu.copy(),
            "rows": rows,
            "q": q,
            "h": h,
            "h_dot": hdot,
            "static_bounds": static_bounds,
            "dynamic_mask": dynamic_mask,
            "mask": mask,
            "system_bounds": np.pad(np.asarray(system["bounds"], dtype=np.float32), (0, max_constraints - count)),
            "constraint_count": count,
            "hocbf_margin": float(system.get("hocbf_margin", np.nan)),
            "min_h": float(system.get("min_h", np.nan)),
        }


class LearnableParameterRollout:
    """Unflattened temporal auxiliary storage kept beside SB3's PPO buffer."""

    def __init__(self, n_steps: int, n_envs: int, *, state_dim: int = 34, max_constraints: int = 18):
        self.n_steps = int(n_steps)
        self.n_envs = int(n_envs)
        self.state_dim = int(state_dim)
        self.max_constraints = int(max_constraints)
        self.reset()

    def reset(self) -> None:
        shape = (self.n_steps, self.n_envs)
        self.state_aug = np.zeros((*shape, self.state_dim), dtype=np.float32)
        for name in ("p", "nu_raw", "nu_safe", "nu_lower", "nu_upper", "p_next", "mu_raw", "mu_safe", "latent_raw", "executed"):
            setattr(self, name, np.zeros((*shape, 2), dtype=np.float32))
        for name in ("rows",):
            setattr(self, name, np.zeros((*shape, self.max_constraints, 2), dtype=np.float32))
        for name in ("q", "h", "h_dot", "static_bounds"):
            setattr(self, name, np.zeros((*shape, self.max_constraints), dtype=np.float32))
        for name in ("dynamic_mask", "mask"):
            setattr(self, name, np.zeros((*shape, self.max_constraints), dtype=bool))
        for name in ("done", "feasible", "fallback", "p_clipped"):
            setattr(self, name, np.zeros(shape, dtype=bool))
        for name in ("slack", "qp_infeasible", "correction", "hocbf_margin"):
            setattr(self, name, np.zeros(shape, dtype=np.float32))

    def add(self, step: int, env_index: int, **values: Any) -> None:
        for name, value in values.items():
            array = getattr(self, name)
            converted = np.asarray(value, dtype=array.dtype)
            target = array[int(step), int(env_index)]
            if converted.shape != target.shape:
                raise ValueError(
                    f"parameter rollout field {name!r} has shape {converted.shape}, "
                    f"expected {target.shape} (value={value!r})"
                )
            array[int(step), int(env_index)] = converted

    def tensors(self, device: th.device) -> dict[str, th.Tensor]:
        names = (
            "state_aug", "p", "nu_raw", "nu_safe", "nu_lower", "nu_upper", "p_next",
            "mu_raw", "mu_safe", "latent_raw", "executed", "rows", "q", "h", "h_dot",
            "static_bounds", "dynamic_mask", "mask", "done", "feasible", "fallback",
            "p_clipped", "slack", "qp_infeasible", "correction", "hocbf_margin",
        )
        return {name: th.as_tensor(getattr(self, name), device=device) for name in names}

    def summary(self, config: LearnableCBFConfig) -> dict[str, float]:
        tolerance = float(config.bound_hit_tolerance)
        result: dict[str, float] = {}
        for index, label in enumerate(("p1", "p2")):
            p = self.p[..., index]
            result[f"{label}_mean"] = float(np.mean(p))
            result[f"{label}_min"] = float(np.min(p))
            result[f"{label}_max"] = float(np.max(p))
            result[f"{label}_lower_hit_rate"] = float(np.mean(np.abs(p - config.p_min[index]) <= tolerance))
            result[f"{label}_upper_hit_rate"] = float(np.mean(np.abs(p - config.p_max[index]) <= tolerance))
            nu = self.nu_safe[..., index]
            nu_raw = self.nu_raw[..., index]
            result[f"nu{index + 1}_raw_mean"] = float(np.mean(nu_raw))
            result[f"nu{index + 1}_mean"] = float(np.mean(nu))
            result[f"nu{index + 1}_lower_hit_rate"] = float(np.mean(np.abs(nu - self.nu_lower[..., index]) <= tolerance))
            result[f"nu{index + 1}_upper_hit_rate"] = float(np.mean(np.abs(nu - self.nu_upper[..., index]) <= tolerance))
        result.update({
            "slack_mean": float(np.mean(self.slack)),
            "slack_max": float(np.max(self.slack)),
            "qp_infeasible_rate": float(np.mean(self.qp_infeasible)),
            "feasible_rate": float(np.mean(self.feasible)),
            "fallback_rate": float(np.mean(self.fallback)),
            "p_clipped_rate": float(np.mean(self.p_clipped)),
            "correction_mean": float(np.mean(self.correction)),
            "hocbf_margin_mean": float(np.nanmean(self.hocbf_margin)) if np.isfinite(self.hocbf_margin).any() else float("nan"),
            "hocbf_margin_min": float(np.nanmin(self.hocbf_margin)) if np.isfinite(self.hocbf_margin).any() else float("nan"),
        })
        return result


def _safe_action_for_parameter_loss(
    raw_action: th.Tensor,
    projection,
) -> th.Tensor:
    # Empty no-slack sets are behavioral fallbacks.  They must not create an
    # artificial action gradient, but the separately computed normalized slack
    # still differentiates through the dynamic bounds.
    return th.where(projection.feasible.unsqueeze(1), projection.action, projection.action.detach())


class LearnableProjectedCBFPPO(ProjectedCBFPPO):
    """Projected PPO plus a fully decoupled HOCBF parameter learner."""

    def __init__(
        self,
        *args,
        parameter_config: Optional[dict[str, Any] | LearnableCBFConfig] = None,
        parameter_hidden_dims: Sequence[int] = (64, 64),
        parameter_learning_rate: float = 1e-4,
        parameter_unroll_horizon: Optional[int] = None,
        learnable_parameters: bool = True,
        **kwargs,
    ) -> None:
        if isinstance(parameter_config, LearnableCBFConfig):
            config = parameter_config.validate()
        else:
            config = config_from_dict(parameter_config)
        self.parameter_config = config
        self.parameter_hidden_dims = tuple(int(x) for x in parameter_hidden_dims)
        self.parameter_learning_rate = float(parameter_learning_rate)
        self.parameter_unroll_horizon = int(parameter_unroll_horizon or config.unroll_horizon)
        self.learnable_parameters = bool(learnable_parameters)
        self.parameter_rollout: Optional[LearnableParameterRollout] = None
        self.parameter_learning_diagnostics: list[dict[str, float]] = []
        # SB3 reconstructs an instance for ``load()`` with only policy/env/
        # device before restoring the saved dictionary.  Supply the augmented
        # pilot layout at that construction point so Gamma is created with
        # 34 inputs instead of the parent class's historical 42D default.
        kwargs.setdefault("cbf_base_observation_dim", 34)
        kwargs.setdefault("cbf_max_constraints", 18)
        super().__init__(*args, **kwargs)
        base_dim = int(getattr(self, "cbf_layout").base_observation_dim)
        if base_dim < 2:
            raise ValueError("learnable projected PPO requires an augmented state with p1,p2")
        self.parameter_state_dim = base_dim
        self.parameter_net = ParameterRateNetwork(
            input_dim=base_dim,
            hidden_dims=self.parameter_hidden_dims,
            output_dim=2,
        ).to(self.device)
        if not np.isfinite(self.parameter_learning_rate) or self.parameter_learning_rate <= 0.0:
            raise ValueError("parameter_learning_rate must be finite and positive")
        self.parameter_optimizer = th.optim.Adam(
            self.parameter_net.parameters(), lr=self.parameter_learning_rate
        )

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        # SB3 expects two lists: state-dict-bearing modules and other Torch
        # variables (typically optimizers).  Keep the parent contract intact
        # while persisting the separate Gamma learner.
        state_dicts, torch_variables = super()._get_torch_save_params()
        state_dicts = list(state_dicts)
        torch_variables = list(torch_variables)
        state_dicts.append("parameter_net")
        # Do not place the Adam object in SB3's ``pytorch_variables.pth``.
        # Recent PyTorch releases load that archive with ``weights_only=True``
        # and reject optimizer pickles.  The optimizer is intentionally
        # recreated from the pilot learning rate when a checkpoint is loaded;
        # all learned CBF state lives in ``parameter_net``.
        return state_dicts, torch_variables

    def _excluded_save_params(self) -> list[str]:
        # The optimizer is deliberately rebuilt on load (see ``load`` below)
        # and must not be cloud-pickled alongside the algorithm data.  In
        # particular, an optimizer deserialized before ``parameter_net`` is
        # restored would retain references to stale parameter objects.
        return [*super()._excluded_save_params(), "parameter_optimizer"]

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a pilot checkpoint and bind a fresh Adam optimizer to Gamma."""

        model = super().load(*args, **kwargs)
        model.parameter_optimizer = th.optim.Adam(
            model.parameter_net.parameters(), lr=float(model.parameter_learning_rate)
        )
        return model

    def _parameter_rates_from_observation(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observation_array = np.asarray(observations, dtype=np.float32)
        state = observation_array[..., : self.parameter_state_dim].reshape(
            (-1, self.parameter_state_dim)
        )
        with th.no_grad():
            state_tensor = obs_as_tensor(state, self.device)
            p = state_tensor[:, -2:]
            raw = self.parameter_net(state_tensor)
            safe, lower, upper = smooth_project_parameter_rates(raw, p, self.parameter_config)
        return (
            raw.cpu().numpy().astype(np.float32),
            safe.cpu().numpy().astype(np.float32),
            lower.cpu().numpy().astype(np.float32),
            upper.cpu().numpy().astype(np.float32),
        )

    def _refresh_worker_parameter_context(self, observations: np.ndarray, *, fixed: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if fixed:
            observation_array = np.asarray(observations, dtype=np.float32)
            count = int(
                observation_array[..., : self.parameter_state_dim]
                .reshape((-1, self.parameter_state_dim))
                .shape[0]
            )
            raw = np.zeros((count, 2), dtype=np.float32)
            safe = raw.copy()
            p = observation_array[..., : self.parameter_state_dim].reshape(
                (-1, self.parameter_state_dim)
            )[:, -2:]
            p_tensor = th.as_tensor(p, dtype=th.float32, device=self.device)
            lower_tensor, upper_tensor = parameter_rate_interval(
                p_tensor, self.parameter_config
            )
            lower = lower_tensor.cpu().numpy().astype(np.float32)
            upper = upper_tensor.cpu().numpy().astype(np.float32)
        else:
            raw, safe, lower, upper = self._parameter_rates_from_observation(observations)
        refreshed: list[np.ndarray] = []
        for env_index in range(int(self.env.num_envs)):
            result = self.env.env_method(
                "refresh_parameter_rate",
                safe[env_index],
                raw[env_index],
                indices=env_index,
            )
            if not result:
                raise RuntimeError("worker did not return a refreshed learnable-CBF observation")
            refreshed.append(np.asarray(result[0], dtype=np.float32))
        return np.asarray(refreshed, dtype=np.float32), raw, safe, lower, upper

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect with a two-stage Gamma/context refresh at every policy step."""

        assert self._last_obs is not None, "No previous observation was provided"
        if not isinstance(self.policy, ProjectedCBFActorCriticPolicy):
            raise TypeError("learnable projected PPO requires ProjectedCBFActorCriticPolicy")
        self.policy.set_training_mode(False)
        self.parameter_net.eval()
        rollout_buffer.reset()
        self.parameter_rollout = LearnableParameterRollout(
            n_rollout_steps,
            env.num_envs,
            state_dim=self.parameter_state_dim,
            max_constraints=int(self.cbf_layout.max_constraints),
        )
        n_steps = 0
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            current_obs = np.asarray(self._last_obs, dtype=np.float32).copy()
            refreshed_obs, nu_raw, nu_safe, nu_lower, nu_upper = self._refresh_worker_parameter_context(
                current_obs, fixed=not self.learnable_parameters
            )
            self._last_obs = refreshed_obs
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                distribution, values, _, mu_raw, mu_safe, _ = self.policy._distribution_and_stages(obs_tensor)
                latent_tensor = distribution.get_actions(deterministic=False)
                log_prob_tensor = distribution.log_prob(latent_tensor)
            latent_actions = latent_tensor.cpu().numpy()
            executed_actions, projection_records = self._execution_actions(
                latent_actions, np.asarray(self._last_obs)
            )
            for env_index, record in enumerate(projection_records):
                env.env_method(
                    "set_projection_record",
                    latent_actions[env_index],
                    executed_actions[env_index],
                    feasible=bool(record["feasible"]),
                    fallback_used=bool(record["fallback_used"]),
                    projection_source=str(record["projection_source"]),
                    max_constraint_violation_safe=float(record["max_constraint_violation_safe"]),
                    active_indices=record["active_indices"],
                    constraint_hash=str(record["constraint_hash"]),
                    cbf_applied=bool(record["cbf_applied"]),
                    indices=env_index,
                )
            snapshots = [
                env.env_method("parameter_snapshot", indices=env_index)[0]
                for env_index in range(env.num_envs)
            ]
            new_obs, rewards, dones, infos = env.step(executed_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            actions = latent_actions
            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape(-1, 1)
            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
            for env_index, snapshot in enumerate(snapshots):
                info = infos[env_index]
                p_next = np.asarray(
                    [
                        float(info.get("learnable_cbf_p1_next", snapshot["p"][0] + self.parameter_config.dt_policy * nu_safe[env_index, 0])),
                        float(info.get("learnable_cbf_p2_next", snapshot["p"][1] + self.parameter_config.dt_policy * nu_safe[env_index, 1])),
                    ],
                    dtype=np.float32,
                )
                max_violation = float(info.get("cbf_max_constraint_violation_safe", 0.0))
                correction = float(
                    np.linalg.norm(
                        (mu_raw[env_index] - mu_safe[env_index]).detach().cpu().numpy()
                    )
                )
                self.parameter_rollout.add(
                    n_steps - 1,
                    env_index,
                    state_aug=self._last_obs[env_index, : self.parameter_state_dim],
                    p=np.asarray(snapshot["p"], dtype=np.float32),
                    nu_raw=nu_raw[env_index],
                    nu_safe=nu_safe[env_index],
                    nu_lower=nu_lower[env_index],
                    nu_upper=nu_upper[env_index],
                    p_next=p_next,
                    mu_raw=mu_raw[env_index].detach().cpu().numpy(),
                    mu_safe=mu_safe[env_index].detach().cpu().numpy(),
                    latent_raw=latent_actions[env_index],
                    executed=executed_actions[env_index],
                    rows=snapshot["rows"],
                    q=snapshot["q"],
                    h=snapshot["h"],
                    h_dot=snapshot["h_dot"],
                    static_bounds=snapshot["static_bounds"],
                    dynamic_mask=snapshot["dynamic_mask"],
                    mask=snapshot["mask"],
                    done=bool(dones[env_index]),
                    feasible=bool(info.get("cbf_qp_success", True)),
                    fallback=bool(info.get("cbf_fallback_used", False)),
                    p_clipped=bool(info.get("learnable_cbf_p_clipped", False)),
                    slack=max(0.0, max_violation),
                    qp_infeasible=not bool(info.get("cbf_qp_success", True)),
                    correction=correction,
                    hocbf_margin=float(info.get("cbf_hocbf_min_margin", np.nan)),
                )
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_prob_tensor,
                **(
                    {
                        "safety_costs": np.asarray(
                            [
                                float(
                                    np.clip(
                                        float(info.get("cbf_correction_norm_normalized", 0.0)) ** 2,
                                        0.0,
                                        rollout_buffer.safety_cost_clip,
                                    )
                                )
                                for info in infos
                            ],
                            dtype=np.float32,
                        ),
                        "safety_fallbacks": np.asarray(
                            [float(bool(info.get("cbf_fallback_used", False))) for info in infos],
                            dtype=np.float32,
                        ),
                    }
                    if hasattr(rollout_buffer, "safety_cost_clip")
                    else {}
                ),
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def _window_loss(
        self,
        data: dict[str, th.Tensor],
        env_index: int,
        start: int,
        stop: int,
    ) -> dict[str, th.Tensor]:
        config = self.parameter_config
        device = self.device
        p = data["p"][start, env_index].detach().clone()
        prev_nu = data["nu_safe"][start, env_index].detach()
        losses = {
            "feas": th.zeros((), device=device),
            "intervention": th.zeros((), device=device),
            "smooth": th.zeros((), device=device),
            "reg": th.zeros((), device=device),
        }
        count = 0
        for step in range(start, stop):
            state_base = data["state_aug"][step, env_index, : self.parameter_state_dim - 2].detach()
            state_aug = th.cat([state_base, p], dim=0).unsqueeze(0)
            nu_raw = self.parameter_net(state_aug).reshape(2)
            nu_safe, nu_lower, nu_upper = smooth_project_parameter_rates(
                nu_raw.unsqueeze(0), p.unsqueeze(0), config
            )
            nu_safe = nu_safe.reshape(2)
            rows = data["rows"][step, env_index].unsqueeze(0).to(device=device, dtype=th.float32)
            q = data["q"][step, env_index].unsqueeze(0).to(device=device, dtype=th.float32)
            h = data["h"][step, env_index].unsqueeze(0).to(device=device, dtype=th.float32)
            h_dot = data["h_dot"][step, env_index].unsqueeze(0).to(device=device, dtype=th.float32)
            static_bounds = data["static_bounds"][step, env_index].unsqueeze(0).to(device=device, dtype=th.float32)
            dynamic_mask = data["dynamic_mask"][step, env_index].unsqueeze(0).to(device=device, dtype=th.bool)
            mask = data["mask"][step, env_index].unsqueeze(0).to(device=device, dtype=th.bool)
            bounds = dynamic_bounds_from_primitives(
                q, h, h_dot, static_bounds, dynamic_mask, p.unsqueeze(0), nu_safe.unsqueeze(0)
            )
            raw_action = data["mu_raw"][step, env_index].detach().to(device=device, dtype=th.float32).unsqueeze(0)
            projection = project_polytope_2d_torch(
                raw_action,
                rows,
                bounds,
                mask,
                feasibility_tol=float(self.cbf_feasibility_tol),
                action_low=th.as_tensor(self.action_space.low, dtype=th.float32, device=device),
                action_high=th.as_tensor(self.action_space.high, dtype=th.float32, device=device),
                detach_constraints=False,
            )
            safe_action = _safe_action_for_parameter_loss(raw_action, projection)
            residual = th.einsum("bd,bmd->bm", safe_action, rows) - bounds
            normalized_violation = F.relu(
                residual / (th.linalg.vector_norm(rows, dim=2) + float(config.feasibility_epsilon))
            )
            dynamic = dynamic_mask & mask
            violation = th.where(dynamic, normalized_violation, th.zeros_like(normalized_violation))
            losses["feas"] = losses["feas"] + violation.amax(dim=1).square().mean()
            action_scale = th.as_tensor(config.action_scale, dtype=th.float32, device=device)
            losses["intervention"] = losses["intervention"] + ((raw_action - safe_action) / action_scale).square().sum(dim=1).mean()
            losses["smooth"] = losses["smooth"] + ((nu_safe - prev_nu) / th.as_tensor(config.nu_max, dtype=th.float32, device=device)).square().mean()
            p_min = th.as_tensor(config.p_min, dtype=th.float32, device=device)
            p_max = th.as_tensor(config.p_max, dtype=th.float32, device=device)
            losses["reg"] = losses["reg"] + ((p - th.as_tensor(config.p_nominal, dtype=th.float32, device=device)) / (p_max - p_min)).square().mean()
            prev_nu = nu_safe
            p_next = p + float(config.dt_policy) * nu_safe
            done = bool(data["done"][step, env_index].detach().cpu().item())
            p = th.where(
                th.full((2,), done, dtype=th.bool, device=device),
                th.as_tensor(config.p_nominal, dtype=th.float32, device=device),
                p_next,
            )
            count += 1
        if count:
            losses = {key: value / float(count) for key, value in losses.items()}
        return losses

    def _train_parameter_learner(self) -> None:
        if not self.learnable_parameters or self.parameter_rollout is None:
            return
        data = self.parameter_rollout.tensors(self.device)
        n_steps, n_envs = self.parameter_rollout.n_steps, self.parameter_rollout.n_envs
        horizon = max(1, min(int(self.parameter_unroll_horizon), n_steps))
        windows = [
            (env_index, start, min(start + horizon, n_steps))
            for env_index in range(n_envs)
            for start in range(0, n_steps, horizon)
        ]
        self.parameter_net.train()
        self.parameter_optimizer.zero_grad(set_to_none=True)
        totals = {key: 0.0 for key in ("feas", "intervention", "smooth", "reg", "total")}
        for env_index, start, stop in windows:
            losses = self._window_loss(data, env_index, start, stop)
            total = (
                self.parameter_config.lambda_feas * losses["feas"]
                + self.parameter_config.lambda_intervention * losses["intervention"]
                + self.parameter_config.lambda_smooth * losses["smooth"]
                + self.parameter_config.lambda_reg * losses["reg"]
            ) / max(len(windows), 1)
            total.backward()
            for key in ("feas", "intervention", "smooth", "reg"):
                totals[key] += float(losses[key].detach().cpu().item()) / max(len(windows), 1)
            totals["total"] += float(total.detach().cpu().item())
        th.nn.utils.clip_grad_norm_(self.parameter_net.parameters(), 1.0)
        self.parameter_optimizer.step()
        summary = self.parameter_rollout.summary(self.parameter_config)
        summary.update({f"loss_{key}": value for key, value in totals.items()})
        self.parameter_learning_diagnostics.append(summary)
        for key, value in summary.items():
            if np.isfinite(value):
                self.logger.record(f"train/learnable_cbf_{key}", float(value))

    def train(self) -> None:
        # Gamma is not registered under policy, but explicit requires_grad
        # guards make the cross-gradient contract testable and obvious.
        for parameter in self.parameter_net.parameters():
            parameter.requires_grad_(False)
        super().train()
        if self.learnable_parameters:
            # The parameter pass consumes stored, detached PPO means and
            # geometry.  Freeze the entire driving policy explicitly while its
            # Gamma graph is reconstructed, making dL_CBF/dtheta=0 by
            # construction rather than only by lack of use.
            policy_flags = [
                parameter.requires_grad for parameter in self.policy.parameters()
            ]
            for parameter in self.policy.parameters():
                parameter.requires_grad_(False)
            for parameter in self.parameter_net.parameters():
                parameter.requires_grad_(True)
            try:
                self._train_parameter_learner()
            finally:
                for parameter, requires_grad in zip(
                    self.policy.parameters(), policy_flags
                ):
                    parameter.requires_grad_(requires_grad)
        else:
            for parameter in self.parameter_net.parameters():
                parameter.requires_grad_(True)


def migrate_projected_policy_weights(source_model: Any, target_model: LearnableProjectedCBFPPO) -> dict[str, Any]:
    """Copy a 32D projected-policy checkpoint into the 34D policy.

    Matching tensors are copied exactly.  A two-column zero extension is used
    for the first actor/value linear layers, preserving the old policy at
    ``p1=p2=2.3``.  The parameter network remains zero-rate initialized.
    """

    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    copied: list[str] = []
    expanded: list[str] = []
    skipped: list[str] = []
    for key, value in source_state.items():
        if key not in target_state:
            skipped.append(key)
            continue
        destination = target_state[key]
        if tuple(destination.shape) == tuple(value.shape):
            destination.copy_(value.detach().to(destination.device, destination.dtype))
            copied.append(key)
            continue
        if (
            value.ndim == 2
            and destination.ndim == 2
            and destination.shape[0] == value.shape[0]
            and destination.shape[1] == value.shape[1] + 2
        ):
            destination.zero_()
            destination[:, : value.shape[1]].copy_(value.detach().to(destination.device, destination.dtype))
            expanded.append(key)
            continue
        skipped.append(key)
    target_model.policy.load_state_dict(target_state, strict=False)
    return {"copied": copied, "expanded": expanded, "skipped": skipped}


def migrate_projected_checkpoint(
    checkpoint: str | Path,
    target_model: LearnableProjectedCBFPPO,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load an existing projected PPO checkpoint and migrate its policy state."""

    source = ProjectedCBFPPO.load(str(checkpoint), device=device)
    result = migrate_projected_policy_weights(source, target_model)
    result["source_checkpoint"] = str(checkpoint)
    return result


__all__ = [
    "AugmentedParameterStateWrapper",
    "LearnableCBFConfig",
    "LearnableCBFContextPhysicalActionWrapper",
    "LearnableParameterRollout",
    "LearnableProjectedCBFPPO",
    "ParameterRateNetwork",
    "build_dynamic_cbf_action_constraints",
    "config_from_dict",
    "dynamic_bounds_from_primitives",
    "migrate_projected_checkpoint",
    "migrate_projected_policy_weights",
    "parameter_rate_interval",
    "smooth_project_parameter_rates",
]
