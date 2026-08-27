
"""Analyze the saved 50k PPO/CBF policies on common traffic states.

This is deliberately a *policy diagnostic*, not another evaluation protocol:

* The seven learned actors are queried on exactly the same state bank.
* No actor's commands are used to score an episode or change the bank.
* Every raw actor mean is compared with the same external CBF polytope.

The primary causal comparison is B3.1 -> B3.2.  Both use the differentiable
mean projection and CBF reward feedback; B3.2 alone adds
``lambda_mean * ||mu_raw - P_s(mu_raw)||^2`` with lambda_mean=0.10.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

# Keep this diagnostic lightweight beside any ongoing rollout training.
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
import torch as th


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_cbf_filter_ablation as protocol
import run_ppo_cbf_progression as progression
from cbf_projection import (
    CBFContextLayout,
    max_constraint_violation_numpy,
    project_polytope_2d_numpy,
    split_cbf_context_numpy,
)
from ppo_reward_safety import install_cbf_violation_reward
from projected_ppo_cbf import ProjectedCBFActorCriticPolicy


VARIANT_ORDER = ("B1", "B2_1", "B2_2", "B2_3", "B3_1", "B3_2", "B3_3")
VARIANT_LABELS = {
    "B1": "B1 nominal",
    "B2_1": "B2.1 reward",
    "B2_2": "B2.2 detached + reward",
    "B2_3": "B2.3 detached only",
    "B3_1": "B3.1 diff. projection",
    "B3_2": "B3.2 diff. + mean loss",
    "B3_3": "B3.3 diff. actor only",
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
PRIMARY_PAIR = ("B3_1", "B3_2")
PROBE_BASE_DIM = 32
PROBE_LAYOUT = CBFContextLayout(base_observation_dim=PROBE_BASE_DIM)
RAW_FEASIBILITY_TOL = 1e-6
INTERVENTION_THRESHOLD_NORMALIZED = 0.03


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if np.isfinite(numeric) else float(default)


def _tail_mean(values: Iterable[float], count: int = 10) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return np.nan
    return float(np.mean(array[-min(int(count), int(array.size)) :]))


def _model_run_dir(project_root: Path, variant_id: str) -> Path:
    return project_root / "artifacts" / "final_Results" / "models" / variant_id / "seed_307"


def load_specs(project_root: Path) -> dict[str, dict[str, Any]]:
    """Read saved model metadata without inferring a method from its name."""

    specs: dict[str, dict[str, Any]] = {}
    for variant_id in VARIANT_ORDER:
        run_dir = _model_run_dir(project_root, variant_id)
        config_path = run_dir / "run_config.json"
        signature_path = run_dir / "training_signature.json"
        if not config_path.is_file() or not signature_path.is_file():
            raise FileNotFoundError(f"Missing saved configuration for {variant_id}: {run_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        signature = json.loads(signature_path.read_text(encoding="utf-8"))
        specs[variant_id] = {
            "variant_id": variant_id,
            "label": VARIANT_LABELS[variant_id],
            "run_dir": run_dir,
            "model_path": run_dir / "model_final.zip",
            "variant": str(config["variant"]),
            "config": config,
            "signature": signature,
        }
        if not specs[variant_id]["model_path"].is_file():
            raise FileNotFoundError(f"Missing model checkpoint: {specs[variant_id]['model_path']}")
    return specs


def ablation_map(specs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Make the experimental comparison structure explicit in a CSV."""

    rows: list[dict[str, Any]] = []
    comparison_roles = {
        "B1": "baseline; cross-method descriptive only",
        "B2_1": "paired with B2.2 for detached-target effect",
        "B2_2": "B2.1 + stopped-gradient hard-projection target",
        "B2_3": "secondary detached actor-only reference",
        "B3_1": "primary control for B3.2",
        "B3_2": "primary differentiable mean-loss treatment",
        "B3_3": "secondary differentiable actor-only reference",
    }
    for variant_id in VARIANT_ORDER:
        spec = specs[variant_id]
        config = spec["config"]
        variant_spec = config.get("variant_spec", {})
        if not isinstance(variant_spec, dict):
            variant_spec = {}
        contract = spec["signature"].get("model_contract", {})
        rows.append(
            {
                "variant_id": variant_id,
                "label": spec["label"],
                "saved_variant": spec["variant"],
                "reward_cbf_penalty": bool(variant_spec.get("reward_penalty", False)),
                "projected_actor_mean": bool(variant_spec.get("projected_mean", False)),
                "differentiable_actor_loss": bool(
                    variant_spec.get("differentiable_actor_loss", False)
                ),
                "detached_actor_loss": bool(variant_spec.get("detached_actor_loss", False)),
                "lambda_mean": _finite(config.get("lambda_mean", 0.0), 0.0),
                "lambda_detached_actor": _finite(
                    config.get("lambda_detached_actor", 0.0), 0.0
                ),
                "lambda_delta": _finite(config.get("lambda_delta", 0.0), 0.0),
                "lambda_intervention": _finite(
                    config.get("lambda_intervention", 0.0), 0.0
                ),
                "actor_cbf_gradient_path": contract.get(
                    "actor_cbf_gradient_path", "none (nominal)"
                ),
                "base_observation_dim": contract.get("base_observation_dim", np.nan),
                "simulation_frequency_hz": config.get("env_config", {}).get(
                    "simulation_frequency", np.nan
                ),
                "policy_frequency_hz": config.get("env_config", {}).get(
                    "policy_frequency", np.nan
                ),
                "comparison_role": comparison_roles[variant_id],
            }
        )
    return pd.DataFrame(rows)


def _deduplicated_tag(
    scalars: pd.DataFrame, variant_id: str, tag: str
) -> pd.DataFrame:
    data = scalars.loc[
        (scalars["variant_id"] == variant_id) & (scalars["tag"] == tag),
        ["step", "value"],
    ].copy()
    if data.empty:
        return data
    data["step"] = pd.to_numeric(data["step"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna().sort_values("step")
    return data.groupby("step", as_index=False)["value"].mean()


def training_summary(
    scalars: pd.DataFrame, gradients: pd.DataFrame, specs: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """Summarize end-of-training behavior and logged gradient diagnostics."""

    tags = {
        "final_ep_rew_mean": "rollout/ep_rew_mean",
        "final_ep_len_mean": "rollout/ep_len_mean",
        "final_return_per_timestep": "rollout/return_per_timestep",
        "final_explained_variance": "train/explained_variance",
        "final_approx_kl": "train/approx_kl",
        "final_cbf_mean_correction": "train/cbf_mean_correction",
        "final_cbf_mean_loss": "train/cbf_mean_loss",
        "final_cbf_mean_infeasible_rate": "train/cbf_mean_infeasible_rate",
    }
    rows: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        row: dict[str, Any] = {
            "variant_id": variant_id,
            "label": specs[variant_id]["label"],
        }
        for column, tag in tags.items():
            data = _deduplicated_tag(scalars, variant_id, tag)
            row[column] = float(data["value"].iloc[-1]) if not data.empty else np.nan
            row[f"tail10_{column.removeprefix('final_')}"] = (
                _tail_mean(data["value"].tolist()) if not data.empty else np.nan
            )
        gradient_data = gradients.loc[gradients["variant_id"] == variant_id].copy()
        for column in (
            "g_ppo_norm",
            "g_cbf_norm",
            "g_cbf_to_g_ppo_ratio",
            "g_ppo_g_cbf_cosine",
        ):
            values = pd.to_numeric(gradient_data.get(column), errors="coerce")
            finite = values[np.isfinite(values)] if values is not None else pd.Series(dtype=float)
            row[f"mean_{column}"] = float(finite.mean()) if len(finite) else np.nan
            row[f"final_{column}"] = float(finite.iloc[-1]) if len(finite) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _find_context_wrapper(env: Any) -> Any:
    current = env
    for _ in range(20):
        if hasattr(current, "current_constraint_system"):
            return current
        if not hasattr(current, "env"):
            break
        current = current.env
    raise RuntimeError("Could not find CBFContextPhysicalActionWrapper in probe env")


def _probe_command(controller: str, step: int) -> np.ndarray:
    """Small fixed commands diversify the state bank without using an actor."""

    if controller == "coast":
        return np.asarray([0.0, 0.0], dtype=np.float32)
    if controller == "accelerate":
        return np.asarray([1.5, 0.0], dtype=np.float32)
    if controller == "brake":
        return np.asarray([-1.5, 0.0], dtype=np.float32)
    if controller == "lateral_sweep":
        lateral = 0.65 if (int(step) // 10) % 2 == 0 else -0.65
        return np.asarray([0.40, lateral], dtype=np.float32)
    raise ValueError(f"Unknown probe controller {controller!r}")


def build_probe_environment(
    project_root: Path, specs: dict[str, dict[str, Any]]
) -> tuple[Any, dict[str, Any]]:
    """Build a common state generator with CBF context but no CBF action filter."""

    protocol.set_stable_native_defaults()
    try:
        th.set_num_threads(1)
        th.set_num_interop_threads(1)
    except RuntimeError:
        pass
    namespace = protocol.bootstrap_notebook_namespace(project_root)
    protocol.exec_required_notebook_cells(
        project_root / "notebooks" / "lanelessKaralakou.ipynb", namespace
    )
    install_cbf_violation_reward(namespace)
    namespace["DEVICE"] = "cpu"

    # B3.2 is the primary treatment config.  Keep its CBF-safe reset sampler
    # but intentionally do not project the scripted probe commands at either
    # policy or physics rate.  Traffic's ordinary simulator guard is untouched.
    config = copy.deepcopy(specs["B3_2"]["config"]["env_config"])
    config["cbf_substep_filtering"] = False
    config["cbf_require_initial_safe_set"] = True
    env = progression.make_ppo_cbf_env(
        namespace,
        env_config=config,
        reward_config=copy.deepcopy(specs["B3_2"]["config"]["reward_config"]),
        project_inputs=False,
        lambda_delta=0.0,
        lambda_intervention=0.0,
        correction_epsilon=INTERVENTION_THRESHOLD_NORMALIZED,
        action_rate_penalty_lambda=0.0,
    )
    if int(np.prod(env.observation_space.shape)) != PROBE_LAYOUT.observation_dim:
        raise RuntimeError(
            "Probe environment has unexpected augmented observation width "
            f"{env.observation_space.shape}; expected {PROBE_LAYOUT.observation_dim}"
        )
    return env, config


def collect_probe_states(
    env: Any,
    *,
    seeds: int,
    steps_per_seed: int,
    seed_start: int,
) -> pd.DataFrame:
    """Collect a fixed bank of pre-action CBF contexts and base observations."""

    if seeds <= 0 or steps_per_seed <= 0:
        raise ValueError("seeds and steps_per_seed must be positive")
    context_wrapper = _find_context_wrapper(env)
    controllers = ("coast", "accelerate", "brake", "lateral_sweep")
    rows_out: list[dict[str, Any]] = []
    state_id = 0
    for controller in controllers:
        for seed_index in range(int(seeds)):
            scenario_seed = int(seed_start) + seed_index
            observation, _info = env.reset(seed=scenario_seed)
            observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            for step in range(int(steps_per_seed)):
                base, constraint_rows, bounds, mask = split_cbf_context_numpy(
                    observation, layout=PROBE_LAYOUT
                )
                system = context_wrapper.current_constraint_system()
                neutral_projection = project_polytope_2d_numpy(
                    np.zeros(2, dtype=np.float32),
                    constraint_rows,
                    bounds,
                    mask,
                    action_low=np.asarray([-3.0, -3.0], dtype=np.float32),
                    action_high=np.asarray([3.0, 3.0], dtype=np.float32),
                )
                command = _probe_command(controller, step)
                rows_out.append(
                    {
                        "state_id": state_id,
                        "controller": controller,
                        "scenario_seed": scenario_seed,
                        "policy_step": int(step),
                        "command_ax": float(command[0]),
                        "command_ay": float(command[1]),
                        "action_set_feasible": bool(neutral_projection.feasible),
                        "constraint_count": int(np.sum(mask > 0.5)),
                        "cbf_min_h": _finite(system.get("min_h")),
                        "cbf_min_center_distance": _finite(system.get("min_center_distance")),
                        "cbf_min_required_distance": _finite(system.get("min_required_distance")),
                        "cbf_min_boundary_h": _finite(system.get("min_boundary_h")),
                        "previous_action_ax_normalized": float(base[30]),
                        "previous_action_ay_normalized": float(base[31]),
                        "observation": observation.copy(),
                    }
                )
                state_id += 1
                observation, _reward, terminated, truncated, _info = env.step(command)
                observation = np.asarray(observation, dtype=np.float32).reshape(-1)
                if bool(terminated) or bool(truncated):
                    break
    if not rows_out:
        raise RuntimeError("Probe state bank is empty")
    return pd.DataFrame(rows_out)


def _model_observation_from_probe(model: Any, probe_observation: np.ndarray) -> np.ndarray:
    """Adapt B1's 30D base input versus the later 32D at-1 base input."""

    observation = np.asarray(probe_observation, dtype=np.float32).reshape(-1)
    if observation.size != PROBE_LAYOUT.observation_dim:
        raise ValueError(f"Unexpected probe observation width {observation.size}")
    extractor = getattr(model.policy, "features_extractor", None)
    base_dim = int(getattr(extractor, "base_observation_dim", PROBE_BASE_DIM))
    if base_dim > PROBE_BASE_DIM:
        raise ValueError(f"Model base observation dimension {base_dim} exceeds probe base")
    context = observation[PROBE_BASE_DIM:]
    model_observation = np.concatenate((observation[:base_dim], context)).astype(
        np.float32, copy=False
    )
    expected = int(np.prod(model.observation_space.shape))
    if model_observation.size != expected:
        raise RuntimeError(
            "Model observation adaptation failed: "
            f"{model_observation.size} != expected {expected}"
        )
    return model_observation


def _raw_actor_mean(model: Any, observation: np.ndarray) -> np.ndarray:
    """Return the unbounded neural actor mean, before CBF/box execution."""

    policy = model.policy
    obs_tensor, _vectorized = policy.obs_to_tensor(observation)
    with th.no_grad():
        if hasattr(policy, "_latents"):
            latent_pi, _latent_vf = policy._latents(obs_tensor)
        else:
            features = policy.extract_features(obs_tensor)
            if bool(getattr(policy, "share_features_extractor", True)):
                latent_pi, _latent_vf = policy.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = policy.mlp_extractor.forward_actor(pi_features)
                _latent_vf = policy.mlp_extractor.forward_critic(vf_features)
        action = policy.action_net(latent_pi)
    return np.asarray(action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)[:2]


def load_models(specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for variant_id in VARIANT_ORDER:
        spec = specs[variant_id]
        model = progression.load_model(spec["variant"], spec["model_path"], "cpu")
        model.policy.set_training_mode(False)
        models[variant_id] = model
    return models


def probe_actions(
    states: pd.DataFrame, models: dict[str, Any]
) -> pd.DataFrame:
    """Query all raw actor means and project each through one shared CBF set."""

    physical_low = np.asarray([-3.0, -3.0], dtype=np.float32)
    physical_high = np.asarray([3.0, 3.0], dtype=np.float32)
    half_range = np.maximum(0.5 * (physical_high - physical_low), 1e-6)
    action_rows: list[dict[str, Any]] = []
    state_columns = [
        "state_id",
        "controller",
        "scenario_seed",
        "policy_step",
        "action_set_feasible",
        "constraint_count",
        "cbf_min_h",
        "cbf_min_center_distance",
        "cbf_min_required_distance",
        "cbf_min_boundary_h",
    ]
    for state in states.itertuples(index=False):
        probe_observation = np.asarray(getattr(state, "observation"), dtype=np.float32)
        _base, constraint_rows, bounds, mask = split_cbf_context_numpy(
            probe_observation, layout=PROBE_LAYOUT
        )
        common = {column: getattr(state, column) for column in state_columns}
        for variant_id in VARIANT_ORDER:
            model = models[variant_id]
            model_observation = _model_observation_from_probe(model, probe_observation)
            raw_action = _raw_actor_mean(model, model_observation)
            external = project_polytope_2d_numpy(
                raw_action,
                constraint_rows,
                bounds,
                mask,
                action_low=physical_low,
                action_high=physical_high,
            )
            raw_max_violation = max_constraint_violation_numpy(
                raw_action, constraint_rows, bounds, mask
            )
            raw_in_box = bool(
                np.all(raw_action >= physical_low - RAW_FEASIBILITY_TOL)
                and np.all(raw_action <= physical_high + RAW_FEASIBILITY_TOL)
            )
            correction = np.asarray(external.action, dtype=np.float32) - raw_action
            correction_physical = float(np.linalg.norm(correction))
            correction_normalized = float(np.linalg.norm(correction / half_range))
            raw_feasible = bool(
                external.feasible
                and raw_in_box
                and raw_max_violation <= RAW_FEASIBILITY_TOL
            )

            internal_safe = np.asarray([np.nan, np.nan], dtype=np.float32)
            internal_feasible: bool | None = None
            internal_match_error = np.nan
            if isinstance(model.policy, ProjectedCBFActorCriticPolicy):
                stages = model.predict_action_stages(model_observation, deterministic=True)
                internal_safe = np.asarray(stages["mu_safe"], dtype=np.float32).reshape(-1)[:2]
                internal_feasible = bool(
                    np.asarray(stages["mean_feasible"], dtype=bool).reshape(-1)[0]
                )
                internal_match_error = float(
                    np.linalg.norm(internal_safe - np.asarray(external.action))
                )

            action_rows.append(
                {
                    **common,
                    "variant_id": variant_id,
                    "label": VARIANT_LABELS[variant_id],
                    "raw_ax": float(raw_action[0]),
                    "raw_ay": float(raw_action[1]),
                    "raw_in_box": raw_in_box,
                    "raw_max_constraint_violation": float(raw_max_violation),
                    "raw_feasible": raw_feasible,
                    "safe_ax_external": float(external.action[0]),
                    "safe_ay_external": float(external.action[1]),
                    "projection_correction_physical": correction_physical,
                    "projection_correction_normalized": correction_normalized,
                    "external_intervention": bool(
                        correction_normalized > INTERVENTION_THRESHOLD_NORMALIZED
                    ),
                    "projection_set_feasible": bool(external.feasible),
                    "projection_fallback_used": bool(external.fallback_used),
                    "projection_source": str(external.source),
                    "internal_safe_ax": float(internal_safe[0]),
                    "internal_safe_ay": float(internal_safe[1]),
                    "internal_mean_feasible": internal_feasible,
                    "internal_external_projection_error": internal_match_error,
                }
            )
    return pd.DataFrame(action_rows)


def summarize_probe(actions: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw-policy alignment only where a CBF feasible set exists."""

    rows: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        data = actions.loc[actions["variant_id"] == variant_id].copy()
        usable = data.loc[data["action_set_feasible"].astype(bool)].copy()
        rows.append(
            {
                "variant_id": variant_id,
                "label": VARIANT_LABELS[variant_id],
                "state_count": int(len(data)),
                "feasible_set_state_count": int(len(usable)),
                "infeasible_set_state_rate": float(
                    1.0 - len(usable) / max(len(data), 1)
                ),
                "mean_raw_ax": float(data["raw_ax"].mean()),
                "mean_raw_ay": float(data["raw_ay"].mean()),
                "raw_in_box_rate": float(usable["raw_in_box"].mean()) if len(usable) else np.nan,
                "raw_feasible_rate": float(usable["raw_feasible"].mean()) if len(usable) else np.nan,
                "mean_projection_correction_physical": float(
                    usable["projection_correction_physical"].mean()
                )
                if len(usable)
                else np.nan,
                "median_projection_correction_physical": float(
                    usable["projection_correction_physical"].median()
                )
                if len(usable)
                else np.nan,
                "p90_projection_correction_physical": float(
                    usable["projection_correction_physical"].quantile(0.90)
                )
                if len(usable)
                else np.nan,
                "mean_projection_correction_normalized": float(
                    usable["projection_correction_normalized"].mean()
                )
                if len(usable)
                else np.nan,
                "external_intervention_rate": float(
                    usable["external_intervention"].mean()
                )
                if len(usable)
                else np.nan,
                "mean_raw_max_constraint_violation": float(
                    usable["raw_max_constraint_violation"].mean()
                )
                if len(usable)
                else np.nan,
                "projected_internal_external_max_error": float(
                    usable["internal_external_projection_error"].max()
                )
                if variant_id.startswith("B3") and len(usable)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_paired_difference(
    left: np.ndarray, right: np.ndarray, *, samples: int = 2000
) -> tuple[float, float, float]:
    """Paired mean difference (right - left) and a nonparametric 95% CI."""

    difference = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    difference = difference[np.isfinite(difference)]
    if not difference.size:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20260826)
    estimates = np.empty(int(samples), dtype=float)
    for index in range(int(samples)):
        sample = rng.integers(0, difference.size, size=difference.size)
        estimates[index] = float(np.mean(difference[sample]))
    return (
        float(np.mean(difference)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def b3_pair_effect(actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return B3.1/B3.2 paired effects plus min-h-conditioned summaries."""

    pair = actions.loc[
        actions["variant_id"].isin(PRIMARY_PAIR) & actions["action_set_feasible"].astype(bool)
    ].copy()
    left = pair.loc[pair["variant_id"] == "B3_1"].set_index("state_id")
    right = pair.loc[pair["variant_id"] == "B3_2"].set_index("state_id")
    shared = left.join(right, how="inner", lsuffix="_B3_1", rsuffix="_B3_2")
    if shared.empty:
        raise RuntimeError("B3.1/B3.2 share no feasible probe states")

    metrics = {
        "projection_correction_physical": "external correction (m/s²)",
        "projection_correction_normalized": "normalized external correction",
        "raw_feasible": "raw CBF-feasible rate",
        "external_intervention": "external intervention rate",
        "raw_max_constraint_violation": "raw max constraint violation",
    }
    effect_rows: list[dict[str, Any]] = []
    for metric, label in metrics.items():
        mean_diff, ci_low, ci_high = _bootstrap_paired_difference(
            shared[f"{metric}_B3_1"].to_numpy(dtype=float),
            shared[f"{metric}_B3_2"].to_numpy(dtype=float),
        )
        effect_rows.append(
            {
                "metric": metric,
                "label": label,
                "n_paired_states": int(len(shared)),
                "B3_1_mean": float(shared[f"{metric}_B3_1"].mean()),
                "B3_2_mean": float(shared[f"{metric}_B3_2"].mean()),
                "B3_2_minus_B3_1": mean_diff,
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
            }
        )

    risk = shared.loc[np.isfinite(shared["cbf_min_h_B3_1"])].copy()
    risk_rows: list[dict[str, Any]] = []
    if len(risk) >= 8:
        try:
            bins = pd.qcut(risk["cbf_min_h_B3_1"], q=4, duplicates="drop")
            risk = risk.assign(min_h_bin=bins)
            for index, (_bin, group) in enumerate(risk.groupby("min_h_bin", observed=True)):
                risk_rows.append(
                    {
                        "risk_bin_index": index + 1,
                        "risk_bin": str(_bin),
                        "n_states": int(len(group)),
                        "mean_min_h": float(group["cbf_min_h_B3_1"].mean()),
                        "B3_1_mean_correction": float(
                            group["projection_correction_physical_B3_1"].mean()
                        ),
                        "B3_2_mean_correction": float(
                            group["projection_correction_physical_B3_2"].mean()
                        ),
                        "B3_1_raw_feasible_rate": float(group["raw_feasible_B3_1"].mean()),
                        "B3_2_raw_feasible_rate": float(group["raw_feasible_B3_2"].mean()),
                    }
                )
        except ValueError:
            pass
    return pd.DataFrame(effect_rows), pd.DataFrame(risk_rows)


def plot_training_evidence(
    scalars: pd.DataFrame, gradients: pd.DataFrame, output_path: Path
) -> None:
    """Plot the clean B3.1/B3.2 learning and gradient evidence."""

    panels = (
        ("rollout/ep_rew_mean", "Episode reward mean"),
        ("rollout/ep_len_mean", "Episode length mean"),
        ("rollout/return_per_timestep", "Return per timestep"),
        ("train/cbf_mean_correction", "Mean CBF correction"),
        ("train/cbf_mean_loss", "Raw-to-projected mean loss"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (tag, title) in zip(axes.flat[:5], panels):
        plotted = False
        for variant_id in PRIMARY_PAIR:
            data = _deduplicated_tag(scalars, variant_id, tag)
            if data.empty:
                continue
            smooth = data["value"].rolling(window=3, min_periods=1).mean()
            axis.plot(
                data["step"] / 1000.0,
                smooth,
                color=VARIANT_COLORS[variant_id],
                linewidth=2.0,
                label=VARIANT_LABELS[variant_id],
            )
            plotted = True
        axis.set_title(title)
        axis.set_xlabel("Training timesteps (thousands)")
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "not logged", ha="center", va="center")

    gradient_axis = axes.flat[5]
    b3_gradient = gradients.loc[gradients["variant_id"] == "B3_2"].copy()
    if not b3_gradient.empty:
        x = pd.to_numeric(b3_gradient["num_timesteps"], errors="coerce") / 1000.0
        ratio = pd.to_numeric(b3_gradient["g_cbf_to_g_ppo_ratio"], errors="coerce")
        cosine = pd.to_numeric(b3_gradient["g_ppo_g_cbf_cosine"], errors="coerce")
        line_ratio = gradient_axis.plot(
            x,
            ratio.rolling(3, min_periods=1).mean(),
            color=VARIANT_COLORS["B3_2"],
            linewidth=2.0,
            label="||g_CBF|| / ||g_PPO||",
        )
        gradient_axis.set_ylabel("Gradient-norm ratio")
        cosine_axis = gradient_axis.twinx()
        line_cosine = cosine_axis.plot(
            x,
            cosine.rolling(3, min_periods=1).mean(),
            color="#333333",
            linewidth=1.7,
            linestyle="--",
            label="cos(g_PPO, g_CBF)",
        )
        cosine_axis.set_ylabel("Gradient cosine")
        gradient_axis.legend(line_ratio + line_cosine, [line.get_label() for line in line_ratio + line_cosine], fontsize=8)
    else:
        gradient_axis.text(0.5, 0.5, "not logged", ha="center", va="center")
    gradient_axis.set_title("B3.2 explicit actor-term gradient")
    gradient_axis.set_xlabel("Training timesteps (thousands)")
    gradient_axis.grid(alpha=0.25)
    figure.suptitle(
        "Primary differentiable mean-loss comparison: B3.1 vs B3.2",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_shared_state_alignment(summary: pd.DataFrame, actions: pd.DataFrame, output_path: Path) -> None:
    """Show each method's raw output alignment on exactly the same contexts."""

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    ordered = summary.set_index("variant_id").loc[list(VARIANT_ORDER)].reset_index()
    x = np.arange(len(ordered))
    colors = [VARIANT_COLORS[variant] for variant in ordered["variant_id"]]
    labels = [VARIANT_LABELS[variant] for variant in ordered["variant_id"]]

    axes[0, 0].bar(x, 100.0 * ordered["raw_feasible_rate"], color=colors)
    axes[0, 0].set_title("Raw mean already CBF-feasible")
    axes[0, 0].set_ylabel("Feasible states (%)")
    axes[0, 0].set_ylim(0.0, 105.0)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(x, ordered["mean_projection_correction_physical"], color=colors)
    axes[0, 1].set_title("External CBF correction required by raw mean")
    axes[0, 1].set_ylabel("Mean correction (m/s²)")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(x, 100.0 * ordered["external_intervention_rate"], color=colors)
    axes[1, 0].set_title("External intervention rate (threshold = 0.03 normalized)")
    axes[1, 0].set_ylabel("Feasible states (%)")
    axes[1, 0].set_ylim(0.0, 105.0)
    axes[1, 0].grid(axis="y", alpha=0.25)

    usable = actions.loc[actions["action_set_feasible"].astype(bool)].copy()
    samples = [
        usable.loc[usable["variant_id"] == variant, "projection_correction_physical"].to_numpy()
        for variant in VARIANT_ORDER
    ]
    box = axes[1, 1].boxplot(samples, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1, 1].set_title("Distribution of external corrections")
    axes[1, 1].set_ylabel("Correction (m/s²)")
    axes[1, 1].grid(axis="y", alpha=0.25)

    for axis in axes.flat:
        if axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
            axis.set_xticks(x, labels, rotation=28, ha="right", fontsize=8)
        else:
            axis.set_xticks(x + 1, labels, rotation=28, ha="right", fontsize=8)
    figure.suptitle("Shared-state raw policy alignment to one external CBF", fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_b3_pair_probe(
    actions: pd.DataFrame, risk_summary: pd.DataFrame, output_path: Path
) -> None:
    """Visualize the paired B3.1/B3.2 raw-policy effect state by state."""

    data = actions.loc[
        actions["variant_id"].isin(PRIMARY_PAIR) & actions["action_set_feasible"].astype(bool)
    ]
    left = data.loc[data["variant_id"] == "B3_1"].set_index("state_id")
    right = data.loc[data["variant_id"] == "B3_2"].set_index("state_id")
    shared = left.join(right, how="inner", lsuffix="_B3_1", rsuffix="_B3_2")

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.7), constrained_layout=True)
    x = shared["projection_correction_physical_B3_1"].to_numpy(dtype=float)
    y = shared["projection_correction_physical_B3_2"].to_numpy(dtype=float)
    limit = max(float(np.nanmax(x)), float(np.nanmax(y)), 1e-3) * 1.04
    axes[0].scatter(x, y, s=13, alpha=0.35, color=VARIANT_COLORS["B3_2"], edgecolor="none")
    axes[0].plot([0.0, limit], [0.0, limit], color="#333333", linestyle="--", linewidth=1.3)
    axes[0].set_xlim(0.0, limit)
    axes[0].set_ylim(0.0, limit)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("B3.1 external correction (m/s²)")
    axes[0].set_ylabel("B3.2 external correction (m/s²)")
    axes[0].set_title("One dot = one identical state\nBelow diagonal favors B3.2")
    axes[0].grid(alpha=0.25)

    if not risk_summary.empty:
        risk_x = risk_summary["risk_bin_index"].to_numpy(dtype=int)
        axes[1].plot(
            risk_x,
            risk_summary["B3_1_mean_correction"],
            marker="o",
            linewidth=2.0,
            color=VARIANT_COLORS["B3_1"],
            label="B3.1",
        )
        axes[1].plot(
            risk_x,
            risk_summary["B3_2_mean_correction"],
            marker="o",
            linewidth=2.0,
            color=VARIANT_COLORS["B3_2"],
            label="B3.2",
        )
        axes[1].set_xticks(risk_x, [f"Q{value}\nlow → high h" for value in risk_x])
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "Insufficient min-h variation", ha="center", va="center")
    axes[1].set_title("Raw-mean correction by CBF min-h quartile")
    axes[1].set_xlabel("State safety-margin bin")
    axes[1].set_ylabel("Mean external correction (m/s²)")
    axes[1].grid(alpha=0.25)
    figure.suptitle("Effect of the B3.2 differentiable mean-alignment term", fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _markdown_number(value: Any, digits: int = 3) -> str:
    numeric = _finite(value)
    return "N/A" if not np.isfinite(numeric) else f"{numeric:.{digits}f}"


def write_readme(
    output_path: Path,
    *,
    summary: pd.DataFrame,
    pair_effect: pd.DataFrame,
    state_count: int,
    feasible_state_count: int,
) -> None:
    """Write the interpretation guardrails next to the generated analysis."""

    primary = summary.set_index("variant_id")
    b31 = primary.loc["B3_1"]
    b32 = primary.loc["B3_2"]
    correction = pair_effect.loc[
        pair_effect["metric"] == "projection_correction_physical"
    ].iloc[0]
    feasible = pair_effect.loc[pair_effect["metric"] == "raw_feasible"].iloc[0]
    text = f"""# Learned-policy analysis

This folder analyzes the seven saved 50k policies without rerunning the prior evaluation. The probe bank contains {state_count} pre-action states, of which {feasible_state_count} have a feasible no-slack CBF action set. Each saved actor is queried on the exact same states; its **raw neural mean** is then projected by the same external CBF for comparison.

## What is cleanly isolated

The primary causal comparison is **B3.1 → B3.2**. Both have the same projected-policy architecture, CBF reward feedback, CBF parameters, environment, optimizer configuration, seed (307), and 50k training budget. B3.2 alone enables the explicit differentiable mean-alignment term `lambda_mean * ||mu_raw - P_s(mu_raw)||²`, with `lambda_mean=0.10` (B3.1: 0.00).

B2.1 → B2.2 is a separate clean comparison for the **stopped-gradient hard-projection target**, not for differentiating through the projection. B2.3 ↔ B3.3 changes both gradient treatment and policy architecture, so it is descriptive rather than a one-variable causal ablation.

## Shared-state B3 result

- B3.1 raw-feasible rate: {_markdown_number(100.0 * b31['raw_feasible_rate'], 1)}%; B3.2: {_markdown_number(100.0 * b32['raw_feasible_rate'], 1)}%.
- B3.1 mean external correction: {_markdown_number(b31['mean_projection_correction_physical'])} m/s²; B3.2: {_markdown_number(b32['mean_projection_correction_physical'])} m/s².
- Paired B3.2 − B3.1 correction: {_markdown_number(correction['B3_2_minus_B3_1'])} m/s² (bootstrap 95% CI {_markdown_number(correction['bootstrap_95_ci_low'])} to {_markdown_number(correction['bootstrap_95_ci_high'])}). Negative favors B3.2 internalizing the CBF.
- Paired B3.2 − B3.1 raw-feasible rate: {_markdown_number(100.0 * feasible['B3_2_minus_B3_1'], 1)} percentage points.

## Interpretation limits

- This is one training seed, so it is evidence about these saved runs—not a seed-level statistical claim.
- The state bank uses CBF-safe resets and fixed scripted commands with CBF action filtering disabled. It is a common **input probe**, not a trajectory-performance or collision evaluation.
- B1 uses the older 30D observation without previous action; its shared-state results are useful descriptively, but should not be treated as a perfectly matched architecture comparison.
- Raw-mean alignment is the right signal for whether the actor itself has internalized the CBF. It is intentionally distinct from safety delivered by an external CBF at execution.

## Files

- `ablation_map.csv`: exact method/gradient-path map from saved configurations.
- `training_effect_summary.csv`: final and tail training scalars plus logged actor-gradient statistics.
- `probe_states.csv` / `probe_actions.csv`: reproducible state/action-level evidence.
- `probe_alignment_summary.csv`: all-method shared-state summary.
- `b3_1_vs_b3_2_paired_effect.csv`: primary paired effect and bootstrap intervals.
- `b3_risk_conditioned_summary.csv`: B3 comparison across min-h quartiles.
- `b3_training_gradient_evidence.png`, `shared_state_action_alignment.png`, and `b3_differentiable_mean_probe.png`: figures.
"""
    output_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "final_Results" / "policy_analysis",
    )
    parser.add_argument("--seeds", type=int, default=2, help="Initial seeds per scripted probe controller")
    parser.add_argument("--steps-per-seed", type=int, default=15, help="Pre-action states collected per probe rollout")
    parser.add_argument("--seed-start", type=int, default=1_300_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[policy-analysis] output: {output_dir}", flush=True)

    specs = load_specs(PROJECT_ROOT)
    ablations = ablation_map(specs)
    ablations.to_csv(output_dir / "ablation_map.csv", index=False)

    tensorboard_dir = PROJECT_ROOT / "artifacts" / "final_Results" / "tensorboard"
    scalars = pd.read_csv(tensorboard_dir / "scalars_long.csv")
    gradients = pd.read_csv(tensorboard_dir / "cbf_training_gradients.csv")
    train_summary = training_summary(scalars, gradients, specs)
    train_summary.to_csv(output_dir / "training_effect_summary.csv", index=False)
    plot_training_evidence(scalars, gradients, output_dir / "b3_training_gradient_evidence.png")

    print("[policy-analysis] constructing common state bank", flush=True)
    env, probe_env_config = build_probe_environment(PROJECT_ROOT, specs)
    try:
        states = collect_probe_states(
            env,
            seeds=int(args.seeds),
            steps_per_seed=int(args.steps_per_seed),
            seed_start=int(args.seed_start),
        )
    finally:
        env.close()

    states_for_csv = states.drop(columns=["observation"])
    states_for_csv.to_csv(output_dir / "probe_states.csv", index=False)
    np.savez_compressed(
        output_dir / "probe_observations.npz",
        state_id=states["state_id"].to_numpy(dtype=np.int64),
        observation=np.stack(states["observation"].to_numpy()).astype(np.float32),
    )

    print("[policy-analysis] loading checkpoints and querying raw actor means", flush=True)
    models = load_models(specs)
    actions = probe_actions(states, models)
    actions.to_csv(output_dir / "probe_actions.csv", index=False)
    probe_summary = summarize_probe(actions)
    probe_summary.to_csv(output_dir / "probe_alignment_summary.csv", index=False)
    pair_effect, risk_summary = b3_pair_effect(actions)
    pair_effect.to_csv(output_dir / "b3_1_vs_b3_2_paired_effect.csv", index=False)
    risk_summary.to_csv(output_dir / "b3_risk_conditioned_summary.csv", index=False)
    plot_shared_state_alignment(
        probe_summary, actions, output_dir / "shared_state_action_alignment.png"
    )
    plot_b3_pair_probe(
        actions, risk_summary, output_dir / "b3_differentiable_mean_probe.png"
    )

    metadata = {
        "analysis": "shared-state raw-actor CBF-alignment probe",
        "created_from": "artifacts/final_Results saved 50k checkpoints and TensorBoard archive",
        "state_bank": {
            "initial_seeds_per_controller": int(args.seeds),
            "steps_per_seed": int(args.steps_per_seed),
            "seed_start": int(args.seed_start),
            "controllers": ["coast", "accelerate", "brake", "lateral_sweep"],
            "cbf_safe_resets": True,
            "policy_rate_cbf_filtering": False,
            "physics_rate_cbf_filtering": False,
            "ordinary_traffic_guard": "preserved from B3.2 config",
        },
        "primary_pair": {
            "control": "B3_1",
            "treatment": "B3_2",
            "change": "lambda_mean 0.00 -> 0.10",
        },
        "probe_states": int(len(states)),
        "feasible_action_set_states": int(states["action_set_feasible"].sum()),
        "probe_env_config": probe_env_config,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    write_readme(
        output_dir / "README.md",
        summary=probe_summary,
        pair_effect=pair_effect,
        state_count=int(len(states)),
        feasible_state_count=int(states["action_set_feasible"].sum()),
    )

    b3_effect = pair_effect.loc[
        pair_effect["metric"] == "projection_correction_physical"
    ].iloc[0]
    print(
        "[policy-analysis] complete "
        f"states={len(states)} feasible={int(states['action_set_feasible'].sum())} "
        f"B3.2-B3.1 correction={float(b3_effect['B3_2_minus_B3_1']):.4f} m/s^2",
        flush=True,
    )


if __name__ == "__main__":
    main()
