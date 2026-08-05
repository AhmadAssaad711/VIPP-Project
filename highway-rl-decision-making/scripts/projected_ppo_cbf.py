"""PPO with a differentiable CBF-projected policy mean.

Data semantics are intentionally explicit:

* the policy distribution is ``Normal(mu_safe, sigma)``;
* the rollout buffer action is the latent Gaussian sample ``z``;
* its stored log probability is ``log pi(z | s)``;
* the simulator receives the separate hard projection ``P_s(z)``.

The constraint context is appended to observations by :mod:`ppo_cbf_env`.
Only the original state features enter the learned actor/value networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from cbf_projection import (
    CBFContextLayout,
    TorchProjection2D,
    project_polytope_2d_numpy,
    project_polytope_2d_torch,
    split_cbf_context_numpy,
    split_cbf_context_torch,
)
from ppo_cbf_env import constraint_system_hash


class CBFBaseFeaturesExtractor(BaseFeaturesExtractor):
    """Expose only the original state to actor and value networks."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        base_observation_dim: int = 42,
    ) -> None:
        self.base_observation_dim = int(base_observation_dim)
        if int(np.prod(observation_space.shape)) < self.base_observation_dim:
            raise ValueError("Observation is narrower than the requested base state")
        super().__init__(observation_space, features_dim=self.base_observation_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations[..., : self.base_observation_dim]


@dataclass(frozen=True)
class ProjectedPolicyEvaluation:
    values: th.Tensor
    log_prob: th.Tensor
    entropy: Optional[th.Tensor]
    distribution: DiagGaussianDistribution
    mu_raw: th.Tensor
    mu_safe: th.Tensor
    projection: TorchProjection2D


class ProjectedCBFActorCriticPolicy(ActorCriticPolicy):
    """Diagonal-Gaussian policy whose mean is an exact CBF-QP projection."""

    def __init__(
        self,
        *args,
        cbf_base_observation_dim: int = 42,
        cbf_max_constraints: int = 18,
        cbf_feasibility_tol: float = 1e-6,
        **kwargs,
    ) -> None:
        if kwargs.get("use_sde", False):
            raise ValueError("ProjectedCBFActorCriticPolicy does not support gSDE")
        extractor_class = kwargs.pop(
            "features_extractor_class", CBFBaseFeaturesExtractor
        )
        if extractor_class is not CBFBaseFeaturesExtractor:
            raise ValueError(
                "Projected CBF policy requires CBFBaseFeaturesExtractor so context "
                "cannot leak into the learned state representation"
            )
        extractor_kwargs = dict(kwargs.pop("features_extractor_kwargs", {}) or {})
        extractor_kwargs["base_observation_dim"] = int(cbf_base_observation_dim)
        kwargs["features_extractor_class"] = CBFBaseFeaturesExtractor
        kwargs["features_extractor_kwargs"] = extractor_kwargs
        self.cbf_layout = CBFContextLayout(
            base_observation_dim=int(cbf_base_observation_dim),
            max_constraints=int(cbf_max_constraints),
        )
        self.cbf_feasibility_tol = float(cbf_feasibility_tol)
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise TypeError("Projected CBF PPO requires a diagonal Gaussian action distribution")
        if not isinstance(self.action_space, spaces.Box) or tuple(self.action_space.shape) != (2,):
            raise TypeError("Projected CBF PPO requires a two-dimensional Box action space")

    def _latents(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi_features, vf_features = features
        return (
            self.mlp_extractor.forward_actor(pi_features),
            self.mlp_extractor.forward_critic(vf_features),
        )

    def project_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> TorchProjection2D:
        _, rows, bounds, mask = split_cbf_context_torch(
            obs, layout=self.cbf_layout
        )
        return project_polytope_2d_torch(
            actions,
            rows,
            bounds,
            mask,
            feasibility_tol=self.cbf_feasibility_tol,
            action_low=th.as_tensor(
                self.action_space.low, dtype=obs.dtype, device=obs.device
            ),
            action_high=th.as_tensor(
                self.action_space.high, dtype=obs.dtype, device=obs.device
            ),
        )

    def _distribution_and_stages(
        self, obs: th.Tensor
    ) -> tuple[
        DiagGaussianDistribution,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        TorchProjection2D,
    ]:
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)
        mu_raw = self.action_net(latent_pi)
        projection = self.project_actions(obs, mu_raw)
        # No mathematical projection exists when the no-slack set is empty.
        # Use the shared labelled fallback for behavior, but do not claim or
        # propagate an optimization-layer Jacobian through that fallback.
        mu_safe = th.where(
            projection.feasible.unsqueeze(1),
            projection.action,
            projection.action.detach(),
        )
        distribution = self.action_dist.proba_distribution(mu_safe, self.log_std)
        assert isinstance(distribution, DiagGaussianDistribution)
        return distribution, values, mu_raw, mu_safe, projection

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        distribution, values, _, _, _ = self._distribution_and_stages(obs)
        latent_z = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(latent_z)
        latent_z = latent_z.reshape((-1, *self.action_space.shape))
        return latent_z, values, log_prob

    def evaluate_actions_with_projection(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> ProjectedPolicyEvaluation:
        distribution, values, mu_raw, mu_safe, projection = (
            self._distribution_and_stages(obs)
        )
        return ProjectedPolicyEvaluation(
            values=values,
            log_prob=distribution.log_prob(actions),
            entropy=distribution.entropy(),
            distribution=distribution,
            mu_raw=mu_raw,
            mu_safe=mu_safe,
            projection=projection,
        )

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        result = self.evaluate_actions_with_projection(obs, actions)
        return result.values, result.log_prob, result.entropy

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        distribution, _, _, _, _ = self._distribution_and_stages(obs)
        return distribution

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

    def action_stages(
        self,
        obs: th.Tensor,
        *,
        deterministic: bool = True,
    ) -> dict[str, th.Tensor]:
        """Return raw mean, safe mean, latent z, and final hard projection."""

        distribution, _, mu_raw, mu_safe, mean_projection = (
            self._distribution_and_stages(obs)
        )
        latent_z = distribution.get_actions(deterministic=deterministic)
        executed_projection = self.project_actions(obs, latent_z)
        return {
            "mu_raw": mu_raw,
            "mu_safe": mu_safe,
            "latent_z": latent_z,
            "executed_action": executed_projection.action,
            "mean_feasible": mean_projection.feasible,
            "sample_feasible": executed_projection.feasible,
        }


def context_ignoring_policy_kwargs(
    *,
    base_observation_dim: int = 42,
    policy_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Attach the no-context-leak feature extractor to a standard PPO policy."""

    result = dict(policy_kwargs or {})
    result["features_extractor_class"] = CBFBaseFeaturesExtractor
    extractor_kwargs = dict(result.get("features_extractor_kwargs", {}) or {})
    extractor_kwargs["base_observation_dim"] = int(base_observation_dim)
    result["features_extractor_kwargs"] = extractor_kwargs
    return result


class LatentActionPPO(PPO):
    """PPO collector that stores ``z`` but executes a separate hard action."""

    def __init__(
        self,
        *args,
        execution_mode: str = "box",
        cbf_base_observation_dim: int = 42,
        cbf_max_constraints: int = 18,
        cbf_feasibility_tol: float = 1e-6,
        **kwargs,
    ) -> None:
        self.execution_mode = str(execution_mode).strip().lower()
        if self.execution_mode not in {"box", "cbf"}:
            raise ValueError("execution_mode must be 'box' or 'cbf'")
        self.cbf_layout = CBFContextLayout(
            base_observation_dim=int(cbf_base_observation_dim),
            max_constraints=int(cbf_max_constraints),
        )
        self.cbf_feasibility_tol = float(cbf_feasibility_tol)
        super().__init__(*args, **kwargs)

    def _execution_actions(
        self, latent_actions: np.ndarray, observations: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        action_dim = int(low.size)
        latent_array = np.asarray(latent_actions, dtype=np.float32)
        if latent_array.size == 0 or latent_array.size % action_dim != 0:
            raise ValueError(
                f"latent actions contain {latent_array.size} values, which cannot "
                f"form actions of width {action_dim}"
            )
        latent_actions = latent_array.reshape((-1, action_dim))
        observation_batch = np.asarray(observations, dtype=np.float32)
        if observation_batch.ndim == 1:
            observation_batch = observation_batch.reshape(1, -1)
        elif observation_batch.ndim < 1:
            raise ValueError("observations must include an observation dimension")
        else:
            observation_batch = observation_batch.reshape(
                (-1, observation_batch.shape[-1])
            )
        batch_size = int(latent_actions.shape[0])
        if int(observation_batch.shape[0]) != batch_size:
            raise ValueError(
                "latent-action and observation batch sizes differ "
                f"({batch_size} != {observation_batch.shape[0]})"
            )
        _, rows, bounds, mask = split_cbf_context_numpy(
            observation_batch, layout=self.cbf_layout
        )
        executed = np.empty_like(latent_actions)
        records: list[dict[str, Any]] = []
        for env_index in range(batch_size):
            raw = latent_actions[env_index]
            active = np.asarray(mask[env_index] > 0.5, dtype=bool)
            active_rows = np.asarray(rows[env_index][active], dtype=np.float32)
            active_bounds = np.asarray(bounds[env_index][active], dtype=np.float32)
            context_hash = constraint_system_hash(active_rows, active_bounds)
            if self.execution_mode == "cbf":
                projection = project_polytope_2d_numpy(
                    raw,
                    rows[env_index],
                    bounds[env_index],
                    mask[env_index],
                    feasibility_tol=self.cbf_feasibility_tol,
                    action_low=low,
                    action_high=high,
                )
                safe = projection.action
                record = {
                    "feasible": projection.feasible,
                    "fallback_used": projection.fallback_used,
                    "projection_source": projection.source,
                    "max_constraint_violation_safe": projection.max_violation,
                    "active_indices": projection.active_indices,
                    "constraint_hash": context_hash,
                    "cbf_applied": True,
                }
            else:
                safe = np.clip(raw, low, high).astype(np.float32)
                record = {
                    "feasible": True,
                    "fallback_used": False,
                    "projection_source": "box",
                    "max_constraint_violation_safe": 0.0,
                    "active_indices": np.zeros(0, dtype=np.int64),
                    "constraint_hash": context_hash,
                    "cbf_applied": False,
                }
            executed[env_index] = safe
            records.append(record)
        return executed, records

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect latent actions while the environment executes ``P_s(z)``."""

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                latent_actions_tensor, values, log_probs = self.policy(obs_tensor)
            latent_actions = latent_actions_tensor.cpu().numpy()
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
                    max_constraint_violation_safe=float(
                        record["max_constraint_violation_safe"]
                    ),
                    active_indices=record["active_indices"],
                    constraint_hash=str(record["constraint_hash"]),
                    cbf_applied=bool(record["cbf_applied"]),
                    indices=env_index,
                )

            # The callback sees both quantities.  Crucially, RolloutBuffer.add
            # below receives latent_actions, never executed_actions.
            actions = latent_actions
            clipped_actions = executed_actions
            new_obs, rewards, dones, infos = env.step(executed_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                latent_actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device)
            )
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def predict_action_stages(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> dict[str, np.ndarray]:
        """Diagnostic API that never confuses latent and executed actions."""

        obs_tensor, vectorized = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            if isinstance(self.policy, ProjectedCBFActorCriticPolicy):
                stages = self.policy.action_stages(
                    obs_tensor, deterministic=deterministic
                )
                result = {
                    key: value.detach().cpu().numpy() for key, value in stages.items()
                }
            else:
                distribution = self.policy.get_distribution(obs_tensor)
                latent = distribution.get_actions(deterministic=deterministic)
                latent_np = latent.detach().cpu().numpy()
                executed, _ = self._execution_actions(
                    latent_np, obs_tensor.detach().cpu().numpy()
                )
                result = {
                    "mu_raw": distribution.distribution.mean.detach().cpu().numpy(),
                    "mu_safe": distribution.distribution.mean.detach().cpu().numpy(),
                    "latent_z": latent_np,
                    "executed_action": executed,
                }
        if not vectorized:
            result = {key: np.asarray(value).squeeze(axis=0) for key, value in result.items()}
        return result


def _gradient_pair_diagnostics(
    parameters: list[th.nn.Parameter],
    primary_loss: th.Tensor,
    auxiliary_loss: th.Tensor,
) -> dict[str, float]:
    """Measure PPO/CBF actor gradient norms without mutating ``.grad``."""

    primary = th.autograd.grad(
        primary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    auxiliary = th.autograd.grad(
        auxiliary_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    primary_norm_sq = th.zeros((), device=primary_loss.device)
    auxiliary_norm_sq = th.zeros((), device=primary_loss.device)
    dot = th.zeros((), device=primary_loss.device)
    for primary_grad, auxiliary_grad in zip(primary, auxiliary):
        if primary_grad is not None:
            primary_norm_sq = primary_norm_sq + primary_grad.detach().square().sum()
        if auxiliary_grad is not None:
            auxiliary_norm_sq = auxiliary_norm_sq + auxiliary_grad.detach().square().sum()
        if primary_grad is not None and auxiliary_grad is not None:
            dot = dot + (primary_grad.detach() * auxiliary_grad.detach()).sum()
    primary_norm = th.sqrt(primary_norm_sq)
    auxiliary_norm = th.sqrt(auxiliary_norm_sq)
    denominator = primary_norm * auxiliary_norm
    cosine = dot / denominator if float(denominator) > 1e-12 else th.full_like(dot, th.nan)
    ratio = auxiliary_norm / primary_norm.clamp_min(1e-12)
    return {
        "g_ppo_norm": float(primary_norm.cpu().item()),
        "g_cbf_norm": float(auxiliary_norm.cpu().item()),
        "g_cbf_to_g_ppo_ratio": float(ratio.cpu().item()),
        "g_ppo_g_cbf_cosine": float(cosine.cpu().item()),
    }


class ProjectedCBFPPO(LatentActionPPO):
    """PPO surrogate plus mean/sample CBF internalization losses."""

    policy_aliases = {
        **PPO.policy_aliases,
        "ProjectedCBFPolicy": ProjectedCBFActorCriticPolicy,
    }

    def __init__(
        self,
        *args,
        lambda_mean: float = 0.10,
        lambda_sample: float = 0.0,
        **kwargs,
    ) -> None:
        self.lambda_mean = float(lambda_mean)
        self.lambda_sample = float(lambda_sample)
        self.cbf_training_diagnostics: list[dict[str, float]] = []
        kwargs.setdefault("execution_mode", "cbf")
        super().__init__(*args, **kwargs)
        # SB3 load() constructs once with _init_setup_model=False and restores
        # the policy immediately afterward.
        if hasattr(self, "policy") and not isinstance(
            self.policy, ProjectedCBFActorCriticPolicy
        ):
            raise TypeError(
                "ProjectedCBFPPO must be constructed with ProjectedCBFActorCriticPolicy"
            )

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        mean_losses: list[float] = []
        sample_losses: list[float] = []
        mean_corrections: list[float] = []
        sample_corrections: list[float] = []
        mean_infeasible_rates: list[float] = []
        sample_infeasible_rates: list[float] = []
        gradient_rows: list[dict[str, float]] = []
        all_approx_kl_divs: list[float] = []
        continue_training = True
        last_loss = th.zeros((), device=self.device)

        actor_parameters = [
            parameter
            for name, parameter in self.policy.named_parameters()
            if parameter.requires_grad and not name.startswith("value_net")
        ]

        for epoch in range(self.n_epochs):
            epoch_approx_kl_divs: list[float] = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                evaluation = self.policy.evaluate_actions_with_projection(
                    rollout_data.observations, actions
                )
                values = evaluation.values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                ratio = th.exp(evaluation.log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(float(policy_loss.detach().cpu().item()))
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(float(clip_fraction))

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(float(value_loss.detach().cpu().item()))

                if evaluation.entropy is None:
                    entropy_loss = -th.mean(-evaluation.log_prob)
                else:
                    entropy_loss = -th.mean(evaluation.entropy)
                entropy_losses.append(float(entropy_loss.detach().cpu().item()))

                mean_target = th.where(
                    evaluation.projection.feasible.unsqueeze(1),
                    evaluation.mu_safe,
                    evaluation.mu_safe.detach(),
                )
                mean_delta = evaluation.mu_raw - mean_target
                mean_feasible = evaluation.projection.feasible.to(mean_delta.dtype)
                mean_denominator = mean_feasible.sum().clamp_min(1.0)
                mean_loss = (
                    mean_delta.square().sum(dim=1) * mean_feasible
                ).sum() / mean_denominator
                mean_losses.append(float(mean_loss.detach().cpu().item()))
                mean_corrections.append(
                    float(
                        (
                            th.linalg.vector_norm(mean_delta.detach(), dim=1)
                            * mean_feasible
                        ).sum()
                        .div(mean_denominator)
                        .cpu()
                        .item()
                    )
                )
                mean_infeasible_rates.append(
                    float((~evaluation.projection.feasible).float().mean().cpu().item())
                )

                sample_loss = th.zeros((), device=self.device)
                if self.lambda_sample != 0.0:
                    # A stored rollout z is constant during this update.  Use a
                    # fresh reparameterized current-policy sample so this term
                    # can train both the mean and log standard deviation.
                    fresh_z = evaluation.distribution.distribution.rsample()
                    fresh_projection = self.policy.project_actions(
                        rollout_data.observations, fresh_z
                    )
                    fresh_target = th.where(
                        fresh_projection.feasible.unsqueeze(1),
                        fresh_projection.action,
                        fresh_projection.action.detach(),
                    )
                    sample_delta = fresh_z - fresh_target
                    sample_feasible = fresh_projection.feasible.to(
                        sample_delta.dtype
                    )
                    sample_denominator = sample_feasible.sum().clamp_min(1.0)
                    sample_loss = (
                        sample_delta.square().sum(dim=1) * sample_feasible
                    ).sum() / sample_denominator
                    sample_corrections.append(
                        float(
                            (
                                th.linalg.vector_norm(sample_delta.detach(), dim=1)
                                * sample_feasible
                            ).sum()
                            .div(sample_denominator)
                            .cpu()
                            .item()
                        )
                    )
                    sample_infeasible_rates.append(
                        float((~fresh_projection.feasible).float().mean().cpu().item())
                    )
                sample_losses.append(float(sample_loss.detach().cpu().item()))

                auxiliary_loss = (
                    self.lambda_mean * mean_loss
                    + self.lambda_sample * sample_loss
                )
                if self.lambda_mean != 0.0 or self.lambda_sample != 0.0:
                    actor_primary_loss = (
                        policy_loss + self.ent_coef * entropy_loss
                    )
                    gradient_rows.append(
                        _gradient_pair_diagnostics(
                            actor_parameters, actor_primary_loss, auxiliary_loss
                        )
                    )
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + auxiliary_loss
                )
                last_loss = loss

                with th.no_grad():
                    log_ratio = evaluation.log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).cpu().item()
                    epoch_approx_kl_divs.append(float(approx_kl_div))
                    all_approx_kl_divs.append(float(approx_kl_div))
                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching "
                            f"max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(all_approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", float(last_loss.detach().cpu().item()))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/cbf_lambda_mean", self.lambda_mean)
        self.logger.record("train/cbf_lambda_sample", self.lambda_sample)
        self.logger.record("train/cbf_mean_loss", np.mean(mean_losses))
        self.logger.record("train/cbf_sample_loss", np.mean(sample_losses))
        self.logger.record("train/cbf_mean_correction", np.mean(mean_corrections))
        self.logger.record("train/cbf_mean_infeasible_rate", np.mean(mean_infeasible_rates))
        self.logger.record(
            "train/cbf_sample_correction",
            np.mean(sample_corrections) if sample_corrections else 0.0,
        )
        self.logger.record(
            "train/cbf_sample_infeasible_rate",
            np.mean(sample_infeasible_rates) if sample_infeasible_rates else 0.0,
        )
        gradient_summary: dict[str, float] = {}
        if gradient_rows:
            for key in gradient_rows[0]:
                finite = [row[key] for row in gradient_rows if np.isfinite(row[key])]
                gradient_summary[key] = float(np.mean(finite)) if finite else np.nan
                self.logger.record(
                    f"train/actor_{key}", gradient_summary[key]
                )
        self.cbf_training_diagnostics.append(
            {
                "n_updates": float(self._n_updates),
                "num_timesteps": float(self.num_timesteps),
                "mean_loss": float(np.mean(mean_losses)),
                "sample_loss": float(np.mean(sample_losses)),
                "mean_correction": float(np.mean(mean_corrections)),
                "mean_infeasible_rate": float(np.mean(mean_infeasible_rates)),
                "sample_correction": float(
                    np.mean(sample_corrections) if sample_corrections else 0.0
                ),
                "sample_infeasible_rate": float(
                    np.mean(sample_infeasible_rates)
                    if sample_infeasible_rates
                    else 0.0
                ),
                **gradient_summary,
            }
        )
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item()
            )
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
