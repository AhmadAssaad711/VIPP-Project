from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cbf_projection import CBFContextLayout, append_cbf_context  # noqa: E402
from projected_ppo_cbf import (  # noqa: E402
    CBFSafetyRolloutBuffer,
    LatentActionPPO,
    ProjectedCBFActorCriticPolicy,
    ProjectedCBFPPO,
    context_ignoring_policy_kwargs,
)


def _augmented_observation() -> np.ndarray:
    # x <= 0 plus the common physical [-3, 3]^2 action box.
    rows = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    bounds = np.asarray([0.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32)
    return append_cbf_context(np.zeros(42, dtype=np.float32), rows, bounds)


def _spaces():
    layout = CBFContextLayout()
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(layout.observation_dim,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(
        low=np.full(2, -3.0, dtype=np.float32),
        high=np.full(2, 3.0, dtype=np.float32),
        dtype=np.float32,
    )
    return observation_space, action_space


def test_projected_policy_log_probability_is_for_latent_z_not_execution():
    observation_space, action_space = _spaces()
    policy = ProjectedCBFActorCriticPolicy(
        observation_space,
        action_space,
        lr_schedule=lambda _: 0.0,
        net_arch={"pi": [8], "vf": [8]},
        ortho_init=False,
        log_std_init=-1.0,
    )
    with th.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.copy_(th.tensor([2.0, 0.25]))
    obs = th.tensor(_augmented_observation()[None], dtype=th.float32)
    evaluation_mean = policy.action_stages(obs, deterministic=True)
    np.testing.assert_allclose(
        evaluation_mean["mu_raw"].detach().numpy()[0], [2.0, 0.25], atol=1e-6
    )
    np.testing.assert_allclose(
        evaluation_mean["mu_safe"].detach().numpy()[0], [0.0, 0.25], atol=1e-6
    )

    latent_z = th.tensor([[0.4, -0.2]], dtype=th.float32)
    evaluation = policy.evaluate_actions_with_projection(obs, latent_z)
    executed = policy.project_actions(obs, latent_z).action
    expected_latent_log_prob = evaluation.distribution.log_prob(latent_z)
    executed_log_prob = evaluation.distribution.log_prob(executed)
    th.testing.assert_close(evaluation.log_prob, expected_latent_log_prob)
    assert not th.allclose(evaluation.log_prob, executed_log_prob)


def test_mean_internalization_has_outward_normal_gradient():
    observation_space, action_space = _spaces()
    policy = ProjectedCBFActorCriticPolicy(
        observation_space,
        action_space,
        lr_schedule=lambda _: 0.0,
        net_arch={"pi": [8], "vf": [8]},
        ortho_init=False,
    )
    with th.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.copy_(th.tensor([2.0, 0.0]))
    obs = th.tensor(_augmented_observation()[None], dtype=th.float32)
    stages = policy.action_stages(obs, deterministic=True)
    loss = (stages["mu_raw"] - stages["mu_safe"]).square().sum()
    gradient = th.autograd.grad(loss, policy.action_net.bias)[0]
    assert float(gradient[0]) > 0.0
    assert abs(float(gradient[1])) < 1e-6


def test_deterministic_safe_mean_needs_no_second_projection():
    observation_space, action_space = _spaces()
    policy = ProjectedCBFActorCriticPolicy(
        observation_space,
        action_space,
        lr_schedule=lambda _: 0.0,
        net_arch={"pi": [8], "vf": [8]},
        ortho_init=False,
    )
    with th.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.copy_(th.tensor([2.0, 0.25]))
    obs = th.tensor(_augmented_observation()[None], dtype=th.float32)
    stages = policy.action_stages(obs, deterministic=True)
    th.testing.assert_close(stages["latent_z"], stages["mu_safe"])
    th.testing.assert_close(stages["executed_action"], stages["mu_safe"])
    assert bool(stages["mean_feasible"].item())
    assert bool(stages["sample_feasible"].item())


def test_safety_critic_head_and_rollout_targets_are_nonnegative():
    observation_space, action_space = _spaces()
    policy = ProjectedCBFActorCriticPolicy(
        observation_space,
        action_space,
        lr_schedule=lambda _: 1e-3,
        net_arch={"pi": [8], "vf": [8]},
        ortho_init=False,
    )
    obs = th.tensor(_augmented_observation()[None], dtype=th.float32)
    safety_value = policy.predict_safety_values(obs)
    assert bool((safety_value >= 0.0).all())
    optimizer_parameters = {
        id(parameter)
        for group in policy.optimizer.param_groups
        for parameter in group["params"]
    }
    assert id(policy.safety_value_net.weight) in optimizer_parameters

    buffer = CBFSafetyRolloutBuffer(
        2,
        observation_space,
        action_space,
        device="cpu",
        gamma=0.99,
        gae_lambda=0.95,
        n_envs=1,
        safety_gamma=0.9,
        safety_cost_clip=1.0,
    )
    for cost in (0.25, 0.5):
        buffer.add(
            np.zeros((1, observation_space.shape[0]), dtype=np.float32),
            np.zeros((1, action_space.shape[0]), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=bool),
            th.zeros(1),
            th.zeros(1),
            safety_costs=np.asarray([cost], dtype=np.float32),
            safety_fallbacks=np.zeros(1, dtype=np.float32),
        )
    buffer.compute_safety_returns(
        last_safety_values=th.tensor([0.0]), dones=np.asarray([True])
    )
    np.testing.assert_allclose(
        buffer.safety_returns.reshape(-1), [0.7, 0.5], atol=1e-6
    )


class _ProjectionRecordEnv(gym.Env):
    metadata = {}

    def __init__(self):
        observation_space, action_space = _spaces()
        self.observation_space = observation_space
        self.action_space = action_space
        self.observation = _augmented_observation()
        self.pending = None
        self.raw_actions: list[np.ndarray] = []
        self.executed_actions: list[np.ndarray] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation.copy(), {}

    def set_projection_record(self, raw_action, safe_action, **kwargs):
        self.pending = (
            np.asarray(raw_action, dtype=np.float32).copy(),
            np.asarray(safe_action, dtype=np.float32).copy(),
            dict(kwargs),
        )

    def step(self, action):
        assert self.pending is not None
        raw, expected_safe, _ = self.pending
        self.pending = None
        action = np.asarray(action, dtype=np.float32)
        np.testing.assert_allclose(action, expected_safe, atol=1e-6)
        self.raw_actions.append(raw)
        self.executed_actions.append(action.copy())
        return self.observation.copy(), 0.0, False, False, {}


def test_rollout_buffer_stores_unclipped_z_while_env_receives_projection():
    env = _ProjectionRecordEnv()
    model = LatentActionPPO(
        "MlpPolicy",
        env,
        execution_mode="cbf",
        learning_rate=0.0,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=9,
        device="cpu",
        policy_kwargs=context_ignoring_policy_kwargs(
            policy_kwargs={
                "net_arch": {"pi": [8], "vf": [8]},
                "ortho_init": False,
                "log_std_init": -20.0,
            }
        ),
    )
    with th.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.copy_(th.tensor([5.0, 0.2]))
    model.learn(total_timesteps=2)

    stored_z = np.asarray(model.rollout_buffer.actions).reshape(-1, 2)
    assert np.all(stored_z[:, 0] > 4.9)
    assert len(env.raw_actions) == 2
    assert len(env.executed_actions) == 2
    np.testing.assert_allclose(np.asarray(env.raw_actions), stored_z, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(env.executed_actions)[:, 0], np.zeros(2), atol=1e-6
    )
    assert np.all(np.abs(np.asarray(env.executed_actions)) <= 3.0 + 1e-6)

    observations = th.as_tensor(
        model.rollout_buffer.observations.reshape(-1, model.observation_space.shape[0]),
        device=model.device,
    )
    actions = th.as_tensor(stored_z, device=model.device)
    with th.no_grad():
        _, recomputed_log_prob, _ = model.policy.evaluate_actions(
            observations, actions
        )
    stored_log_prob = th.as_tensor(
        model.rollout_buffer.log_probs.reshape(-1), device=model.device
    )
    th.testing.assert_close(recomputed_log_prob, stored_log_prob, atol=1e-5, rtol=1e-5)


def test_loaded_parallel_model_predicts_stages_for_one_state(tmp_path):
    env = _ProjectionRecordEnv()
    model = LatentActionPPO(
        "MlpPolicy",
        env,
        execution_mode="cbf",
        learning_rate=0.0,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=11,
        device="cpu",
        policy_kwargs=context_ignoring_policy_kwargs(
            policy_kwargs={
                "net_arch": {"pi": [8], "vf": [8]},
                "ortho_init": False,
                "log_std_init": -20.0,
            }
        ),
    )
    with th.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.copy_(th.tensor([5.0, 0.2]))

    # Saved parallel models retain their training topology. Diagnostic and
    # counterfactual APIs still need to support a single evaluation state.
    model.n_envs = 8
    path = tmp_path / "parallel_latent_ppo"
    model.save(path)
    loaded = LatentActionPPO.load(path, device="cpu")
    assert loaded.n_envs == 8

    stages = loaded.predict_action_stages(
        _augmented_observation(), deterministic=True
    )

    assert stages["latent_z"].shape == (2,)
    assert stages["executed_action"].shape == (2,)
    np.testing.assert_allclose(stages["latent_z"], [5.0, 0.2], atol=1e-5)
    np.testing.assert_allclose(stages["executed_action"], [0.0, 0.2], atol=1e-5)


def test_projected_ppo_save_load_preserves_action_stages(tmp_path):
    env = _ProjectionRecordEnv()
    model = ProjectedCBFPPO(
        ProjectedCBFActorCriticPolicy,
        env,
        execution_mode="cbf",
        lambda_mean=0.1,
        lambda_sample=0.0,
        learning_rate=0.0,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=17,
        device="cpu",
        policy_kwargs={
            "net_arch": {"pi": [8], "vf": [8]},
            "ortho_init": False,
            "log_std_init": -1.0,
        },
    )
    with th.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.copy_(th.tensor([2.0, -0.4]))
    expected = model.predict_action_stages(
        _augmented_observation(), deterministic=True
    )
    path = tmp_path / "projected_ppo"
    model.save(path)
    loaded = ProjectedCBFPPO.load(path, device="cpu")
    actual = loaded.predict_action_stages(
        _augmented_observation(), deterministic=True
    )
    for key in ("mu_raw", "mu_safe", "latent_z", "executed_action"):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-6)


def test_projected_ppo_collector_uses_projected_mean_logprob_and_hard_sample():
    env = _ProjectionRecordEnv()
    model = ProjectedCBFPPO(
        ProjectedCBFActorCriticPolicy,
        env,
        execution_mode="cbf",
        lambda_mean=0.0,
        lambda_sample=0.0,
        learning_rate=0.0,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        seed=23,
        device="cpu",
        policy_kwargs={
            "net_arch": {"pi": [8], "vf": [8]},
            "ortho_init": False,
            "log_std_init": -0.5,
        },
    )
    with th.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.copy_(th.tensor([2.0, 0.2]))
    model.learn(total_timesteps=16)

    stored_z = np.asarray(model.rollout_buffer.actions).reshape(-1, 2)
    raw = np.asarray(env.raw_actions)
    executed = np.asarray(env.executed_actions)
    np.testing.assert_allclose(raw, stored_z, atol=1e-6)
    assert np.any(stored_z[:, 0] > 0.0)
    np.testing.assert_allclose(
        executed[:, 0], np.minimum(stored_z[:, 0], 0.0), atol=1e-6
    )

    observations = th.as_tensor(
        model.rollout_buffer.observations.reshape(-1, model.observation_space.shape[0]),
        device=model.device,
    )
    actions = th.as_tensor(stored_z, device=model.device)
    with th.no_grad():
        evaluation = model.policy.evaluate_actions_with_projection(
            observations, actions
        )
    th.testing.assert_close(
        evaluation.mu_safe[:, 0],
        th.zeros_like(evaluation.mu_safe[:, 0]),
        atol=1e-6,
        rtol=0.0,
    )
    stored_log_prob = th.as_tensor(
        model.rollout_buffer.log_probs.reshape(-1), device=model.device
    )
    th.testing.assert_close(
        evaluation.log_prob, stored_log_prob, atol=1e-5, rtol=1e-5
    )
