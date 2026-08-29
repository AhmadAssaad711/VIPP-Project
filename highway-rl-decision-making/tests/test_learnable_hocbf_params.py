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

from cbf_projection import project_polytope_2d_torch  # noqa: E402
from learnable_hocbf_params import (  # noqa: E402
    LearnableCBFConfig,
    LearnableProjectedCBFPPO,
    ParameterRateNetwork,
    dynamic_bounds_from_primitives,
    migrate_projected_policy_weights,
    parameter_rate_interval,
    smooth_project_parameter_rates,
)


def test_parameter_rate_interval_and_euler_step_stay_inside_bounds():
    config = LearnableCBFConfig().validate()
    p = th.tensor([[1.15, 3.45], [2.3, 2.3]], dtype=th.float32)
    raw = th.tensor([[0.0, 0.0], [8.0, -8.0]], dtype=th.float32)
    safe, lower, upper = smooth_project_parameter_rates(raw, p, config)

    # Auxiliary parameter CBFs collapse the outward rate at either boundary.
    assert float(lower[0, 0]) == 0.0
    assert float(upper[0, 1]) == 0.0
    p_next = p + config.dt_policy * safe
    p_min = th.as_tensor(config.p_min)
    p_max = th.as_tensor(config.p_max)
    assert bool((p_next >= p_min - 1e-6).all())
    assert bool((p_next <= p_max + 1e-6).all())
    assert bool((safe >= lower - 1e-6).all())
    assert bool((safe <= upper + 1e-6).all())


def test_dynamic_bounds_use_p1_p2_and_nu1():
    q = th.tensor([[-2.0, 3.0]])
    h = th.tensor([[1.5, 0.0]])
    h_dot = th.tensor([[0.25, -1.0]])
    static = th.tensor([[99.0, 7.0]])
    dynamic_mask = th.tensor([[True, False]])
    p = th.tensor([[2.0, 3.0]], requires_grad=True)
    nu = th.tensor([[0.4, -9.0]], requires_grad=True)

    bounds = dynamic_bounds_from_primitives(
        q, h, h_dot, static, dynamic_mask, p, nu
    )
    expected = -2.0 + (2.0 + 3.0) * 0.25 + (2.0 * 3.0 + 0.4) * 1.5
    assert np.isclose(float(bounds[0, 0]), expected)
    assert float(bounds[0, 1]) == 7.0
    bounds.sum().backward()
    assert p.grad is not None and bool((p.grad.abs() > 0).any())
    assert nu.grad is not None and float(nu.grad[0, 0].abs()) > 0.0
    assert float(nu.grad[0, 1]) == 0.0


def test_projection_constraint_detach_switch_routes_bound_gradient():
    target = th.tensor([[2.0, 0.0]])
    rows = th.tensor([[[1.0, 0.0]]])
    bounds = th.tensor([[1.0]], requires_grad=True)

    detached = project_polytope_2d_torch(target, rows, bounds)
    assert not detached.action.requires_grad

    differentiable = project_polytope_2d_torch(
        target, rows, bounds, detach_constraints=False
    )
    differentiable.action.sum().backward()
    assert bounds.grad is not None
    assert float(bounds.grad.abs().sum()) > 0.0


def test_truncated_unroll_provides_future_nu2_gradient():
    # Build only the small object surface consumed by _window_loss.  This
    # avoids starting a simulator while checking the differentiable recurrence.
    model = object.__new__(LearnableProjectedCBFPPO)
    model.parameter_config = LearnableCBFConfig(
        lambda_feas=0.0,
        lambda_intervention=1.0,
        lambda_smooth=0.0,
        lambda_reg=0.0,
        unroll_horizon=2,
    ).validate()
    model.device = th.device("cpu")
    model.parameter_state_dim = 4
    model.cbf_feasibility_tol = 1e-5
    model.action_space = gym.spaces.Box(
        low=np.asarray([-10.0, -10.0], dtype=np.float32),
        high=np.asarray([10.0, 10.0], dtype=np.float32),
        dtype=np.float32,
    )
    model.parameter_net = ParameterRateNetwork(4, hidden_dims=(8, 8), output_dim=2)

    # At t=0 h=0, so the first intervention does not depend on either rate.
    # At t=1, p2 reconstructed from nu2,0 enters p1*p2*h and changes the
    # projected mean action.  The gradient below therefore certifies the
    # intended future-step nu2 path.
    data = {
        "state_aug": th.tensor(
            [
                [[0.0, 0.0, 2.3, 2.3]],
                [[0.0, 0.0, 2.3, 2.3]],
            ]
        ),
        "p": th.tensor([[[2.3, 2.3]], [[2.3, 2.3]]]),
        "nu_safe": th.zeros((2, 1, 2)),
        "rows": th.tensor(
            [[[[1.0, 0.0]]], [[[1.0, 0.0]]]], dtype=th.float32
        ),
        "q": th.tensor([[[-1.0]], [[-5.29]]]),
        "h": th.tensor([[[0.0]], [[1.0]]]),
        "h_dot": th.zeros((2, 1, 1)),
        "static_bounds": th.zeros((2, 1, 1)),
        "dynamic_mask": th.ones((2, 1, 1), dtype=th.bool),
        "mask": th.ones((2, 1, 1), dtype=th.bool),
        "mu_raw": th.tensor([[[2.0, 0.0]], [[2.0, 0.0]]]),
        "done": th.zeros((2, 1), dtype=th.bool),
    }
    losses = model._window_loss(data, env_index=0, start=0, stop=2)
    losses["intervention"].backward()
    final_weight_grad = model.parameter_net.net[-1].weight.grad
    assert final_weight_grad is not None
    assert float(final_weight_grad[1].abs().sum()) > 1e-8


def test_policy_migration_zero_extends_32d_first_layers():
    class Holder:
        def __init__(self, module):
            self.policy = module

    source_policy = th.nn.Sequential(th.nn.Linear(32, 4), th.nn.Linear(4, 2))
    target_policy = th.nn.Sequential(th.nn.Linear(34, 4), th.nn.Linear(4, 2))
    with th.no_grad():
        source_policy[0].weight.fill_(0.25)
        source_policy[0].bias.fill_(0.5)
        source_policy[1].weight.fill_(-0.75)
        source_policy[1].bias.fill_(0.1)
    source = Holder(source_policy)
    target = Holder(target_policy)

    result = migrate_projected_policy_weights(source, target)
    assert "0.weight" in result["expanded"]
    th.testing.assert_close(target.policy[0].weight[:, :32], source.policy[0].weight)
    assert bool((target.policy[0].weight[:, 32:] == 0.0).all())
    th.testing.assert_close(target.policy[1].weight, source.policy[1].weight)
    th.testing.assert_close(target.policy[1].bias, source.policy[1].bias)
