# Structured baseline DQN notebook

Notebook: [`baseline_dqn.ipynb`](baseline_dqn.ipynb)

## Purpose

This notebook establishes the standard discrete DQN reference for native
HighwayEnv `highway-v0`. It is the control policy for the attention and
reward/safety comparisons. The notebook is an orchestrator; the learner is
implemented in [`elurant_dqn.py`](../../../src/deep_learning/DQN/elurant_dqn.py)
and the shared setup is in
[`dqn_notebook_utils.py`](../../_shared/dqn_notebook_utils.py).

## Protocol

- Environment: `structured_baseline`, three lanes, 20 vehicles, duration 40,
  density 1.0, IDM traffic, simulation frequency 15, policy frequency 1.
- Observation: five nearby vehicles with relative `presence`, `x`, `y`, `vx`,
  and `vy` features.
- Action: `DiscreteMetaAction`.
- Reward: normalized native reward with collision `-1.0`, right-lane `0.1`,
  high-speed `0.4`, and lane-change `0.0` terms.
- Training: 20,000 timesteps, four environments, seed 42, learning rate
  `2.5e-4`, discount `0.95`.
- Final named run: `baseline_dqn_driver_spectrum_20k`; its optional driver
  spectrum samples surrounding IDM/MOBIL behaviour continuously from scores
  0--100 while leaving the ego policy unchanged.

## Cell guide

1. Locate the repository and import shared DQN helpers.
2. Build and preview the training/evaluation configurations.
3. Train the baseline and save its summary, model, evaluation metrics, and
   plots.
4. Reload the saved model for a 1,000-episode evaluation.
5. Run 100-episode congestion diagnostics that label bad actions, unavailable
   good actions, earlier lane mistakes, and rear-pressure cases.
6. Optionally train a separate TTC/flow reward baseline and render a policy
   panel. These are follow-up runs, not replacements for the control result.

## Outputs

The default output root is `artifacts/dqn/baseline_dqn/`. A run contains a
`summary.json`, model checkpoint, per-episode `evaluation_metrics.json`,
summary plots, and optional diagnostic CSV/JSON files. All are ignored by Git.

## Recorded result snapshot

The clean staging repository intentionally excludes raw artifacts. The table
below records the matching source-workspace snapshot that was available during
the repository split; it is provenance, not a committed checkpoint.

| Metric | 20k training evaluation (5 episodes) | Saved-model evaluation (1,000 episodes) |
| --- | ---: | ---: |
| Mean reward | 18.644 | 18.665 ± 10.401 |
| Collision rate | 80.0% | 80.3% (803/1,000) |
| Average speed | 26.657 m/s | 27.079 ± 2.313 m/s |
| Overtakes | 3.600 | 3.622 ± 2.613 |
| Average TTC | 7.119 s | 6.734 ± 1.797 s |
| Minimum TTC | 1.293 s | 1.392 ± 2.260 s |

Source provenance path:
`artifacts/dqn/baseline_dqn/baseline_dqn_driver_spectrum_20k/` in the
migration workspace. The result should be rerun before publication, with the
full configuration and evaluation seed recorded alongside it.
