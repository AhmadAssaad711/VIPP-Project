# Highway RL Decision Making

Notebook and paper workspace for reinforcement-learning experiments in autonomous highway decision making.

## Start here

The canonical scientific map is [docs/research_flow.md](docs/research_flow.md).
It defines the laned/structured and laneless/unstructured tracks, their
execution order, code paths, artifact ownership, and eventual split boundary.

Use [notebooks/README.md](notebooks/README.md) for notebook flow and
[scripts/README.md](scripts/README.md) for reusable script responsibilities.

## Scope

This work focuses on decision-level behavior:

- structured highway policies
- congestion-aware decision making
- reward and safety-factor studies
- laneless and unstructured highway environments
- planning baselines for comparison

## Research Flow

The work has two parallel boundaries: laned/structured HighwayEnv work
around attention and reward/safety integration, and laneless/unstructured
work around the custom lane-free environment and CBF formulation. Read
docs/research_flow.md for the detailed notebook-to-script flow.

Paper: [`docs/paper/highway-rl-decision-making-paper.pdf`](docs/paper/highway-rl-decision-making-paper.pdf)

## Notebooks

| Folder | Purpose |
| --- | --- |
| [`structured_highway/`](notebooks/structured_highway/) | DQN, attention DQN, PPO, hybrid PPO, and reproduction notebooks. |
| [`congested_traffic/`](notebooks/congested_traffic/) | Dense traffic policy experiments and reward-safety studies. |
| [`laneless_unstructured/`](notebooks/laneless_unstructured/) | Laneless highway environment experiments. |
| [`planning/`](notebooks/planning/) | CEM planning trials used as decision-level comparisons. |

Full notebook list: [`notebooks/README.md`](notebooks/README.md)

CBF reward-by-actor-loss study: [`docs/cbf_factorial_ablation.md`](docs/cbf_factorial_ablation.md)

Diagnostic scenario registry: [`docs/diagnostic_scenarios.md`](docs/diagnostic_scenarios.md)

## Install

```powershell
py -3.12 -m pip install -r requirements.txt
```

## Exact PPO/DDPG baseline comparison

To compare the two algorithms before changing the reward or safety design,
run the frozen nominal P0/Q0 formulation:

```powershell
python scripts\compare_nominal_ppo_ddpg.py
```

The runner uses PPO `Q0_current_aligned` and DDPG `P0_current` with the same
environment, 42-D observation, normalized acceleration action, reciprocal
reward, collision protocol, training seed, and fixed evaluation seeds. It
writes `final_comparison.csv`, `checkpoint_comparison.csv`, and the shared
formulation manifest under `artifacts\ppo_ddpg_exact_p0`.

## Notes

Generated outputs, checkpoints, videos, and Python caches are not source
documentation and should not be committed by default. The local virtual
environments are development environments rather than repository content;
remove duplicates only after confirming which interpreter is still needed.
Read [docs/repo_split_plan.md](docs/repo_split_plan.md) for the concrete
file ownership and migration sequence.
