# Highway RL Decision Making

Structured/lane-based notebook and paper workspace for reinforcement-learning
experiments in autonomous highway decision making.

## Start here

The canonical scientific map is [docs/research_flow.md](docs/research_flow.md).
It defines the retained structured track, its execution order, code paths,
and artifact ownership.

The structured staging copy is `../laned-highway-rl/`. This source directory
does not depend on any sibling checkout.

Use [notebooks/README.md](notebooks/README.md) for notebook flow and
[docs/research_flow.md](docs/research_flow.md) for reusable module
responsibilities.

## Scope

This work focuses on structured decision-level behavior:

- structured highway policies
- congestion-aware decision making
- reward and safety-factor studies
- planning baselines for comparison

## Research Flow

The retained work uses the lane-based HighwayEnv contract. Read
docs/research_flow.md for the detailed notebook-to-module flow.

Paper: [`docs/paper/highway-rl-decision-making-paper.pdf`](docs/paper/highway-rl-decision-making-paper.pdf)

## Notebooks

| Folder | Purpose |
| --- | --- |
| [`structured_highway/`](notebooks/structured_highway/) | DQN, attention DQN, PPO, hybrid PPO, and reproduction notebooks. |
| [`congested_traffic/`](notebooks/congested_traffic/) | Dense traffic policy experiments and reward-safety studies. |
| [`planning/`](notebooks/planning/) | CEM planning trials used as decision-level comparisons. |

Full notebook list: [`notebooks/README.md`](notebooks/README.md)

Rendered result figures and the archived result summaries are indexed in the
parent workspace's [`docs/visualizations.md`](../docs/visualizations.md).

## Install

```powershell
py -3.12 -m pip install -r requirements.txt
```

## Notes

Generated outputs, checkpoints, videos, and Python caches are not source
documentation and should not be committed by default. The local virtual
environments are development environments rather than repository content;
remove duplicates only after confirming which interpreter is still needed.
Read [docs/repo_split_plan.md](docs/repo_split_plan.md) for the completed
file-ownership decision and cleanup record.
