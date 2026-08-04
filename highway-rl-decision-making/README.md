# Highway RL Decision Making

Notebook and paper workspace for reinforcement-learning experiments in autonomous highway decision making.

## Scope

This work focuses on decision-level behavior:

- structured highway policies
- congestion-aware decision making
- reward and safety-factor studies
- laneless and unstructured highway environments
- planning baselines for comparison

## Research Flow

1. Establish baselines in structured highway environments.
2. Improve the baselines with attention models, PPO variants, and reward-safety studies.
3. Extend and evaluate the approach in unstructured settings: congested traffic and laneless highways.

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

This public version keeps the notebooks and associated paper. Generated outputs, source scripts, old practice work, and vendor copies are intentionally excluded.
