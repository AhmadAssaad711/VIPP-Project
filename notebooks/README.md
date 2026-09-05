# Notebook map

Run notebooks from the repository root after installing `requirements.txt`.
Each notebook is an experiment narrative; the reusable learner and wrapper
code belongs under `src/deep_learning/DQN/`.

## Recommended order

1. [`structured_highway/baseline_dqn/baseline_dqn.ipynb`](structured_highway/baseline_dqn/baseline_dqn.ipynb)
   establishes the standard discrete DQN reference.
2. [`structured_highway/attention_dqn/attention_dqn.ipynb`](structured_highway/attention_dqn/attention_dqn.ipynb)
   changes the representation while holding the structured task fixed.
3. [`congested_traffic/congested_traffic_four_experiments.ipynb`](congested_traffic/congested_traffic_four_experiments.ipynb)
   runs the controlled attention-by-TTC-safety 2x2 comparison.
4. The remaining congested-traffic notebooks are follow-up combinations of
   reward, traffic, and safety terms and should not be reported as the same
   factorial study.
5. [`planning/CEM_planning_trials.ipynb`](planning/CEM_planning_trials.ipynb)
   is a historical planning comparison and requires an external `rl-agents`
   checkout.

## Notebook inventory

| Area | Notebooks |
| --- | --- |
| Baseline and attention DQN | `structured_highway/baseline_dqn/`, `structured_highway/attention_dqn/` |
| Structured PPO history | `structured_highway/ppo/` |
| Congested traffic | `congested_traffic/` |
| Planning comparison | `planning/CEM_planning_trials.ipynb` |

The environment remains the lane-based HighwayEnv contract: lane counts and
lane indices are meaningful, and the policy uses discrete lane-level actions.
Profiles called `semi_unstructured` and `unstructured_stress` remain as
historical structured stress configurations because they still use that
structured contract.

See [the visualization index](../docs/visualizations.md) for the retained
graphs, diagram, archived result values, and the source notebook for each
figure.

Folder-level READMEs document the supported baseline, attention, congested,
and historical PPO notebook families. Each one records its purpose, cell
flow, expected outputs, and result status.

## Artifact rule

Training outputs belong under an ignored `artifacts/` directory or an
explicit external result root. Preserve the environment profile, reward
switches, timestep budget, seed, checkpoint, and evaluation protocol with
every promoted result; similarly named notebook folders are not sufficient
provenance.
