# Notebook map

Notebooks are experiment narratives. Run setup/configuration cells first,
then training/evaluation cells in order. Reusable implementation belongs in
src/ or scripts/; the notebook should make the hypothesis, controlled
variables, inputs, outputs, and interpretation visible.

The complete dependency and artifact flow is in
[../docs/research_flow.md](../docs/research_flow.md).

Recommended order: structured baseline DQN, attention DQN, the controlled
congested four-experiment ablation, broader structured reward studies, then
the laneless environment smoke test and canonical lanelessKaralakou study.

Associated paper: [`../docs/paper/highway-rl-decision-making-paper.pdf`](../docs/paper/highway-rl-decision-making-paper.pdf)

## Structured Highway

| Notebook | Focus |
| --- | --- |
| [`baseline_dqn.ipynb`](structured_highway/baseline_dqn/baseline_dqn.ipynb) | Baseline DQN policy experiments. |
| [`attention_dqn.ipynb`](structured_highway/attention_dqn/attention_dqn.ipynb) | Attention-based DQN experiments. |
| [`congested_traffic_four_experiments.ipynb`](congested_traffic/congested_traffic_four_experiments.ipynb) | Controlled 2x2 comparison: attention x TTC safety reward. |
| [`PPO_trials.ipynb`](structured_highway/ppo/PPO_trials.ipynb) | PPO training trials. |
| [`Hybrid_PPO_baseline.ipynb`](structured_highway/ppo/Hybrid_PPO_baseline.ipynb) | Hybrid PPO baseline experiments. |
| [`Paper_PPO_reproduction.ipynb`](structured_highway/ppo/Paper_PPO_reproduction.ipynb) | PPO reproduction notebook. |
| [`Attention_PPO_baseline.ipynb`](structured_highway/ppo/Attention_PPO_baseline.ipynb) | Attention PPO baseline experiments. |

The PPO notebooks currently reference backend modules that are not visible
in the source tree. Treat them as historical/incomplete until the imports
are restored or the notebooks are rewritten.

## Congested Traffic

| Notebook | Focus |
| --- | --- |
| [`congested_traffic_policy.ipynb`](congested_traffic/congested_traffic_policy.ipynb) | Dense traffic policy experiment. |
| [`congested_traffic_policy_v2.ipynb`](congested_traffic/congested_traffic_policy_v2.ipynb) | Second congested traffic policy variant. |
| [`congested_reward_safety_factor_study.ipynb`](congested_traffic/congested_reward_safety_factor_study.ipynb) | Attention DQN with base reward plus potential-field reward shaping. |

## Laneless and Unstructured

| Notebook | Focus |
| --- | --- |
| [`laneless_highway_env.ipynb`](laneless_unstructured/laneless_highway_env.ipynb) | Laneless highway environment study. |
| [`lanelessKaralakou.ipynb`](lanelessKaralakou.ipynb) | Canonical seven-policy PPO/CBF formulation and paired evaluation. |

The laneless notebooks use the custom lane-free-v0 environment and
continuous [a_x, a_y] actions; they are separate from the lane-based
structured profiles named semi_unstructured and unstructured_stress.

## Planning

| Notebook | Focus |
| --- | --- |
| [`CEM_planning_trials.ipynb`](planning/CEM_planning_trials.ipynb) | Cross-entropy-method planning trials. |
