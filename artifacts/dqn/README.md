# Structured DQN visualizations

This directory contains the curated rendered figures for the structured
Highway RL experiments. The complete visual index, research-flow diagram,
archived result table, and notebook provenance are in
[`docs/visualizations.md`](../../docs/visualizations.md).

The figures are grouped by experiment:

| Experiment | Figures |
| --- | --- |
| Baseline DQN driver spectrum | [`evaluation_metrics.png`](baseline_dqn/baseline_dqn_driver_spectrum_20k/evaluation_metrics.png), [`1000-episode metrics`](baseline_dqn/baseline_dqn_driver_spectrum_20k/saved_model_eval_1000_episodes/evaluation_metrics.png), [`1000-episode summary`](baseline_dqn/baseline_dqn_driver_spectrum_20k/saved_model_eval_1000_episodes/evaluation_summary.png), [`training/evaluation summary`](baseline_dqn/baseline_dqn_driver_spectrum_20k/training_evaluation_summary.png) |
| Attention DQN potential-field study | [`evaluation_metrics.png`](congested_potential_field_reward_test/attention_dqn_base_potential_field_20k/evaluation_metrics.png) |
| Controlled congested-traffic study | [`single-episode metrics`](congested_traffic_four_experiments/four_way_baseline_dqn_20k/eval1/metrics.png), [`single-episode summary`](congested_traffic_four_experiments/four_way_baseline_dqn_20k/eval1/summary.png), [`evaluation metrics`](congested_traffic_four_experiments/four_way_baseline_dqn_20k/evaluation_metrics.png), [`training/evaluation summary`](congested_traffic_four_experiments/four_way_baseline_dqn_20k/training_evaluation_summary.png) |
| Attention potential-field evaluation | [`detailed metrics`](ct_pf/attn_pf20k/eval1000/metrics.png), [`summary`](ct_pf/attn_pf20k/eval1000/summary.png), [`evaluation metrics`](ct_pf/attn_pf20k/evaluation_metrics.png), [`training/evaluation summary`](ct_pf/attn_pf20k/training_evaluation_summary.png) |

Only rendered figures are retained here. Checkpoints, TensorBoard events,
and other runtime-generated files remain outside version control.
