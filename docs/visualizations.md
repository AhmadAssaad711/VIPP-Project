# Structured graphs, diagrams, and visualizations

This page is the visual index for the retained structured Highway RL work.
The 13 rendered PNG figures are versioned under [`artifacts/dqn/`](../artifacts/dqn/)
and are linked below by experiment. Model checkpoints, TensorBoard event
files, and other generated runtime output remain excluded.

## Research pipeline

```mermaid
flowchart LR
    N[Notebook narrative] --> U[Structured notebook utilities]
    U --> B[Baseline DQN]
    U --> A[Attention DQN]
    B --> H[HighwayEnv highway-v0]
    A --> H
    H --> R[Reward and safety wrappers]
    R --> M[Metrics and archived figures]
```

The notebooks define the experiment narrative and controlled variables. The
shared adapter builds the structured environment and learner configuration;
the DQN modules execute training/evaluation; and the archived figures show
the resulting metrics or summary dashboards.

## Archived result summary

The values below are transcribed from the retained local run summaries. They
describe archived experiments and are not a claim that the cleanup reran
training.

| Experiment | Steps | Eval episodes | Mean reward | Collision rate | Mean speed | Mean TTC | Minimum TTC | Figures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline DQN driver-spectrum run | 20,000 | 5 | 18.64 | 80% | 26.66 | 7.12 | 1.29 | [4 figures](#baseline-dqn-driver-spectrum) |
| Attention DQN potential-field run | 20,000 | 10 | 11.62 | 20% | 21.36 | 8.80 | 3.18 | [1 figure](#attention-dqn-potential-field-study) |
| Four-experiment baseline run | 20,000 | 10 | 2.67 | 100% | 29.13 | 4.15 | 0.17 | [4 figures](#controlled-congested-traffic-study) |
| Attention potential-field run (`ct_pf`) | 20,000 | 10 | 11.62 | 20% | 21.36 | 8.80 | 3.18 | [4 figures](#attention-potential-field-evaluation) |

These runs use different evaluation setups and should not be treated as one
causal comparison table without checking the full configuration metadata.

## Figure catalog

### Baseline DQN driver spectrum

Source notebook: [`baseline_dqn.ipynb`](../notebooks/structured_highway/baseline_dqn/baseline_dqn.ipynb)

![Baseline evaluation metrics](../artifacts/dqn/baseline_dqn/baseline_dqn_driver_spectrum_20k/evaluation_metrics.png)

![Baseline 1000-episode metrics](../artifacts/dqn/baseline_dqn/baseline_dqn_driver_spectrum_20k/saved_model_eval_1000_episodes/evaluation_metrics.png)

![Baseline 1000-episode summary](../artifacts/dqn/baseline_dqn/baseline_dqn_driver_spectrum_20k/saved_model_eval_1000_episodes/evaluation_summary.png)

![Baseline training and evaluation summary](../artifacts/dqn/baseline_dqn/baseline_dqn_driver_spectrum_20k/training_evaluation_summary.png)

### Attention DQN potential-field study

Source notebook: [`congested_reward_safety_factor_study.ipynb`](../notebooks/congested_traffic/congested_reward_safety_factor_study.ipynb)

![Attention DQN potential-field evaluation metrics](../artifacts/dqn/congested_potential_field_reward_test/attention_dqn_base_potential_field_20k/evaluation_metrics.png)

### Controlled congested-traffic study

Source notebook: [`congested_traffic_four_experiments.ipynb`](../notebooks/congested_traffic/congested_traffic_four_experiments.ipynb)

![Single-episode detailed metrics](../artifacts/dqn/congested_traffic_four_experiments/four_way_baseline_dqn_20k/eval1/metrics.png)

![Single-episode summary](../artifacts/dqn/congested_traffic_four_experiments/four_way_baseline_dqn_20k/eval1/summary.png)

![Controlled evaluation metrics](../artifacts/dqn/congested_traffic_four_experiments/four_way_baseline_dqn_20k/evaluation_metrics.png)

![Controlled training and evaluation summary](../artifacts/dqn/congested_traffic_four_experiments/four_way_baseline_dqn_20k/training_evaluation_summary.png)

### Attention potential-field evaluation

![Attention potential-field detailed metrics](../artifacts/dqn/ct_pf/attn_pf20k/eval1000/metrics.png)

![Attention potential-field summary](../artifacts/dqn/ct_pf/attn_pf20k/eval1000/summary.png)

![Attention potential-field evaluation metrics](../artifacts/dqn/ct_pf/attn_pf20k/evaluation_metrics.png)

![Attention potential-field training and evaluation summary](../artifacts/dqn/ct_pf/attn_pf20k/training_evaluation_summary.png)

## Notebook visualization status

The retained DQN and PPO notebook files are output-stripped, so their plots
are generated when the notebooks run rather than embedded in the notebooks.
The planning notebook currently contains text output and an environment error,
but no saved graph. The 13 PNGs above are the complete curated set of
rendered structured result visuals available in this checkout.

When adding a new experiment, store its rendered figures beside a compact
configuration/result manifest, link every figure from this page, and record
the source notebook, seed, evaluation protocol, and artifact revision.
