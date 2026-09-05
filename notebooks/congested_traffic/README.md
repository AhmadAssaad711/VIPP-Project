# Congested-traffic notebooks

This directory contains three structured studies. All use native lane-based
`highway-v0`; none changes the repository into a different environment or
action space.

## Shared congested protocol

The notebooks use three lanes, 30 vehicles, duration 40, ego spacing 1.8,
density 1.2, IDM traffic, simulation frequency 15, and policy frequency 3.
The observation contains 12 relative kinematic vehicles and sees behind the
ego. The base reward is unnormalized with collision `-5.0`, right-lane `0.03`,
high-speed `0.25`, and lane-change `-0.02`. Surrounding drivers use a uniform
conservative-to-aggressive spectrum. Training runs use 20,000 timesteps, seed
42, and ten short evaluation episodes; saved models are evaluated for 1,000
episodes.

## `congested_traffic_four_experiments.ipynb`

### Question and design

This is the clean 2x2 study of representation and TTC safety reward:

| Representation | TTC safety reward | Run label |
| --- | --- | --- |
| Baseline DQN | Off | `base20k` |
| Attention DQN | Off | `attn20k` |
| Baseline DQN | On | `safe20k` |
| Attention DQN | On | `attn_safe20k` |

Adaptive speed control, rear-flow injection, TTC observations, and lane-change
safety penalties remain disabled so the two named factors stay interpretable.
Each run writes a summary and a saved-model evaluation; the final cell joins
the four summaries into `comparison.csv`.

### Cell guide

The first cells load both DQN backends and define the shared configuration. The
runner builds one environment per factor combination, trains it, evaluates the
saved checkpoint, and stores the result in `artifacts/dqn/ct4/`. The final cell
is a reporting step and does not retrain anything.

### Available result

The source-workspace snapshot contains only the baseline training row, not a
complete four-cell comparison:

| Metric | Baseline DQN snapshot (20k training evaluation, 10 episodes) |
| --- | ---: |
| Mean reward | 2.669 |
| Collision rate | 100.0% |
| Average speed | 29.132 m/s |
| Overtakes | 3.400 |
| Average TTC | 4.147 s |
| Minimum TTC | 0.168 s |

Because the attention and safety cells are absent from that snapshot, these
values must not be presented as a causal 2x2 conclusion. Rerun all four cells
under the same protocol before comparing them.

## `congested_reward_safety_factor_study.ipynb`

### Question and design

This focused study adds only a proximity-based potential-field reward to an
attention DQN. It disables adaptive control, rear-flow injection, TTC reward,
TTC observation, and lane-change safety terms. The field uses weight `0.25`,
120 m sensing range, and the configured longitudinal/lateral/time-gap terms.

The notebook first performs a five-step wrapper smoke test, then trains
`attn_pf20k`, evaluates it for 1,000 episodes, and writes a one-row
`summary.csv` under `artifacts/dqn/ct_pf/`.

### Recorded result snapshot

The following source-workspace snapshot is recorded for provenance; its raw
model and plots are not part of this clean repository.

| Metric | Saved-model evaluation (1,000 episodes) |
| --- | ---: |
| Mean reward | 9.084 ± 6.806 |
| Collision rate | 39.9% (399/1,000) |
| Average speed | 21.810 ± 1.270 m/s |
| Overtakes | 3.700 ± 2.389 |
| Average TTC | 8.113 ± 1.672 s |
| Minimum TTC | 2.311 ± 2.631 s |
| Potential-field cost | 0.161 ± 0.072 |
| Potential-field penalty | 0.040 ± 0.018 |

This is a separate congested protocol and should not be compared directly with
the baseline-DQN snapshot without matching all environment and reward settings.

## `congested_traffic_policy.ipynb`

### Question and design

This notebook expands the congested study to seven labelled DQN variants: a
baseline, baseline plus safety reward, attention, attention plus safety
reward, adaptive TTC wide-band control, adaptive control plus safety reward,
and adaptive attention. Each variant is intended to use the same congested
traffic distribution, 20,000 training steps, and a 1,000-episode saved-model
evaluation.

### Cell guide and outputs

The setup cells define the common traffic, observation, reward, and driver
configuration. Seven experiment cells call the shared DQN runner. The
diagnostic cells label collision episodes using action quality, lane quality,
TTC, and rear pressure, then the final cell writes a comparison table. Outputs
are placed under `artifacts/dqn/congested_traffic_policy/` when the notebook is
run.

### Result status

No completed result summary for this seven-variant notebook is present in the
clean staging repository or the source snapshot used for this documentation
pass. The notebook is retained because all of its experiment cells use the
structured DQN contract; run all seven variants before making a comparison.
