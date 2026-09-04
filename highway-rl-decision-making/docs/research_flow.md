# Research flow

This checkout contains the retained structured/lane-based research track. It
is self-contained and is not evaluated through any sibling checkout. The
intended structured flow is:

| Track | Environment | Scientific question | Main entry points |
| --- | --- | --- | --- |
| Laned / structured | highway-v0 from HighwayEnv | Does better scene representation, followed by carefully isolated reward and safety terms, improve lane-based decision making? | notebooks/structured_highway/, notebooks/congested_traffic/, src/deep_learning/DQN/ |

Notebooks are the scientific narratives and experiment orchestrators. Python
modules contain reusable environment construction, model,
reward, safety, evaluation, and reporting logic. A notebook should explain
the hypothesis, define the controlled variables, call those modules, and
collect the resulting artifacts; it should not be the only place where the
core algorithm exists.

## Track A: laned / structured HighwayEnv

### Recommended order

1. Run [baseline_dqn.ipynb](../notebooks/structured_highway/baseline_dqn/baseline_dqn.ipynb)
   to establish the standard Leurent-style DQN behavior in highway-v0.
2. Run [attention_dqn.ipynb](../notebooks/structured_highway/attention_dqn/attention_dqn.ipynb)
   with the same structured configuration to test whether ego-centered
   attention changes the policy.
3. Run [congested_traffic_four_experiments.ipynb](../notebooks/congested_traffic/congested_traffic_four_experiments.ipynb)
   for the clean 2x2 factorial comparison: baseline versus attention, each
   with and without the TTC safety reward.
4. Use [congested_traffic_policy.ipynb](../notebooks/congested_traffic/congested_traffic_policy.ipynb)
   for the broader combination of traffic-flow,
   adaptive-longitudinal, driver-aggressiveness, TTC-observation, potential
   field, and lane-change-safety terms.
5. Run [congested_reward_safety_factor_study.ipynb](../notebooks/congested_traffic/congested_reward_safety_factor_study.ipynb)
   only as the focused potential-field test. It remains a laned
   highway-v0 experiment and should be interpreted as a single-factor
   reward study.

The PPO notebooks under structured_highway/ppo/ are part of the intended
laned track, but their imports should be checked before treating them as
reproducible entry points. The current checkout references modules such as
attention_ppo, elurant_ppo, ppo_overtake_lab, and
paper_ppo_reproduction that are not present in the visible source tree.

### Code path

    structured notebook
      -> notebooks/_shared/dqn_notebook_utils.py
          -> elurant_dqn.py or attention_dqn.py
              -> highway-v0
              -> adaptive_longitudinal.py wrappers
              -> congestion_diagnostics.py
      -> artifacts/dqn/<experiment>

dqn_notebook_utils.py owns the shared structured configuration and the
notebook-facing train/evaluate helpers. The two DQN backends differ mainly in
the feature extractor: the baseline uses the standard structured input, while
the attention backend uses the ego-attention extractor. The wrapper module
adds optional reward and observation terms around the native HighwayEnv
reward. Each experiment must record which terms are enabled; similarly named
folders are not enough to establish comparability.

The four-experiment notebook intentionally disables adaptive speed, lane
context, and lane-change safety. That is a control experiment for isolating
the two intended factors: attention and TTC safety reward. The larger
congested notebooks are follow-up studies and should not be presented as the
same ablation.

## Planning comparison

[CEM_planning_trials.ipynb](../notebooks/planning/CEM_planning_trials.ipynb) is
retained as a historical planning comparison. It depends on an external
`rl-agents` checkout that is not included here, so it is not a clean smoke-test
entry point until that dependency is pinned.

## Artifact and reproducibility rules

Generated checkpoints, TensorBoard logs, monitor CSVs, videos, and Python
caches belong outside version control. Commit small manifests, configuration
snapshots, KPI tables, and selected figures. The retained `artifacts/dqn/`
tree contains the structured result summaries; new generated output should
remain outside Git.

Every promoted result should identify:

- track and environment ID;
- HighwayEnv revision and experiment-code revision;
- observation and action spaces;
- complete reward/safety configuration;
- traffic model and route/episode protocol;
- random seeds;
- checkpoint and artifact root;
- whether CBF was present during training, evaluation, both, or neither.

## Repository boundary

This source tree owns the native highway-v0 experiments, attention DQN,
baseline DQN, structured reward/safety wrappers, structured diagnostics, and
their curated `artifacts/dqn/` results. Generated checkpoints, logs, videos,
and caches remain outside version control.
