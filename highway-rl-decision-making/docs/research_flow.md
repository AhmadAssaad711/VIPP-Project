# Research flow

This checkout currently contains two related, but scientifically different,
research tracks. The intended split is by environment and research question,
not by the order in which files were created:

| Track | Environment | Scientific question | Main entry points |
| --- | --- | --- | --- |
| Laned / structured | highway-v0 from HighwayEnv | Does better scene representation, followed by carefully isolated reward and safety terms, improve lane-based decision making? | notebooks/structured_highway/, notebooks/congested_traffic/, src/deep_learning/DQN/ |
| Laneless / unstructured | lane-free-v0 from laneless highway env/lane_free_env.py | Can a continuous policy and a high-order CBF formulation produce useful, collision-aware behavior without lane labels? | notebooks/lanelessKaralakou.ipynb, scripts/*ppo*cbf*, scripts/run_ppo_cbf_progression.py |

Notebooks are the scientific narratives and experiment orchestrators. Python
modules and scripts contain reusable environment construction, model,
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
   and its v2 variant for the broader combination of traffic-flow,
   adaptive-longitudinal, driver-aggressiveness, TTC-observation, potential
   field, and lane-change-safety terms.
5. Run [congested_reward_safety_factor_study.ipynb](../notebooks/congested_traffic/congested_reward_safety_factor_study.ipynb)
   only as the focused potential-field test. It borrows the proximity-field
   idea from the laneless work but remains a laned highway-v0 experiment.

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

## Track B: laneless / unstructured HighwayEnv

### Recommended order

1. Start with [laneless_highway_env.ipynb](../notebooks/laneless_unstructured/laneless_highway_env.ipynb)
   to inspect and smoke-test the custom lane-free-v0 environment.
2. Read and run [lanelessKaralakou.ipynb](../notebooks/lanelessKaralakou.ipynb) as the
   canonical scientific formulation. It defines the shared continuous
   observation/action contract, the Karalakou-style reward path, the CBF
   geometry, the seven-policy progression, and the paired evaluation.
3. Use [run_ppo_cbf_progression.py](../scripts/run_ppo_cbf_progression.py) for
   repeatable command-line training of the PPO/CBF ladder. The progression
   separates nominal PPO, non-differentiable hard projection, and
   differentiable projection with reward and/or actor-internalization terms.
4. Use the evaluate_ppo_cbf_*.py scripts for deployment, timing,
   counterfactual, gain-grid, and CBF-free evaluation. Keep CBF-OFF and
   CBF-ON results separate: removing a projection at evaluation time is not
   the same as evaluating a policy that was trained without the CBF path.
5. Use the render scripts only after the numeric evaluation has completed.
   Videos are qualitative evidence and should be linked to a seed, checkpoint,
   and evaluation manifest.

### Code path

    laneless notebook or CLI configuration
      -> lane_free_env.py
          -> ppo_cbf_env.py
              -> cbf_projection.py
              -> cbf_ray_mask.py (legacy DDPG path only)
          -> ppo_reward_safety.py
          -> ppo_observation_variants.py
      -> projected_ppo_cbf.py
          -> run_ppo_cbf_progression.py
      -> finalResults/ or artifacts/<run>

The custom environment has no lane index and no lane-change action. The ego
action is a continuous longitudinal/lateral acceleration command. The CBF
formulation inflates an elliptical clearance set, differentiates the barrier
through the relative dynamics, imposes the HOCBF condition
h_ddot + k1 h_dot + k0 h >= 0, and converts those conditions into
two-dimensional linear inequalities. cbf_projection.py enumerates the target,
one-face, and two-face candidates exactly. If the no-slack set is empty, it
returns an explicitly labelled least-violating fallback; that fallback must
not be reported as a safe QP solution.

The canonical 1M study currently describes seven policies, five physics
substeps per policy step, a 32-dimensional learned state, and paired CBF-OFF
/ CBF-ON evaluation. Keep those protocol values in the run manifest and
avoid comparing runs with different route length, episode count, or traffic
model as though they were one leaderboard.

## Artifact and reproducibility rules

Generated checkpoints, TensorBoard logs, monitor CSVs, videos, and Python
caches belong outside version control. Commit small manifests, configuration
snapshots, KPI tables, and selected figures. The current artifacts/ tree is
useful for local continuation but is not a clean publication boundary.

Every promoted result should identify:

- track and environment ID;
- HighwayEnv revision and experiment-code revision;
- observation and action spaces;
- complete reward/safety configuration;
- traffic model and route/episode protocol;
- random seeds;
- checkpoint and artifact root;
- whether CBF was present during training, evaluation, both, or neither.

finalResults/ is a curated laneless package, not a replacement for the
training manifest. Its current package contains a provenance inconsistency:
the top-level manifest and the true-CBF-free evaluation metadata report
different episode counts. Resolve that mismatch before using the package for
publication claims.

## Boundary for the eventual split

The laned repository should contain the native highway-v0 experiments,
attention DQN, baseline DQN, structured reward/safety wrappers, structured
diagnostics, and their curated artifacts/dqn/* results.

The laneless repository should contain lane_free_env.py, the
lanelessKaralakou notebook, the PPO/CBF training and evaluation scripts,
CBF tests, laneless registries/configuration, and curated laneless results.
The upstream HighwayEnv checkout is currently a clean v1.11 gitlink with only
a local .DS_Store deletion in its nested status; it should be pinned as a
common dependency or explicit third-party revision rather than duplicated as
part of the laneless implementation.
