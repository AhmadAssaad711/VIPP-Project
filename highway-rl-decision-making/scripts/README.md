# Script guide

The notebooks describe experiments; the scripts implement reusable execution
and evaluation paths. Run commands from this project directory after
installing the selected environment:

    cd highway-rl-decision-making

Before starting a long job, inspect its options and the output directory:

    python scripts/run_ppo_cbf_progression.py --help
    python scripts/evaluate_ppo_cbf_free.py --help

Do not infer an experiment from a filename alone. Record the environment,
reward/safety switches, seed, checkpoint, and evaluation protocol in the
result manifest.

## Structured / laned track

These modules use the upstream HighwayEnv highway-v0 environment and
lane-based observations/actions.

| File | Responsibility | Used by |
| --- | --- | --- |
| src/deep_learning/DQN/elurant_dqn.py | Standard Leurent-style DQN baseline. Builds the native structured observation, trains/evaluates a discrete policy, and writes model/evaluation outputs. | notebooks/structured_highway/baseline_dqn/ |
| src/deep_learning/DQN/attention_dqn.py | Attention DQN backend. Replaces the baseline feature extraction path with the ego-attention network while preserving the structured HighwayEnv task. | notebooks/structured_highway/attention_dqn/ |
| notebooks/_shared/dqn_notebook_utils.py | Notebook adapter. Defines environment profiles, merges native reward configuration with optional custom wrappers, loads the selected DQN backend, and provides train/evaluate/display helpers. | All active structured DQN notebooks |
| src/deep_learning/DQN/adaptive_longitudinal.py | Optional structured wrappers for adaptive TTC-based speed control, rear-flow pressure, traffic-flow reward, TTC safety reward, potential-field reward, driver-aggressiveness signals, TTC observations, and lane-change safety. Disabled by default unless the notebook enables them. | Congested-traffic notebooks |
| src/deep_learning/DQN/congestion_diagnostics.py | Computes lane/TTC/flow labels and aggregates policy-risk diagnostics for structured evaluations. It is analysis code, not a training reward. | Congested-traffic analysis |
| notebooks/congested_traffic_policy/learned_behaviour_analysis.py | Post-training analysis for learned behavior and intervention/risk summaries. | run_learned_behaviour_analysis.py |

The cleanest structured ablation is
congested_traffic_four_experiments.ipynb: it changes only attention and
TTC safety reward. The other congested notebooks combine more terms and must
be interpreted as follow-up studies.

The structured PPO notebooks currently reference backend modules that are not
present in the visible source tree. Treat them as historical/incomplete until
those imports are restored or the notebooks are rewritten to use an available
backend.

## Laneless / unstructured track

These modules use the custom lane-free-v0 environment. They do not use lane
indices or discrete lane-change actions.

| File | Responsibility | Used by |
| --- | --- | --- |
| laneless highway env/lane_free_env.py | Custom wide-corridor/ring-road simulation, traffic models, continuous ego acceleration, collision/off-road handling, and laneless observation construction. | Laneless notebook and PPO/CBF scripts |
| scripts/laneless_script_config.py | Shared CLI/config merging. Applies JSON-file and inline JSON overrides without mutating the base configuration. | Training/evaluation entry points |
| scripts/laneless_training_registry.py | Resolves named training requests, checkpoint locations, and configuration fingerprints. | PPO/CBF training |
| scripts/laneless_evaluation_registry.py | Resolves evaluation requests, cache paths, manifests, and matching completed runs. | Laneless evaluation |
| scripts/cbf_projection.py | Exact two-dimensional NumPy/Torch projection for state-dependent linear CBF action constraints. Returns feasibility and fallback metadata. | ppo_cbf_env.py, projected PPO |
| scripts/ppo_cbf_env.py | Builds the laneless PPO environment wrapper, fixed-size CBF context, hard-filter execution, and safety metrics. | projected_ppo_cbf.py and progression runs |
| scripts/projected_ppo_cbf.py | PPO learner/policy path for nominal, hard-projected, and differentiable CBF variants. | run_ppo_cbf_progression.py |
| scripts/learnable_hocbf_params.py | Optional learnable HOCBF parameterization and actor-internalization support for differentiable variants. | Learnable CBF pilots |
| scripts/ppo_reward_safety.py | Installs the laneless Karalakou/additive reward and CBF safety-cost path into notebook/worker namespaces, including scalar geometry hardening. | PPO safety pilots |
| scripts/ppo_observation_variants.py | Adds picklable observation variants such as previous executed action to worker environments. | PPO observation pilots |
| scripts/run_ppo_cbf_progression.py | Orchestrates the named seven-policy progression, training metadata, and checkpoint output. | Canonical laneless study |
| scripts/evaluate_ppo_cbf_free.py | Evaluates checkpoints with the CBF removed from both the policy/runtime path when requested by the protocol. | True CBF-free package |
| scripts/evaluate_ppo_cbf_deployment.py | Deployment-style evaluation of a trained checkpoint with runtime safety instrumentation. | Deployment study |
| scripts/evaluate_ppo_cbf_counterfactuals.py | Runs controlled counterfactual CBF settings against a fixed policy/checkpoint. | CBF sensitivity study |
| scripts/evaluate_ppo_cbf_gain_grid_stage1.py | Sweeps first-stage HOCBF gains and records feasibility/safety outcomes. | Gain-grid pilot |
| scripts/evaluate_ppo_cbf_timing_pilot.py | Measures policy/filter/environment timing and failure behavior. | Timing pilot |
| scripts/evaluate_laneless_karalakou.py | Replays registered laneless checkpoints, aggregates KPI blocks, and writes evaluation manifests. | Karalakou/PPO result tables |
| scripts/cbf_ray_mask.py | Legacy ray-mask CBF path for the earlier DDPG formulation. Keep separate from the current HOCBF projection path. | Legacy DDPG comparisons |

Rendering scripts (render_laneless_* and render_ppo_*) consume a selected
checkpoint and evaluation configuration. They are downstream visualization
tools, not training entry points.

## Tests and reports

Tests are grouped by behavior rather than script filename:

- tests/test_lane_free_*: custom environment and traffic behavior;
- tests/test_cbf_*, tests/test_ppo_cbf*: geometry, projection, wrapper, and
  execution semantics;
- tests/test_learnable_hocbf_params.py and related pilot tests: learnable
  parameter/actor paths;
- report/audit scripts: manifests, KPI aggregation, and artifact audits.

Run targeted tests before a long experiment, for example:

    python -m pytest tests/test_cbf_projection.py tests/test_ppo_cbf_env.py

## Historical files

Files with suffixes such as -DSP000107027L.py, one-off debug names, and the
old DDPG scripts are retained for provenance but are not part of the active
run flow. They should be moved into an explicit archive/ area during the
repository split rather than receiving new experiment logic.
