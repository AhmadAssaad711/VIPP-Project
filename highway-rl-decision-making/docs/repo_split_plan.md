# Repository split plan

The split boundary is scientific and architectural:

- laned-highway-rl: native lane-based HighwayEnv and the structured study of
  baseline DQN, attention representation, and reward/safety-term integration;
- laneless-karalakou-cbf: the custom lane-free environment and the current
  continuous PPO, Karalakou reward, and CBF formulation.

The current checkout should remain as a read-only migration source until both
new repositories pass their smoke tests. Do not copy virtual environments,
TensorBoard caches, checkpoints, videos, or the full raw artifacts tree into
either Git repository.

## Proposed laned repository

Copy and then flatten these paths so they become repository-root paths:

| Current path | Destination role |
| --- | --- |
| highway-rl-decision-making/src/deep_learning/DQN/ | Structured DQN backends, reward/observation wrappers, and diagnostics. |
| highway-rl-decision-making/notebooks/structured_highway/ | Baseline, attention, and structured PPO narratives. |
| highway-rl-decision-making/notebooks/congested_traffic/ | Controlled four-way ablation and follow-up reward/safety studies. |
| highway-rl-decision-making/notebooks/congested_traffic_policy/ | Structured learned-behavior analysis. |
| highway-rl-decision-making/notebooks/_shared/dqn_notebook_utils.py | Structured notebook adapter; rename to structured_dqn_utils.py if desired. |
| Relevant structured tests | DQN backend, wrapper, diagnostic, and structured environment tests. |
| highway-rl-decision-making/artifacts/dqn/ | Curated summaries/figures only; keep raw checkpoints external. |
| highway-rl-decision-making/docs/paper/ | Copy only if the paper is maintained in this repository. |

The native environment contract is highway-v0 with lane counts, lane indices,
discrete lane-level actions, and lane-based TTC/flow diagnostics. The names
semi_unstructured and unstructured_stress do not move this code to laneless:
they still use the structured HighwayEnv API.

## Proposed laneless repository

Copy and then flatten these paths:

| Current path | Destination role |
| --- | --- |
| highway-rl-decision-making/laneless highway env/lane_free_env.py | Custom lane-free-v0 simulator and Gymnasium registration. |
| highway-rl-decision-making/laneless highway env/demo_lane_free.py and README.md | Environment smoke/demo path. |
| highway-rl-decision-making/notebooks/laneless_unstructured/ | Environment configuration and smoke notebook. |
| highway-rl-decision-making/notebooks/lanelessKaralakou.ipynb | Canonical PPO/CBF scientific narrative. |
| Active scripts with laneless_, cbf_, ppo_, render_laneless_, render_ppo_, or run_ppo_cbf_ names | Training, projection, evaluation, rendering, registries, and audits. |
| Laneless/CBF tests | Environment, projection, wrapper, timing, and learnable-HOCBF tests. |
| highway-rl-decision-making/configs/current_mtm_live.json and ppo_cbf_policy_rate.json | Laneless runtime configuration. |
| highway-rl-decision-making/finalResults/ | Curated seven-variant laneless package after episode-count provenance is fixed. |
| Outer ppo_* result directories | Historical laneless study manifests only; keep large raw outputs outside Git. |
| highway-rl-decision-making/docs/cbf_factorial_ablation.md and diagnostic_scenarios.md | CBF study protocol and diagnostic scenario registry. |

The laneless environment contract is lane-free-v0 with continuous [a_x, a_y]
ego acceleration and Cartesian neighbor state. CBF projection remains in this
repository initially because it is coupled to the laneless physical state and
observation context. Extract a shared safety package only after that API has
stabilized.

## Shared dependency policy

HighwayEnv is currently an upstream v1.11 gitlink nested under the laneless
environment directory. Keep one pinned upstream revision as a submodule or
normal dependency in both repositories; do not fork or duplicate it inside
the laneless implementation. The current nested checkout has no source
changes, aside from a local .DS_Store deletion.

Create separate requirements files from the actual imports. Both repositories
may share Gymnasium, NumPy, PyTorch, Stable-Baselines3, and HighwayEnv, but
the laned repository needs its structured DQN stack while the laneless
repository needs CBF projection, QP/testing, rendering, and worker support.

## Archive policy

Move historical copies and one-off experiments into an explicit archive/
area or a separate results archive:

- files with -DSP000107027L suffixes;
- old DDPG ray-mask/guided-CBF experiments when they are not part of the
  current PPO formulation;
- pre_reset_backup_* snapshots and temporary notebook cell files;
- raw checkpoints, TensorBoard event files, videos, and generated CSV trees.

Archive metadata should retain the original path, commit, environment
configuration, seed, and reason for archival. A filename suffix alone is not
enough provenance.

## Migration sequence

1. Finish documentation and remove disposable caches in this source checkout.
2. Freeze a migration commit or tag so paths and manifests can be referenced.
3. Build source-only laned and laneless working copies without environments
   or generated outputs.
4. Extract KaralakouRewardWrapper and related notebook-only helpers into an
   importable laneless module; update the notebook to call that module.
5. Replace fragile notebook sys.path assumptions with repository-root imports
   or a small installable package in each new repository.
6. Add a manifest schema containing track, environment ID, dependency revision,
   code revision, observation/action contract, full reward/CBF configuration,
   seed set, checkpoint, artifact root, and evaluation protocol.
7. Run JSON parsing, import checks, environment smoke tests, targeted unit
   tests, and one short training/evaluation smoke run in each repository.
8. Only after those checks pass, archive or remove the combined workspace and
   push the two repositories to their intended remotes.

The split is complete when a fresh clone of either repository can follow its
README from setup to a short smoke experiment without importing code from the
other track.
