# Repository split plan

## Current state (2026-09-04)

The source-only extraction is complete in two independent local Git roots
beside this workspace:

- `../../laned-highway-rl/` — structured/lane-based HighwayEnv code and
  notebooks;
- `../../laneless-karalakou-cbf/` — the custom lane-free environment, PPO/CBF
  code, tests, and lightweight result provenance.

The parent checkout now retains the structured/lane-based source and planning
notebook only. The laneless implementation, laneless notebooks, PPO/CBF
scripts/tests, result manifests, and generated laneless output were removed
from the parent after the standalone extraction was verified. The staging
repositories do not contain virtual environments, TensorBoard caches, raw
checkpoints, videos, or generated CSV trees. See
[repository_split_status.md](repository_split_status.md) for the exact
boundary and validation results.

The extraction is source-complete for the active code paths, but it is not yet
a publication handoff: remotes still need to be configured, the standalone
laneless result provenance must be reconciled, and historical or
notebook-only helpers should be promoted only after their dependencies are
made explicit.

The split boundary is scientific and architectural:

- laned-highway-rl: native lane-based HighwayEnv and the structured study of
  baseline DQN, attention representation, and reward/safety-term integration;
- laneless-karalakou-cbf: the custom lane-free environment and the current
  continuous PPO, Karalakou reward, and CBF formulation.

The parent checkout is now a structured source/archive. Do not copy virtual
environments, TensorBoard caches, checkpoints, videos, or the full raw
artifacts tree into either Git repository.

## Proposed laned repository

Copy and then flatten these paths so they become repository-root paths:

| Current path | Destination role |
| --- | --- |
| highway-rl-decision-making/src/deep_learning/DQN/ | Structured DQN backends, reward/observation wrappers, and diagnostics. |
| highway-rl-decision-making/notebooks/structured_highway/ | Baseline, attention, and structured PPO narratives. |
| highway-rl-decision-making/notebooks/congested_traffic/ | Controlled four-way ablation and follow-up reward/safety studies. |
| highway-rl-decision-making/notebooks/_shared/dqn_notebook_utils.py | Structured notebook adapter; rename to structured_dqn_utils.py if desired. |
| Relevant structured tests | DQN backend, wrapper, diagnostic, and structured environment tests. |
| highway-rl-decision-making/artifacts/dqn/ | Curated summaries/figures only; keep raw checkpoints external. |
| highway-rl-decision-making/docs/paper/ | Copy only if the paper is maintained in this repository. |

The native environment contract is highway-v0 with lane counts, lane indices,
discrete lane-level actions, and lane-based TTC/flow diagnostics. The names
semi_unstructured and unstructured_stress do not move this code to laneless:
they still use the structured HighwayEnv API.

## Laneless source of record

The standalone `../../laneless-karalakou-cbf/` repository is the source of
record for the custom `lane-free-v0` simulator, continuous PPO, Karalakou
reward, CBF projection, evaluations, tests, and lightweight provenance. The
parent copy was intentionally removed after extraction. Shared dependencies
such as Gymnasium, NumPy, PyTorch, Stable-Baselines3, and HighwayEnv should be
installed from the standalone repository's requirements rather than copied
back into this structured source.

## Shared dependency policy

HighwayEnv is a normal dependency of the retained structured source. The
standalone laneless repository should pin its own compatible revision rather
than relying on a checkout nested in this parent.

Create requirements files from the actual imports. The retained parent
requirements cover the structured DQN and notebook stack; the standalone
laneless repository owns its CBF, QP/testing, rendering, and worker
dependencies.

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
   importable laneless module; update the standalone notebook to call that
   module.
5. Replace fragile notebook sys.path assumptions with repository-root imports
   or a small installable package in each new repository.
6. Add a manifest schema containing track, environment ID, dependency revision,
   code revision, observation/action contract, full reward/CBF configuration,
   seed set, checkpoint, artifact root, and evaluation protocol.
7. Run JSON parsing, import checks, environment smoke tests, targeted unit
   tests, and one short training/evaluation smoke run in each repository.
8. Review the staged parent deletions, commit the parent and standalone
   repository changes separately, and push each repository to its intended
   remote.

The split is complete when a fresh clone of either repository can follow its
README from setup to a short smoke experiment without importing code from the
other track.
