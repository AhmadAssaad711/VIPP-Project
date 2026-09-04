# Repository split status

Updated: 2026-09-04

This directory now retains the structured/lane-based source and planning
notebook only. The two track-specific repositories are staged as local sibling
directories and are ignored by the parent repository so that their Git
histories remain separate. The laneless SafeRL implementation and generated
results were removed from this parent after the standalone extraction was
verified.

## Staging roots

| Repository | Local root | Current local HEAD | Scope |
| --- | --- | --- | --- |
| Laned Highway RL | `../../laned-highway-rl/` | `0206a33` | Native lane-based `highway-v0`, structured DQN/attention DQN, congested-traffic studies, and structured notebooks. |
| Laneless Karalakou CBF | `../../laneless-karalakou-cbf/` | `4628c01` | Custom `lane-free-v0`, continuous PPO, Karalakou reward, CBF projection, laneless evaluations, and tests. |

Both paths resolve to independent Git roots with clean local working trees.
The extraction commits were `b79fd88` and `a9a8227`; the current heads also
include the standalone documentation and development-requirements pass. The
files copied into the staging roots were checked against the source by
SHA-256 for the active code and notebook paths.

## Ownership boundary

The laned repository owns:

- `src/deep_learning/DQN/`;
- `notebooks/structured_highway/`;
- `notebooks/congested_traffic/`;
- the shared structured notebook adapter under `notebooks/_shared/`.

The standalone laneless repository is the source of record for:

- the custom `lane-free-v0` environment;
- the laneless notebooks and continuous PPO/CBF formulation;
- the PPO/CBF training, evaluation, rendering, audit, and registry code;
- laneless/CBF tests and configuration; and
- lightweight result provenance.

No copy of those implementation or result paths remains in this parent.

## Intentionally not copied

The staging repositories exclude disposable or unresolved material:

- `.venv*`, Python caches, TensorBoard event caches, logs, videos, raw model
  checkpoints, generated CSV/plot trees, and temporary notebook-cell files;
- `-DSP000107027L` snapshots and the legacy duplicate environment file;
- the vendored HighwayEnv gitlink, which is consumed through the declared
  `highway-env` dependency instead of being duplicated in the laneless repo;
- the planning CEM notebook, which depends on an `rl-agents` checkout that is
  not part of this source tree and needs an explicit dependency decision; and
- the paper, which remains as a historical combined-project deliverable until
  its final ownership is confirmed.

These omissions are deliberate. They prevent a clean clone from silently
depending on a local cache or on the other research track.

## Validation performed

Using the available system Python 3.13 environment:

- structured DQN modules imported successfully from the laned staging root;
- `lane-free-v0` was registered, reset with a fixed seed, stepped once, and
  returned a valid continuous two-action space from the laneless staging root;
- both staging trees passed `compileall`; and
- source/staging file hashes matched for the extracted active code, notebooks,
  laneless scripts, and tests.

## Parent cleanup

The parent cleanup staged removal of the laneless environment gitlink and
source files, laneless notebooks, PPO/CBF scripts and tests, laneless
configuration/result trees, laneless-only handoff notes, mixed SAC/laneless
notebook content, historical top-level PPO result directories, and generated
laneless artifacts. The structured `artifacts/dqn/` tree was retained. Local
Python/TensorBoard/pytest caches in the combined source were also removed.

The workspace `.venv` was not used because its `pyvenv.cfg` still points to the
removed `C:\Program Files\Python312\python.exe`. Recreate an environment from
the destination repository's requirements before running training or the full
test suite.

## Remaining release gates

1. Configure the intended remote for each independent repository and push only
   after reviewing its first post-extraction commit.
2. Reconcile the standalone laneless result manifests and evaluation metadata
   before using that repository's package for publication claims.
3. Extract the notebook-owned Karalakou reward wrapper into an importable
   module before treating the notebook as a fully reproducible API.
4. Decide whether the CEM planner belongs in the laned repository and pin or
   vendor its `rl-agents` dependency if it is promoted.
5. Recreate clean virtual environments and run the targeted test suites from
   fresh clones of both repositories.

Until these gates are closed, the source checkout and its staged repositories
are migration artifacts, not publication-ready result packages.
