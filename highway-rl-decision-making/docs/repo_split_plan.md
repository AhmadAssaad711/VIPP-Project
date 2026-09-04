# Structured repository plan

## Current state (2026-09-04)

This checkout retains the structured/lane-based HighwayEnv source, notebooks,
documentation, and curated structured artifacts. The independent structured
extraction is available beside this workspace at `../../laned-highway-rl/`.
The checkout is self-contained and does not require source files from a
sibling repository.

The retained work covers baseline DQN, attention DQN, congested-traffic
experiments, structured reward and safety wrappers, diagnostics, and the
planning notebook as a documented historical comparison.

## Ownership boundary

The structured repository owns:

- `src/deep_learning/DQN/` for reusable learners, wrappers, and diagnostics;
- `notebooks/structured_highway/` for baseline and attention narratives;
- `notebooks/congested_traffic/` for controlled reward and safety studies;
- `notebooks/_shared/` for notebook-facing structured configuration helpers;
- `docs/` for the research flow, paper context, and provenance; and
- `artifacts/dqn/` for curated summaries and figures only.

The native environment contract is `highway-v0` with lane counts, lane
indices, discrete lane-level actions, and lane-based TTC/flow diagnostics.
The historical profile names `semi_unstructured` and `unstructured_stress`
still use this structured HighwayEnv contract.

## Generated data policy

Virtual environments, Python caches, TensorBoard events, logs, videos, raw
checkpoints, generated CSV trees, and temporary notebook-cell files stay
outside version control. Curated manifests, configuration snapshots, KPI
tables, and selected figures may be committed under the structured artifact
root.

Every promoted result should record the environment revision, profile,
observation and action configuration, enabled reward/safety wrappers,
training budget, seed, checkpoint path, evaluation seeds, episode count, and
reported metrics.

## Validation and handoff

Before a release or fresh-clone handoff:

1. recreate the virtual environment from `requirements.txt`;
2. run the DQN import/help checks and targeted unit tests;
3. parse the retained notebooks and validate their local links;
4. run a short training/evaluation smoke experiment; and
5. review generated output to ensure that only curated provenance is staged.

The planning CEM notebook remains historical because it depends on an
external `rl-agents` checkout. It should be promoted only after that
dependency is pinned or vendored and its execution path is tested.
