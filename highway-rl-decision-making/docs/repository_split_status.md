# Structured repository status

Updated: 2026-09-04

The parent checkout now retains a self-contained structured/lane-based
research source and its curated documentation. The independent structured
extraction is kept as a separate local Git root so its history can be managed
independently.

## Retained roots

| Repository | Local root | Current local HEAD | Scope |
| --- | --- | --- | --- |
| Structured parent | `./` | current `main` | Structured HighwayEnv source, notebooks, documentation, and curated DQN artifacts. |
| Structured extraction | `../../laned-highway-rl/` | `eb03796` | Standalone structured DQN, attention DQN, congested studies, tests, and documentation. |

## Ownership boundary

The retained structured source owns:

- `src/deep_learning/DQN/`;
- `notebooks/structured_highway/`;
- `notebooks/congested_traffic/`;
- `notebooks/_shared/`;
- the structured tests and documentation; and
- curated `artifacts/dqn/` summaries.

Obsolete alternate-track implementations, notebooks, scripts, tests,
configuration files, generated result trees, and temporary caches were
removed from this checkout. No sibling source checkout is required by the
retained code or notebooks.

## Validation performed

The cleanup was checked with:

- JSON parsing for all retained notebooks;
- Python AST parsing for retained source modules;
- local Markdown-link validation;
- DQN import/help smoke checks;
- staged-diff whitespace checks; and
- a scan for tracked and non-virtual-environment cache files.

The structured environment tests cover profile reset/step behavior,
deterministic seeding, observation-space bounds, and the main opt-in wrappers.
Full notebook training is intentionally not part of the fast validation path
because it creates model and TensorBoard output and can be lengthy.

## Handoff requirements

1. Recreate a clean virtual environment from the retained requirements.
2. Run the targeted tests and a short training/evaluation smoke experiment.
3. Pin the external `rl-agents` dependency before promoting the planning
   notebook as a reproducible entry point.
4. Preserve configuration, seed, checkpoint, artifact-root, and evaluation
   metadata for every reported result.
