# Laned Highway RL

Standalone repository for the structured/lane-based track of the Highway RL
project. It studies baseline DQN, ego-attention DQN, and controlled
reward/safety integration in native HighwayEnv `highway-v0`.

This repository contains only the structured/lane-based track. The historical
combined migration source is not a runtime dependency.

## Scientific flow

1. Run `notebooks/structured_highway/baseline_dqn/baseline_dqn.ipynb`.
2. Run `notebooks/structured_highway/attention_dqn/attention_dqn.ipynb` with
   the same traffic, reward, seed, and evaluation protocol.
3. Run `notebooks/congested_traffic/congested_traffic_four_experiments.ipynb`
   for the controlled attention x TTC-safety 2x2 study.
4. Use the other congested-traffic notebooks for explicitly labelled
   follow-up reward/control combinations.
5. Use `notebooks/planning/CEM_planning_trials.ipynb` only as a historical
   planning comparison until its external dependency is pinned.

The notebooks are experiment narratives. Reusable code lives in
`src/deep_learning/DQN/` and `notebooks/_shared/dqn_notebook_utils.py`.

See [notebooks/README.md](notebooks/README.md) for the notebook map and
[docs/research_flow.md](docs/research_flow.md) for the controlled comparison
protocol and validation commands. The retained graph and diagram index is in
[docs/visualizations.md](docs/visualizations.md).

The associated paper is available at
[docs/paper/highway-rl-decision-making-paper.pdf](docs/paper/highway-rl-decision-making-paper.pdf).

## Environment and code

- Environment: native HighwayEnv `highway-v0`.
- Action/observation semantics: lane-based, discrete HighwayEnv policy.
- Learners: `elurant_dqn.py` baseline and `attention_dqn.py`.
- Reward/control terms: `adaptive_longitudinal.py`.
- Analysis: `congestion_diagnostics.py`.

## Setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open notebooks from this repository root so their project-root discovery
finds `src/` and `notebooks/`. Generated artifacts should stay outside Git.

For a dependency/import check before opening a notebook:

```powershell
python src\deep_learning\DQN\elurant_dqn.py --help
python src\deep_learning\DQN\attention_dqn.py --help
python -m unittest discover -s tests -p "test_*.py" -v
```

The structured PPO notebooks are retained as historical narratives. They
currently reference backend modules that are not present in this repository;
restore those backends or rewrite the notebooks before presenting them as
reproducible entry points.

The CEM planning notebook is retained as a historical comparison, but it
requires an unpinned external `rl-agents` checkout and is not a clean
reproducible entry point yet.
