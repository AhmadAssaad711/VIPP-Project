<div align="center">

# Highway RL Decision Making

**Structured highway decision-making research, with the unstructured track maintained separately.**

[Project Workspace](highway-rl-decision-making/) | [Notebook Map](highway-rl-decision-making/notebooks/) | [Paper](highway-rl-decision-making/docs/paper/highway-rl-decision-making-paper.pdf) | [Setup](#setup)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Notebook-first](https://img.shields.io/badge/Notebook--first-research-111111?style=flat-square)
![RL](https://img.shields.io/badge/RL-highway_decision_making-0B7285?style=flat-square)
![Paper](https://img.shields.io/badge/Paper-included-495057?style=flat-square)

</div>

---

## Focus

This repository presents work on high-level decision making for autonomous driving:

- lane-based highway policies
- dense traffic behavior
- safety-aware reward design
- planning comparisons for decision-level behavior

The repo is intentionally notebook-first. Each notebook captures a specific experiment, baseline, reproduction, or environment study.

## Research Flow

The structured source retained in this workspace covers baseline and
attention DQN, controlled reward/safety studies in congested traffic, and
the historical planning notebook. The unstructured SafeRL track was extracted
to the standalone `laneless-karalakou-cbf` repository and removed from this
source tree.

The detailed notebook-to-script flow is documented in
[research_flow.md](highway-rl-decision-making/docs/research_flow.md).

## Split repositories

The local staging roots are intentionally ignored by this parent repository:

- [laned-highway-rl/](laned-highway-rl/) contains the structured, lane-based
  HighwayEnv experiments.
- [laneless-karalakou-cbf/](laneless-karalakou-cbf/) is the preserved
  standalone source for the custom lane-free environment and PPO/CBF
  experiments.

Each staging root has its own `.git` directory and can be cloned, tested, and
published independently. The parent checkout no longer contains the
unstructured implementation or its generated result trees. The extraction
boundary and remaining publication gates are recorded in
[repository_split_status.md](highway-rl-decision-making/docs/repository_split_status.md).

Associated paper: [`highway-rl-decision-making-paper.pdf`](highway-rl-decision-making/docs/paper/highway-rl-decision-making-paper.pdf)

## Research Map

| Area | Core Question | Entry Points |
| --- | --- | --- |
| Structured highway RL | Baseline DQN and PPO behavior in lane-based highway settings. | [`baseline_dqn`](highway-rl-decision-making/notebooks/structured_highway/baseline_dqn/baseline_dqn.ipynb), [`attention_dqn`](highway-rl-decision-making/notebooks/structured_highway/attention_dqn/attention_dqn.ipynb), [`PPO_trials`](highway-rl-decision-making/notebooks/structured_highway/ppo/PPO_trials.ipynb) |
| Attention and hybrid PPO | Improvements over baseline policy structure. | [`Attention_PPO_baseline`](highway-rl-decision-making/notebooks/structured_highway/ppo/Attention_PPO_baseline.ipynb), [`Hybrid_PPO_baseline`](highway-rl-decision-making/notebooks/structured_highway/ppo/Hybrid_PPO_baseline.ipynb) |
| Congested traffic | Baseline extensions under dense traffic and safety constraints. | [`congested_traffic_policy`](highway-rl-decision-making/notebooks/congested_traffic/congested_traffic_policy.ipynb), [`potential_field_reward_test`](highway-rl-decision-making/notebooks/congested_traffic/congested_reward_safety_factor_study.ipynb) |
| Planning comparison | How do planning-based methods compare as decision baselines? | [`CEM_planning_trials`](highway-rl-decision-making/notebooks/planning/CEM_planning_trials.ipynb) |

## Repository Layout

```text
highway-rl-decision-making/
  README.md
  requirements.txt
  docs/
    paper/
  notebooks/
    structured_highway/
    congested_traffic/
    planning/
```

## What Is Included

The active reusable implementation is in the nested project workspace and
contains the structured DQN modules, structured notebook flow, planning
notebook, and associated paper.

The nested project remains a read-only migration source while the two
track-specific repositories are reviewed. Do not add new experiments to the
parent workspace when they belong in one of the staging repositories.

- clean notebook portfolio
- associated paper
- reproducible environment requirements
- grouped experiments by research theme
- public-facing structure around Highway RL Decision Making

## What Is Excluded

- unrelated practice problems
- old experimental folders
- vendored external repositories
- the extracted unstructured SafeRL implementation and result trees
- generated logs, videos, checkpoints, and artifacts
- material outside decision-level highway RL

Python caches and local virtual environments are also excluded from the
source package. The standalone laneless repository remains beside this
checkout; it is not part of the parent source tree.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r highway-rl-decision-making\requirements.txt
```

Then open the notebooks from:

```text
highway-rl-decision-making/notebooks/
```
