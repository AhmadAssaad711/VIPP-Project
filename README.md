# Laned Highway RL

This repository is the structured/laned track extracted from the combined
workspace. It studies baseline DQN, ego-attention DQN, and controlled
reward/safety integration in native HighwayEnv highway-v0.

## Scientific flow

1. Run notebooks/structured_highway/baseline_dqn/baseline_dqn.ipynb.
2. Run notebooks/structured_highway/attention_dqn/attention_dqn.ipynb with
   the same traffic, reward, seed, and evaluation protocol.
3. Run notebooks/congested_traffic/congested_traffic_four_experiments.ipynb
   for the controlled attention x TTC-safety 2x2 study.
4. Use the other congested-traffic notebooks for explicitly labelled
   follow-up reward/control combinations.

The notebooks are experiment narratives. Reusable code lives in
src/deep_learning/DQN/ and notebooks/_shared/dqn_notebook_utils.py.

## Environment and code

- Environment: native HighwayEnv highway-v0.
- Action/observation semantics: lane-based, discrete HighwayEnv policy.
- Learners: elurant_dqn.py baseline and attention_dqn.py.
- Reward/control terms: adaptive_longitudinal.py.
- Analysis: congestion_diagnostics.py.

The profiles named semi_unstructured and unstructured_stress remain in the
structured repository because they still use lane counts and lane-based
HighwayEnv semantics.

## Setup

    python -m venv .venv
    .venv/Scripts/Activate.ps1
    python -m pip install -r requirements.txt

Open notebooks from this repository root so their project-root discovery
finds src/ and notebooks/. Generated artifacts should stay outside Git.

The source workspace contains the full combined documentation and split map;
the structured PPO notebooks in this extracted snapshot still reference
backends that must be restored or rewritten before they are reproducible.
