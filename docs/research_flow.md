# Structured research flow

This repository contains the lane-based `highway-v0` track only. The central
comparison asks whether an ego-centered representation and carefully isolated
reward/safety terms improve decision making when the environment and protocol
are held fixed.

## Execution order

1. Run the baseline DQN notebook to establish the standard structured policy.
2. Run the attention DQN notebook with the same environment profile, reward,
   training budget, and evaluation seeds.
3. Run `congested_traffic_four_experiments.ipynb` for the pre-registered 2x2
   comparison: attention on/off by TTC safety reward on/off.
4. Use the other congested notebooks as labelled follow-up studies. They add
   combinations such as adaptive speed, flow pressure, potential-field
   shaping, driver aggressiveness, TTC observations, and lane-change safety.

The four-experiment notebook is the clean causal comparison. Its controls
intentionally disable the additional follow-up terms so that the two stated
factors remain interpretable.

## Code path

```text
notebook
  -> notebooks/_shared/dqn_notebook_utils.py
      -> src/deep_learning/DQN/elurant_dqn.py
         or src/deep_learning/DQN/attention_dqn.py
          -> HighwayEnv highway-v0
          -> optional adaptive_longitudinal.py wrappers
          -> congestion_diagnostics.py for post-run labels
```

The baseline backend uses the native structured feature representation. The
attention backend uses an ego-centered extractor while preserving the same
structured observation and action contract. Reward wrappers are opt-in and
must be recorded in the run configuration.

## Quick checks

From the repository root:

```powershell
python src\deep_learning\DQN\elurant_dqn.py --help
python src\deep_learning\DQN\attention_dqn.py --help
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q src notebooks
```

The help commands validate imports without starting training, while the
laned-environment tests check profile reset/step behavior, deterministic
seeding, observation-space bounds, and the main opt-in wrappers. A full
notebook run is intentionally not part of the smoke check because it creates
model and TensorBoard outputs and can be lengthy.

## Reproducibility

Every comparison must record:

- the HighwayEnv version;
- environment profile and traffic configuration;
- observation and action configuration;
- enabled reward/safety wrappers;
- training budget, seed, and device;
- checkpoint path; and
- evaluation seeds, episode count, and metrics.

Compare only runs with matching state semantics, action spaces, environment
contracts, and evaluation protocols.
