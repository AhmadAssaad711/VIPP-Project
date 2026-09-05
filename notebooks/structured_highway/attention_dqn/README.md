# Structured attention DQN notebook

Notebook: [`attention_dqn.ipynb`](attention_dqn.ipynb)

## Purpose

This notebook tests whether an ego-centered attention representation improves
decision making relative to the baseline DQN. The environment, reward,
training budget, seeds, and discrete action contract are held fixed; the
representation is the intended controlled factor. The reusable implementation
is [`attention_dqn.py`](../../../src/deep_learning/DQN/attention_dqn.py).

## Protocol

The default environment is the same `structured_baseline` protocol used by
the baseline notebook: three lanes, 20 vehicles, 40-step duration, density
1.0, IDM traffic, simulation frequency 15, and policy frequency 1. Observations
contain five relative kinematic entities and actions use `DiscreteMetaAction`.
The native reward uses collision `-1.0`, right-lane `0.1`, high-speed `0.4`,
and lane-change `0.0` terms with normalization enabled.

Training uses 20,000 timesteps, seed 42, a `2.5e-4` learning rate, discount
`0.95`, replay buffer size 50,000, and the same DQN schedule as the baseline.
The attention policy uses a 64-dimensional feature representation, two heads,
`64,64` entity embeddings, and a `64,64` Q-network head.

## Cell guide

1. Discover the repository root and load the shared attention backend.
2. Build the common training and saved-model evaluation configurations.
3. Train or resume the attention policy and display training metrics.
4. Evaluate the saved checkpoint for 1,000 episodes using seed `10042`.
5. Run 100-episode TTC/lane-gap diagnostics and save per-step traces.
6. Optionally train the aggressive TTC-safety follow-up and render a policy
   panel; those settings are separate from the clean representation test.

## Outputs

The default output root is `artifacts/dqn/attention_dqn/`. Expected files are
`summary.json`, a model checkpoint, per-episode evaluation JSON, summary and
detailed plots, and optional congestion-diagnostic files. Generated outputs are
ignored by Git.

## Result status

No attention-DQN result artifact is present in the clean staging repository or
in the source snapshot used for this documentation pass. Consequently, this
README does not claim an attention-versus-baseline improvement. Run the
notebook with the baseline protocol, then promote a result only with the
checkpoint, environment configuration, seed, episode count, and metric table.
