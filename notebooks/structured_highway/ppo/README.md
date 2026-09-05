# Historical structured PPO notebooks

The four notebooks in this directory are lane-based historical experiment
narratives. They are retained so the original research progression is visible,
but their original PPO implementation modules were not included in the
source-only DQN repository. They are therefore not fresh-clone entry points
until the missing backend is restored or rewritten against the current code.

## `PPO_trials.ipynb` — PPO Overtake Lab

This is an interactive overtaking experiment. It exposes reward shaping,
traffic density, steering range, lead-vehicle spacing, and PPO settings; trains
`overtake_push_v1`; evaluates base, dense-traffic, and narrow-highway variants
for ten episodes each; and renders three human-view episodes. The default run
uses 30,000 timesteps, 24 environments, seed 42, three lanes, five vehicles per
lane, and an overtaking-focused reward.

Missing dependency: `ppo_overtake_lab` under the historical PPO source tree.
Result status: no result is stored in this repository.

## `Attention_PPO_baseline.ipynb` — Attention PPO Baseline

This notebook configures an attention PPO policy for native `highway-v0` with
24 environments, 7,680 timesteps, eight evaluation episodes, seed 42, one
attention head, and a 32-dimensional feature representation. It saves an
evaluation JSON plus summary and episode-trace plots, and can optionally render
human-view episodes.

Missing dependency: `attention_ppo`. Result status: no result is stored here.

## `Hybrid_PPO_baseline.ipynb` — Hybrid PPO Baseline

This notebook trains a discrete-lane plus continuous-throttle PPO baseline for
10,000 timesteps and evaluates five episodes. Lane-action intent, lane-safety
checks, and throttle-safety checks are enabled. It writes a model, TensorBoard
directory, and `summary.json` below `artifacts/ppo/hybrid_ppo_baseline/`.

Missing dependency: the historical `elurant_ppo` module. Result status: no
result is stored here.

## `Paper_PPO_reproduction.ipynb` — Paper PPO Reproduction

This notebook recreates the paper-style PPO configuration for 10,000 timesteps
with three environments, seed 42, five short evaluation episodes, and an
optional 1,000-episode learned-policy evaluation. It records best/final models,
evaluation JSON, logs, and plots under
`artifacts/ppo/paper_ppo_reproduction/`.

Missing dependency: the historical `paper_ppo_reproduction` module. Result
status: no result is stored here.

## Promotion rule

Do not mix these historical PPO descriptions with the supported DQN results.
To promote one, restore its backend, run it from the repository root, record
the exact environment/reward configuration and dependency versions, and add a
result summary with the evaluation seed and episode count.
