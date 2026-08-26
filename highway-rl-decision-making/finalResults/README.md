# Final results package

This folder contains the 50k PPO pilot-study artifacts for all seven variants. The original files under `artifacts/` were copied, not moved or deleted.

## Contents

- `evaluation/true_cbf_free/` contains the true CBF-free evaluation: aggregate KPIs, episode/block/progress CSVs for every variant, metadata, and the nominal render. “CBF-free” means the CBF was removed from both the environment and the policy path.
- `videos/true_cbf_free/` contains three reproducible videos for each policy (seeds `1100001`, `1100002`, and `1100003`), plus a preview PNG and JSON summary for every video.
- `evaluation/external_cbf_on_off_pilot/` contains the earlier external-CBF ON/OFF KPI tables, retained as a reference and kept separate from the true CBF-free evaluation.
- `models_and_checkpoints/` contains each seed-307 model, run configuration, training metadata/logs, monitor files, and the checkpoint directory. The nominal run has `rollout_50000_steps.zip`; the six CBF-variant source checkpoint directories were empty at packaging time.
- `manifest.json` records the variant mapping, source artifact paths, and packaged result locations.

The seven variants are B1 nominal, B2.1 non-differentiable reward, B2.2 non-differentiable reward with detached actor term, B2.3 non-differentiable detached actor-only, B3.1 differentiable reward-only, B3.2 differentiable reward plus actor, and B3.3 differentiable actor-only.
