# Diagnostic scenario registry

This is the canonical registry for the designed lane-free highway scenes and
the common state-bank strata used in policy, actor, critic, and CBF-filter
comparisons.

The primary suite for our B2 learned-policy comparison is the six-scene core
below. The renderer also contains four additional scenes that are useful for
expanded qualitative coverage.

## Core fixed diagnostic suite

These scenes are designed to expose passing decisions, gap rejection, pressure
handling, boundary recovery, and reactive safety. Each scene should be run
from the same initial condition and seed for every policy being compared.

| ID | Scenario | Intended diagnostic | Expected behavior |
| --- | --- | --- | --- |
| `safe_overtake_open_upper_gap` | Open Passing Gap | A clean side gap is available behind a slow leader. | Commit smoothly to the viable passing gap when appropriate. |
| `unsafe_overtake_fast_closing_upper` | Fast Closing Side Vehicle | A tempting side gap is closing quickly. | Reject the closing gap and preserve separation. |
| `boxed_in_hold_position` | Boxed In | Both nearby side gaps are occupied. | Avoid forcing a pass; hold or react conservatively. |
| `rear_pressure_escape` | Rear Pressure Escape | A fast rear vehicle closes while one side is cleaner. | Escape through the cleaner side without creating a new conflict. |
| `boundary_recovery_no_upper_squeeze` | Boundary Recovery | Ego starts near the road edge with a blocker ahead. | Recover inward and avoid the edge squeeze. |
| `sudden_lead_slowdown` | Sudden Lead Slowdown | The lead vehicle becomes sharply slower while side gaps are risky. | React safely without diving into occupied traffic. |

Notebook render hooks for these scenes use `steps=240` and `episodes=1` by
default. The scene definitions and vehicle placements live in
`scripts/render_policy_scenarios.py`.

## Extended rendering catalog

These existing renderer scenes are not part of the six-scene core, but can be
added when broader behavior coverage is wanted.

| ID | Scenario | Intended diagnostic |
| --- | --- | --- |
| `open_road_no_neighbors` | Open Road | Steady progress and low lateral drift without nearby traffic. |
| `narrow_gap_wait_or_upper_escape` | Tight-Slot Trap | Avoid diving into a tight slot; observe shield redirection. |
| `opposite_edge_recovery` | Opposite-Edge Recovery | Recover inward from the opposite road edge. |
| `staggered_gap_selection` | Staggered Traffic | Select the later clean gap rather than the closer blocked side. |

## Common state-bank strata

For same-state actor and critic comparisons, use a deterministic common bank
with these five strata:

| Stratum | Meaning |
| --- | --- |
| `normal` | Ordinary states away from safety and density extremes. |
| `near_boundary` | States close to a road or CBF safety boundary. |
| `intervention` | States where the CBF is likely to modify the policy action. |
| `dense_traffic` | States with unusually high nearby-vehicle density. |
| `overtaking` | States involving a blocker, a passing opportunity, or an overtake setup. |

The state bank must pass the exact same observations and reconstructed simulator
states through every policy. Occupancy-only comparisons should remain separate
from fixed-state comparisons because a policy can improve by visiting different
states rather than by changing its action at the same state.

## Recommended use for B2

For B2.1, B2.2, and B2.3, report:

1. the six core scenes as paired qualitative rollouts;
2. the five state-bank strata for fixed-state actor, critic, and CBF-action
   comparisons; and
3. the four extended scenes only as an optional sensitivity suite.

Critic values should be compared primarily by within-variant calibration and
ranking because reward shaping differs across B2 variants. Actor outputs should
be separated into raw policy action, CBF-projected action, and the correction
between them.

## Sources

- `notebooks/lanelessKaralakou.ipynb`, section E.2.1--E.2.6
- `scripts/render_policy_scenarios.py`
- `docs/cbf_factorial_ablation.md`
