# VIPP Project — Path Following on Racetrack

Comparison of two RL steering controllers on the `highway-env` racetrack:

| # | Approach | Script | Description |
|---|----------|--------|-------------|
| 1 | **Basic RL Agent** | `2_basic_rl_agent/basic_rl_agent.py` | Tabular Q-learning with state = (lateral error, heading error). |
| 2 | **Physics-Informed Agent** | `3_physics_informed_agent/physics_informed_agent.py` | Tabular Q-learning with state = (lateral error, heading error, **lane curvature κ**). |

## Repository Structure

```
vipp_project/
│
├── 2_basic_rl_agent/
│   └── basic_rl_agent.py
│
├── 3_physics_informed_agent/
│   └── physics_informed_agent.py
│
├── results/
│   ├── basic_rl_agent/              # training curves + policy heatmap
│   └── physics_informed_agent/      # training curves + κ-slice heatmaps
│
└── README.md
```

## Running

Each script is self-contained. Run from the repo root:

```bash
python 2_basic_rl_agent/basic_rl_agent.py
python 3_physics_informed_agent/physics_informed_agent.py
```

Figures are automatically saved to the corresponding `results/<approach>/` folder.
