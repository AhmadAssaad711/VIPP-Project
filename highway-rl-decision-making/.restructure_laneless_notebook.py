from __future__ import annotations

import copy
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parent / "notebooks" / "lanelessKaralakou.ipynb"


def source_lines(text: str) -> list[str]:
    text = text.strip("\n")
    lines = text.splitlines(keepends=True)
    return lines or [""]


def markdown_cell(cell_id: str, text: str, *, tags: list[str] | None = None) -> dict:
    metadata: dict[str, object] = {}
    if tags:
        metadata["tags"] = tags
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": metadata,
        "source": source_lines(text),
    }


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    original_cells = notebook["cells"]
    if len(original_cells) != 85:
        raise RuntimeError(f"Expected 85 cells before restructuring, found {len(original_cells)}")

    by_id = {cell["id"]: cell for cell in original_cells}
    if len(by_id) != len(original_cells):
        raise RuntimeError("Notebook contains duplicate cell ids before restructuring")

    required_ids = {
        "959ff31d",
        "332a85a33082b94b",
        "26a35305",
        "8ce1b274",
        "old_reward_cbf_pair_heading",
        "old_reward_cbf_pair_run",
        "ddpg_cbf_safety_reward_heading",
        "ddpg_cbf_safety_reward_train",
        "ab05d7ca",
        "d7e9f2a1",
        "cd3efcc1",
        "88be826b162c026e",
        "80f0eaaa4d9cc25c",
    }
    missing = sorted(required_ids - set(by_id))
    if missing:
        raise RuntimeError(f"Required notebook cells are missing: {missing}")

    original_code_cells = {
        cell["id"]: copy.deepcopy(cell)
        for cell in original_cells
        if cell["cell_type"] == "code"
    }

    replacements = {
        "959ff31d": """# Laneless Karalakou RL Experiments

This notebook trains, evaluates, and compares lane-free highway policies using the Karalakou reward formulation and CBF safety mechanisms. The simulator remains isolated in `laneless highway env/lane_free_env.py`; this notebook owns experiment configuration, wrappers, training launchers, evaluation, and rendering.

## Notebook Map

1. **Shared Foundation** — imports, reward/environment configuration, observation and task wrappers, KPI helpers, and TensorBoard logging.
2. **Baseline Policy Experiments** — PPO and nominal DDPG without a CBF shield.
3. **CBF Policy Experiments** — shared shield implementation, reward-penalty training, historical-reward and safety-potential subexperiments, and the reward-plus-actor-loss policy.
4. **Cross-Experiment Analysis** — trained-model progress and frozen-policy test-time shield ablations.
5. **Environment and Scenario Subexperiments** — environment activation, common launch helpers, normal renders, and fixed diagnostic scenes.

> Review every `RUN_*` flag before using **Run All**. Long training and GUI-render cells are intentionally controlled independently.""",
        "332a85a33082b94b": """## 1. Shared Foundation

Run this section from top to bottom after starting or restarting the kernel. It defines the common state, reward, environment, evaluation, and logging machinery used by every experiment below.""",
        "8356d30d": """### 1.3 Environment, Reward, and Training Configuration

These values mirror the paper's simulation setup as closely as the current lane-free simulator allows. They define the common road, traffic, action bounds, reward weights, training budgets, and evaluation protocol used by downstream policy sections.""",
        "c1ba47b5": """### 1.5 Evaluation Helpers

The shared callbacks evaluate current policies during training and record speed tracking, collision, task-progress, action, comfort, traffic-density, and safety metrics.""",
        "26a35305": """## 2. Baseline Policy Experiments

This umbrella contains the two policies that do not use a CBF safety mechanism: the PPO baseline and the nominal DDPG baseline. Each policy keeps its training, plots, final evaluation, and optional render together.""",
        "aa19eda9": "#### 2.1.1 Plot PPO Training Results",
        "6ffd8341": "#### 2.1.2 PPO Final Evaluation",
        "050cad8d": """#### 2.1.3 Render the Trained PPO Policy

Set `RUN_RENDER = True` only when you want a live highway-env render window.""",
        "5bbbff3f": """### 2.2 DDPG Baseline — No CBF

This variant uses the same lane-free environment, Karalakou reward wrapper, flat 42D legacy vehicle-table observation, zero-centered normalized action interface, and evaluation metrics as PPO. It has no CBF shield, correction reward, or actor safety loss.""",
        "485eb7d6": """#### 2.2.1 Observation Preflight

Run this immediately before DDPG training to verify that the environment exposes the expected flat 42D legacy vehicle-table observation.""",
        "a107c49f": "#### 2.2.2 Train DDPG Without CBF",
        "36fde7b1": "#### 2.2.3 Plot DDPG Training Results",
        "04701389": "#### 2.2.4 DDPG Final Evaluation",
        "f2baf6d2": """#### 2.2.5 Render the Trained DDPG Policy

Set `RUN_DDPG_RENDER = True` only when you want a live highway-env render window.""",
        "8ce1b274": """## 3. CBF Policy Experiments

This is the main CBF umbrella. Shared shield definitions appear once, followed by the reward-only experiment, its reward/deployment subexperiments, and the guided reward-plus-actor-loss experiment.""",
        "bf35ce81": "#### 3.1.1 QP Solver Dependencies",
        "cd7a1a4d": "#### 3.1.2 Barrier Geometry and Hyperparameters",
        "d81f0b23": "#### 3.1.3 Action Conversion and Vehicle-State Helpers",
        "e358492e": "#### 3.1.4 Safety-Filter Environment Wrapper",
        "fbfb1d77": "#### 3.1.5 Tuned No-Slack Shield Override",
        "da9561f1": """### 3.2 Shared CBF Evaluation Helpers

These helpers evaluate a policy through the shielded environment and add intervention, action-correction, feasibility, barrier, and QP diagnostics to the common policy metrics.""",
        "59bffad4": """### 3.3 Main Experiment — DDPG with CBF Reward Penalty

The actor proposes a raw physical acceleration. The non-differentiable CBF-QP shield filters it before execution, and correction/intervention penalties enter the reward. The DDPG actor and critic objectives otherwise remain standard.""",
        "cb5acbf4": "#### 3.3.1 Plot Reward-Penalty Training Results",
        "3acee561": "#### 3.3.2 Reward-Penalty Final Evaluation",
        "3f003a13": """#### 3.3.3 Render the Reward-Penalty Policy

Set `RUN_DDPG_CBF_RENDER = True` only for a live render. Execute the tuned no-slack shield cell first so deployment matches training.""",
        "old_reward_cbf_pair_heading": """### 3.4 Subexperiment — Historical Safety Reward Deployment Comparison

This opt-in subexperiment trains one shielded DDPG policy using the historical Karalakou safety potential (`cf`), then evaluates that same saved actor twice: raw and with the CBF shield. Common episode seeds, reward settings, environment settings, and evaluation protocol isolate the deployment-time contribution of the shield.

The runner fingerprints the complete experiment specification, resumes only a matching manifest, and writes the model, logs, per-episode metrics, paired comparison, delta, and plot below `artifacts/oldcbf_<fingerprint>/`.""",
        "ddpg_cbf_safety_reward_heading": """### 3.5 Subexperiment — CBF Safety-Potential Reward

This reward-side control trains plain DDPG without a CBF action shield, replacing the historical vehicle potential with the CBF-clearance-based `safety_cf` reward term. Its separate artifact directory prevents accidental resume from a legacy DDPG run.""",
        "ab05d7ca": """### 3.6 Main Experiment — DDPG with CBF Reward and Actor Loss

This guided variant retains the shield and reward penalty, then adds a CBF-informed imitation term to the actor update. Successful interventions stored in replay teach the raw actor to approach the safe CBF-filtered action directly.""",
        "c86a1f2e": """#### 3.6.1 Guided Reward + Actor-Loss Final Evaluation

Evaluate the saved guided policy out of process using the same shield and final-evaluation protocol as the reward-only DDPG-CBF policy.""",
        "d7e9f2a1": """## 4. Cross-Experiment Analysis

These cells consume frozen artifacts from the policy sections above. They compare learning progress, final model behavior, and the contribution of enabling the CBF shield at deployment; they do not train policies.""",
        "a731e6cf": """#### 4.1.1 Current Trained/Evaluated Model Figures

These previews are loaded from the canonical model histories and final-evaluation artifacts.

![Current DDPG training-evaluation progress](../artifacts/lanelessKaralakou/ddpg_evaluated_training_progress.png)

![Current saved DDPG final evaluation](../artifacts/lanelessKaralakou/ddpg_evaluated_final_comparison.png)""",
        "cbf_filter_ablation_heading": """### 4.2 Test-Time CBF Filter Ablation

Evaluate the three frozen DDPG policies with the tuned CBF filter disabled and enabled. Common episode seeds pair both conditions. This cell performs evaluation only and never retrains or modifies a model.""",
        "cd3efcc1": """## 5. Environment and Scenario Subexperiments

This umbrella contains environment selection, shared script launchers, ordinary policy renders, and fixed diagnostic scenario simulations. Run one activation cell before using the launch or render cells.""",
        "88be826b162c026e": """<!-- GENERATED_POLICY_RENDER_SECTION -->
### 5.3 Normal Policy Renders

Run one activation cell first. Each labeled cell below renders five ordinary episodes for one frozen policy in the active traffic environment.""",
        "80f0eaaa4d9cc25c": """<!-- GENERATED_POLICY_RENDER_SECTION -->
### 5.4 Fixed Diagnostic Scenario Simulations

Each subexperiment launches the same designed scene for DDPG, DDPG-CBF reward, and DDPG-CBF reward plus actor loss, sequentially, in the active traffic environment.""",
        "1d6bda411c367949": """<!-- GENERATED_POLICY_RENDER_SECTION -->
#### 5.4.1 Open Passing Gap

A clear side gap exists behind a slow leader; the policy should commit smoothly if it learned passing.""",
        "97b341f7839efa5a": """<!-- GENERATED_POLICY_RENDER_SECTION -->
#### 5.4.2 Fast Closing Side Vehicle

A tempting side gap is actually closing fast; the policy should reject it.""",
        "14aa60252878bc19": """<!-- GENERATED_POLICY_RENDER_SECTION -->
#### 5.4.3 Boxed In

No clean nearby side gap exists; the policy should avoid forcing a pass.""",
        "922d0833840e23c7": """<!-- GENERATED_POLICY_RENDER_SECTION -->
#### 5.4.4 Rear Pressure Escape

A fast rear vehicle closes on ego; the policy should escape only through the cleaner side.""",
        "373f41fe5ae33b99": """<!-- GENERATED_POLICY_RENDER_SECTION -->
#### 5.4.5 Boundary Recovery

Ego starts near the road edge with a blocker ahead; the policy should recover inward.""",
        "c039432db4bdac09": """<!-- GENERATED_POLICY_RENDER_SECTION -->
#### 5.4.6 Sudden Lead Slowdown

The leader is much slower and both side gaps are risky; the policy should react without diving into traffic.""",
    }

    for cell_id, text in replacements.items():
        cell = by_id[cell_id]
        if cell["cell_type"] != "markdown":
            raise RuntimeError(f"Expected markdown cell for replacement: {cell_id}")
        cell["source"] = source_lines(text)

    inserted = {
        "shared_dependencies_heading": markdown_cell(
            "shared_dependencies_heading",
            """### 1.1 Dependencies and Project Paths

Import the common libraries, locate the project and custom environment, select the compute device, and create the shared artifact directory. If an import fails, install dependencies into the selected notebook interpreter:

```python
#!python -m pip install -U stable-baselines3[extra] gymnasium highway-env torch numpy pandas matplotlib pygame qpsolvers osqp clarabel
```""",
            tags=["section", "shared-foundation"],
        ),
        "reward_wrapper_heading": markdown_cell(
            "reward_wrapper_heading",
            """### 1.2 Karalakou Reward Wrapper

Define the common reward terms and expose the task, action, collision, and traffic information required by training and evaluation.""",
            tags=["section", "shared-foundation"],
        ),
        "observation_task_kpi_heading": markdown_cell(
            "observation_task_kpi_heading",
            """### 1.4 Observation, Task, and KPI Wrappers

Define observation normalization, distance-task termination, episode summaries, action conversions, and the shared KPI schema used across policy families.""",
            tags=["section", "shared-foundation"],
        ),
        "tensorboard_helpers_heading": markdown_cell(
            "tensorboard_helpers_heading",
            """### 1.6 TensorBoard Metric Bridge

Map the shared KPI schema into consistent TensorBoard namespaces for training, checkpoint evaluation, and final evaluation.""",
            tags=["section", "shared-foundation"],
        ),
        "ppo_baseline_heading": markdown_cell(
            "ppo_baseline_heading",
            """### 2.1 PPO Baseline

The PPO launcher delegates to the canonical nominal-PPO pilot runner using the fixed MTM environment and corrected timestep-based evaluation protocol.""",
            tags=["experiment", "baseline", "ppo"],
        ),
        "shared_cbf_shield_heading": markdown_cell(
            "shared_cbf_shield_heading",
            """### 3.1 Shared CBF Safety Shield

Define the non-differentiable two-dimensional CBF-QP projection used by all shielded experiments. The actor proposes a raw physical acceleration and the shield minimally modifies it before environment execution.""",
            tags=["section", "cbf", "shared-shield"],
        ),
        "evaluated_progress_heading": markdown_cell(
            "evaluated_progress_heading",
            """### 4.1 Evaluated DDPG Model Progress

Compare nominal DDPG, reward-penalty DDPG-CBF, and guided reward-plus-actor-loss DDPG-CBF using matching training-evaluation histories and final-evaluation artifacts.""",
            tags=["analysis", "model-comparison"],
        ),
        "environment_activation_heading": markdown_cell(
            "environment_activation_heading",
            """### 5.1 Environment Activation

Run exactly one of the next two cells. Each defines its environment and training parameters, then sets `ACTIVE_ENV_CONFIG` and `ACTIVE_TRAINING_CONFIG` for later script and render cells.""",
            tags=["subexperiment", "environment"],
        ),
        "common_script_runner_heading": markdown_cell(
            "common_script_runner_heading",
            """### 5.2 Common Active-Environment Script Runners

Define reusable launch helpers that consume whichever activation cell ran most recently. The `RUN_ACTIVE_*` flags at the bottom opt into individual training or diagnostic scripts.""",
            tags=["section", "environment", "runners"],
        ),
        "normal_render_ddpg_heading": markdown_cell(
            "normal_render_ddpg_heading",
            "#### 5.3.1 Nominal DDPG Render",
            tags=["subexperiment", "render", "ddpg"],
        ),
        "normal_render_cbf_heading": markdown_cell(
            "normal_render_cbf_heading",
            "#### 5.3.2 DDPG-CBF Reward-Penalty Render",
            tags=["subexperiment", "render", "ddpg-cbf"],
        ),
        "normal_render_guided_heading": markdown_cell(
            "normal_render_guided_heading",
            "#### 5.3.3 DDPG-CBF Reward + Actor-Loss Render",
            tags=["subexperiment", "render", "guided-ddpg-cbf"],
        ),
    }

    ordered_ids = [
        "959ff31d",
        "332a85a33082b94b",
        "shared_dependencies_heading",
        "3c4e2e4cd8681b85",
        "reward_wrapper_heading",
        "56179f4b9b480dbf",
        "8356d30d",
        "c9f74b85",
        "observation_task_kpi_heading",
        "2c6e9a65",
        "c1ba47b5",
        "25ee1b53",
        "tensorboard_helpers_heading",
        "7a3442b464f8f11e",
        "26a35305",
        "ppo_baseline_heading",
        "eb9eade5",
        "aa19eda9",
        "729d0c1b",
        "6ffd8341",
        "3b70133c",
        "050cad8d",
        "57a1ec06",
        "5bbbff3f",
        "485eb7d6",
        "aa94309f",
        "a107c49f",
        "91abe55c",
        "36fde7b1",
        "db5d8bf6",
        "04701389",
        "6373b733",
        "f2baf6d2",
        "74cf00ce",
        "8ce1b274",
        "shared_cbf_shield_heading",
        "bf35ce81",
        "7eff7b9a",
        "cd7a1a4d",
        "c9383270",
        "d81f0b23",
        "7f6e5854",
        "e358492e",
        "c1a0a1a6",
        "fbfb1d77",
        "6d0a567b",
        "da9561f1",
        "840106dd",
        "59bffad4",
        "3131359d",
        "cb5acbf4",
        "0a1bbef3",
        "3acee561",
        "44c85806",
        "3f003a13",
        "bb415e47",
        "old_reward_cbf_pair_heading",
        "old_reward_cbf_pair_run",
        "ddpg_cbf_safety_reward_heading",
        "ddpg_cbf_safety_reward_train",
        "ab05d7ca",
        "0ebd383e",
        "c86a1f2e",
        "a9d40e7b",
        "d7e9f2a1",
        "evaluated_progress_heading",
        "cbf3a2d4",
        "e46d1c90",
        "a731e6cf",
        "cbf_filter_ablation_heading",
        "cbf_filter_ablation_eval",
        "cd3efcc1",
        "environment_activation_heading",
        "74fde712",
        "f6c0d5d6",
        "common_script_runner_heading",
        "f7b6efd6",
        "88be826b162c026e",
        "normal_render_ddpg_heading",
        "4a79001d22c29596",
        "normal_render_cbf_heading",
        "35a2130515ae8f12",
        "normal_render_guided_heading",
        "1918f9a745eaaaff",
        "80f0eaaa4d9cc25c",
        "1d6bda411c367949",
        "3bf9413b3308a6ff",
        "97b341f7839efa5a",
        "9eed8941c1fd11e2",
        "14aa60252878bc19",
        "f584445064193440",
        "922d0833840e23c7",
        "e0df6b4af5ae5f8c",
        "373f41fe5ae33b99",
        "72f07073c6b4e30e",
        "c039432db4bdac09",
        "c9c1fedeeb88eb59",
    ]

    combined = {**by_id, **inserted}
    if len(ordered_ids) != len(set(ordered_ids)):
        raise RuntimeError("The target order contains duplicate cell ids")
    if set(ordered_ids) != set(combined):
        missing_from_order = sorted(set(combined) - set(ordered_ids))
        unknown_in_order = sorted(set(ordered_ids) - set(combined))
        raise RuntimeError(
            f"Target order mismatch; missing={missing_from_order}, unknown={unknown_in_order}"
        )

    notebook["cells"] = [combined[cell_id] for cell_id in ordered_ids]

    final_ids = [cell["id"] for cell in notebook["cells"]]
    if len(final_ids) != 97 or len(final_ids) != len(set(final_ids)):
        raise RuntimeError("Restructured notebook must contain 97 uniquely identified cells")
    if not set(by_id).issubset(final_ids):
        raise RuntimeError("An original cell was lost during restructuring")

    final_code_cells = {
        cell["id"]: cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }
    if final_code_cells != original_code_cells:
        raise RuntimeError("A code cell or its output/metadata changed during restructuring")

    old_index = final_ids.index("old_reward_cbf_pair_run")
    safety_index = final_ids.index("ddpg_cbf_safety_reward_train")
    guided_index = final_ids.index("0ebd383e")
    if not old_index < safety_index < guided_index:
        raise RuntimeError("CBF reward subexperiments are not grouped in the intended order")

    serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    NOTEBOOK.write_bytes(serialized.replace("\n", "\r\n").encode("utf-8"))
    print(f"Restructured {NOTEBOOK} ({len(original_cells)} -> {len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
