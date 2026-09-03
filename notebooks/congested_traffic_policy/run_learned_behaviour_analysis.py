from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src").exists() and (candidate / "notebooks").exists():
            return candidate
        nested = candidate / "highway-rl-decision-making"
        if (nested / "src").exists() and (nested / "notebooks").exists():
            return nested
    raise RuntimeError("Could not locate the project root.")


def load_notebook_setup(project_root: Path) -> dict[str, object]:
    notebook_path = (
        project_root
        / "notebooks"
        / "congested_traffic"
        / "congested_traffic_policy_v2.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {
        "__name__": "__learned_behaviour_runner__",
        "__file__": str(notebook_path),
    }

    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "experiment_1_lane_safety_baseline_dqn = run_congested_experiment" in source:
            break
        if source.strip():
            exec(compile(source, f"{notebook_path}:cell_{cell_index}", "exec"), namespace)
    return namespace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-episodes", type=int, default=20)
    parser.add_argument("--force-replay", action="store_true")
    args = parser.parse_args()

    project_root = find_project_root()
    os.chdir(project_root)
    namespace = load_notebook_setup(project_root)

    notebook_dir = str(namespace["NOTEBOOK_DIR"])
    if notebook_dir not in sys.path:
        sys.path.insert(0, notebook_dir)

    import learned_behaviour_analysis

    learned_behaviour_analysis = importlib.reload(learned_behaviour_analysis)
    outputs = learned_behaviour_analysis.run_learned_behaviour_analysis(
        project_root=namespace["PROJECT_ROOT"],
        results_dir=namespace["RESULTS_DIR"],
        results_subdir=namespace["RESULTS_SUBDIR"],
        make_congested_env_config=namespace["make_congested_env_config"],
        load_dqn_backend=namespace["load_dqn_backend"],
        seed=namespace["seed"],
        replay_episodes=args.replay_episodes,
        force_replay=args.force_replay,
    )

    print(f"output_dir={outputs['output_dir']}", flush=True)
    print(outputs["metric_summary"].to_string(index=False), flush=True)
    if "replay_summary" in outputs:
        print(outputs["replay_summary"].to_string(index=False), flush=True)
        print(f"matched_trace_episode={outputs['matched_trace_episode']}", flush=True)
    for plot_path in outputs["plot_paths"]:
        print(f"plot={plot_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
