from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_laneless_notebook_task as notebook_task  # noqa: E402
import run_ppo_cbf_progression as progression  # noqa: E402
from evaluate_laneless_karalakou import TEN_KPI_SPECS  # noqa: E402


def _signature_namespace() -> dict[str, object]:
    return {
        "CBF_AX_BOUNDS": (-3.0, 3.0),
        "CBF_AY_BOUNDS": (-3.0, 3.0),
        "CBF_EPS_SIDE": 0.10,
        "CBF_K0": 5.29,
        "CBF_K1": 3.68,
        "CBF_NEIGHBOR_RANGE": 90.0,
        "CBF_MAX_NEIGHBOR_CONSTRAINTS": 12,
        "CBF_QP_FEASIBILITY_TOL": 1e-3,
        "CBF_TARGET_PAIR_DY": 3.0,
    }


def _signature_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "timesteps": 50_000,
        "ppo_config": "Q0_current_aligned",
        "n_steps": None,
        "batch_size": None,
        "n_epochs": None,
        "lambda_delta": 0.05,
        "lambda_intervention": 0.10,
        "lambda_mean": 0.10,
        "lambda_sample": 0.0,
        "correction_epsilon": 0.03,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _training_signature(
    variant: str = "ppo_cbf_projected",
    **arg_overrides: object,
) -> dict[str, object]:
    return progression.training_signature(
        _signature_namespace(),
        variant=variant,
        training_seed=307,
        env_config={"traffic_model": "mtm", "vehicles_count": 55},
        reward_config={"collision_reward": -1.0, "speed_reward": 1.0},
        args=_signature_args(**arg_overrides),
    )


def _write_completed_checkpoint(
    output_dir: Path,
    *,
    variant: str,
    signature: dict[str, object],
) -> Path:
    run_dir = progression._variant_dir(output_dir, variant, 307)
    run_dir.mkdir(parents=True)
    model_path = run_dir / "model_final.zip"
    model_path.write_bytes(b"verified PPO checkpoint")
    progression._signature_path(run_dir).write_text(
        json.dumps(signature, indent=2), encoding="utf-8"
    )
    progression._completion_path(run_dir).write_text(
        json.dumps(
            {
                "schema_version": progression.PROGRESSION_SCHEMA_VERSION,
                "training_signature_hash": progression.protocol.canonical_config_hash(
                    signature
                ),
                "model_file": model_path.name,
                "model_sha256": progression.protocol.file_sha256(model_path),
                "num_timesteps": signature["timesteps"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path


def test_progression_contains_the_required_causal_controls():
    assert tuple(progression.VARIANT_SPECS) == (
        "ppo_nominal",
        "ppo_cbf_shield_only",
        "ppo_cbf_reward",
        "ppo_cbf_projected_reward_off",
        "ppo_cbf_projected",
    )
    nominal = progression.VARIANT_SPECS["ppo_nominal"]
    shield_only = progression.VARIANT_SPECS["ppo_cbf_shield_only"]
    reward = progression.VARIANT_SPECS["ppo_cbf_reward"]
    projected_reward_off = progression.VARIANT_SPECS[
        "ppo_cbf_projected_reward_off"
    ]
    projected = progression.VARIANT_SPECS["ppo_cbf_projected"]
    assert nominal["execution_mode"] == "box"
    assert not nominal["reward_penalty"]
    assert shield_only["execution_mode"] == "cbf"
    assert not shield_only["reward_penalty"]
    assert reward["execution_mode"] == "cbf"
    assert reward["reward_penalty"]
    assert projected_reward_off["projected_mean"]
    assert not projected_reward_off["reward_penalty"]
    assert projected["projected_mean"]
    assert progression.FILTERED_FACTORIAL_VARIANTS == {
        (False, False): "ppo_cbf_shield_only",
        (True, False): "ppo_cbf_reward",
        (False, True): "ppo_cbf_projected_reward_off",
        (True, True): "ppo_cbf_projected",
    }
    assert progression.EVALUATION_MODES == ("raw", "cbf")


def test_ten_kpi_summary_has_exactly_ten_rows_per_deployment():
    rows = []
    for mode in progression.EVALUATION_MODES:
        for scenario_seed in (1, 2):
            row = {
                "variant": "ppo_nominal",
                "mode": mode,
                "scenario_seed": scenario_seed,
            }
            for index, (_, column) in enumerate(TEN_KPI_SPECS):
                row[column] = float(index + scenario_seed)
            rows.append(row)
    table = progression.ten_kpi_table(pd.DataFrame(rows))
    assert len(TEN_KPI_SPECS) == 10
    assert table.groupby(["variant", "mode"]).size().eq(10).all()
    assert set(table["KPI"]) == {label for label, _ in TEN_KPI_SPECS}
    assert np.isfinite(table["Mean"]).all()


def test_post_training_evaluation_has_exact_paired_episode_counts(
    tmp_path, monkeypatch
):
    variant = "ppo_nominal"
    training_seed = 307
    run_dir = progression._variant_dir(tmp_path, variant, training_seed)
    run_dir.mkdir(parents=True)
    model_path = run_dir / "model_final.zip"
    model_path.write_bytes(b"model for immediate evaluation")
    calls: list[tuple[str, int, int]] = []

    def fake_episode(*_args, **kwargs):
        mode = str(kwargs["mode"])
        episode_index = int(kwargs["episode_index"])
        episode_seed = int(kwargs["episode_seed"])
        calls.append((mode, episode_index, episode_seed))
        row: dict[str, object] = {
            "variant": variant,
            "variant_label": progression.VARIANT_SPECS[variant]["label"],
            "mode": mode,
            "training_seed": training_seed,
            "scenario_seed": episode_seed,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "timesteps": 10,
            "total_distance_m": 100.0,
            "distinct_ego_collision_events": 0,
        }
        for index, (_, column) in enumerate(TEN_KPI_SPECS):
            row[column] = float(index + episode_index)
        return row

    monkeypatch.setattr(progression, "load_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(progression, "evaluate_completed_episode", fake_episode)
    args = SimpleNamespace(
        post_train_eval_episodes=3,
        post_train_eval_seed_start=1_100_000,
        device="cpu",
    )

    table = progression.evaluate_post_training_model(
        _signature_namespace(),
        model_path=model_path,
        variant=variant,
        training_seed=training_seed,
        env_config={},
        reward_config={},
        args=args,
        output_dir=tmp_path,
    )

    assert len(calls) == 6
    assert [entry[0] for entry in calls[:3]] == ["raw"] * 3
    assert [entry[0] for entry in calls[3:]] == ["cbf"] * 3
    assert [entry[2] for entry in calls[:3]] == [1_100_000, 1_100_001, 1_100_002]
    assert [entry[2] for entry in calls[3:]] == [1_100_000, 1_100_001, 1_100_002]
    assert set(table["external_cbf"]) == {"OFF", "ON"}
    assert table.groupby("mode")["N"].first().to_dict() == {"raw": 3, "cbf": 3}
    assert (run_dir / "pe" / "e.csv").is_file()
    assert (run_dir / "pe" / "b.csv").is_file()
    assert (run_dir / "pe" / "kpi.csv").is_file()
    assert (run_dir / "pe" / "m.json").is_file()
    assert (tmp_path / "post_train_200ep_kpis.csv").is_file()


def test_post_training_summary_pools_rates_before_averaging():
    rows: list[dict[str, object]] = []
    for mode in progression.EVALUATION_MODES:
        for episode_index in range(1, 21):
            early_collision = episode_index % 2 == 1
            row: dict[str, object] = {
                "variant": "ppo_nominal",
                "variant_label": progression.VARIANT_SPECS["ppo_nominal"]["label"],
                "mode": mode,
                "training_seed": 307,
                "episode_index": episode_index,
                "episode_seed": 1_100_000 + episode_index - 1,
                "timesteps": 1 if early_collision else 99,
                "total_distance_m": 1.0 if early_collision else 99.0,
                "distinct_ego_collision_events": int(early_collision),
            }
            for _label, column in TEN_KPI_SPECS:
                row[column] = 1.0 if early_collision else 0.0
            # The special KPI values need their actual physical meanings.
            row["episode_return"] = 1.0
            row["episode_length_steps"] = float(row["timesteps"])
            row["ego_collisions_per_km"] = 1000.0 if early_collision else 0.0
            row["h_min"] = -0.1 if early_collision else -1.0
            rows.append(row)

    blocks, table, geometry = progression.summarize_post_training_episodes(
        pd.DataFrame(rows)
    )
    raw = table.loc[table["mode"].eq("raw")].set_index("KPI")

    assert geometry == {
        "episodes_per_mode": 20,
        "summary_blocks": 10,
        "episodes_per_summary_block": 2,
    }
    assert len(blocks) == 20
    # Each two-episode block has one ego collision over 100 m: 10/km,
    # rather than the invalid mean of 1000/km and 0/km (=500/km).
    assert raw.loc["Ego collisions / km", "Mean"] == 10.0
    # The timestep mean is 1 event in 100 steps, not an equal episode mean.
    assert np.isclose(raw.loc["QP failure rate", "Mean"], 0.01)
    assert raw.loc["Minimum h", "Mean"] == -1.0
    assert raw.loc["Ego collisions / km", "N"] == 10


def test_complete_episode_evaluation_waits_for_a_terminal_episode(monkeypatch):
    class _Vehicle:
        vx = 12.0
        desired_speed = 15.0

    class _Base:
        vehicle = _Vehicle()
        _last_accelerations = np.asarray([[1.0, 0.0]], dtype=float)

    class _Env:
        def __init__(self):
            self.unwrapped = _Base()
            self.steps = 0
            self.closed = False

        def reset(self, *, seed):
            assert seed == 1_100_000
            return np.zeros(2, dtype=np.float32), {}

        def get_wrapper_attr(self, name):
            assert name == "project_current_action"
            return lambda action: (
                action,
                {"correction_norm_normalized": 0.0},
            )

        def step(self, _action):
            self.steps += 1
            return (
                np.zeros(2, dtype=np.float32),
                2.0,
                self.steps == 3,
                False,
                {
                    "pipeline_distance_step_m": 10.0,
                    "ego_collision_events": 0,
                    "karalakou_abs_speed_deviation": 3.0,
                    "karalakou_lat_y_error_m": 0.5,
                },
            )

        def close(self):
            self.closed = True

    env = _Env()
    monkeypatch.setattr(progression, "make_evaluation_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(progression.protocol, "_policy_dt", lambda _env: 0.1)
    monkeypatch.setattr(
        progression.protocol,
        "cbf_state_occupancy_metrics",
        lambda *_args, **_kwargs: {"h_min": -0.2},
    )
    model = SimpleNamespace(
        predict=lambda _obs, deterministic: (np.asarray([0.0, 0.0]), None)
    )
    args = SimpleNamespace(correction_epsilon=0.03, ttc_cap=30.0)

    row = progression.evaluate_completed_episode(
        {"CBF_EPS_SIDE": 0.1},
        model=model,
        variant="ppo_nominal",
        mode="raw",
        training_seed=307,
        episode_index=1,
        episode_seed=1_100_000,
        env_config={},
        reward_config={},
        args=args,
    )

    assert env.steps == 3
    assert env.closed
    assert row["episode_length_steps"] == 3.0
    assert row["episode_return"] == 6.0
    assert row["h_min"] == -0.2
    assert row["ego_collisions_per_km"] == 0.0


def test_long_windows_tensorboard_path_uses_durable_artifacts(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    long_run_dir = tmp_path / ("x" * 180) / "seed_307"
    monkeypatch.setattr(progression.os, "name", "nt")

    log_dir = progression._tensorboard_log_dir(
        {"PROJECT_ROOT": project_root}, long_run_dir, "ppo_nominal", 307
    )

    assert log_dir == project_root.parent / "artifacts" / "tb" / "ppo" / "nom_307"
    assert log_dir.is_dir()
    assert progression._tensorboard_path_is_safe(log_dir)


def test_tensorboard_run_label_isolated_for_new_budget(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    long_run_dir = tmp_path / ("x" * 180) / "seed_307"
    monkeypatch.setattr(progression.os, "name", "nt")

    log_dir = progression._tensorboard_log_dir(
        {"PROJECT_ROOT": project_root},
        long_run_dir,
        "ppo_nominal",
        307,
        tensorboard_run_label="nominal_500k",
    )

    assert log_dir == (
        project_root.parent
        / "artifacts"
        / "tb"
        / "ppo"
        / "nominal_500k_nom_307"
    )
    assert log_dir.is_dir()


def test_legacy_tensorboard_events_are_copied_to_artifacts(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    temp_root = tmp_path / "temp"
    monkeypatch.setattr(progression.tempfile, "gettempdir", lambda: str(temp_root))
    legacy_dir = progression._legacy_tensorboard_log_dir(
        run_dir, "ppo_nominal", 307
    )
    source_event = legacy_dir / "train_1" / "events.out.tfevents.legacy"
    source_event.parent.mkdir(parents=True)
    source_event.write_bytes(b"legacy tensorboard event")
    durable_dir = tmp_path / "artifacts" / "tb" / "ppo" / "nom_307"
    monkeypatch.setattr(
        progression, "_tensorboard_log_dir", lambda *_args, **_kwargs: durable_dir
    )

    restored = progression.restore_legacy_tensorboard_artifacts(
        {"PROJECT_ROOT": tmp_path / "project"},
        run_dir=run_dir,
        variant="ppo_nominal",
        training_seed=307,
    )

    copied = durable_dir / "legacy" / "train_1" / source_event.name
    assert restored == durable_dir
    assert copied.read_bytes() == b"legacy tensorboard event"
    manifest = json.loads((run_dir / "tb.json").read_text(encoding="utf-8"))
    assert manifest["legacy_source_dir"] == str(legacy_dir.resolve())
    assert str(copied.resolve()) in manifest["legacy_event_files"]


def test_filtered_factorial_effects_use_paired_main_and_interaction_contrasts():
    cell_values = {
        "ppo_cbf_shield_only": 1.0,
        "ppo_cbf_reward": 3.0,
        "ppo_cbf_projected_reward_off": 4.0,
        "ppo_cbf_projected": 10.0,
    }
    rows = []
    for variant, base in cell_values.items():
        row = {
            "training_seed": 307,
            "scenario_seed": 900000,
            "mode": "raw",
            "variant": variant,
        }
        for _, column in TEN_KPI_SPECS:
            row[column] = base
        rows.append(row)
    effects, summary = progression.factorial_effects(pd.DataFrame(rows))
    first_kpi = TEN_KPI_SPECS[0][0]
    observed = effects.loc[effects["KPI"].eq(first_kpi)].set_index("effect")[
        "value"
    ]
    assert observed["reward_main"] == 4.0
    assert observed["projected_actor_main"] == 5.0
    assert observed["reward_x_projected_actor"] == 4.0
    assert len(summary) == 3 * len(TEN_KPI_SPECS)


def test_exact_completed_checkpoint_is_reused(tmp_path):
    signature = _training_signature()
    model_path = _write_completed_checkpoint(
        tmp_path,
        variant="ppo_cbf_projected",
        signature=signature,
    )

    resolved = progression.resolve_existing_variant_checkpoint(
        tmp_path,
        variant="ppo_cbf_projected",
        training_seed=307,
        expected_signature=signature,
    )

    assert resolved == model_path


def test_evaluation_only_rejects_nonexact_or_incomplete_checkpoint(tmp_path):
    expected = _training_signature()
    observed = copy.deepcopy(expected)
    observed["env_config"]["vehicles_count"] = 56
    _write_completed_checkpoint(
        tmp_path,
        variant="ppo_cbf_projected",
        signature=observed,
    )

    with np.testing.assert_raises_regex(RuntimeError, "not an exact match"):
        progression.resolve_existing_variant_checkpoint(
            tmp_path,
            variant="ppo_cbf_projected",
            training_seed=307,
            expected_signature=expected,
        )

    run_dir = progression._variant_dir(tmp_path, "ppo_cbf_projected", 307)
    progression._completion_path(run_dir).unlink()
    with np.testing.assert_raises_regex(RuntimeError, "completion record is missing"):
        progression.resolve_existing_variant_checkpoint(
            tmp_path,
            variant="ppo_cbf_projected",
            training_seed=307,
            expected_signature=observed,
        )


def test_evaluation_only_rejects_a_checkpoint_with_a_changed_file(tmp_path):
    signature = _training_signature()
    model_path = _write_completed_checkpoint(
        tmp_path,
        variant="ppo_cbf_projected",
        signature=signature,
    )
    model_path.write_bytes(b"changed after completion")

    with np.testing.assert_raises_regex(RuntimeError, "checksum does not match"):
        progression.resolve_existing_variant_checkpoint(
            tmp_path,
            variant="ppo_cbf_projected",
            training_seed=307,
            expected_signature=signature,
        )


def test_inactive_variant_lambdas_do_not_change_checkpoint_identity():
    first = _training_signature(
        variant="ppo_nominal",
        lambda_delta=0.05,
        lambda_intervention=0.10,
        lambda_mean=0.10,
    )
    second = _training_signature(
        variant="ppo_nominal",
        lambda_delta=99.0,
        lambda_intervention=88.0,
        lambda_mean=77.0,
    )
    assert first == second


def test_cli_defaults_ensure_training_and_alias_supports_existing_only(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_ppo_cbf_progression.py"])
    defaults = progression.parse_args()
    assert not defaults.skip_training
    assert not defaults.force_retrain
    assert defaults.post_train_eval_episodes == 200
    assert not defaults.skip_post_train_evaluation

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ppo_cbf_progression.py", "--use-existing-results"],
    )
    evaluation_only = progression.parse_args()
    assert evaluation_only.skip_training


def test_parallel_rollout_config_preserves_the_global_ppo_batch_geometry():
    config = progression.resolved_ppo_config(_signature_args(n_envs=8))
    assert config["n_envs"] == 8
    assert config["n_steps"] == 125
    assert config["global_rollout_steps"] == 1_000
    assert config["global_rollout_steps"] % config["batch_size"] == 0
    assert progression.training_topology(_signature_args(n_envs=8)) == {
        "n_envs": 8,
        "backend": "subproc",
        "start_method": "spawn",
    }


def test_parallel_rollout_rejects_an_unshardable_global_rollout():
    with np.testing.assert_raises_regex(ValueError, "divisible by n_envs"):
        progression.resolved_ppo_config(_signature_args(n_envs=8, n_steps=1_001))


def test_parallel_topology_changes_checkpoint_identity():
    one_env = _training_signature(n_envs=1)
    eight_envs = _training_signature(n_envs=8)
    assert one_env != eight_envs


def test_parallel_worker_monitor_paths_fit_the_windows_legacy_limit():
    run_monitor = (
        PROJECT_ROOT
        / "artifacts"
        / "ppo_cbf_progression_parallel_v3"
        / "ppo_cbf_projected_reward_off"
        / "seed_307"
        / "training.monitor.csv"
    )
    paths = [
        progression.compact_worker_monitor_path(run_monitor, rank)
        for rank in range(8)
    ]

    assert [path.name for path in paths] == [
        f"m{rank}.monitor.csv" for rank in range(8)
    ]
    assert len({str(path) for path in paths}) == 8
    assert max(len(str(path)) for path in paths) < 260


def test_pending_signature_only_artifacts_can_be_retried(tmp_path):
    run_dir = tmp_path / "ppo_cbf_projected_reward_off" / "seed_307"
    run_dir.mkdir(parents=True)
    pending = progression._pending_signature_path(run_dir)
    pending.write_text("{}", encoding="utf-8")
    (run_dir / "training_monitors").mkdir()

    assert progression._is_retryable_pending_run(run_dir, pending)

    (run_dir / "training_episodes.csv").write_text("episode_index\n", encoding="utf-8")
    assert not progression._is_retryable_pending_run(run_dir, pending)


def test_action_clip_callback_accepts_batched_parallel_actions():
    class _Logger:
        def record(self, *_args, **_kwargs):
            return None

    callback = progression.PPOActionClipCallback()
    callback.model = SimpleNamespace(
        action_space=gym.spaces.Box(
            low=np.full(2, -3.0, dtype=np.float32),
            high=np.full(2, 3.0, dtype=np.float32),
            dtype=np.float32,
        ),
        logger=_Logger(),
    )
    callback.locals = {
        "actions": np.asarray([[0.0, 4.0], [-5.0, 1.0]], dtype=float),
        "clipped_actions": np.asarray([[0.0, 3.0], [-3.0, 1.0]], dtype=float),
    }
    assert callback._on_step()
    assert callback.total_components == 4
    assert callback.clipped_components == 2


def test_notebook_primary_ladder_is_ppo_first_and_streams_inline():
    notebook_path = PROJECT_ROOT / "notebooks" / "lanelessKaralakou.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 86
    sources = {
        cell.get("id"): "".join(cell.get("source", []))
        for cell in notebook["cells"]
    }
    assert "PPO–CBF Research Ladder" in sources["959ff31d"]
    assert "Primary PPO Experiment Ladder" in sources["26a35305"]
    launcher = sources["eb9eade5"]
    assert "run_ppo_cbf_progression.py" in launcher
    assert "PPO_RUN_TRAINING = True" in launcher
    assert "PPO_FORCE_RETRAIN = False" in launcher
    assert "PPO_RUN_POST_TRAIN_EVALUATION = True" in launcher
    assert "PPO_POST_TRAIN_EVAL_EPISODES = 200" in launcher
    assert "PPO_POST_TRAIN_EVALUATE_REUSED =" in launcher
    assert "PPO_PROGRESSION_NUM_ENVS" in launcher
    assert "PPO_PROGRESSION_COMPLETED = True" in launcher
    assert '"--skip-training"' in launcher
    assert "RUN_PPO_PROGRESSION" not in launcher
    assert '"--traffic-model", "mtm"' in launcher
    assert '"--n-envs", str(PPO_PROGRESSION_NUM_ENVS)' in launcher
    assert '"--post-train-eval-episodes", str(PPO_POST_TRAIN_EVAL_EPISODES)' in launcher
    assert '"--skip-post-train-evaluation"' in launcher
    assert '"--post-train-evaluate-reused"' in launcher
    assert "subprocess.Popen" in launcher
    assert "stdout=subprocess.PIPE" in launcher
    assert "stderr=subprocess.STDOUT" in launcher
    assert "PPO_PROGRESSION_LOG_PATH" in launcher
    elongated = sources["ppo_nominal_500k"]
    assert "PPO_NOMINAL_500K_TIMESTEPS = 500_000" in elongated
    assert "PPO_NOMINAL_500K_PPO_CONFIG = 'Q1_stable'" in elongated
    assert "'--variants', 'ppo_nominal'" in elongated
    assert "'--post-train-eval-episodes', str(PPO_NOMINAL_500K_EVAL_EPISODES)" in elongated
    assert "'--skip-evaluation'" in elongated
    assert "'--skip-counterfactual'" in elongated
    assert "plot_nominal_ppo_results.py" in elongated
    assert "06_tensorboard_rollout_scalars.png" in elongated
    assert "02_post_training_400_episode_kpis.png" in elongated
    assert "Differentiable CBF Final Optimization/Action Layer" in sources["050cad8d"]
    assert '"ego_dimensions": [3.5, 1.8]' in sources["c9f74b85"]
    assert "smoke_ego_dimensions" in sources["2c6e9a65"]
    assert "Re-run All PPO Evaluations Inline Without Retraining" in sources[
        "cbf_filter_ablation_heading"
    ]
    assert '"--skip-training"' in sources["cbf_filter_ablation_eval"]
    active_runner = sources["f7b6efd6"]
    assert "PPO is launched only by the primary workflow cell" in active_runner
    assert "RUN_ACTIVE_PPO_TRAIN" not in active_runner


def test_notebook_task_ppo_entry_targets_the_progression():
    task = notebook_task.TASKS["ppo-train"]
    assert task["cell"] == 11
    assert task["flag"] == "PPO_RUN_TRAINING"
    assert task["timesteps_key"] == "_PPO_TASK_TIMESTEPS_OVERRIDE"


def test_latest_checkpoint_selection_is_timestep_ordered(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for step in (1_000, 20_000, 9_000):
        (checkpoint_dir / f"rollout_{step}_steps.zip").write_bytes(b"checkpoint")
    (checkpoint_dir / "unrelated.zip").write_bytes(b"ignore")
    latest = progression._latest_rollout_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "rollout_20000_steps.zip"
