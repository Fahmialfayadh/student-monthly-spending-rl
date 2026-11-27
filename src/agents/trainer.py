from datetime import datetime
import json
import shutil
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from src.agents.evaluator import StudentEvaluator
from src.agents.utils import compute_file_sha256, resolve_algorithm_class


def compute_checkpoint_score(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    """
    Selaras dengan PPO reward seimbang:
    1. Bertahan 30 hari penuh terlebih dahulu (completion_rate)
    2. Setelah 100% survive, utamakan mean_episode_reward (menggabungkan
       kesejahteraan, kehadiran kuliah, dan bonus tabungan)
    3. Tie-breaker: sisa saldo dan tingkat kehadiran kuliah
    """
    survival_target_met = float(summary["completion_rate"]) >= 1.0
    secondary = (
        float(summary["mean_episode_reward"])
        if survival_target_met
        else float(summary["mean_days_survived"])
    )
    return (
        float(summary["completion_rate"]),
        secondary,
        float(summary["mean_final_balance_idr"]),
        float(summary["mean_class_attendance_rate"]),
    )


class PeriodicEvalCallback(BaseCallback):
    """
    Menjalankan evaluasi berkala, seleksi best checkpoint, dan pencatatan history
    di dalam satu panggilan model.learn() tanpa memutus/mereset episode paralel.
    """

    def __init__(
        self,
        agent: Any,
        evaluator: StudentEvaluator,
        eval_interval_steps: int,
        eval_episodes: int,
        eval_seed: int,
        total_timesteps: int,
        survival_phase_timesteps: int,
        run_dir: Path,
        best_model_path: Path,
        best_vec_path: Path,
        lr_state: dict[str, Any],
        eval_history_rows: list[dict[str, Any]],
        best_checkpoint_score: tuple[float, float, float, float] | None,
        best_timestep: int,
        survival_achieved: bool,
        milestone_announced: bool,
    ):
        super().__init__(verbose=0)
        self.agent = agent
        self.evaluator = evaluator
        self.eval_interval_steps = int(eval_interval_steps)
        self.eval_episodes = int(eval_episodes)
        self.eval_seed = int(eval_seed)
        self.total_timesteps = int(total_timesteps)
        self.survival_phase_timesteps = int(survival_phase_timesteps)
        self.run_dir = run_dir
        self.best_model_path = best_model_path
        self.best_vec_path = best_vec_path
        self.lr_state = lr_state
        self.eval_history_rows = eval_history_rows
        self.best_checkpoint_score = best_checkpoint_score
        self.best_timestep = int(best_timestep)
        self.survival_achieved = bool(survival_achieved)
        self.milestone_announced = bool(milestone_announced)
        self.next_eval_step = 0

    def _on_training_start(self) -> None:
        if not self.eval_history_rows:
            self._run_eval_checkpoint()
        self.next_eval_step = int(self.num_timesteps) + self.eval_interval_steps

    def _run_eval_checkpoint(self) -> None:
        timesteps_completed = int(self.num_timesteps)
        eval_summary, _, _ = self.evaluator.evaluate_detailed(
            num_episodes=self.eval_episodes,
            deterministic=True,
            eval_seed=self.eval_seed,
            record_best_episode_trace=False,
        )
        if float(eval_summary["completion_rate"]) >= 1.0:
            self.survival_achieved = True
            if self.lr_state["decay_start"] is None:
                self.lr_state["decay_start"] = timesteps_completed
                print(
                    f"Survival achieved at {timesteps_completed:,} steps; LR now starts decaying."
                )

        training_phase = "optimize" if self.survival_achieved else "survive"
        eval_row = {
            "timesteps": timesteps_completed,
            "training_phase": training_phase,
            **eval_summary,
        }
        self.eval_history_rows.append(eval_row)

        current_score = compute_checkpoint_score(eval_summary)
        is_new_best = (
            self.best_checkpoint_score is None or current_score > self.best_checkpoint_score
        )
        if is_new_best:
            self.best_checkpoint_score = current_score
            self.best_timestep = timesteps_completed
            self.agent.save(str(self.best_model_path), str(self.best_vec_path))

        if timesteps_completed >= self.survival_phase_timesteps and not self.milestone_announced:
            status = "entering optimization" if self.survival_achieved else "still in survival phase"
            print(
                f"Milestone {self.survival_phase_timesteps:,} steps reached; "
                f"completion={eval_summary['completion_rate'] * 100:.0f}% ({status})."
            )
            self.milestone_announced = True

        home_meals = (
            eval_summary["mean_cook_fresh_count"]
            + eval_summary["mean_meal_prep_count"]
            + eval_summary["mean_reheat_prep_count"]
            + eval_summary["mean_eat_bekal_count"]
        )
        best_tag = " [BEST]" if is_new_best else ""
        print(
            f"Step {timesteps_completed:>7d}/{self.total_timesteps} | "
            f"Fase: {training_phase:>8} | "
            f"Reward: {eval_summary['mean_episode_reward']:>7.2f} +/- {eval_summary['std_episode_reward']:>5.2f} | "
            f"Complete: {eval_summary['completion_rate'] * 100:>5.1f}% | "
            f"Hari: {eval_summary['mean_days_survived']:>4.1f}/30 | "
            f"Kelas: {eval_summary['mean_class_attendance_rate'] * 100:>5.1f}% | "
            f"Sisa: Rp {eval_summary['mean_final_balance_idr']:>9,.0f} | "
            f"Masak/Bekal: {home_meals:>4.1f} | "
            f"CCWS/Kafe: {eval_summary['mean_ccws_stay_count'] + eval_summary['mean_cafe_session_count']:>4.1f}"
            f"{best_tag}"
        )
        pd.DataFrame(self.eval_history_rows).to_csv(
            self.run_dir / "eval_history.csv", index=False
        )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval_step:
            self._run_eval_checkpoint()
            while self.next_eval_step <= self.num_timesteps:
                self.next_eval_step += self.eval_interval_steps
        return True

    def _on_training_end(self) -> None:
        if not self.eval_history_rows or int(self.eval_history_rows[-1]["timesteps"]) < int(
            self.num_timesteps
        ):
            self._run_eval_checkpoint()


class ExperimentTrainer:
    """
    Mengelola siklus eksperimen terstruktur: snapshot env_config.yaml, resume checkpoint,
    LR schedule berbasis survival, evaluasi berkala via PeriodicEvalCallback, serta
    pencatatan artefak & master experiments_summary.csv.
    """

    def __init__(self, agent: Any):
        self.agent = agent
        self.evaluator = StudentEvaluator(agent)

    def _collect_library_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {
            "gymnasium": gym.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        if self.agent.use_action_masking:
            import sb3_contrib

            versions["sb3_contrib"] = sb3_contrib.__version__
        return versions

    def run(
        self,
        total_timesteps: int = 250_000,
        eval_interval_steps: int = 25_000,
        eval_episodes: int = 10,
        final_eval_episodes: int = 20,
        eval_seed: int = 1000,
        survival_phase_timesteps: int = 500_000,
        experiment_name: str = "maskable_ppo_5step",
        runs_root_dir: str = "runs",
        progress_bar: bool = False,
        resume_from_run: str | Path | None = None,
        resume_checkpoint: str = "best",
    ) -> dict[str, Any]:
        config_sha256 = compute_file_sha256(self.agent.yaml_path)
        safe_experiment_name = experiment_name.strip().replace(" ", "_")
        resumed_run_config: dict[str, Any] | None = None

        if resume_from_run is not None:
            resumed_dir = Path(resume_from_run)
            resumed_run_config = json.loads(
                (resumed_dir / "train_config.json").read_text(encoding="utf-8")
            )
            if resumed_run_config["env_config_sha256"] != config_sha256:
                raise ValueError("Resume requires the same env_config.yaml as the original run.")
            run_dir = resumed_dir
            run_id = run_dir.name
            timestamp_str = str(resumed_run_config["timestamp"])
            safe_experiment_name = str(resumed_run_config["experiment_name"])
            snapshot_yaml_path = run_dir / "env_config_snapshot.yaml"
        else:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{timestamp_str}_{safe_experiment_name}_{config_sha256[:8]}"
            run_dir = Path(runs_root_dir) / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            snapshot_yaml_path = run_dir / "env_config_snapshot.yaml"
            shutil.copy2(self.agent.yaml_path, snapshot_yaml_path)

        train_metadata: dict[str, Any] = {
            "run_id": run_id,
            "experiment_name": safe_experiment_name,
            "timestamp": timestamp_str,
            "env_config_sha256": config_sha256,
            "env_config_snapshot": str(snapshot_yaml_path),
            "total_timesteps": int(total_timesteps),
            "eval_interval_steps": int(eval_interval_steps),
            "eval_episodes": int(eval_episodes),
            "final_eval_episodes": int(final_eval_episodes),
            "eval_seed": int(eval_seed),
            "survival_phase_timesteps": int(survival_phase_timesteps),
            "hyperparameters": self.agent.hyperparameters,
            "library_versions": self._collect_library_versions(),
        }
        if resumed_run_config is not None:
            train_metadata["resumed_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            train_metadata["resume_from_checkpoint"] = str(resume_checkpoint)
            train_metadata["original_hyperparameters"] = resumed_run_config["hyperparameters"]
        (run_dir / "train_config.json").write_text(
            json.dumps(train_metadata, indent=2), encoding="utf-8"
        )

        # LR tetap konstan sampai survival tercapai, lalu anneal halus hingga 10% base LR
        base_learning_rate = float(self.agent.hyperparameters["learning_rate"])
        lr_state: dict[str, Any] = {"decay_start": None}

        def current_learning_rate(_progress_remaining: float) -> float:
            decay_start = lr_state["decay_start"]
            if decay_start is None:
                return base_learning_rate
            decay_steps = max(1, total_timesteps - int(decay_start))
            progress = min(
                1.0,
                max(0.0, (self.agent.model.num_timesteps - int(decay_start)) / decay_steps),
            )
            return base_learning_rate * (1.0 - 0.9 * progress)

        self.agent.model.lr_schedule = current_learning_rate

        eval_history_rows: list[dict[str, Any]] = []
        best_checkpoint_score: tuple[float, float, float, float] | None = None
        best_timestep = 0
        survival_achieved = False
        milestone_announced = False
        timesteps_completed = 0
        is_first_run = True

        if resumed_run_config is not None:
            algorithm_class = resolve_algorithm_class(self.agent.use_action_masking)
            self.agent.model = algorithm_class.load(
                str(run_dir / f"{resume_checkpoint}_model.zip"),
                env=self.agent.training_env,
                device=str(self.agent.hyperparameters["device"]),
                verbose=0,
            )
            self.agent.training_env = VecNormalize.load(
                str(run_dir / f"{resume_checkpoint}_vec_normalize.pkl"),
                self.agent.training_env.venv,
            )
            self.agent.training_env.training = True
            self.agent.training_env.norm_reward = True
            self.agent.training_env.gamma = self.agent.discount_factor_gamma
            self.agent.model.set_env(self.agent.training_env)
            self.agent.model.lr_schedule = current_learning_rate

            history_path = run_dir / "eval_history.csv"
            if history_path.exists():
                eval_history_rows = pd.read_csv(history_path).to_dict("records")
                completed_rows = [
                    row for row in eval_history_rows if float(row["completion_rate"]) >= 1.0
                ]
                if completed_rows:
                    survival_achieved = True
                    lr_state["decay_start"] = int(completed_rows[0]["timesteps"])
                best_row = max(eval_history_rows, key=compute_checkpoint_score)
                best_checkpoint_score = compute_checkpoint_score(best_row)
                best_timestep = int(best_row["timesteps"])
                milestone_announced = any(
                    int(row["timesteps"]) >= survival_phase_timesteps
                    for row in eval_history_rows
                )
            timesteps_completed = int(self.agent.model.num_timesteps)
            is_first_run = False
            print(f"Resumed {run_id} from '{resume_checkpoint}' at {timesteps_completed:,} steps.")

        best_model_path = run_dir / "best_model.zip"
        best_vec_path = run_dir / "best_vec_normalize.pkl"
        final_model_path = run_dir / "final_model.zip"
        final_vec_path = run_dir / "final_vec_normalize.pkl"

        remaining_timesteps = max(0, int(total_timesteps) - timesteps_completed)
        callback = PeriodicEvalCallback(
            agent=self.agent,
            evaluator=self.evaluator,
            eval_interval_steps=eval_interval_steps,
            eval_episodes=eval_episodes,
            eval_seed=eval_seed,
            total_timesteps=total_timesteps,
            survival_phase_timesteps=survival_phase_timesteps,
            run_dir=run_dir,
            best_model_path=best_model_path,
            best_vec_path=best_vec_path,
            lr_state=lr_state,
            eval_history_rows=eval_history_rows,
            best_checkpoint_score=best_checkpoint_score,
            best_timestep=best_timestep,
            survival_achieved=survival_achieved,
            milestone_announced=milestone_announced,
        )

        if remaining_timesteps > 0:
            self.agent.training_env.training = True
            self.agent.training_env.norm_reward = True
            self.agent.model.learn(
                total_timesteps=remaining_timesteps,
                reset_num_timesteps=is_first_run,
                callback=callback,
                progress_bar=progress_bar,
            )

        timesteps_completed = int(self.agent.model.num_timesteps)
        best_checkpoint_score = callback.best_checkpoint_score
        best_timestep = callback.best_timestep
        self.agent.save(str(final_model_path), str(final_vec_path))

        best_agent = self.agent.__class__.load(
            model_path=str(best_model_path),
            vec_normalize_path=str(best_vec_path),
            yaml_path=str(snapshot_yaml_path),
            use_action_masking=self.agent.use_action_masking,
            device=str(self.agent.hyperparameters["device"]),
        )
        final_summary, final_episodes_df, best_episode_trace_df = best_agent.evaluate_detailed(
            num_episodes=final_eval_episodes,
            deterministic=True,
            eval_seed=eval_seed + 10_000,
            record_best_episode_trace=True,
        )

        final_summary_record = {
            "run_id": run_id,
            "experiment_name": safe_experiment_name,
            "env_config_sha256": config_sha256[:12],
            "seed": self.agent.seed,
            "use_action_masking": self.agent.use_action_masking,
            "total_timesteps": int(timesteps_completed),
            "survival_phase_timesteps": int(survival_phase_timesteps),
            "best_checkpoint_timestep": int(best_timestep),
            "best_checkpoint_score": list(best_checkpoint_score or ()),
            **final_summary,
        }

        (run_dir / "final_eval_summary.json").write_text(
            json.dumps(final_summary_record, indent=2), encoding="utf-8"
        )
        final_episodes_df.to_csv(run_dir / "final_eval_episodes.csv", index=False)
        best_episode_trace_df.to_csv(run_dir / "best_episode_trace.csv", index=False)

        self._update_master_summary(Path(runs_root_dir), run_id, final_summary_record)

        eval_history_df = pd.DataFrame(eval_history_rows)
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "best_checkpoint_timestep": best_timestep,
            "eval_history_df": eval_history_df,
            "final_summary": final_summary_record,
            "final_episodes_df": final_episodes_df,
            "best_episode_trace_df": best_episode_trace_df,
            "sample_trace_df": best_episode_trace_df,
            "best_agent": best_agent,
        }

    @staticmethod
    def _update_master_summary(
        runs_root_dir: Path, run_id: str, final_summary_record: dict[str, Any]
    ) -> None:
        master_summary_path = runs_root_dir / "experiments_summary.csv"
        summary_row_df = pd.DataFrame([final_summary_record])
        if master_summary_path.exists():
            existing_master_df = pd.read_csv(master_summary_path)
            existing_master_df = existing_master_df[existing_master_df["run_id"] != run_id]
            updated_master_df = pd.concat([existing_master_df, summary_row_df], ignore_index=True)
        else:
            updated_master_df = summary_row_df
        updated_master_df.to_csv(master_summary_path, index=False)
