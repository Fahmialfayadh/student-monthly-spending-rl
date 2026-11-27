import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agents.evaluator import StudentEvaluator
from src.agents.networks import StudentFeaturesExtractor
from src.agents.pretrain import BehaviorCloningPretrainer, default_balanced_expert_policy
from src.agents.trainer import ExperimentTrainer
from src.agents.utils import (
    compute_file_sha256,
    create_vectorized_env,
    make_single_env,
    resolve_algorithm_class,
    set_global_seed,
)

__all__ = [
    "StudentFeaturesExtractor",
    "StudentPPOAgent",
    "compute_file_sha256",
    "set_global_seed",
]


class StudentPPOAgent:
    """
    Wrapper inti MaskablePPO / PPO untuk StudentLifeEnv.
    Berfokus pada siklus hidup model (inisialisasi, train, predict, save, load),
    sementara evaluasi didelegasikan ke StudentEvaluator dan eksperimen/checkpointing
    ke ExperimentTrainer.
    """

    def __init__(
        self,
        yaml_path: str = "env_config.yaml",
        num_environments: int = 4,
        features_dim: int = 128,
        hidden_dim: int = 128,
        policy_hidden_layers: list[int] | None = None,
        value_hidden_layers: list[int] | None = None,
        learning_rate: float = 3e-4,
        rollout_steps_per_env: int = 300,
        batch_size: int = 120,
        num_epochs: int = 10,
        discount_factor_gamma: float = 0.995,
        gae_lambda: float = 0.98,
        clip_range: float = 0.2,
        entropy_coefficient: float = 0.025,
        target_kl: float | None = None,
        use_action_masking: bool = True,
        share_features_extractor: bool = False,
        seed: int = 42,
        device: str = "auto",
        tensorboard_log_dir: str | None = None,
        verbose: int = 1,
    ):
        set_global_seed(seed)

        self.yaml_path = str(yaml_path)
        self.num_environments = int(num_environments)
        self.use_action_masking = bool(use_action_masking)
        self.discount_factor_gamma = float(discount_factor_gamma)
        self.seed = int(seed)

        self._validate_shaping_gamma(self.yaml_path, self.discount_factor_gamma)

        policy_layers = list(policy_hidden_layers) if policy_hidden_layers is not None else [128, 128]
        value_layers = list(value_hidden_layers) if value_hidden_layers is not None else [128, 128]

        self.hyperparameters: dict[str, Any] = {
            "yaml_path": self.yaml_path,
            "num_environments": self.num_environments,
            "features_dim": int(features_dim),
            "hidden_dim": int(hidden_dim),
            "policy_hidden_layers": policy_layers,
            "value_hidden_layers": value_layers,
            "learning_rate": float(learning_rate),
            "rollout_steps_per_env": int(rollout_steps_per_env),
            "batch_size": int(batch_size),
            "num_epochs": int(num_epochs),
            "discount_factor_gamma": self.discount_factor_gamma,
            "gae_lambda": float(gae_lambda),
            "clip_range": float(clip_range),
            "entropy_coefficient": float(entropy_coefficient),
            "target_kl": float(target_kl) if target_kl is not None else None,
            "use_action_masking": self.use_action_masking,
            "share_features_extractor": bool(share_features_extractor),
            "seed": self.seed,
            "device": str(device),
        }

        self.training_env = create_vectorized_env(
            yaml_path=self.yaml_path,
            num_environments=self.num_environments,
            seed=self.seed,
            training_mode=True,
            discount_factor_gamma=self.discount_factor_gamma,
            use_action_masking=self.use_action_masking,
        )

        policy_kwargs: dict[str, Any] = {
            "features_extractor_class": StudentFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": int(features_dim),
                "hidden_dim": int(hidden_dim),
            },
            "share_features_extractor": bool(share_features_extractor),
            "net_arch": {
                "pi": policy_layers,
                "vf": value_layers,
            },
            "activation_fn": nn.SiLU,
            "ortho_init": True,
        }

        algorithm_class = resolve_algorithm_class(self.use_action_masking)
        self.model = algorithm_class(
            policy="MlpPolicy",
            env=self.training_env,
            learning_rate=learning_rate,
            n_steps=rollout_steps_per_env,
            batch_size=batch_size,
            n_epochs=num_epochs,
            gamma=self.discount_factor_gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=entropy_coefficient,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log_dir,
            verbose=verbose,
        )

        self.evaluator = StudentEvaluator(self)
        self.trainer = ExperimentTrainer(self)
        self.pretrainer = BehaviorCloningPretrainer(self)

    def pretrain_bc(
        self,
        expert_fn=default_balanced_expert_policy,
        num_episodes: int = 60,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        entropy_coef: float = 0.05,
        vf_coef: float = 0.5,
        demo_seed: int = 5000,
        verbose: bool = True,
        log_callback: Any = None,
    ) -> dict[str, Any]:
        return self.pretrainer.pretrain(
            expert_fn=expert_fn,
            num_episodes=num_episodes,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            entropy_coef=entropy_coef,
            vf_coef=vf_coef,
            demo_seed=demo_seed,
            verbose=verbose,
            log_callback=log_callback,
        )

    @staticmethod
    def _validate_shaping_gamma(yaml_path: str, discount_factor_gamma: float) -> None:
        with open(yaml_path, "r", encoding="utf-8") as cfg_file:
            env_cfg = yaml.safe_load(cfg_file)
        shaping_cfg = env_cfg.get("rewards", {}).get("budget_potential_shaping", {})
        if bool(shaping_cfg.get("enabled", False)):
            shaping_gamma = float(shaping_cfg["gamma"])
            if not np.isclose(discount_factor_gamma, shaping_gamma, atol=1e-6):
                raise ValueError(
                    f"discount_factor_gamma ({discount_factor_gamma}) harus sama dengan "
                    f"rewards.budget_potential_shaping.gamma ({shaping_gamma}) di {yaml_path}."
                )

    def train(self, total_timesteps: int = 150_000, progress_bar: bool = False) -> None:
        self.training_env.training = True
        self.training_env.norm_reward = True
        self.model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)

    def predict(
        self,
        normalized_observation: np.ndarray,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, Any]:
        if self.use_action_masking and action_masks is not None:
            return self.model.predict(
                normalized_observation,
                deterministic=deterministic,
                action_masks=action_masks,
            )
        return self.model.predict(normalized_observation, deterministic=deterministic)

    def evaluate_detailed(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        eval_seed: int = 1000,
        record_best_episode_trace: bool = False,
        record_first_episode_trace: bool = False,
        render: bool = False,
    ) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
        return self.evaluator.evaluate_detailed(
            num_episodes=num_episodes,
            deterministic=deterministic,
            eval_seed=eval_seed,
            record_best_episode_trace=record_best_episode_trace,
            record_first_episode_trace=record_first_episode_trace,
            render=render,
        )

    def evaluate(
        self,
        num_episodes: int = 5,
        deterministic: bool = True,
        eval_seed: int = 1000,
        render: bool = False,
    ) -> dict[str, float]:
        return self.evaluator.evaluate(
            num_episodes=num_episodes,
            deterministic=deterministic,
            eval_seed=eval_seed,
            render=render,
        )

    def record_episode_trace(
        self,
        episode_seed: int,
        deterministic: bool = True,
        render: bool = False,
    ) -> pd.DataFrame:
        return self.evaluator.record_episode_trace(
            episode_seed=episode_seed,
            deterministic=deterministic,
            render=render,
        )

    def train_with_logging(
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
        stop_event: Any = None,
        progress_callback: Any = None,
        log_callback: Any = None,
    ) -> dict[str, Any]:
        return self.trainer.run(
            total_timesteps=total_timesteps,
            eval_interval_steps=eval_interval_steps,
            eval_episodes=eval_episodes,
            final_eval_episodes=final_eval_episodes,
            eval_seed=eval_seed,
            survival_phase_timesteps=survival_phase_timesteps,
            experiment_name=experiment_name,
            runs_root_dir=runs_root_dir,
            progress_bar=progress_bar,
            resume_from_run=resume_from_run,
            resume_checkpoint=resume_checkpoint,
            stop_event=stop_event,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )

    def save(
        self,
        model_path: str = "ppo_student_life.zip",
        vec_normalize_path: str = "vec_normalize.pkl",
    ) -> None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(vec_normalize_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        self.training_env.save(vec_normalize_path)

    @classmethod
    def load(
        cls,
        model_path: str = "ppo_student_life.zip",
        vec_normalize_path: str = "vec_normalize.pkl",
        yaml_path: str = "env_config.yaml",
        use_action_masking: bool = True,
        device: str = "auto",
    ) -> "StudentPPOAgent":
        with open(yaml_path, "r", encoding="utf-8") as cfg_file:
            env_cfg = yaml.safe_load(cfg_file)
        shaping_gamma = float(
            env_cfg.get("rewards", {}).get("budget_potential_shaping", {}).get("gamma", 0.995)
        )

        agent = cls.__new__(cls)
        agent.yaml_path = str(yaml_path)
        agent.num_environments = 1
        agent.use_action_masking = bool(use_action_masking)
        agent.discount_factor_gamma = shaping_gamma
        agent.seed = 42
        agent.hyperparameters = {
            "yaml_path": agent.yaml_path,
            "use_action_masking": agent.use_action_masking,
            "discount_factor_gamma": agent.discount_factor_gamma,
            "device": str(device),
        }

        base_vec_env = DummyVecEnv(
            [
                make_single_env(
                    yaml_path=agent.yaml_path,
                    env_seed=42,
                    use_action_masking=agent.use_action_masking,
                )
            ]
        )
        agent.training_env = VecNormalize.load(vec_normalize_path, base_vec_env)
        agent.training_env.training = False
        agent.training_env.norm_reward = False

        algorithm_class = resolve_algorithm_class(agent.use_action_masking)
        agent.model = algorithm_class.load(
            model_path,
            env=agent.training_env,
            device=device,
        )
        agent.evaluator = StudentEvaluator(agent)
        agent.trainer = ExperimentTrainer(agent)
        agent.pretrainer = BehaviorCloningPretrainer(agent)
        return agent

    @classmethod
    def load_from_run(
        cls,
        run_dir: str | Path,
        checkpoint: str = "best",
        device: str = "auto",
    ) -> "StudentPPOAgent":
        """Memuat ulang model bersama env_config_snapshot.yaml dan VecNormalize run tersebut."""
        run_path = Path(run_dir)
        train_config = json.loads((run_path / "train_config.json").read_text(encoding="utf-8"))
        use_action_masking = bool(train_config["hyperparameters"].get("use_action_masking", False))

        model_file = run_path / f"{checkpoint}_model.zip"
        vec_file = run_path / f"{checkpoint}_vec_normalize.pkl"
        snapshot_yaml = run_path / "env_config_snapshot.yaml"

        return cls.load(
            model_path=str(model_file),
            vec_normalize_path=str(vec_file),
            yaml_path=str(snapshot_yaml),
            use_action_masking=use_action_masking,
            device=device,
        )
