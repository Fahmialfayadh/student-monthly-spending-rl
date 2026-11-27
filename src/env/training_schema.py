from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.env.schema import WorldConfig


class PPOHyperparamsConfig(BaseModel):
    """Hyperparameter MaskablePPO / PPO & arsitektur jaringan Actor-Critic."""

    seed: int = Field(default=42, ge=0)
    num_environments: int = Field(default=4, ge=1, le=32)
    features_dim: int = Field(default=128, ge=16, le=1024)
    hidden_dim: int = Field(default=128, ge=16, le=1024)
    policy_hidden_layers: list[int] = Field(default_factory=lambda: [128, 128], min_length=1)
    value_hidden_layers: list[int] = Field(default_factory=lambda: [128, 128], min_length=1)
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0)
    rollout_steps_per_env: int = Field(default=300, ge=16, le=8192)
    batch_size: int = Field(default=120, ge=8, le=4096)
    num_epochs: int = Field(default=10, ge=1, le=100)
    discount_factor_gamma: float = Field(default=0.995, gt=0.0, le=1.0)
    gae_lambda: float = Field(default=0.98, ge=0.0, le=1.0)
    clip_range: float = Field(default=0.2, gt=0.0, le=1.0)
    entropy_coefficient: float = Field(default=0.015, ge=0.0, le=1.0)
    target_kl: float | None = Field(default=0.02, gt=0.0, le=1.0)
    use_action_masking: bool = True
    share_features_extractor: bool = False
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @model_validator(mode="after")
    def _validate_rollout_batch(self) -> "PPOHyperparamsConfig":
        total_rollout = self.rollout_steps_per_env * self.num_environments
        if self.batch_size > total_rollout:
            raise ValueError(
                f"batch_size ({self.batch_size}) tidak boleh melebihi "
                f"rollout_steps_per_env * num_environments ({total_rollout})."
            )
        return self


class BehaviorCloningConfig(BaseModel):
    """Hyperparameter warm-start Behavior Cloning sebelum fine-tuning PPO."""

    enabled: bool = True
    num_episodes: int = Field(default=80, ge=1, le=2000)
    epochs: int = Field(default=25, ge=1, le=500)
    batch_size: int = Field(default=256, ge=16, le=4096)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    entropy_coef: float = Field(default=0.03, ge=0.0, le=1.0)
    vf_coef: float = Field(default=0.5, ge=0.0, le=10.0)
    demo_seed: int = Field(default=5000, ge=0)


class TrainingScheduleConfig(BaseModel):
    """Jadwal durasi training, checkpointing, dan evaluasi berkala."""

    total_timesteps: int = Field(default=3_000_000, ge=64)
    eval_interval_steps: int = Field(default=100_000, ge=64)
    eval_episodes: int = Field(default=10, ge=1, le=200)
    final_eval_episodes: int = Field(default=20, ge=1, le=200)
    eval_seed: int = Field(default=1000, ge=0)
    survival_phase_timesteps: int = Field(default=1_000_000, ge=0)


class TrainingProfileConfig(BaseModel):
    """Konfigurasi lengkap training RL (PPO + Behavior Cloning + Schedule)."""

    experiment_name: str = Field(default="student_agent", min_length=1, max_length=64)
    runs_root_dir: str = "runs"
    ppo: PPOHyperparamsConfig = Field(default_factory=PPOHyperparamsConfig)
    bc: BehaviorCloningConfig = Field(default_factory=BehaviorCloningConfig)
    schedule: TrainingScheduleConfig = Field(default_factory=TrainingScheduleConfig)


def validate_training_against_world(
    training_cfg: TrainingProfileConfig,
    world_cfg: WorldConfig,
) -> list[str]:
    """
    Validasi silang antara TrainingProfileConfig dan WorldConfig.
    Mengembalikan daftar peringatan non-fatal; mengangkat ValueError bila inkompatibel.
    """
    shaping = world_cfg.rewards.budget_potential_shaping
    if shaping.enabled and abs(training_cfg.ppo.discount_factor_gamma - shaping.gamma) > 1e-6:
        raise ValueError(
            f"ppo.discount_factor_gamma ({training_cfg.ppo.discount_factor_gamma}) harus sama "
            f"dengan rewards.budget_potential_shaping.gamma ({shaping.gamma}) saat shaping aktif."
        )

    warnings: list[str] = []
    total_rollout = training_cfg.ppo.rollout_steps_per_env * training_cfg.ppo.num_environments
    if total_rollout % training_cfg.ppo.batch_size != 0:
        warnings.append(
            f"rollout_steps_per_env * num_environments ({total_rollout}) tidak habis dibagi "
            f"batch_size ({training_cfg.ppo.batch_size}); minibatch terakhir akan terpotong."
        )
    return warnings


def export_training_json_schema() -> dict[str, Any]:
    """Mengekspor JSON Schema lengkap TrainingProfileConfig untuk form hyperparameter UI."""
    return TrainingProfileConfig.model_json_schema()
