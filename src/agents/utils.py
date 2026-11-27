import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env import StudentLifeEnv


def set_global_seed(seed: int) -> None:
    """Mengunci seluruh sumber randomness agar eksperimen reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_file_sha256(file_path: str | Path) -> str:
    content_bytes = Path(file_path).read_bytes()
    return hashlib.sha256(content_bytes).hexdigest()


def resolve_algorithm_class(use_action_masking: bool):
    if not use_action_masking:
        return PPO
    try:
        from sb3_contrib import MaskablePPO

        return MaskablePPO
    except ImportError as import_error:
        raise ImportError(
            "use_action_masking=True membutuhkan paket 'sb3-contrib'. "
            "Jalankan dengan use_action_masking=False atau instal sb3-contrib."
        ) from import_error


def make_single_env(
    yaml_path: str,
    env_seed: int,
    use_action_masking: bool = True,
    render_mode: str | None = None,
    exploring_starts_pantry: bool = False,
):
    def _init():
        environment = StudentLifeEnv(
            yaml_path=yaml_path,
            render_mode=render_mode,
            exploring_starts_pantry=exploring_starts_pantry,
        )
        environment = Monitor(environment)
        if use_action_masking:
            from sb3_contrib.common.wrappers import ActionMasker

            environment = ActionMasker(environment, lambda env: env.unwrapped.action_masks())
        environment.reset(seed=env_seed)
        return environment

    return _init


def create_vectorized_env(
    yaml_path: str,
    num_environments: int,
    seed: int,
    training_mode: bool,
    discount_factor_gamma: float = 0.995,
    use_action_masking: bool = True,
    render_mode: str | None = None,
    exploring_starts_pantry: bool = False,
) -> VecNormalize:
    env_factories = [
        make_single_env(
            yaml_path=yaml_path,
            env_seed=seed + env_index,
            use_action_masking=use_action_masking,
            render_mode=render_mode,
            exploring_starts_pantry=bool(training_mode and exploring_starts_pantry),
        )
        for env_index in range(num_environments)
    ]
    vectorized_env = DummyVecEnv(env_factories)
    return VecNormalize(
        vectorized_env,
        training=training_mode,
        norm_obs=True,
        norm_reward=training_mode,
        clip_obs=10.0,
        clip_reward=25.0,
        gamma=float(discount_factor_gamma),
    )
