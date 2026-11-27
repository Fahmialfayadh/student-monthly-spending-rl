import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class StudentFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom PyTorch feature extractor untuk observasi tabular StudentLifeEnv.
    Menggunakan proyeksi input + LayerNorm + SiLU serta satu residual block.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        input_dimension = int(observation_space.shape[0])

        self.input_projection = nn.Sequential(
            nn.Linear(input_dimension, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.residual_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.residual_activation = nn.SiLU()

        self.output_projection = (
            nn.Sequential(
                nn.Linear(hidden_dim, features_dim),
                nn.LayerNorm(features_dim),
                nn.SiLU(),
            )
            if hidden_dim != features_dim
            else nn.Identity()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden_representation = self.input_projection(observations)
        residual_output = self.residual_activation(
            hidden_representation + self.residual_block(hidden_representation)
        )
        return self.output_projection(residual_output)
