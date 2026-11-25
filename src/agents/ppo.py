from datetime import datetime
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import stable_baselines3
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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


class StudentPPOAgent:
    """
    Wrapper konkret untuk melatih, mengevaluasi secara menyeluruh, menyimpan,
    dan memuat ulang agen MaskablePPO / PPO pada StudentLifeEnv dengan
    experiment tracking lengkap dan reproduksibilitas penuh.
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
        batch_size: int = 64,
        num_epochs: int = 10,
        discount_factor_gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coefficient: float = 0.015,
        use_action_masking: bool = True,
        seed: int = 42,
        device: str = "auto",
        tensorboard_log_dir: str | None = None,
        verbose: int = 1,
    ):
        set_global_seed(seed)

        self.yaml_path = str(yaml_path)
        self.num_environments = int(num_environments)
        self.use_action_masking = bool(use_action_masking)
        self.seed = int(seed)

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
            "discount_factor_gamma": float(discount_factor_gamma),
            "gae_lambda": float(gae_lambda),
            "clip_range": float(clip_range),
            "entropy_coefficient": float(entropy_coefficient),
            "use_action_masking": self.use_action_masking,
            "seed": self.seed,
            "device": str(device),
        }

        self.training_env = self._create_vectorized_env(
            num_environments=self.num_environments,
            seed=self.seed,
            training_mode=True,
        )

        policy_kwargs: dict[str, Any] = {
            "features_extractor_class": StudentFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": int(features_dim),
                "hidden_dim": int(hidden_dim),
            },
            "net_arch": {
                "pi": policy_layers,
                "vf": value_layers,
            },
            "activation_fn": nn.SiLU,
            "ortho_init": True,
        }

        algorithm_class = self._resolve_algorithm_class(self.use_action_masking)
        self.model = algorithm_class(
            policy="MlpPolicy",
            env=self.training_env,
            learning_rate=learning_rate,
            n_steps=rollout_steps_per_env,
            batch_size=batch_size,
            n_epochs=num_epochs,
            gamma=discount_factor_gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=entropy_coefficient,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log_dir,
            verbose=verbose,
        )

    def _resolve_algorithm_class(self, use_action_masking: bool):
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

    def _make_single_env(self, env_seed: int, render_mode: str | None = None):
        def _init():
            environment = StudentLifeEnv(yaml_path=self.yaml_path, render_mode=render_mode)
            environment = Monitor(environment)
            if self.use_action_masking:
                from sb3_contrib.common.wrappers import ActionMasker

                environment = ActionMasker(environment, lambda env: env.unwrapped.action_masks())
            environment.reset(seed=env_seed)
            return environment

        return _init

    def _create_vectorized_env(
        self,
        num_environments: int,
        seed: int,
        training_mode: bool,
        render_mode: str | None = None,
    ) -> VecNormalize:
        env_factories = [
            self._make_single_env(env_seed=seed + env_index, render_mode=render_mode)
            for env_index in range(num_environments)
        ]
        vectorized_env = DummyVecEnv(env_factories)
        return VecNormalize(
            vectorized_env,
            training=training_mode,
            norm_obs=True,
            norm_reward=training_mode,
            clip_obs=10.0,
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

    def _format_step_events_summary(self, step_events: dict[str, Any]) -> str:
        event_tags: list[str] = []
        if step_events.get("new_daily_task_added"):
            event_tags.append("new_daily_task")
        if step_events.get("forced_rest_due_to_mood_burnout"):
            event_tags.append("forced_rest_mood_burnout")
        if step_events.get("campus_activity_cancelled_no_transport"):
            event_tags.append("campus_cancelled_no_transport")
        if step_events.get("campus_activity_cancelled_insufficient_funds"):
            event_tags.append("campus_cancelled_low_balance")
        if step_events.get("transit_minimarket_impulse"):
            event_tags.append("transit_minimarket_impulse")
        if step_events.get("ccws_minimarket_impulse"):
            event_tags.append("ccws_minimarket_impulse")
        if "cafe_spend_idr" in step_events:
            cafe_total = int(step_events["cafe_spend_idr"]) + int(
                step_events.get("cafe_transport_idr", 0)
            )
            event_tags.append(f"cafe_total_Rp{cafe_total:,}")
        if step_events.get("purchased_groceries"):
            items = ",".join(str(item) for item in step_events["purchased_groceries"])
            event_tags.append(f"bought({items})")
        if step_events.get("meal_prep_created_signature"):
            event_tags.append(f"meal_prep({step_events['meal_prep_created_signature']})")
        if step_events.get("packed_bekal_signature"):
            event_tags.append(f"packed_bekal({step_events['packed_bekal_signature']})")
        if step_events.get("ate_bekal_at_campus"):
            event_tags.append("ate_bekal_campus")
        if step_events.get("morning_heavy_cooking"):
            event_tags.append("morning_heavy_cooking")
        boredom_penalty = float(step_events.get("boredom_mood_penalty", 0.0))
        if boredom_penalty < 0.0:
            event_tags.append(f"boredom({boredom_penalty:+.0f})")
        if step_events.get("skipped_meal_while_hungry"):
            event_tags.append("skipped_meal_hungry")
        if step_events.get("missed_scheduled_class"):
            event_tags.append("missed_class")
        if step_events.get("night_cramming"):
            event_tags.append("night_cramming")
        if step_events.get("distracted_by_craving"):
            event_tags.append("distracted_craving")
        if step_events.get("starvation_active"):
            event_tags.append("starvation")
        if step_events.get("mood_burnout_penalty_applied"):
            event_tags.append("mood_burnout")
        if step_events.get("stress_burnout_penalty_applied"):
            event_tags.append("stress_burnout")
        spoiled_units = int(step_events.get("spoiled_units", 0))
        if spoiled_units > 0:
            event_tags.append(f"spoiled({spoiled_units})")
        if step_events.get("termination_reason"):
            event_tags.append(f"end:{step_events['termination_reason']}")
        return " | ".join(event_tags)

    def _decode_requested_action(
        self, base_env: StudentLifeEnv, raw_action: np.ndarray
    ) -> dict[str, str]:
        action_values = [int(value) for value in np.asarray(raw_action).reshape(-1)]
        (
            meal_mode_index,
            cook_rice_index,
            cook_egg_index,
            cook_chicken_index,
            cook_tempe_index,
            mobility_index,
            activity_index,
            buy_rice_index,
            buy_egg_index,
            buy_chicken_index,
            buy_tempe_index,
        ) = action_values

        cook_parts: list[str] = []
        rice_qty = int(base_env.cook_rice_options[cook_rice_index])
        egg_qty = int(base_env.cook_egg_options[cook_egg_index])
        chicken_qty = int(base_env.cook_chicken_options[cook_chicken_index])
        tempe_qty = int(base_env.cook_tempe_options[cook_tempe_index])
        if rice_qty > 0:
            cook_parts.append(f"{rice_qty}rice")
        if egg_qty > 0:
            cook_parts.append(f"{egg_qty}egg")
        if chicken_qty > 0:
            cook_parts.append(f"{chicken_qty}chicken")
        if tempe_qty > 0:
            cook_parts.append(f"{tempe_qty}tempe")

        buy_parts: list[str] = []
        if bool(base_env.buy_rice_options[buy_rice_index]):
            buy_parts.append("rice")
        if bool(base_env.buy_egg_options[buy_egg_index]):
            buy_parts.append("egg")
        if bool(base_env.buy_chicken_options[buy_chicken_index]):
            buy_parts.append("chicken")
        if bool(base_env.buy_tempe_options[buy_tempe_index]):
            buy_parts.append("tempe")

        return {
            "requested_meal_mode": base_env.meal_mode_names[meal_mode_index],
            "requested_cook_combo": "+".join(cook_parts),
            "requested_mobility": base_env.mobility_names[mobility_index],
            "requested_activity": base_env.activity_names[activity_index],
            "requested_groceries": ",".join(buy_parts),
        }

    def evaluate_detailed(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        eval_seed: int = 1000,
        record_best_episode_trace: bool = False,
        record_first_episode_trace: bool = False,
        render: bool = False,
    ) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
        """
        Mengevaluasi model pada environment terpisah dengan VecNormalize dibekukan.
        Mengembalikan (summary_metrics, episodes_dataframe, trace_dataframe).
        """
        should_record_traces = bool(record_best_episode_trace or record_first_episode_trace)
        render_mode = "human" if render else None
        evaluation_env = self._create_vectorized_env(
            num_environments=1,
            seed=eval_seed,
            training_mode=False,
            render_mode=render_mode,
        )
        evaluation_env.obs_rms = self.training_env.obs_rms
        evaluation_env.ret_rms = self.training_env.ret_rms

        base_env: StudentLifeEnv = evaluation_env.envs[0].unwrapped
        feature_index = {
            name: idx for idx, name in enumerate(base_env.observation_feature_names)
        }
        energy_idx = feature_index["energy"]
        hunger_idx = feature_index["hunger"]
        mood_idx = feature_index["mood"]
        stress_idx = feature_index["academic_stress"]
        craving_idx = feature_index["cafe_craving"]

        episode_records: list[dict[str, Any]] = []
        best_episode_reward = -np.inf
        best_episode_trace_records: list[dict[str, Any]] = []
        first_episode_trace_records: list[dict[str, Any]] = []

        for episode_index in range(num_episodes):
            episode_seed = eval_seed + episode_index
            evaluation_env.seed(episode_seed)
            observation = evaluation_env.reset()

            episode_done = False
            cumulative_reward = 0.0
            steps_taken = 0
            terminal_info: dict[str, Any] = {}
            current_episode_trace: list[dict[str, Any]] = []

            energy_history: list[float] = []
            hunger_history: list[float] = []
            mood_history: list[float] = []
            stress_history: list[float] = []
            craving_history: list[float] = []

            meal_counts = {
                "skip": 0,
                "cook_fresh": 0,
                "meal_prep": 0,
                "reheat_prep": 0,
                "eat_bekal": 0,
                "pack_bekal": 0,
                "warung_hemat": 0,
                "warung_variasi": 0,
                "craving_snack": 0,
            }
            activity_counts = {name: 0 for name in base_env.activity_names}
            route_counts = {name: 0 for name in base_env.mobility_names}
            route_counts["emergency_return"] = 0

            starvation_steps = 0
            mood_burnout_steps = 0
            stress_burnout_steps = 0
            night_cramming_count = 0
            minimarket_impulse_count = 0

            while not episode_done:
                step_before = base_env.current_step
                location_before = base_env.current_location
                balance_before = float(base_env.balance_idr)

                day_before, _, step_in_day_before, day_name_before = (
                    base_env.academics.get_time_context(step_before)
                )
                step_definition_before = base_env.academics.get_step_definition(step_in_day_before)
                class_scheduled_before = base_env.academics.is_class_scheduled(
                    day_name_before, step_in_day_before
                )

                if render:
                    base_env.render()

                current_masks = base_env.action_masks() if self.use_action_masking else None
                action, _ = self.predict(
                    normalized_observation=observation,
                    deterministic=deterministic,
                    action_masks=current_masks,
                )
                decoded_request = (
                    self._decode_requested_action(base_env, action)
                    if should_record_traces
                    else {}
                )

                observation, rewards, dones, infos = evaluation_env.step(action)

                step_reward = float(rewards[0])
                cumulative_reward += step_reward
                steps_taken += 1
                episode_done = bool(dones[0])
                step_info = infos[0]
                step_events = step_info.get("step_events", {})

                if episode_done and "terminal_observation" in step_info:
                    raw_observation = evaluation_env.unnormalize_obs(
                        step_info["terminal_observation"]
                    )
                    terminal_info = step_info
                else:
                    raw_observation = evaluation_env.get_original_obs()[0]

                current_energy = float(raw_observation[energy_idx])
                current_hunger = float(raw_observation[hunger_idx])
                current_mood = float(raw_observation[mood_idx])
                current_stress = float(raw_observation[stress_idx])
                current_craving = float(raw_observation[craving_idx])

                energy_history.append(current_energy)
                hunger_history.append(current_hunger)
                mood_history.append(current_mood)
                stress_history.append(current_stress)
                craving_history.append(current_craving)

                executed_meal = str(step_events.get("executed_meal_mode", "skip"))
                executed_activity = str(step_events.get("executed_activity", "rest_or_sleep"))
                executed_route = str(step_events.get("executed_route", "stay"))

                if executed_meal in meal_counts:
                    meal_counts[executed_meal] += 1
                if executed_activity in activity_counts:
                    activity_counts[executed_activity] += 1
                if executed_route in route_counts:
                    route_counts[executed_route] += 1

                if step_events.get("starvation_active"):
                    starvation_steps += 1
                if step_events.get("mood_burnout_penalty_applied"):
                    mood_burnout_steps += 1
                if step_events.get("stress_burnout_penalty_applied"):
                    stress_burnout_steps += 1
                if step_events.get("night_cramming"):
                    night_cramming_count += 1
                if step_events.get("transit_minimarket_impulse") or step_events.get(
                    "ccws_minimarket_impulse"
                ):
                    minimarket_impulse_count += 1

                if should_record_traces:
                    pantry_totals = step_info.get("pantry_totals", {})
                    balance_after = float(step_info.get("balance_idr", 0.0))
                    purchased_groceries = step_events.get("purchased_groceries", [])

                    current_episode_trace.append({
                        "episode_index": episode_index,
                        "episode_seed": episode_seed,
                        "step": step_before,
                        "day": day_before,
                        "day_name": day_name_before,
                        "step_in_day": step_in_day_before,
                        "time_slot": str(step_definition_before["name"]),
                        "hours": str(step_definition_before["hours"]),
                        "climate": str(step_definition_before["climate"]),
                        "ojol_tariff": str(step_definition_before["ojol_tariff"]),
                        "class_scheduled": bool(class_scheduled_before),
                        "location_before": location_before,
                        "location_after": str(step_info.get("current_location", "")),
                        "requested_mobility": decoded_request["requested_mobility"],
                        "executed_route": executed_route,
                        "requested_activity": decoded_request["requested_activity"],
                        "executed_activity": executed_activity,
                        "requested_meal_mode": decoded_request["requested_meal_mode"],
                        "requested_cook_combo": decoded_request["requested_cook_combo"],
                        "executed_meal_mode": executed_meal,
                        "meal_signature": str(step_events.get("meal_signature", "")),
                        "purchased_groceries": ",".join(str(item) for item in purchased_groceries),
                        "step_spend_idr": max(0.0, balance_before - balance_after),
                        "balance_idr": balance_after,
                        "energy": current_energy,
                        "hunger": current_hunger,
                        "mood": current_mood,
                        "academic_stress": current_stress,
                        "cafe_craving": current_craving,
                        "rice_stock": int(pantry_totals.get("rice", 0)),
                        "egg_stock": int(pantry_totals.get("egg", 0)),
                        "chicken_stock": int(pantry_totals.get("chicken", 0)),
                        "tempe_stock": int(pantry_totals.get("tempe", 0)),
                        "meal_prep_stock": int(pantry_totals.get("meal_prep", 0)),
                        "packed_lunch_stock": int(pantry_totals.get("packed_lunch", 0)),
                        "spoiled_units_step": int(step_events.get("spoiled_units", 0)),
                        "missed_class": bool(step_events.get("missed_scheduled_class", False)),
                        "events_summary": self._format_step_events_summary(step_events),
                        "step_reward": step_reward,
                        "cumulative_reward": cumulative_reward,
                    })

            if should_record_traces:
                if episode_index == 0:
                    first_episode_trace_records = current_episode_trace
                if cumulative_reward > best_episode_reward:
                    best_episode_reward = cumulative_reward
                    best_episode_trace_records = current_episode_trace

            terminal_events = terminal_info.get("step_events", {})
            termination_reason = str(terminal_events.get("termination_reason", "unknown"))
            classes_scheduled = int(terminal_info.get("total_classes_scheduled", 0))
            classes_attended = int(terminal_info.get("total_classes_attended", 0))
            attendance_rate = (
                classes_attended / classes_scheduled if classes_scheduled > 0 else 1.0
            )
            spending = terminal_info.get("spending_by_category_idr", {})

            episode_records.append({
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "episode_reward": cumulative_reward,
                "episode_steps": steps_taken,
                "termination_reason": termination_reason,
                "completed_30_days": int(termination_reason == "completed_30_days"),
                "final_balance_idr": float(terminal_info.get("balance_idr", 0.0)),
                "spent_transport_idr": float(spending.get("transport", 0.0)),
                "spent_grocery_idr": float(spending.get("grocery", 0.0)),
                "spent_external_food_idr": float(spending.get("external_food", 0.0)),
                "spent_cafe_idr": float(spending.get("cafe", 0.0)),
                "spent_minimarket_impulse_idr": float(spending.get("minimarket_impulse", 0.0)),
                "classes_attended": classes_attended,
                "classes_scheduled": classes_scheduled,
                "class_attendance_rate": attendance_rate,
                "mean_energy": float(np.mean(energy_history)),
                "mean_hunger": float(np.mean(hunger_history)),
                "mean_mood": float(np.mean(mood_history)),
                "mean_academic_stress": float(np.mean(stress_history)),
                "mean_cafe_craving": float(np.mean(craving_history)),
                "starvation_steps": starvation_steps,
                "mood_burnout_steps": mood_burnout_steps,
                "stress_burnout_steps": stress_burnout_steps,
                "night_cramming_count": night_cramming_count,
                "total_spoiled_units": int(terminal_info.get("total_spoiled_units", 0)),
                "minimarket_impulse_count": minimarket_impulse_count,
                "cook_fresh_count": meal_counts["cook_fresh"],
                "meal_prep_count": meal_counts["meal_prep"],
                "reheat_prep_count": meal_counts["reheat_prep"],
                "pack_bekal_count": meal_counts["pack_bekal"],
                "eat_bekal_count": meal_counts["eat_bekal"],
                "warung_hemat_count": meal_counts["warung_hemat"],
                "warung_variasi_count": meal_counts["warung_variasi"],
                "craving_snack_count": meal_counts["craving_snack"],
                "skip_meal_count": meal_counts["skip"],
                "grocery_shop_count": activity_counts.get("grocery_shop", 0),
                "room_study_count": activity_counts.get("room_study", 0),
                "ccws_stay_count": activity_counts.get("ccws_stay", 0),
                "cafe_session_count": activity_counts.get("cafe_session", 0),
                "wait_at_department_count": activity_counts.get("wait_at_department", 0),
                "direct_ojol_count": route_counts.get("direct_ojol", 0),
                "transit_ccws_ojol_count": route_counts.get("transit_ccws_ojol", 0),
                "emergency_return_count": route_counts.get("emergency_return", 0),
            })

        evaluation_env.close()

        episodes_dataframe = pd.DataFrame(episode_records)
        selected_trace_records = (
            best_episode_trace_records if record_best_episode_trace else first_episode_trace_records
        )
        trace_dataframe = pd.DataFrame(selected_trace_records)

        best_row = episodes_dataframe.loc[episodes_dataframe["episode_reward"].idxmax()]
        summary_metrics = {
            "mean_episode_reward": float(episodes_dataframe["episode_reward"].mean()),
            "std_episode_reward": float(episodes_dataframe["episode_reward"].std(ddof=0)),
            "min_episode_reward": float(episodes_dataframe["episode_reward"].min()),
            "max_episode_reward": float(episodes_dataframe["episode_reward"].max()),
            "best_eval_episode_index": int(best_row["episode_index"]),
            "best_eval_episode_seed": int(best_row["episode_seed"]),
            "mean_episode_steps": float(episodes_dataframe["episode_steps"].mean()),
            "completion_rate": float(episodes_dataframe["completed_30_days"].mean()),
            "mean_class_attendance_rate": float(episodes_dataframe["class_attendance_rate"].mean()),
            "mean_final_balance_idr": float(episodes_dataframe["final_balance_idr"].mean()),
            "mean_spent_transport_idr": float(episodes_dataframe["spent_transport_idr"].mean()),
            "mean_spent_grocery_idr": float(episodes_dataframe["spent_grocery_idr"].mean()),
            "mean_spent_external_food_idr": float(
                episodes_dataframe["spent_external_food_idr"].mean()
            ),
            "mean_spent_cafe_idr": float(episodes_dataframe["spent_cafe_idr"].mean()),
            "mean_spent_minimarket_impulse_idr": float(
                episodes_dataframe["spent_minimarket_impulse_idr"].mean()
            ),
            "mean_energy": float(episodes_dataframe["mean_energy"].mean()),
            "mean_hunger": float(episodes_dataframe["mean_hunger"].mean()),
            "mean_mood": float(episodes_dataframe["mean_mood"].mean()),
            "mean_academic_stress": float(episodes_dataframe["mean_academic_stress"].mean()),
            "mean_cafe_craving": float(episodes_dataframe["mean_cafe_craving"].mean()),
            "mean_starvation_steps": float(episodes_dataframe["starvation_steps"].mean()),
            "mean_spoiled_units": float(episodes_dataframe["total_spoiled_units"].mean()),
            "mean_grocery_shop_count": float(episodes_dataframe["grocery_shop_count"].mean()),
            "mean_cook_fresh_count": float(episodes_dataframe["cook_fresh_count"].mean()),
            "mean_meal_prep_count": float(episodes_dataframe["meal_prep_count"].mean()),
            "mean_reheat_prep_count": float(episodes_dataframe["reheat_prep_count"].mean()),
            "mean_pack_bekal_count": float(episodes_dataframe["pack_bekal_count"].mean()),
            "mean_eat_bekal_count": float(episodes_dataframe["eat_bekal_count"].mean()),
            "mean_warung_hemat_count": float(episodes_dataframe["warung_hemat_count"].mean()),
            "mean_warung_variasi_count": float(episodes_dataframe["warung_variasi_count"].mean()),
            "mean_cafe_session_count": float(episodes_dataframe["cafe_session_count"].mean()),
            "mean_night_cramming_count": float(episodes_dataframe["night_cramming_count"].mean()),
            "mean_direct_ojol_count": float(episodes_dataframe["direct_ojol_count"].mean()),
            "mean_transit_ccws_ojol_count": float(
                episodes_dataframe["transit_ccws_ojol_count"].mean()
            ),
        }

        return summary_metrics, episodes_dataframe, trace_dataframe

    def record_episode_trace(
        self,
        episode_seed: int,
        deterministic: bool = True,
        render: bool = False,
    ) -> pd.DataFrame:
        """Merekam trace step-by-step lengkap (150 step) untuk satu seed episode tertentu."""
        _, _, trace_dataframe = self.evaluate_detailed(
            num_episodes=1,
            deterministic=deterministic,
            eval_seed=episode_seed,
            record_best_episode_trace=True,
            render=render,
        )
        return trace_dataframe

    def evaluate(
        self,
        num_episodes: int = 5,
        deterministic: bool = True,
        eval_seed: int = 1000,
        render: bool = False,
    ) -> dict[str, float]:
        summary_metrics, _, _ = self.evaluate_detailed(
            num_episodes=num_episodes,
            deterministic=deterministic,
            eval_seed=eval_seed,
            record_best_episode_trace=False,
            render=render,
        )
        return summary_metrics

    def train_with_logging(
        self,
        total_timesteps: int = 250_000,
        eval_interval_steps: int = 25_000,
        eval_episodes: int = 10,
        final_eval_episodes: int = 20,
        eval_seed: int = 1000,
        experiment_name: str = "maskable_ppo_5step",
        runs_root_dir: str = "runs",
        progress_bar: bool = False,
    ) -> dict[str, Any]:
        """
        Menjalankan training loop terstruktur dengan snapshot config, best checkpoint,
        riwayat evaluasi berkala, trace episode terbaik, dan master log eksperimen.
        """
        config_sha256 = compute_file_sha256(self.yaml_path)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_experiment_name = experiment_name.strip().replace(" ", "_")
        run_id = f"{timestamp_str}_{safe_experiment_name}_{config_sha256[:8]}"
        run_dir = Path(runs_root_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        snapshot_yaml_path = run_dir / "env_config_snapshot.yaml"
        shutil.copy2(self.yaml_path, snapshot_yaml_path)

        library_versions: dict[str, str] = {
            "gymnasium": gym.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        if self.use_action_masking:
            import sb3_contrib

            library_versions["sb3_contrib"] = sb3_contrib.__version__

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
            "hyperparameters": self.hyperparameters,
            "library_versions": library_versions,
        }
        (run_dir / "train_config.json").write_text(
            json.dumps(train_metadata, indent=2), encoding="utf-8"
        )

        eval_history_rows: list[dict[str, Any]] = []
        best_mean_reward = -np.inf
        best_timestep = 0

        best_model_path = run_dir / "best_model.zip"
        best_vec_path = run_dir / "best_vec_normalize.pkl"
        final_model_path = run_dir / "final_model.zip"
        final_vec_path = run_dir / "final_vec_normalize.pkl"

        timesteps_completed = 0
        is_first_chunk = True

        while timesteps_completed < total_timesteps:
            chunk_timesteps = min(eval_interval_steps, total_timesteps - timesteps_completed)
            self.training_env.training = True
            self.training_env.norm_reward = True
            self.model.learn(
                total_timesteps=chunk_timesteps,
                reset_num_timesteps=is_first_chunk,
                progress_bar=progress_bar,
            )
            is_first_chunk = False
            timesteps_completed = int(self.model.num_timesteps)

            eval_summary, _, _ = self.evaluate_detailed(
                num_episodes=eval_episodes,
                deterministic=True,
                eval_seed=eval_seed,
                record_best_episode_trace=False,
            )
            eval_row = {"timesteps": timesteps_completed, **eval_summary}
            eval_history_rows.append(eval_row)

            current_mean_reward = eval_summary["mean_episode_reward"]
            is_new_best = current_mean_reward > best_mean_reward
            if is_new_best:
                best_mean_reward = current_mean_reward
                best_timestep = timesteps_completed
                self.save(str(best_model_path), str(best_vec_path))

            best_tag = " [BEST]" if is_new_best else ""
            print(
                f"Step {timesteps_completed:>7d}/{total_timesteps} | "
                f"Reward: {current_mean_reward:>7.2f} +/- {eval_summary['std_episode_reward']:>5.2f} | "
                f"Complete: {eval_summary['completion_rate'] * 100:>5.1f}% | "
                f"Kelas: {eval_summary['mean_class_attendance_rate'] * 100:>5.1f}% | "
                f"Sisa: Rp {eval_summary['mean_final_balance_idr']:>9,.0f} | "
                f"Masak/Bekal: {eval_summary['mean_cook_fresh_count'] + eval_summary['mean_reheat_prep_count'] + eval_summary['mean_eat_bekal_count']:>4.1f}"
                f"{best_tag}"
            )

            pd.DataFrame(eval_history_rows).to_csv(run_dir / "eval_history.csv", index=False)

        self.save(str(final_model_path), str(final_vec_path))

        best_agent = self.load(
            model_path=str(best_model_path),
            vec_normalize_path=str(best_vec_path),
            yaml_path=str(snapshot_yaml_path),
            use_action_masking=self.use_action_masking,
            device=str(self.hyperparameters["device"]),
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
            "seed": self.seed,
            "use_action_masking": self.use_action_masking,
            "total_timesteps": int(timesteps_completed),
            "best_checkpoint_timestep": int(best_timestep),
            **final_summary,
        }

        (run_dir / "final_eval_summary.json").write_text(
            json.dumps(final_summary_record, indent=2), encoding="utf-8"
        )
        final_episodes_df.to_csv(run_dir / "final_eval_episodes.csv", index=False)
        best_episode_trace_df.to_csv(run_dir / "best_episode_trace.csv", index=False)

        master_summary_path = Path(runs_root_dir) / "experiments_summary.csv"
        summary_row_df = pd.DataFrame([final_summary_record])
        if master_summary_path.exists():
            existing_master_df = pd.read_csv(master_summary_path)
            updated_master_df = pd.concat([existing_master_df, summary_row_df], ignore_index=True)
        else:
            updated_master_df = summary_row_df
        updated_master_df.to_csv(master_summary_path, index=False)

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
        agent = cls.__new__(cls)
        agent.yaml_path = str(yaml_path)
        agent.num_environments = 1
        agent.use_action_masking = bool(use_action_masking)
        agent.seed = 42
        agent.hyperparameters = {
            "yaml_path": agent.yaml_path,
            "use_action_masking": agent.use_action_masking,
            "device": str(device),
        }

        base_vec_env = DummyVecEnv([agent._make_single_env(env_seed=42)])
        agent.training_env = VecNormalize.load(vec_normalize_path, base_vec_env)
        agent.training_env.training = False
        agent.training_env.norm_reward = False

        algorithm_class = agent._resolve_algorithm_class(agent.use_action_masking)
        agent.model = algorithm_class.load(
            model_path,
            env=agent.training_env,
            device=device,
        )
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
