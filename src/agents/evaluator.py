from typing import Any

import numpy as np
import pandas as pd

from src.agents.utils import create_vectorized_env
from src.env import StudentLifeEnv
from src.env.pantry import PantryManager


def format_step_events_summary(step_events: dict[str, Any]) -> str:
    event_tags: list[str] = []
    if step_events.get("is_raining"):
        event_tags.append("rain")
    if step_events.get("new_daily_task_added"):
        event_tags.append("new_daily_task")
    if step_events.get("forced_rest_due_to_mood_burnout"):
        event_tags.append("forced_rest_mood_burnout")
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
        portions = int(step_events.get("meal_prep_portions_created", 0))
        event_tags.append(f"meal_prep({step_events['meal_prep_created_signature']}x{portions})")
    if step_events.get("ate_fresh_from_meal_prep"):
        event_tags.append("ate_fresh_prep")
    if step_events.get("packed_bekal_signature"):
        event_tags.append(f"packed_bekal({step_events['packed_bekal_signature']})")
    if step_events.get("ate_bekal_at_campus"):
        event_tags.append("ate_bekal_campus")
    if step_events.get("morning_heavy_cooking"):
        event_tags.append("morning_heavy_cooking")
    boredom_penalty = float(step_events.get("boredom_mood_penalty", 0.0))
    if boredom_penalty < 0.0:
        event_tags.append(f"boredom({boredom_penalty:+.1f})")
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
    if step_events.get("low_attendance_penalty_applied"):
        event_tags.append("low_attendance_penalty")
    spoiled_units = int(step_events.get("spoiled_units", 0))
    if spoiled_units > 0:
        event_tags.append(f"spoiled({spoiled_units})")
    if step_events.get("termination_reason"):
        event_tags.append(f"end:{step_events['termination_reason']}")
    return " | ".join(event_tags)


def decode_requested_action(
    base_env: StudentLifeEnv, raw_action: np.ndarray
) -> dict[str, str]:
    action_values = [int(value) for value in np.asarray(raw_action).reshape(-1)]
    (
        activity_index,
        route_index,
        meal_mode_index,
        cook_combo_index,
        bring_bekal_index,
        grocery_bundle_index,
    ) = action_values

    cook_qty = PantryManager.decode_cook_combo(cook_combo_index)
    cook_parts = [f"{qty}{name}" for name, qty in cook_qty.items() if qty > 0]

    buy_flags = PantryManager.decode_grocery_bundle(grocery_bundle_index)
    buy_parts = [name for name, flag in buy_flags.items() if flag]

    return {
        "requested_activity": base_env.activity_names[activity_index],
        "requested_mobility": base_env.route_names[route_index],
        "requested_meal_mode": base_env.meal_mode_names[meal_mode_index],
        "requested_cook_combo": "+".join(cook_parts),
        "requested_bring_bekal": str(
            bool(base_env.action_dimensions["bring_bekal"][bring_bekal_index])
        ),
        "requested_groceries": ",".join(buy_parts),
    }


class StudentEvaluator:
    """
    Menjalankan evaluasi deterministik/stokastik pada environment terpisah dengan
    statistik VecNormalize dibekukan, mengumpulkan ringkasan metrik per episode
    serta trace step-by-step untuk analisis perilaku agen.
    """

    def __init__(self, agent: Any):
        self.agent = agent

    def evaluate_detailed(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        eval_seed: int = 1000,
        record_best_episode_trace: bool = False,
        record_first_episode_trace: bool = False,
        render: bool = False,
    ) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
        should_record_traces = bool(record_best_episode_trace or record_first_episode_trace)
        render_mode = "human" if render else None
        evaluation_env = create_vectorized_env(
            yaml_path=self.agent.yaml_path,
            num_environments=1,
            seed=eval_seed,
            training_mode=False,
            discount_factor_gamma=self.agent.discount_factor_gamma,
            use_action_masking=self.agent.use_action_masking,
            render_mode=render_mode,
        )
        evaluation_env.obs_rms = self.agent.training_env.obs_rms
        evaluation_env.ret_rms = self.agent.training_env.ret_rms

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
                "apartemen_warung": 0,
                "delivery_food": 0,
                "warung_hemat": 0,
                "warung_variasi": 0,
            }
            pack_bekal_count = 0
            activity_counts = {name: 0 for name in base_env.activity_names}
            route_counts = {
                "stay": 0,
                "direct_ojol": 0,
                "transit_ccws_ojol": 0,
                "emergency_return": 0,
            }

            starvation_steps = 0
            mood_burnout_steps = 0
            stress_burnout_steps = 0
            night_cramming_count = 0
            minimarket_impulse_count = 0

            while not episode_done:
                step_before = base_env.current_step
                location_before = base_env.current_location
                balance_before = float(base_env.balance_idr)
                raining_before = bool(base_env.current_is_raining)

                day_before, _, step_in_day_before, day_name_before = (
                    base_env.academics.get_time_context(step_before)
                )
                step_definition_before = base_env.academics.get_step_definition(step_in_day_before)
                class_scheduled_before = base_env.academics.is_class_scheduled(
                    day_name_before, step_in_day_before
                )

                if render:
                    base_env.render()

                current_masks = base_env.action_masks() if self.agent.use_action_masking else None
                action, _ = self.agent.predict(
                    normalized_observation=observation,
                    deterministic=deterministic,
                    action_masks=current_masks,
                )
                decoded_request = (
                    decode_requested_action(base_env, action)
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
                if step_events.get("packed_bekal_signature"):
                    pack_bekal_count += 1
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
                        "climate": (
                            "rain" if raining_before else str(step_definition_before["climate"])
                        ),
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
                        "events_summary": format_step_events_summary(step_events),
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
                "days_survived": float(steps_taken / base_env.steps_per_day),
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
                "pack_bekal_count": pack_bekal_count,
                "eat_bekal_count": meal_counts["eat_bekal"],
                "apartemen_warung_count": meal_counts["apartemen_warung"],
                "delivery_food_count": meal_counts["delivery_food"],
                "warung_hemat_count": meal_counts["warung_hemat"],
                "warung_variasi_count": meal_counts["warung_variasi"],
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

        summary_metrics = self._aggregate_episode_metrics(episodes_dataframe)
        return summary_metrics, episodes_dataframe, trace_dataframe

    @staticmethod
    def _aggregate_episode_metrics(episodes_dataframe: pd.DataFrame) -> dict[str, float]:
        best_row = episodes_dataframe.loc[episodes_dataframe["episode_reward"].idxmax()]
        return {
            "mean_episode_reward": float(episodes_dataframe["episode_reward"].mean()),
            "std_episode_reward": float(episodes_dataframe["episode_reward"].std(ddof=0)),
            "min_episode_reward": float(episodes_dataframe["episode_reward"].min()),
            "max_episode_reward": float(episodes_dataframe["episode_reward"].max()),
            "best_eval_episode_index": int(best_row["episode_index"]),
            "best_eval_episode_seed": int(best_row["episode_seed"]),
            "mean_episode_steps": float(episodes_dataframe["episode_steps"].mean()),
            "mean_days_survived": float(episodes_dataframe["days_survived"].mean()),
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
            "mean_apartemen_warung_count": float(
                episodes_dataframe["apartemen_warung_count"].mean()
            ),
            "mean_delivery_food_count": float(episodes_dataframe["delivery_food_count"].mean()),
            "mean_warung_hemat_count": float(episodes_dataframe["warung_hemat_count"].mean()),
            "mean_warung_variasi_count": float(episodes_dataframe["warung_variasi_count"].mean()),
            "mean_ccws_stay_count": float(episodes_dataframe["ccws_stay_count"].mean()),
            "mean_cafe_session_count": float(episodes_dataframe["cafe_session_count"].mean()),
            "mean_night_cramming_count": float(episodes_dataframe["night_cramming_count"].mean()),
            "mean_direct_ojol_count": float(episodes_dataframe["direct_ojol_count"].mean()),
            "mean_transit_ccws_ojol_count": float(
                episodes_dataframe["transit_ccws_ojol_count"].mean()
            ),
        }

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
