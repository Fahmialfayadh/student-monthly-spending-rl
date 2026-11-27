from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.academics import AcademicSchedule
from src.env.actions import (
    build_action_masks,
    execute_activity_action,
    execute_meal_action,
    parse_action_vector,
    select_food_for_node,
    try_pack_bekal_before_departure,
)
from src.env.biometrics import BiometricsState
from src.env.finance import FinanceManager
from src.env.mobility import resolve_graph_mobility
from src.env.pantry import PantryManager
from src.env.schema import (
    ActivitySpecConfig,
    ExternalFoodSpecConfig,
    WorldConfig,
    export_world_json_schema,
    load_world_config,
)
from src.env.world_graph import RoutePreference, WorldGraph


class StudentLifeEnv(gym.Env):
    """
    Environment Gymnasium modular berbasis Spatial World Graph (5 step/hari, 150 step/bulan).
    Seluruh lokasi, rute transport, aktivitas, opsi makanan, dan dinamika domain
    dikonfigurasi secara deklaratif melalui WorldConfig (YAML/JSON/UI).
    """

    metadata = {"render_modes": ["human"]}

    ROUTE_PREFERENCES: list[RoutePreference] = ["fastest", "cheapest"]
    MEAL_MODES: list[str] = [
        "skip",
        "cook_fresh",
        "meal_prep",
        "reheat_or_eat_bekal",
        "budget_external_meal",
        "varied_external_meal",
    ]

    def __init__(
        self,
        yaml_path: str | Path | dict[str, Any] = "env_config.yaml",
        render_mode: str | None = None,
        exploring_starts_pantry: bool = False,
    ):
        super().__init__()
        self.yaml_path = str(yaml_path) if not isinstance(yaml_path, dict) else "<dict>"
        self.render_mode = render_mode
        self.exploring_starts_pantry = bool(exploring_starts_pantry)

        self.world_config: WorldConfig = load_world_config(yaml_path)
        self.config: dict[str, Any] = self.world_config.model_dump()

        self.environment_config = self.config["environment"]
        self.finance_config = self.config["finance"]
        self.pantry_config = self.config["pantry_and_cooking"]
        self.biometrics_config = self.config["biometrics_and_psychology"]
        self.academics_config = self.config["academics"]
        self.termination_config = self.config["termination"]
        self.rewards_config = self.config["rewards"]

        self.steps_per_day = int(self.world_config.environment.steps_per_day)
        self.total_steps = int(self.world_config.environment.total_steps)

        # Graf Spasial & Katalog Dinamis
        self.world_graph = WorldGraph(self.world_config)
        self.home_node_id = self.world_graph.home_node_id
        self.activity_specs: dict[str, ActivitySpecConfig] = {
            spec.id: spec for spec in self.world_config.activities
        }
        self.activity_names: list[str] = [spec.id for spec in self.world_config.activities]
        self.external_foods: dict[str, ExternalFoodSpecConfig] = dict(
            self.world_config.external_foods
        )
        self.external_food_ids: tuple[str, ...] = tuple(self.external_foods.keys())

        # Modul domain inti
        self.finance = FinanceManager(self.finance_config)
        self.biometrics = BiometricsState(self.biometrics_config, self.termination_config)
        self.pantry = PantryManager(self.pantry_config, external_food_ids=self.external_food_ids)
        self.academics = AcademicSchedule(self.environment_config, self.academics_config)

        # Action space MultiDiscrete terpadu (6 head; jumlah aktivitas dinamis sesuai config)
        self.route_names: list[RoutePreference] = list(self.ROUTE_PREFERENCES)
        self.meal_mode_names: list[str] = list(self.MEAL_MODES)
        self.action_dimensions: dict[str, list] = {
            "activity": list(self.activity_names),
            "route": list(self.route_names),
            "meal_mode": list(self.meal_mode_names),
            "cook_combo": list(range(len(self.pantry.cook_combos))),
            "bring_bekal": [0, 1],
            "grocery_bundle": list(range(len(self.pantry.grocery_bundles))),
        }

        self.action_space = spaces.MultiDiscrete([
            len(self.activity_names),
            len(self.route_names),
            len(self.meal_mode_names),
            len(self.pantry.cook_combos),
            2,
            len(self.pantry.grocery_bundles),
        ])

        self.observation_feature_names = self._build_observation_feature_names()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=np.inf,
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )

        self.current_step = 0
        self.current_node_id = self.home_node_id
        self.current_is_raining = False

    def _build_observation_feature_names(self) -> list[str]:
        node_features = [f"at_node_{nid}" for nid in self.world_graph.node_ids]
        return [
            "days_remaining",
            "day_of_week_sin",
            "day_of_week_cos",
            "step_in_day_sin",
            "step_in_day_cos",
            "is_late_month",
            "is_midday_climate",
            "is_raining",
            "is_peak_ojol",
            "is_class_scheduled_now",
            "is_class_scheduled_next_step",
            "has_later_class_today",
            "attendance_rate_so_far",
            *node_features,
            "is_at_home",
            "current_node_closes_at_night",
            "current_node_has_kitchen",
            "balance_idr",
            "pantry_inventory_value_idr",
            "is_balance_low",
            "burn_rate_per_day_idr",
            "budget_runway_days",
            "energy",
            "hunger",
            "mood",
            "academic_stress",
            "cafe_craving",
            "cafe_buff_remaining_steps",
            "hub_visits_today",
            "consecutive_same_meal_count",
            "recent_budget_food_count_here",
            "recent_varied_food_count_here",
            "consecutive_fatal_hunger_steps",
            "rice_stock_units",
            "egg_stock_units",
            "chicken_stock_units",
            "tempe_stock_units",
            *[f"{name}_stock_units" for name in self.pantry.extra_ingredient_names],
            "meal_prep_stock_portions",
            "packed_lunch_stock_portions",
            "min_shelf_life_egg_steps",
            "min_shelf_life_chicken_steps",
            "min_shelf_life_tempe_steps",
            *[f"min_shelf_life_{name}_steps" for name in self.pantry.extra_ingredient_names],
            "min_shelf_life_meal_prep_steps",
            "has_non_repeating_meal_prep",
            "has_non_repeating_packed_lunch",
            *self.pantry.get_last_meal_flag_names(),
        ]

    @property
    def current_location(self) -> str:
        """Alias lokasi aktif (node ID pada WorldGraph) untuk kompatibilitas UI & logger."""
        return self.current_node_id

    @property
    def balance_idr(self) -> float:
        return self.finance.balance_idr

    @balance_idr.setter
    def balance_idr(self, value: float) -> None:
        self.finance.balance_idr = float(value)

    @property
    def energy(self) -> float:
        return self.biometrics.energy

    @property
    def hunger(self) -> float:
        return self.biometrics.hunger

    @property
    def mood(self) -> float:
        return self.biometrics.mood

    @property
    def academic_stress(self) -> float:
        return self.biometrics.academic_stress

    @property
    def cafe_craving(self) -> float:
        return self.biometrics.cafe_craving

    @staticmethod
    def export_schema() -> dict[str, Any]:
        """Mengekspor JSON Schema konfigurasi untuk frontend / UI builder."""
        return export_world_json_schema()

    def get_graph_state(self) -> dict[str, Any]:
        """Snapshot status lokasi, waktu, biometrik, stok, dan graf untuk inspeksi API/UI."""
        node = self.world_graph.get_node(self.current_node_id)
        current_day, _, step_in_day, day_name = self.academics.get_time_context(self.current_step)
        step_def = self.academics.get_step_definition(step_in_day)
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "episode_days": int(self.world_config.environment.episode_days),
            "current_day": current_day,
            "day_name": day_name,
            "step_in_day": step_in_day,
            "time_slot_name": str(step_def.get("name", f"slot_{step_in_day}")),
            "time_slot_hours": str(step_def.get("hours", "")),
            "is_class_scheduled_now": self.academics.is_class_scheduled(day_name, step_in_day),
            "current_node_id": self.current_node_id,
            "current_node_label": node.label,
            "current_zone": node.zone,
            "is_raining": self.current_is_raining,
            "balance_idr": self.finance.balance_idr,
            "initial_operational_budget_idr": self.finance.initial_operational_budget_idr,
            "spending_by_category_idr": self.finance.get_spending_summary(),
            "biometrics": {
                "energy": self.biometrics.energy,
                "hunger": self.biometrics.hunger,
                "mood": self.biometrics.mood,
                "academic_stress": self.biometrics.academic_stress,
                "cafe_craving": self.biometrics.cafe_craving,
            },
            "pantry": self.pantry.get_stock_summary(),
        }

    def get_action_labels(self) -> dict[str, list[Any]]:
        cook_labels = [
            " + ".join(
                f"{qty} {self.pantry.get_ingredient_label(name)}"
                for name, qty in self.pantry.decode_cook_combo_dict(i).items()
                if qty > 0
            )
            for i in range(len(self.pantry.cook_combos))
        ]
        grocery_labels = [
            "Beli: "
            + ", ".join(
                self.pantry.get_ingredient_label(name)
                for name, flag in self.pantry.decode_grocery_bundle_dict(i).items()
                if flag
            )
            for i in range(len(self.pantry.grocery_bundles))
        ]
        return {
            "activity": [self.activity_specs[aid].label for aid in self.activity_names],
            "route": ["Cepat (minim lelah)", "Hemat (ongkos terendah)"],
            "meal_mode": [
                "Tidak makan (skip)",
                "Masak segar",
                "Meal prep (batch)",
                "Panaskan / makan bekal",
                "Makan luar hemat",
                "Makan luar variasi",
            ],
            "cook_combo": cook_labels,
            "bring_bekal": ["Tidak bawa bekal", "Bawa bekal"],
            "grocery_bundle": grocery_labels,
        }

    def _roll_rain_for_step(self, step_index: int) -> bool:
        _, _, step_in_day, _ = self.academics.get_time_context(step_index)
        step_def = self.academics.get_step_definition(step_in_day)
        rain_prob = float(step_def.get("rain_prob", 0.0))
        return bool(self.np_random.random() < rain_prob)

    def _get_budget_potential(self) -> float:
        shaping_config = self.rewards_config["budget_potential_shaping"]
        initial_budget = max(1.0, self.finance.initial_operational_budget_idr)
        effective_wealth = self.finance.balance_idr + self.pantry.get_inventory_value_idr()
        wealth_ratio = max(0.0, effective_wealth / initial_budget)
        return float(shaping_config["potential_scale"]) * wealth_ratio

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_node_id = self.home_node_id

        self.finance.reset()
        self.biometrics.reset()
        self.pantry.reset()
        self.academics.reset()

        if self.exploring_starts_pantry and self.np_random.random() < 0.60:
            self._apply_exploring_start_pantry()

        self.current_is_raining = self._roll_rain_for_step(0)
        return self._get_observation(), self._build_info_dictionary({})

    def _apply_exploring_start_pantry(self) -> None:
        starter_bundles = (
            ("rice", "egg", "tempe"),
            ("rice", "egg", "chicken"),
            ("rice", "egg", "chicken", "tempe"),
            ("rice", "chicken", "tempe"),
        )
        chosen_bundle = starter_bundles[int(self.np_random.integers(0, len(starter_bundles)))]
        ingredients_cfg = self.pantry_config["ingredients"]
        for item in chosen_bundle:
            price = float(ingredients_cfg[item]["pack_price_idr"])
            if self.finance.pay(price, category="grocery"):
                self.pantry.add_grocery_pack(item)

    def step(self, action):
        step_reward = 0.0
        step_events: dict[str, object] = {}
        potential_before = self._get_budget_potential()

        current_day, day_of_week, step_in_day, day_name = self.academics.get_time_context(
            self.current_step
        )
        step_definition = self.academics.get_step_definition(step_in_day)
        current_climate = str(step_definition["climate"])
        current_ojol_tariff = str(step_definition["ojol_tariff"])
        step_hunger_gain = float(step_definition["hunger_gain"])
        is_raining_now = self.current_is_raining
        is_class_scheduled_now = self.academics.is_class_scheduled(day_name, step_in_day)

        if is_raining_now:
            step_events["is_raining"] = True
        if is_class_scheduled_now:
            self.academics.total_classes_scheduled += 1

        # 1. Tugas harian baru di awal hari
        if step_in_day == 0:
            task_stress_gain = self.academics.roll_daily_task_stress_gain(
                self.np_random, day_of_week, day_name
            )
            if task_stress_gain > 0.0:
                self.biometrics.academic_stress += task_stress_gain
                step_events["new_daily_task_added"] = True

        # 2. Bongkar aksi & cek soft mood burnout
        parsed_action = parse_action_vector(action, self.action_dimensions, pantry=self.pantry)
        is_stress_burned_out = self.biometrics.is_stress_burned_out()
        effective_activity_id = str(parsed_action["activity"])
        activity_spec = self.activity_specs[effective_activity_id]

        if self.biometrics.is_mood_burned_out() and activity_spec.handler in (
            "attend_schedule",
            "study",
            "grocery_shop",
            "night_cram",
        ):
            fallback_handler = (
                "wait"
                if (
                    self.world_graph.get_node(self.current_node_id).closes_at_night
                    and step_in_day < (self.steps_per_day - 1)
                )
                else "rest"
            )
            effective_activity_id = next(
                (aid for aid, s in self.activity_specs.items() if s.handler == fallback_handler),
                self.activity_names[0],
            )
            activity_spec = self.activity_specs[effective_activity_id]
            step_events["forced_rest_due_to_mood_burnout"] = True

        # 3a. Pra-keberangkatan di lokasi berdapur: masak sebelum pindah & bungkus bekal
        requested_meal_mode = str(parsed_action["meal_mode"])
        executed_meal_mode: str | None = None
        node_before = self.world_graph.get_node(self.current_node_id)
        target_candidates = self.world_graph.resolve_activity_target_nodes(activity_spec)
        will_leave_kitchen = (
            node_before.has_kitchen
            and self.current_node_id not in target_candidates
            and step_in_day < (self.steps_per_day - 1)
        )

        if node_before.has_kitchen:
            if requested_meal_mode in ("cook_fresh", "meal_prep"):
                executed_meal_mode = execute_meal_action(
                    requested_meal_mode=requested_meal_mode,
                    requested_cook_quantities=parsed_action["cook_quantities"],
                    current_node=node_before,
                    step_in_day=step_in_day,
                    is_stress_burned_out=is_stress_burned_out,
                    finance_manager=self.finance,
                    biometrics_state=self.biometrics,
                    pantry_manager=self.pantry,
                    pantry_config=self.pantry_config,
                    biometrics_config=self.biometrics_config,
                    external_foods=self.external_foods,
                    step_events=step_events,
                )
            if will_leave_kitchen and bool(parsed_action["bring_bekal"]):
                try_pack_bekal_before_departure(
                    bring_bekal=True,
                    biometrics_state=self.biometrics,
                    pantry_manager=self.pantry,
                    pantry_config=self.pantry_config,
                    step_events=step_events,
                )

        # 3b. Fase Mobilitas Berbasis Graf Spasial (src/env/mobility.py)
        self.current_node_id, effective_activity_id, mobility_reward = resolve_graph_mobility(
            current_node_id=self.current_node_id,
            effective_activity_id=effective_activity_id,
            activity_specs=self.activity_specs,
            world_graph=self.world_graph,
            requested_route_pref=parsed_action["route"],
            is_class_scheduled_now=is_class_scheduled_now,
            step_in_day=step_in_day,
            steps_per_day=self.steps_per_day,
            current_climate=current_climate,
            current_ojol_tariff=current_ojol_tariff,
            is_raining=is_raining_now,
            current_day=current_day,
            random_generator=self.np_random,
            finance_manager=self.finance,
            biometrics_state=self.biometrics,
            finance_config=self.finance_config,
            rewards_config=self.rewards_config,
            step_events=step_events,
        )
        step_reward += mobility_reward
        current_node = self.world_graph.get_node(self.current_node_id)

        # 4. Fase Makan di Node Tujuan
        if executed_meal_mode is None:
            executed_meal_mode = execute_meal_action(
                requested_meal_mode=requested_meal_mode,
                requested_cook_quantities=parsed_action["cook_quantities"],
                current_node=current_node,
                step_in_day=step_in_day,
                is_stress_burned_out=is_stress_burned_out,
                finance_manager=self.finance,
                biometrics_state=self.biometrics,
                pantry_manager=self.pantry,
                pantry_config=self.pantry_config,
                biometrics_config=self.biometrics_config,
                external_foods=self.external_foods,
                step_events=step_events,
            )
        step_events["executed_meal_mode"] = executed_meal_mode

        # 5. Fase Aktivitas & Kehadiran Kuliah
        attended_class, executed_activity_id = execute_activity_action(
            effective_activity_id=effective_activity_id,
            activity_specs=self.activity_specs,
            current_node=current_node,
            world_graph=self.world_graph,
            is_class_scheduled_now=is_class_scheduled_now,
            is_stress_burned_out=is_stress_burned_out,
            step_in_day=step_in_day,
            steps_per_day=self.steps_per_day,
            current_climate=current_climate,
            is_raining=is_raining_now,
            current_day=current_day,
            requested_grocery_flags=parsed_action["grocery_flags"],
            random_generator=self.np_random,
            finance_manager=self.finance,
            biometrics_state=self.biometrics,
            pantry_manager=self.pantry,
            finance_config=self.finance_config,
            academics_config=self.academics_config,
            biometrics_config=self.biometrics_config,
            pantry_config=self.pantry_config,
            step_events=step_events,
        )
        step_events["executed_activity"] = executed_activity_id
        executed_spec = self.activity_specs[executed_activity_id]

        if is_class_scheduled_now:
            if attended_class:
                self.academics.total_classes_attended += 1
                step_reward += float(self.rewards_config["class_attendance_bonus"])
            else:
                step_reward += float(self.academics_config["absent_class_reward_penalty"])
                self.biometrics.academic_stress += float(
                    self.academics_config["absent_class_stress_gain"]
                )
                step_events["missed_scheduled_class"] = True

        # 6. Dinamika Akhir Step & Penuaan Stok
        effective_climate = "rain" if is_raining_now else current_climate
        step_reward += self.biometrics.apply_end_of_step_dynamics(
            current_day=current_day,
            step_in_day=step_in_day,
            steps_per_day=self.steps_per_day,
            step_hunger_gain=step_hunger_gain,
            current_climate=effective_climate,
            current_node=current_node,
            executed_activity_spec=executed_spec,
            finance_config=self.finance_config,
            rewards_config=self.rewards_config,
            step_events=step_events,
        )

        spoiled_units = self.pantry.age_and_spoil_batches()
        if step_in_day == (self.steps_per_day - 1):
            spoiled_units += self.pantry.spoil_uneaten_packed_lunch_end_of_day()

        if spoiled_units > 0:
            spoilage_penalty = float(
                self.rewards_config["critical_state_penalties"]["food_spoilage_per_unit"]
            )
            step_reward += spoiled_units * spoilage_penalty
            step_events["spoiled_units"] = spoiled_units

        # 7. Reward Kesejahteraan, Terminasi, & Potential Shaping
        step_reward += self.biometrics.compute_wellbeing_reward(self.rewards_config, step_events)
        self.current_step += 1

        terminated = False
        fatal_reason = self.biometrics.check_fatal_termination()
        if fatal_reason is not None:
            terminated = True
            step_reward += float(self.termination_config["fatal_termination_penalty"])
            step_events["termination_reason"] = fatal_reason
        elif self.current_step >= self.total_steps:
            terminated = True
            completion_rewards = self.rewards_config["terminal_episode_completion"]
            step_reward += float(completion_rewards["survived_30_days_bonus"])
            step_reward += (self.finance.balance_idr / 100_000.0) * float(
                completion_rewards["remaining_balance_reward_per_100k"]
            )
            min_attendance = float(self.academics_config["min_attendance_rate_for_exam"])
            if self.academics.get_attendance_rate() < min_attendance:
                step_reward += float(completion_rewards["low_attendance_terminal_penalty"])
                step_events["low_attendance_penalty_applied"] = True
            step_events["termination_reason"] = "completed_30_days"
        else:
            self.current_is_raining = self._roll_rain_for_step(self.current_step)

        shaping_config = self.rewards_config["budget_potential_shaping"]
        if bool(shaping_config["enabled"]):
            potential_after = 0.0 if terminated else self._get_budget_potential()
            shaping_reward = float(shaping_config["gamma"]) * potential_after - potential_before
            step_reward += shaping_reward
            step_events["budget_potential_shaping_reward"] = shaping_reward

        return (
            self._get_observation(),
            float(step_reward),
            terminated,
            False,
            self._build_info_dictionary(step_events),
        )

    def action_masks(self) -> np.ndarray:
        return build_action_masks(
            current_step=self.current_step,
            current_node_id=self.current_node_id,
            is_raining=self.current_is_raining,
            steps_per_day=self.steps_per_day,
            world_graph=self.world_graph,
            activity_specs=self.activity_specs,
            activity_order=self.activity_names,
            route_preferences=self.route_names,
            external_foods=self.external_foods,
            academics=self.academics,
            finance=self.finance,
            biometrics=self.biometrics,
            pantry=self.pantry,
            config=self.config,
        )

    def _get_observation(self) -> np.ndarray:
        current_day, day_of_week, step_in_day, day_name = self.academics.get_time_context(
            self.current_step
        )
        step_def = self.academics.get_step_definition(step_in_day)
        is_class_now = 1.0 if self.academics.is_class_scheduled(day_name, step_in_day) else 0.0

        if self.current_step + 1 < self.total_steps:
            _, _, next_step_in_day, next_day_name = self.academics.get_time_context(
                self.current_step + 1
            )
            is_class_next = (
                1.0 if self.academics.is_class_scheduled(next_day_name, next_step_in_day) else 0.0
            )
        else:
            is_class_next = 0.0

        current_node = self.world_graph.get_node(self.current_node_id)
        node_one_hot = [1.0 if nid == self.current_node_id else 0.0 for nid in self.world_graph.node_ids]

        budget_food_here = select_food_for_node(current_node, "budget", self.external_foods)
        varied_food_here = select_food_for_node(current_node, "varied", self.external_foods)
        recent_budget_here = (
            float(self.pantry.get_recent_meal_count(budget_food_here.id))
            if budget_food_here is not None
            else 0.0
        )
        recent_varied_here = (
            float(self.pantry.get_recent_meal_count(varied_food_here.id))
            if varied_food_here is not None
            else 0.0
        )

        last_meal_flags = self.pantry.get_last_meal_one_hot_flags()
        days_remaining = max(0.0, (self.total_steps - self.current_step) / self.steps_per_day)
        elapsed_days = max(1.0, self.current_step / self.steps_per_day)
        burn_rate_per_day = self.finance.get_total_spent_idr() / elapsed_days
        if burn_rate_per_day > 0.0:
            budget_runway_days = min(
                float(self.environment_config["episode_days"]),
                self.finance.balance_idr / burn_rate_per_day,
            )
        else:
            budget_runway_days = float(self.environment_config["episode_days"])

        dow_angle = 2.0 * math.pi * (day_of_week / 7.0)
        step_angle = 2.0 * math.pi * (step_in_day / self.steps_per_day)
        is_late_month = (
            1.0
            if current_day >= int(self.finance_config["late_month_impulsivity"]["start_day"])
            else 0.0
        )
        is_midday = 1.0 if (str(step_def["climate"]) == "midday" and not self.current_is_raining) else 0.0
        is_peak_ojol = 1.0 if str(step_def["ojol_tariff"]) == "peak" else 0.0

        return np.array(
            [
                days_remaining,
                math.sin(dow_angle),
                math.cos(dow_angle),
                math.sin(step_angle),
                math.cos(step_angle),
                is_late_month,
                is_midday,
                1.0 if self.current_is_raining else 0.0,
                is_peak_ojol,
                is_class_now,
                is_class_next,
                1.0 if self.academics.has_later_class_today(day_name, step_in_day) else 0.0,
                self.academics.get_attendance_rate(),
                *node_one_hot,
                1.0 if self.current_node_id == self.home_node_id else 0.0,
                1.0 if current_node.closes_at_night else 0.0,
                1.0 if current_node.has_kitchen else 0.0,
                self.finance.balance_idr,
                self.pantry.get_inventory_value_idr(),
                1.0 if self.finance.is_balance_low() else 0.0,
                burn_rate_per_day,
                budget_runway_days,
                self.biometrics.energy,
                self.biometrics.hunger,
                self.biometrics.mood,
                self.biometrics.academic_stress,
                self.biometrics.cafe_craving,
                float(self.biometrics.cafe_buff_remaining_steps),
                float(self.biometrics.hub_visits_today),
                float(self.pantry.consecutive_same_meal_count),
                recent_budget_here,
                recent_varied_here,
                float(self.biometrics.consecutive_fatal_hunger_steps),
                float(self.pantry.get_ingredient_total_units("rice")),
                float(self.pantry.get_ingredient_total_units("egg")),
                float(self.pantry.get_ingredient_total_units("chicken")),
                float(self.pantry.get_ingredient_total_units("tempe")),
                *[
                    float(self.pantry.get_ingredient_total_units(name))
                    for name in self.pantry.extra_ingredient_names
                ],
                float(self.pantry.get_meal_prep_total_portions()),
                float(self.pantry.get_packed_lunch_total_portions()),
                float(self.pantry.get_min_ingredient_shelf_life("egg")),
                float(self.pantry.get_min_ingredient_shelf_life("chicken")),
                float(self.pantry.get_min_ingredient_shelf_life("tempe")),
                *[
                    float(self.pantry.get_min_ingredient_shelf_life(name))
                    for name in self.pantry.extra_ingredient_names
                ],
                float(self.pantry.get_min_meal_prep_shelf_life()),
                1.0 if self.pantry.has_non_repeating_meal_prep() else 0.0,
                1.0 if self.pantry.has_non_repeating_packed_lunch() else 0.0,
                *last_meal_flags,
            ],
            dtype=np.float32,
        )

    def _build_info_dictionary(self, step_events: dict[str, object]) -> dict[str, object]:
        return {
            "current_step": self.current_step,
            "current_location": self.current_node_id,
            "current_node_id": self.current_node_id,
            "is_raining": self.current_is_raining,
            "balance_idr": self.finance.balance_idr,
            "pantry_inventory_value_idr": self.pantry.get_inventory_value_idr(),
            "spending_by_category_idr": self.finance.get_spending_summary(),
            "total_classes_scheduled": self.academics.total_classes_scheduled,
            "total_classes_attended": self.academics.total_classes_attended,
            "total_spoiled_units": self.pantry.total_spoiled_units,
            "pantry_totals": self.pantry.get_stock_summary(),
            "step_events": step_events,
        }

    def render(self) -> None:
        if self.render_mode != "human":
            return
        current_day, _, step_in_day, day_name = self.academics.get_time_context(self.current_step)
        step_name = self.academics.get_step_definition(step_in_day)["name"]
        rain_str = " | HUJAN" if self.current_is_raining else ""
        print(
            f"[Hari {current_day:02d} | {day_name.capitalize()} - {step_name} | {self.current_node_id}{rain_str}] "
            f"Saldo: Rp {self.finance.balance_idr:,.0f} | Bekal: {self.pantry.get_packed_lunch_total_portions()} | "
            f"Energi: {self.biometrics.energy:.1f} | Lapar: {self.biometrics.hunger:.1f} | "
            f"Mood: {self.biometrics.mood:.1f} | Stres: {self.biometrics.academic_stress:.1f} | "
            f"Craving: {self.biometrics.cafe_craving:.1f}"
        )
