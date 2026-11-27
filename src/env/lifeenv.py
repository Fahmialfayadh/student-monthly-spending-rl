import math
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.academics import AcademicSchedule
from src.env.actions import (
    build_action_masks,
    execute_activity_action,
    execute_meal_action,
    parse_action_vector,
    try_pack_bekal_before_departure,
)
from src.env.biometrics import BiometricsState
from src.env.finance import FinanceManager
from src.env.mobility import resolve_mobility_phase
from src.env.pantry import PantryManager


class StudentLifeEnv(gym.Env):
    """
    Environment Gymnasium realistis (5 step/hari, 150 step/bulan, 1 saldo operasional bersih)
    yang menghubungkan 5 modul domain: finance, biometrics, pantry, mobility, dan academics.
    """

    metadata = {"render_modes": ["human"]}

    OBSERVATION_FEATURE_NAMES = [
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
        "is_at_campus",
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
        "ccws_visits_today",
        "consecutive_same_meal_count",
        "recent_apartemen_warung_count",
        "recent_warung_hemat_count",
        "consecutive_fatal_hunger_steps",
        "rice_stock_units",
        "egg_stock_units",
        "chicken_stock_units",
        "tempe_stock_units",
        "meal_prep_stock_portions",
        "packed_lunch_stock_portions",
        "min_shelf_life_egg_steps",
        "min_shelf_life_chicken_steps",
        "min_shelf_life_tempe_steps",
        "min_shelf_life_meal_prep_steps",
        "has_non_repeating_meal_prep",
        "has_non_repeating_packed_lunch",
        "last_meal_had_egg",
        "last_meal_had_chicken",
        "last_meal_had_tempe",
        "last_meal_was_apartemen_warung",
        "last_meal_was_delivery_food",
        "last_meal_was_warung_hemat",
        "last_meal_was_warung_variasi",
        "last_meal_was_plain_rice",
    ]

    def __init__(
        self,
        yaml_path: str = "env_config.yaml",
        render_mode: str | None = None,
        exploring_starts_pantry: bool = False,
    ):
        super().__init__()
        self.yaml_path = yaml_path
        self.render_mode = render_mode
        self.exploring_starts_pantry = bool(exploring_starts_pantry)

        with open(yaml_path, "r", encoding="utf-8") as config_file:
            self.config = yaml.safe_load(config_file)

        self.environment_config = self.config["environment"]
        self.finance_config = self.config["finance"]
        self.pantry_config = self.config["pantry_and_cooking"]
        self.biometrics_config = self.config["biometrics_and_psychology"]
        self.mobility_config = self.config["mobility_and_surabaya_climate"]
        self.academics_config = self.config["academics"]
        self.termination_config = self.config["termination"]
        self.rewards_config = self.config["rewards"]

        self.steps_per_day = int(self.environment_config["steps_per_day"])
        self.total_steps = int(self.environment_config["total_steps"])

        # Modul domain konkret
        self.finance = FinanceManager(self.finance_config)
        self.biometrics = BiometricsState(self.biometrics_config, self.termination_config)
        self.pantry = PantryManager(self.pantry_config)
        self.academics = AcademicSchedule(self.environment_config, self.academics_config)

        # Action space MultiDiscrete terpadu (6 dimensi / 51 logits)
        raw_dimensions = self.config["action_space"]["dimensions"]
        self.action_dimensions = {
            "activity": list(raw_dimensions["activity"]),
            "route": list(raw_dimensions["route"]),
            "meal_mode": list(raw_dimensions["meal_mode"]),
            "cook_combo": list(range(len(PantryManager.COOK_COMBOS))),
            "bring_bekal": list(raw_dimensions["bring_bekal"]),
            "grocery_bundle": list(range(len(PantryManager.GROCERY_BUNDLES))),
        }
        self.activity_names = self.action_dimensions["activity"]
        self.route_names = self.action_dimensions["route"]
        self.meal_mode_names = self.action_dimensions["meal_mode"]

        self.action_space = spaces.MultiDiscrete([
            len(self.activity_names),
            len(self.route_names),
            len(self.meal_mode_names),
            len(PantryManager.COOK_COMBOS),
            len(self.action_dimensions["bring_bekal"]),
            len(PantryManager.GROCERY_BUNDLES),
        ])

        self.observation_feature_names = list(self.OBSERVATION_FEATURE_NAMES)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=np.inf,
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )

        self.current_step = 0
        self.current_location = "apartment"
        self.current_campus_sub_location: str | None = None
        self.current_is_raining = False

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

    def _roll_rain_for_step(self, step_index: int) -> bool:
        _, _, step_in_day, _ = self.academics.get_time_context(step_index)
        step_def = self.academics.get_step_definition(step_in_day)
        rain_prob = float(step_def.get("rain_prob", 0.0))
        return bool(self.np_random.random() < rain_prob)

    def _get_budget_potential(self) -> float:
        """
        Phi(s): rasio kekayaan bersih operasional (saldo + nilai stok dapur 1:1)
        terhadap saldo awal. Tetap 100% policy-invariant karena Phi(terminal) = 0.
        """
        shaping_config = self.rewards_config["budget_potential_shaping"]
        initial_budget = max(1.0, self.finance.initial_operational_budget_idr)
        effective_wealth = self.finance.balance_idr + self.pantry.get_inventory_value_idr()
        wealth_ratio = max(0.0, effective_wealth / initial_budget)
        return float(shaping_config["potential_scale"]) * wealth_ratio

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_location = "apartment"
        self.current_campus_sub_location = None

        self.finance.reset()
        self.biometrics.reset()
        self.pantry.reset()
        self.academics.reset()

        # Training-only Exploring Starts (Sutton & Barto): membuka akses aksi masak/prep/bekal
        # sejak awal tanpa memberi uang gratis (harga paket tetap dipotong penuh dari saldo).
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

        # Sebagian kecil episode training juga mulai dengan 1 batch meal_prep (dipotong dari stok mentah)
        if self.np_random.random() < 0.40:
            prep_qty = {"rice": 1, "egg": 1, "chicken": 0, "tempe": 1}
            if not self.pantry.has_enough_ingredients(prep_qty, batch_multiplier=4):
                prep_qty = {"rice": 1, "egg": 0, "chicken": 1, "tempe": 0}
            if self.pantry.has_enough_ingredients(prep_qty, batch_multiplier=4):
                self.pantry.consume_ingredients_fifo(prep_qty, batch_multiplier=4)
                h_d, e_d, m_d, sig = self.pantry.calculate_custom_meal_profile(prep_qty)
                val = self.pantry.compute_combo_ingredient_value_idr(prep_qty)
                self.pantry.store_meal_prep_batch(
                    portions=4,
                    hunger_delta=h_d,
                    energy_delta=e_d,
                    mood_delta=m_d,
                    meal_signature=sig,
                    portion_value_idr=val,
                )

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

        # 1. Tugas harian baru di awal hari (proporsional dengan jadwal weekday vs weekend)
        if step_in_day == 0:
            task_stress_gain = self.academics.roll_daily_task_stress_gain(
                self.np_random, day_of_week, day_name
            )
            if task_stress_gain > 0.0:
                self.biometrics.academic_stress += task_stress_gain
                step_events["new_daily_task_added"] = True

        # 2. Bongkar aksi & cek soft burnout
        parsed_action = parse_action_vector(action, self.action_dimensions)
        is_stress_burned_out = self.biometrics.is_stress_burned_out()
        effective_activity = str(parsed_action["activity"])

        if self.biometrics.is_mood_burned_out() and effective_activity in (
            "attend_class",
            "room_study",
            "grocery_shop",
            "night_cramming",
        ):
            effective_activity = (
                "wait_at_department"
                if self.current_location == "campus" and step_in_day < (self.steps_per_day - 1)
                else "rest_or_sleep"
            )
            step_events["forced_rest_due_to_mood_burnout"] = True

        # 3a. Pra-keberangkatan di apartemen: masak sebelum berangkat & bungkus bekal ringan
        requested_meal_mode = str(parsed_action["meal_mode"])
        executed_meal_mode: str | None = None
        campus_activities = {"attend_class", "ccws_stay", "wait_at_department"}
        heading_to_campus = (
            self.current_location == "apartment"
            and effective_activity in campus_activities
            and step_in_day < (self.steps_per_day - 1)
        )

        if self.current_location == "apartment":
            if requested_meal_mode in ("cook_fresh", "meal_prep"):
                executed_meal_mode = execute_meal_action(
                    requested_meal_mode=requested_meal_mode,
                    requested_cook_quantities=parsed_action["cook_quantities"],
                    current_location="apartment",
                    step_in_day=step_in_day,
                    is_stress_burned_out=is_stress_burned_out,
                    finance_manager=self.finance,
                    biometrics_state=self.biometrics,
                    pantry_manager=self.pantry,
                    pantry_config=self.pantry_config,
                    biometrics_config=self.biometrics_config,
                    step_events=step_events,
                )
            if heading_to_campus and bool(parsed_action["bring_bekal"]):
                try_pack_bekal_before_departure(
                    bring_bekal=True,
                    biometrics_state=self.biometrics,
                    pantry_manager=self.pantry,
                    pantry_config=self.pantry_config,
                    step_events=step_events,
                )

        # 3b. Fase Mobilitas (src/env/mobility.py)
        (
            self.current_location,
            self.current_campus_sub_location,
            effective_activity,
            mobility_reward,
        ) = resolve_mobility_phase(
            current_location=self.current_location,
            current_campus_sub_location=self.current_campus_sub_location,
            effective_activity=effective_activity,
            requested_route=str(parsed_action["route"]),
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
            mobility_config=self.mobility_config,
            rewards_config=self.rewards_config,
            step_events=step_events,
        )
        step_reward += mobility_reward

        # 4. Fase Makan di Lokasi Tujuan (jika belum dieksekusi di pra-keberangkatan)
        if executed_meal_mode is None:
            executed_meal_mode = execute_meal_action(
                requested_meal_mode=requested_meal_mode,
                requested_cook_quantities=parsed_action["cook_quantities"],
                current_location=self.current_location,
                step_in_day=step_in_day,
                is_stress_burned_out=is_stress_burned_out,
                finance_manager=self.finance,
                biometrics_state=self.biometrics,
                pantry_manager=self.pantry,
                pantry_config=self.pantry_config,
                biometrics_config=self.biometrics_config,
                step_events=step_events,
            )
        step_events["executed_meal_mode"] = executed_meal_mode

        # 5. Fase Aktivitas & Kehadiran Kuliah (src/env/actions.py)
        attended_class, executed_activity, self.current_location = execute_activity_action(
            effective_activity=effective_activity,
            current_location=self.current_location,
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
            mobility_config=self.mobility_config,
            biometrics_config=self.biometrics_config,
            pantry_config=self.pantry_config,
            step_events=step_events,
        )
        if self.current_location == "apartment":
            self.current_campus_sub_location = None
        elif executed_activity == "ccws_stay":
            self.current_campus_sub_location = "ccws"
        elif executed_activity in ("attend_class", "wait_at_department"):
            self.current_campus_sub_location = "department"
        step_events["executed_activity"] = executed_activity

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

        # 6. Dinamika Akhir Step (Tidur otomatis, lapar, craving, pembusukan stok & bekal)
        effective_climate = "rain" if is_raining_now else current_climate
        step_reward += self.biometrics.apply_end_of_step_dynamics(
            current_day=current_day,
            step_in_day=step_in_day,
            steps_per_day=self.steps_per_day,
            step_hunger_gain=step_hunger_gain,
            current_climate=effective_climate,
            current_location=self.current_location,
            executed_activity=executed_activity,
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

        # 7. Reward Kesejahteraan & Evaluasi Terminasi
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
            current_location=self.current_location,
            is_raining=self.current_is_raining,
            steps_per_day=self.steps_per_day,
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
            1.0 if current_day >= int(self.finance_config["late_month_impulsivity"]["start_day"]) else 0.0
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
                1.0 if self.current_location == "campus" else 0.0,
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
                float(self.biometrics.ccws_visits_today),
                float(self.pantry.consecutive_same_meal_count),
                float(self.pantry.get_recent_meal_count("apartemen_warung")),
                float(self.pantry.get_recent_meal_count("warung_hemat")),
                float(self.biometrics.consecutive_fatal_hunger_steps),
                float(self.pantry.get_ingredient_total_units("rice")),
                float(self.pantry.get_ingredient_total_units("egg")),
                float(self.pantry.get_ingredient_total_units("chicken")),
                float(self.pantry.get_ingredient_total_units("tempe")),
                float(self.pantry.get_meal_prep_total_portions()),
                float(self.pantry.get_packed_lunch_total_portions()),
                float(self.pantry.get_min_ingredient_shelf_life("egg")),
                float(self.pantry.get_min_ingredient_shelf_life("chicken")),
                float(self.pantry.get_min_ingredient_shelf_life("tempe")),
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
            "current_location": self.current_location,
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
            f"[Hari {current_day:02d} | {day_name.capitalize()} - {step_name} | {self.current_location}{rain_str}] "
            f"Saldo: Rp {self.finance.balance_idr:,.0f} | Bekal: {self.pantry.get_packed_lunch_total_portions()} | "
            f"Energi: {self.biometrics.energy:.1f} | Lapar: {self.biometrics.hunger:.1f} | "
            f"Mood: {self.biometrics.mood:.1f} | Stres: {self.biometrics.academic_stress:.1f} | "
            f"Craving: {self.biometrics.cafe_craving:.1f}"
        )
