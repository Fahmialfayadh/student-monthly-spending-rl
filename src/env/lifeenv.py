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
)
from src.env.biometrics import BiometricsState
from src.env.finance import FinanceManager
from src.env.mobility import resolve_mobility_phase
from src.env.pantry import PantryManager


class StudentLifeEnv(gym.Env):
    """
    Environment Gymnasium utama (5 step/hari, 150 step/bulan, 1 saldo operasional)
    yang menghubungkan 5 modul domain: finance, biometrics, pantry, mobility, dan academics.
    """

    metadata = {"render_modes": ["human"]}

    OBSERVATION_FEATURE_NAMES = [
        "current_step",
        "current_day",
        "day_of_week",
        "step_in_day",
        "is_class_scheduled_now",
        "is_class_scheduled_next_step",
        "has_later_class_today",
        "is_at_campus",
        "waited_until_evening",
        "balance_idr",
        "is_balance_low",
        "energy",
        "hunger",
        "mood",
        "academic_stress",
        "cafe_craving",
        "cafe_buff_remaining_steps",
        "consecutive_same_meal_count",
        "consecutive_fatal_hunger_steps",
        "rice_stock_units",
        "egg_stock_units",
        "chicken_stock_units",
        "tempe_stock_units",
        "meal_prep_stock_portions",
        "packed_lunch_stock_portions",
        "min_shelf_life_chicken_steps",
        "min_shelf_life_tempe_steps",
        "min_shelf_life_meal_prep_steps",
        "last_meal_had_egg",
        "last_meal_had_chicken",
        "last_meal_had_tempe",
        "last_meal_was_warung_hemat",
        "last_meal_was_warung_variasi",
        "last_meal_was_craving_snack",
        "last_meal_was_plain_rice",
    ]

    def __init__(self, yaml_path: str = "env_config.yaml", render_mode: str | None = None):
        super().__init__()
        self.yaml_path = yaml_path
        self.render_mode = render_mode

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

        # Action space MultiDiscrete (11 dimensi / 38 logits)
        self.action_dimensions = {
            key: list(values)
            for key, values in self.config["action_space"]["dimensions"].items()
        }
        self.meal_mode_names = self.action_dimensions["meal_mode"]
        self.cook_rice_options = self.action_dimensions["cook_rice"]
        self.cook_egg_options = self.action_dimensions["cook_egg"]
        self.cook_chicken_options = self.action_dimensions["cook_chicken"]
        self.cook_tempe_options = self.action_dimensions["cook_tempe"]
        self.mobility_names = self.action_dimensions["mobility"]
        self.activity_names = self.action_dimensions["activity"]
        self.buy_rice_options = self.action_dimensions["buy_rice"]
        self.buy_egg_options = self.action_dimensions["buy_egg"]
        self.buy_chicken_options = self.action_dimensions["buy_chicken"]
        self.buy_tempe_options = self.action_dimensions["buy_tempe"]

        self.action_space = spaces.MultiDiscrete([
            len(options) for options in self.action_dimensions.values()
        ])

        self.observation_feature_names = list(self.OBSERVATION_FEATURE_NAMES)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )

        self.current_step = 0
        self.current_location = "apartment"
        self.waited_until_evening = False

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

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_location = "apartment"
        self.waited_until_evening = False

        self.finance.reset()
        self.biometrics.reset()
        self.pantry.reset()
        self.academics.reset()

        return self._get_observation(), self._build_info_dictionary({})

    def step(self, action):
        step_reward = 0.0
        step_events: dict[str, object] = {}

        current_day, _, step_in_day, day_name = self.academics.get_time_context(self.current_step)
        step_definition = self.academics.get_step_definition(step_in_day)
        current_climate = str(step_definition["climate"])
        current_ojol_tariff = str(step_definition["ojol_tariff"])
        step_hunger_gain = float(step_definition["hunger_gain"])
        is_class_scheduled_now = self.academics.is_class_scheduled(day_name, step_in_day)

        if is_class_scheduled_now:
            self.academics.total_classes_scheduled += 1

        # 1. Tugas harian baru di awal hari (step pagi_awal)
        if step_in_day == 0:
            task_stress_gain = self.academics.roll_daily_task_stress_gain(self.np_random)
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
            effective_activity = "rest_or_sleep"
            step_events["forced_rest_due_to_mood_burnout"] = True

        # 3. Fase Mobilitas (src/env/mobility.py)
        (
            self.current_location,
            self.waited_until_evening,
            effective_activity,
            mobility_reward,
        ) = resolve_mobility_phase(
            current_location=self.current_location,
            waited_until_evening=self.waited_until_evening,
            effective_activity=effective_activity,
            requested_mobility=str(parsed_action["mobility"]),
            is_class_scheduled_now=is_class_scheduled_now,
            step_in_day=step_in_day,
            steps_per_day=self.steps_per_day,
            current_climate=current_climate,
            current_ojol_tariff=current_ojol_tariff,
            random_generator=self.np_random,
            finance_manager=self.finance,
            biometrics_state=self.biometrics,
            mobility_config=self.mobility_config,
            rewards_config=self.rewards_config,
            step_events=step_events,
        )
        step_reward += mobility_reward

        # 4. Fase Makan, Masak Pagi Berat, Meal Prep, & Bekal (src/env/actions.py)
        executed_meal_mode = execute_meal_action(
            requested_meal_mode=str(parsed_action["meal_mode"]),
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
        attended_class, executed_activity, self.waited_until_evening = execute_activity_action(
            effective_activity=effective_activity,
            current_location=self.current_location,
            waited_until_evening=self.waited_until_evening,
            is_class_scheduled_now=is_class_scheduled_now,
            is_stress_burned_out=is_stress_burned_out,
            step_in_day=step_in_day,
            steps_per_day=self.steps_per_day,
            requested_grocery_flags=parsed_action["grocery_flags"],
            random_generator=self.np_random,
            finance_manager=self.finance,
            biometrics_state=self.biometrics,
            pantry_manager=self.pantry,
            academics_config=self.academics_config,
            mobility_config=self.mobility_config,
            biometrics_config=self.biometrics_config,
            pantry_config=self.pantry_config,
            step_events=step_events,
        )
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

        # 6. Dinamika Akhir Step (Tidur otomatis, lapar per blok, craving, pembusukan stok & bekal)
        step_reward += self.biometrics.apply_end_of_step_dynamics(
            current_day=current_day,
            step_in_day=step_in_day,
            steps_per_day=self.steps_per_day,
            step_hunger_gain=step_hunger_gain,
            current_climate=current_climate,
            executed_activity=executed_activity,
            finance_config=self.finance_config,
            rewards_config=self.rewards_config,
            step_events=step_events,
        )

        spoiled_units = self.pantry.age_and_spoil_batches()
        if step_in_day == (self.steps_per_day - 1):
            self.waited_until_evening = False
            spoiled_units += self.pantry.spoil_uneaten_packed_lunch_end_of_day()

        if spoiled_units > 0:
            spoilage_penalty = float(
                self.rewards_config["critical_state_penalties"]["food_spoilage_per_unit"]
            )
            step_reward += spoiled_units * spoilage_penalty
            step_events["spoiled_units"] = spoiled_units

        # 7. Reward Kesejahteraan & Evaluasi Terminasi Hybrid
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
            step_events["termination_reason"] = "completed_30_days"

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
            steps_per_day=self.steps_per_day,
            action_dimensions=self.action_dimensions,
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

        return np.array(
            [
                float(self.current_step),
                float(current_day),
                float(day_of_week),
                float(step_in_day),
                is_class_now,
                is_class_next,
                1.0 if self.academics.has_later_class_today(day_name, step_in_day) else 0.0,
                1.0 if self.current_location == "campus" else 0.0,
                1.0 if self.waited_until_evening else 0.0,
                self.finance.balance_idr,
                1.0 if self.finance.is_balance_low() else 0.0,
                self.biometrics.energy,
                self.biometrics.hunger,
                self.biometrics.mood,
                self.biometrics.academic_stress,
                self.biometrics.cafe_craving,
                float(self.biometrics.cafe_buff_remaining_steps),
                float(self.pantry.consecutive_same_meal_count),
                float(self.biometrics.consecutive_fatal_hunger_steps),
                float(self.pantry.get_ingredient_total_units("rice")),
                float(self.pantry.get_ingredient_total_units("egg")),
                float(self.pantry.get_ingredient_total_units("chicken")),
                float(self.pantry.get_ingredient_total_units("tempe")),
                float(self.pantry.get_meal_prep_total_portions()),
                float(self.pantry.get_packed_lunch_total_portions()),
                float(self.pantry.get_min_ingredient_shelf_life("chicken")),
                float(self.pantry.get_min_ingredient_shelf_life("tempe")),
                float(self.pantry.get_min_meal_prep_shelf_life()),
                *last_meal_flags,
            ],
            dtype=np.float32,
        )

    def _build_info_dictionary(self, step_events: dict[str, object]) -> dict[str, object]:
        return {
            "current_step": self.current_step,
            "current_location": self.current_location,
            "balance_idr": self.finance.balance_idr,
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
        print(
            f"[Hari {current_day:02d} | {day_name.capitalize()} - {step_name} | {self.current_location}] "
            f"Saldo: Rp {self.finance.balance_idr:,.0f} | Bekal: {self.pantry.get_packed_lunch_total_portions()} | "
            f"Energi: {self.biometrics.energy:.1f} | Lapar: {self.biometrics.hunger:.1f} | "
            f"Mood: {self.biometrics.mood:.1f} | Stres: {self.biometrics.academic_stress:.1f} | "
            f"Craving: {self.biometrics.cafe_craving:.1f}"
        )
