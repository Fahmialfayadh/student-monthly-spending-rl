import numpy as np


class BiometricsState:
    """
    Menyimpan dan memperbarui kondisi fisiologis serta psikologis mahasiswa:
    energy, hunger, mood, academic_stress, cafe_craving, buff kafe, dan
    counter kelaparan fatal.
    """

    def __init__(self, biometrics_config: dict, termination_config: dict):
        self.biometrics_config = biometrics_config
        self.termination_config = termination_config
        self.step_dynamics = biometrics_config["step_dynamics"]
        self.craving_config = biometrics_config["craving_and_cafe"]
        self.state_bounds = biometrics_config["state_bounds"]

        self.energy = 0.0
        self.hunger = 0.0
        self.mood = 0.0
        self.academic_stress = 0.0
        self.cafe_craving = 0.0
        self.cafe_buff_remaining_steps = 0
        self.consecutive_fatal_hunger_steps = 0

    def reset(self) -> None:
        initial_state = self.biometrics_config["initial_state"]
        self.energy = float(initial_state["energy"])
        self.hunger = float(initial_state["hunger"])
        self.mood = float(initial_state["mood"])
        self.academic_stress = float(initial_state["academic_stress"])
        self.cafe_craving = float(initial_state["cafe_craving"])
        self.cafe_buff_remaining_steps = 0
        self.consecutive_fatal_hunger_steps = 0

    def is_mood_burned_out(self) -> bool:
        return self.mood <= float(self.termination_config["burnout_mood_threshold"])

    def is_stress_burned_out(self) -> bool:
        return self.academic_stress >= float(self.termination_config["burnout_stress_threshold"])

    def get_study_efficiency(self) -> float:
        if self.cafe_craving > float(self.craving_config["distraction_threshold"]):
            return float(self.craving_config["distraction_study_efficiency_multiplier"])
        return 1.0

    def clamp_all_values(self) -> None:
        min_bound, max_bound = self.state_bounds
        self.energy = float(np.clip(self.energy, min_bound, max_bound))
        self.hunger = float(np.clip(self.hunger, min_bound, max_bound))
        self.mood = float(np.clip(self.mood, min_bound, max_bound))
        self.academic_stress = float(np.clip(self.academic_stress, min_bound, max_bound))
        self.cafe_craving = float(np.clip(self.cafe_craving, min_bound, max_bound))

    def apply_end_of_step_dynamics(
        self,
        current_day: int,
        step_in_day: int,
        steps_per_day: int,
        step_hunger_gain: float,
        current_climate: str,
        executed_activity: str,
        finance_config: dict,
        rewards_config: dict,
        step_events: dict[str, object],
    ) -> float:
        """
        Menjalankan dinamika pasif akhir step:
        - Tidur malam otomatis (22:30-06:00) setelah step malam selesai; hanya
          dianggap begadang jika agen secara eksplisit memilih night_cramming.
        - Kenaikan lapar per blok waktu (step_hunger_gain), craving, distraksi,
          starvation, dan clamp nilai ke [0, 100].
        """
        starvation_reward_penalty = 0.0
        critical_penalties = rewards_config["critical_state_penalties"]

        # Tidur otomatis di akhir hari (setelah blok malam 18:30-22:30)
        if step_in_day == (steps_per_day - 1):
            sleep_energy_cap = float(self.step_dynamics["sleep_normal_energy_cap"])
            if executed_activity == "night_cramming":
                self.energy = min(
                    sleep_energy_cap,
                    self.energy + float(self.step_dynamics["sleep_cramming_energy_gain"]),
                )
                self.mood += float(self.step_dynamics["sleep_cramming_mood_delta"])
                step_events["night_cramming"] = True
            else:
                self.energy = min(
                    sleep_energy_cap,
                    self.energy + float(self.step_dynamics["sleep_normal_energy_gain"]),
                )
                self.mood += float(self.step_dynamics["sleep_normal_mood_gain"])

        # Kenaikan lapar sesuai durasi blok waktu
        self.hunger += float(step_hunger_gain)

        # Kenaikan dorongan ngopi/ngafe
        craving_gain = float(self.craving_config["base_craving_gain_per_step"])
        if (
            self.academic_stress > float(self.craving_config["stress_bonus_threshold"])
            or current_climate == "midday"
        ):
            craving_gain += float(self.craving_config["stress_or_heat_craving_bonus"])

        impulsivity_config = finance_config["late_month_impulsivity"]
        if current_day >= int(impulsivity_config["start_day"]):
            craving_gain *= float(impulsivity_config["craving_gain_multiplier"])

        if self.cafe_buff_remaining_steps > 0:
            craving_gain *= float(
                self.craving_config["cafe_session"]["buff_craving_rate_multiplier"]
            )
            self.cafe_buff_remaining_steps -= 1

        if executed_activity != "cafe_session":
            self.cafe_craving += craving_gain

        if self.cafe_craving > float(self.craving_config["distraction_threshold"]):
            self.mood += float(self.craving_config["distraction_mood_penalty_per_step"])
            step_events["distracted_by_craving"] = True

        if self.hunger > float(self.step_dynamics["starvation_threshold"]):
            self.energy += float(self.step_dynamics["starvation_energy_penalty_per_step"])
            self.mood += float(self.step_dynamics["starvation_mood_penalty_per_step"])
            starvation_reward_penalty += float(critical_penalties["starvation_per_step"])
            step_events["starvation_active"] = True

        self.clamp_all_values()

        if self.hunger >= float(self.termination_config["fatal_hunger_threshold"]):
            self.consecutive_fatal_hunger_steps += 1
        else:
            self.consecutive_fatal_hunger_steps = 0

        return starvation_reward_penalty

    def compute_wellbeing_reward(
        self, rewards_config: dict, step_events: dict[str, object]
    ) -> float:
        wellbeing_weights = rewards_config["wellbeing_weights"]
        critical_penalties = rewards_config["critical_state_penalties"]

        wellbeing_reward = float(rewards_config["step_survival_bonus"])
        wellbeing_reward += float(wellbeing_weights["energy_weight"]) * self.energy
        wellbeing_reward += float(wellbeing_weights["mood_weight"]) * self.mood
        wellbeing_reward += float(wellbeing_weights["hunger_penalty_weight"]) * self.hunger
        wellbeing_reward += float(wellbeing_weights["stress_penalty_weight"]) * self.academic_stress

        if self.is_mood_burned_out():
            wellbeing_reward += float(critical_penalties["burnout_mood_per_step"])
            step_events["mood_burnout_penalty_applied"] = True

        if self.is_stress_burned_out():
            wellbeing_reward += float(critical_penalties["burnout_stress_per_step"])
            step_events["stress_burnout_penalty_applied"] = True

        return wellbeing_reward

    def check_fatal_termination(self) -> str | None:
        if self.energy <= float(self.termination_config["fatal_energy_threshold"]):
            return "fatal_energy_collapse"
        if self.consecutive_fatal_hunger_steps >= int(
            self.termination_config["fatal_hunger_consecutive_steps"]
        ):
            return "fatal_starvation"
        return None
