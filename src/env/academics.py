import numpy as np


class AcademicSchedule:
    """
    Mengelola kalender waktu simulasi (hari, hari dalam minggu, blok waktu),
    matriks jadwal kuliah mingguan, serta kemunculan tugas harian.
    """

    DAY_NAMES = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )

    def __init__(self, environment_config: dict, academics_config: dict):
        self.steps_per_day = int(environment_config["steps_per_day"])
        self.total_steps = int(environment_config["total_steps"])
        self.start_day_of_week = int(environment_config["start_day_of_week"])
        self.step_definitions = environment_config["step_definitions"]
        self.academics_config = academics_config
        self.weekly_schedule = academics_config["weekly_schedule"]

        self.total_classes_scheduled = 0
        self.total_classes_attended = 0

    def reset(self) -> None:
        self.total_classes_scheduled = 0
        self.total_classes_attended = 0

    def get_time_context(self, step_index: int) -> tuple[int, int, int, str]:
        clamped_step = min(step_index, self.total_steps - 1)
        day_index = clamped_step // self.steps_per_day
        current_day = day_index + 1
        day_of_week = (self.start_day_of_week + day_index) % 7
        step_in_day = clamped_step % self.steps_per_day
        day_name = self.DAY_NAMES[day_of_week]
        return current_day, day_of_week, step_in_day, day_name

    def get_step_definition(self, step_in_day: int) -> dict:
        return self.step_definitions[step_in_day]

    def is_class_scheduled(self, day_name: str, step_in_day: int) -> bool:
        return bool(self.weekly_schedule[day_name][step_in_day])

    def has_later_class_today(self, day_name: str, step_in_day: int) -> bool:
        daily_schedule = self.weekly_schedule[day_name]
        return any(bool(slot) for slot in daily_schedule[step_in_day + 1 :])

    def roll_daily_task_stress_gain(self, random_generator: np.random.Generator) -> float:
        daily_task_probability = float(self.academics_config["daily_task_probability"])
        if random_generator.random() < daily_task_probability:
            return float(self.academics_config["new_task_stress_gain"])
        return 0.0
