from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class StepDefinitionConfig(BaseModel):
    name: str
    hours: str
    climate: Literal["cool", "midday"] = "cool"
    ojol_tariff: Literal["normal", "peak"] = "normal"
    hunger_gain: float = Field(ge=0.0)
    rain_prob: float = Field(default=0.0, ge=0.0, le=1.0)


class EnvironmentConfig(BaseModel):
    episode_days: int = Field(default=30, ge=1)
    steps_per_day: int = Field(default=5, ge=2)
    total_steps: int = Field(default=150, ge=2)
    start_day_of_week: int = Field(default=0, ge=0, le=6)
    step_definitions: dict[int, StepDefinitionConfig]

    @model_validator(mode="after")
    def _validate_steps(self) -> "EnvironmentConfig":
        for idx in range(self.steps_per_day):
            if idx not in self.step_definitions:
                raise ValueError(f"step_definitions harus memiliki key {idx}.")
        return self


class FinanceConfig(BaseModel):
    monthly_budget: float = Field(gt=0.0)
    fixed_expenses: dict[str, float] = Field(default_factory=dict)
    alerts: dict[str, float] = Field(default_factory=lambda: {"low_balance_threshold": 250000.0})
    late_month_impulsivity: dict[str, float] = Field(
        default_factory=lambda: {"start_day": 22, "weight": 0.50}
    )


class IngredientConfig(BaseModel):
    pack_price_idr: float = Field(gt=0.0)
    units_per_pack: int = Field(ge=1)
    shelf_life_steps: int = Field(ge=1)
    nutrition_per_unit: dict[str, float]


class PantryAndCookingConfig(BaseModel):
    initial_pantry: dict[str, int] = Field(default_factory=dict)
    ingredients: dict[str, IngredientConfig]
    meal_composition_rules: dict[str, float]
    grocery_shopping: dict[str, float]
    cooking_costs_and_gates: dict[str, float]
    boredom_mechanics: dict[str, Any]


class BiometricsAndPsychologyConfig(BaseModel):
    state_bounds: list[float] = Field(default_factory=lambda: [0.0, 100.0], min_length=2, max_length=2)
    initial_state: dict[str, float]
    step_dynamics: dict[str, float]
    craving_and_cafe: dict[str, Any]


class WalkPhysicsConfig(BaseModel):
    energy_cost_per_km_cool: float = Field(default=3.3333, ge=0.0)
    energy_cost_per_km_midday: float = Field(default=10.0, ge=0.0)
    mood_delta_per_km_cool: float = Field(default=-1.3333)
    mood_delta_per_km_midday: float = Field(default=-8.0)
    rain_extra_energy_per_km: float = Field(default=6.6667, ge=0.0)
    rain_extra_mood_per_km: float = Field(default=-5.3333)


class MotorizedPhysicsConfig(BaseModel):
    default_energy_cost: float = Field(default=3.0, ge=0.0)
    mood_delta_cool: float = Field(default=0.0)
    mood_delta_midday: float = Field(default=-2.0)
    mood_delta_rain: float = Field(default=-3.0)


class EdgePricingConfig(BaseModel):
    pricing_model: Literal["free", "manual_range", "per_km_custom", "ride_hailing_zone"] = "manual_range"
    # manual_range
    cost_normal_idr: list[int] | None = None
    cost_peak_idr: list[int] | None = None
    # per_km_custom & ride_hailing_zone overrides
    zone_preset: Literal["goride_zona_1", "goride_zona_2", "goride_zona_3", "gocar"] | None = None
    min_distance_km: float | None = Field(default=None, ge=0.0)
    min_fare_idr: list[int] | None = None
    base_fare_idr: list[int] | None = None
    per_km_idr: list[int] | None = None
    app_fee_idr: list[int] | None = None
    peak_multiplier: float | None = Field(default=None, ge=1.0)
    rain_surge_multiplier: float | None = Field(default=1.25, ge=1.0)

    @model_validator(mode="after")
    def _validate_pricing(self) -> "EdgePricingConfig":
        if self.pricing_model == "manual_range" and not self.cost_normal_idr:
            raise ValueError("pricing_model='manual_range' membutuhkan cost_normal_idr.")
        if self.pricing_model == "per_km_custom" and not self.per_km_idr:
            raise ValueError("pricing_model='per_km_custom' membutuhkan per_km_idr.")
        if self.pricing_model == "ride_hailing_zone" and not self.zone_preset:
            self.zone_preset = "goride_zona_1"
        return self


class ImpulseItemConfig(BaseModel):
    id: str = "minimarket_impulse"
    label: str = "Jajan Minimarket"
    spend_idr: list[int] = Field(default_factory=lambda: [8000, 15000])
    hunger_delta: float = -12.0
    mood_delta: float = 4.0
    hungry_prob_bonus: float = 0.15
    midday_prob_bonus: float = 0.12
    late_month_prob_bonus: float = 0.12
    low_balance_prob_multiplier: float = 0.40


class NodeStudyProfile(BaseModel):
    energy_cost: float = Field(default=10.0, ge=0.0)
    stress_reduction: float = Field(default=12.0, ge=0.0)
    mood_delta: float = 0.0
    max_stress_gate: float = Field(default=100.0, ge=0.0, le=100.0)
    affected_by_craving_distraction: bool = False
    craving_relief: float = Field(default=0.0, ge=0.0)
    craving_floor: float = Field(default=0.0, ge=0.0)
    craving_reset: bool = False
    buff_duration_steps: int = Field(default=0, ge=0)
    buff_craving_rate_multiplier: float = Field(default=1.0, gt=0.0, le=1.0)
    daily_diminishing_returns: bool = False
    repeat_craving_relief: float = Field(default=0.0, ge=0.0)
    first_visit_mood_gain: float = 0.0


class LocationNodeConfig(BaseModel):
    id: str
    label: str
    zone: Literal["home", "campus", "third_place", "commercial", "custom"] = "custom"
    # Efek pasif saat berada di lokasi ini per step
    ambient_mood_delta: float = 0.0
    ambient_stress_delta: float = 0.0
    craving_gain_per_step: float = Field(default=3.0, ge=0.0)
    # Kapabilitas & aturan tempat
    can_sleep: bool = False
    has_kitchen: bool = False
    has_grocery: bool = False
    is_class_venue: bool = False
    can_study: bool = False
    closes_at_night: bool = False
    study_profile: NodeStudyProfile = Field(default_factory=NodeStudyProfile)
    # Kafe / leisure reset & jajan impulsif di lokasi ini
    leisure_spend_idr: list[int] | None = None
    leisure_hunger_delta: float = 0.0
    leisure_mood_gain: float = 0.0
    leisure_stress_reduction: float = 0.0
    leisure_energy_cost: float = 0.0
    leisure_buff_duration_steps: int = 0
    leisure_buff_craving_rate_multiplier: float = 1.0
    wait_energy_cost: float = 2.0
    wait_mood_delta: float = -4.0
    temptation_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    impulse_item_id: str | None = None
    food_options: list[str] = Field(default_factory=list)
    ui_position: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})


class EdgeRouteConfig(BaseModel):
    id: str
    label: str | None = None
    source: str
    target: str
    bidirectional: bool = True
    mode: Literal["walk", "ojol", "goride", "gocar", "public_transit", "personal", "custom"] = "ojol"
    distance_km: float = Field(gt=0.0)
    pricing: EdgePricingConfig = Field(default_factory=EdgePricingConfig)
    # Override opsional beban fisik (jika None, dihitung otomatis dari distance_km)
    energy_cost_cool: float | None = None
    energy_cost_midday: float | None = None
    energy_cost_rain: float | None = None
    mood_delta_cool: float | None = None
    mood_delta_midday: float | None = None
    mood_delta_rain: float | None = None
    temptation_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    impulse_item_id: str | None = None


class ActivitySpecConfig(BaseModel):
    id: str
    label: str
    handler: Literal[
        "rest",
        "attend_schedule",
        "study",
        "hub_stay",
        "leisure_reset",
        "wait",
        "grocery_shop",
        "night_cram",
    ]
    target_nodes: list[str] | None = None


class ExternalFoodSpecConfig(BaseModel):
    id: str = ""
    label: str
    tier: Literal["budget", "varied"] = "budget"
    cost_idr: float = Field(ge=0.0)
    hunger: float
    energy: float
    energy_cost: float = Field(default=0.0, ge=0.0)
    mood: float


class WorldGraphConfig(BaseModel):
    home_node_id: str
    locations: list[LocationNodeConfig] = Field(min_length=1)
    edges: list[EdgeRouteConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_graph_references(self) -> "WorldGraphConfig":
        node_ids = {node.id for node in self.locations}
        if len(node_ids) != len(self.locations):
            raise ValueError("Duplicate location id ditemukan dalam world_graph.locations.")
        if self.home_node_id not in node_ids:
            raise ValueError(f"home_node_id '{self.home_node_id}' tidak ada di world_graph.locations.")
        home_node = next(n for n in self.locations if n.id == self.home_node_id)
        if not home_node.can_sleep:
            raise ValueError(f"home_node_id '{self.home_node_id}' wajib memiliki can_sleep=true.")
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(
                    f"Edge '{edge.id}' menghubungkan node yang tidak terdaftar: {edge.source} -> {edge.target}."
                )
        return self


class MobilityAndClimateConfig(BaseModel):
    heat_sensitivity: float = Field(default=0.95, ge=0.0)
    walk_physics: WalkPhysicsConfig = Field(default_factory=WalkPhysicsConfig)
    motorized_physics: MotorizedPhysicsConfig = Field(default_factory=MotorizedPhysicsConfig)
    impulse_items: dict[str, ImpulseItemConfig] = Field(default_factory=dict)


class AcademicsConfig(BaseModel):
    weekly_schedule: dict[str, list[int]]
    weekday_task_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    weekday_base_task_stress_gain: float = 14.0
    per_class_daily_stress_gain: float = 4.0
    weekend_task_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    weekend_task_stress_gain: float = 6.0
    room_study_max_stress_gate: float = 80.0
    room_study_stress_reduction: float = 12.0
    room_study_energy_cost: float = 10.0
    night_cramming_stress_reduction: float = 24.0
    night_cramming_energy_cost: float = 14.0
    class_attendance_energy_cost: float = 10.0
    absent_class_reward_penalty: float = -50.0
    absent_class_stress_gain: float = 12.0
    min_attendance_rate_for_exam: float = Field(default=0.80, ge=0.0, le=1.0)


class TerminationConfig(BaseModel):
    mode: str = "hybrid"
    fatal_energy_threshold: float = 0.0
    fatal_hunger_threshold: float = 100.0
    fatal_hunger_consecutive_steps: int = Field(default=2, ge=1)
    burnout_mood_threshold: float = 0.0
    burnout_stress_threshold: float = 100.0
    fatal_termination_penalty: float = -800.0


class RewardsConfig(BaseModel):
    step_survival_bonus: float = 1.0
    wellbeing_weights: dict[str, float]
    critical_state_penalties: dict[str, float]
    class_attendance_bonus: float = 6.0
    terminal_episode_completion: dict[str, float]
    budget_potential_shaping: dict[str, Any]


class WorldConfig(BaseModel):
    """Skema tervalidasi penuh untuk seluruh environment & spatial graph UI."""

    schema_version: str = "2.0"
    preset_name: str = "custom"
    environment: EnvironmentConfig
    finance: FinanceConfig
    pantry_and_cooking: PantryAndCookingConfig
    biometrics_and_psychology: BiometricsAndPsychologyConfig
    mobility_and_climate: MobilityAndClimateConfig
    world_graph: WorldGraphConfig
    activities: list[ActivitySpecConfig] = Field(min_length=1)
    external_foods: dict[str, ExternalFoodSpecConfig] = Field(default_factory=dict)
    academics: AcademicsConfig
    termination: TerminationConfig
    rewards: RewardsConfig

    @model_validator(mode="before")
    @classmethod
    def _normalize_food_ids(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("external_foods"), dict):
            normalized_foods: dict[str, Any] = {}
            for food_id, spec in data["external_foods"].items():
                if isinstance(spec, dict) and not spec.get("id"):
                    normalized_foods[food_id] = {**spec, "id": str(food_id)}
                else:
                    normalized_foods[food_id] = spec
            data = {**data, "external_foods": normalized_foods}
        return data

    @model_validator(mode="after")
    def _validate_references(self) -> "WorldConfig":
        node_map = {node.id: node for node in self.world_graph.locations}
        for node in self.world_graph.locations:
            for food_id in node.food_options:
                if food_id not in self.external_foods:
                    raise ValueError(
                        f"Node '{node.id}' mereferensikan food_option '{food_id}' yang tidak ada di external_foods."
                    )
            if node.impulse_item_id and node.impulse_item_id not in self.mobility_and_climate.impulse_items:
                raise ValueError(
                    f"Node '{node.id}' mereferensikan impulse_item_id '{node.impulse_item_id}' yang tidak terdaftar."
                )
        for edge in self.world_graph.edges:
            if edge.impulse_item_id and edge.impulse_item_id not in self.mobility_and_climate.impulse_items:
                raise ValueError(
                    f"Edge '{edge.id}' mereferensikan impulse_item_id '{edge.impulse_item_id}' yang tidak terdaftar."
                )
        for act in self.activities:
            if act.target_nodes:
                for nid in act.target_nodes:
                    if nid not in node_map:
                        raise ValueError(f"Activity '{act.id}' mereferensikan node '{nid}' yang tidak valid.")
        return self


def load_world_config(yaml_or_dict: str | Path | dict[str, Any]) -> WorldConfig:
    if isinstance(yaml_or_dict, (str, Path)):
        raw = yaml.safe_load(Path(yaml_or_dict).read_text(encoding="utf-8"))
    else:
        raw = dict(yaml_or_dict)
    return WorldConfig.model_validate(raw)


def export_world_json_schema() -> dict[str, Any]:
    """Mengekspor JSON Schema lengkap WorldConfig untuk generator form & graph UI."""
    return WorldConfig.model_json_schema()
