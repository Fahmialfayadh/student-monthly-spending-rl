from __future__ import annotations

import numpy as np

from src.env.academics import AcademicSchedule
from src.env.biometrics import BiometricsState
from src.env.finance import FinanceManager
from src.env.mobility import try_impulse_temptation
from src.env.pantry import PantryManager
from src.env.pricing import sample_idr_from_range
from src.env.schema import (
    ActivitySpecConfig,
    ExternalFoodSpecConfig,
    LocationNodeConfig,
)
from src.env.world_graph import RoutePreference, WorldGraph


def parse_action_vector(
    action,
    action_dimensions: dict[str, list],
    pantry: PantryManager | None = None,
) -> dict[str, object]:
    """Mengubah vektor integer MultiDiscrete (6 elemen) menjadi dict nama aksi yang jelas."""
    (
        activity_index,
        route_index,
        meal_mode_index,
        cook_combo_index,
        bring_bekal_index,
        grocery_bundle_index,
    ) = [int(value) for value in action]

    cook_quantities = (
        pantry.decode_cook_combo_dict(cook_combo_index)
        if pantry is not None
        else PantryManager.decode_cook_combo(cook_combo_index)
    )
    grocery_flags = (
        pantry.decode_grocery_bundle_dict(grocery_bundle_index)
        if pantry is not None
        else PantryManager.decode_grocery_bundle(grocery_bundle_index)
    )

    return {
        "activity": action_dimensions["activity"][activity_index],
        "route": action_dimensions["route"][route_index],
        "meal_mode": action_dimensions["meal_mode"][meal_mode_index],
        "cook_combo_index": cook_combo_index,
        "cook_quantities": cook_quantities,
        "bring_bekal": bool(action_dimensions["bring_bekal"][bring_bekal_index]),
        "grocery_bundle_index": grocery_bundle_index,
        "grocery_flags": grocery_flags,
    }


def _get_cooking_energy_requirements(
    cooking_gates: dict, mode_name: str, is_morning_step: bool
) -> tuple[float, float]:
    if mode_name == "cook_fresh":
        base_cost = float(cooking_gates["cook_fresh_energy_cost"])
        min_energy = float(cooking_gates["cook_fresh_min_energy"])
        if is_morning_step:
            base_cost += float(cooking_gates["morning_cook_fresh_extra_energy_cost"])
            min_energy = float(cooking_gates["morning_cook_fresh_min_energy"])
        return base_cost, min_energy

    base_cost = float(cooking_gates["meal_prep_energy_cost"])
    min_energy = float(cooking_gates["meal_prep_min_energy"])
    if is_morning_step:
        base_cost += float(cooking_gates["morning_meal_prep_extra_energy_cost"])
        min_energy = float(cooking_gates["morning_meal_prep_min_energy"])
    return base_cost, min_energy


def _apply_eaten_meal(
    biometrics_state: BiometricsState,
    pantry_manager: PantryManager,
    hunger_delta: float,
    energy_gain: float,
    mood_delta: float,
    meal_signature: str,
    step_events: dict[str, object],
) -> None:
    biometrics_state.hunger += hunger_delta
    biometrics_state.energy += energy_gain
    boredom_penalty = pantry_manager.compute_boredom_mood_penalty(meal_signature)
    biometrics_state.mood += mood_delta + boredom_penalty
    step_events["meal_signature"] = meal_signature
    step_events["boredom_mood_penalty"] = boredom_penalty


def select_food_for_node(
    node: LocationNodeConfig,
    tier: str,
    external_foods: dict[str, ExternalFoodSpecConfig],
) -> ExternalFoodSpecConfig | None:
    """Memilih makanan eksternal pada node berdasarkan tier ('budget' atau 'varied')."""
    if not node.food_options:
        return None
    node_foods = [external_foods[fid] for fid in node.food_options if fid in external_foods]
    tier_matches = [f for f in node_foods if f.tier == tier]
    if tier_matches:
        return tier_matches[0]
    return node_foods[0] if node_foods else None


def try_pack_bekal_before_departure(
    bring_bekal: bool,
    biometrics_state: BiometricsState,
    pantry_manager: PantryManager,
    pantry_config: dict,
    step_events: dict[str, object],
) -> bool:
    if not bring_bekal:
        return False

    cooking_gates = pantry_config["cooking_costs_and_gates"]
    max_packed = int(cooking_gates["max_packed_lunch_portions"])
    if (
        pantry_manager.get_meal_prep_total_portions() >= 1
        and pantry_manager.get_packed_lunch_total_portions() < max_packed
        and biometrics_state.energy >= float(cooking_gates["pack_bekal_min_energy"])
        and biometrics_state.mood >= float(cooking_gates["pack_bekal_min_mood"])
    ):
        biometrics_state.energy -= float(cooking_gates["pack_bekal_energy_cost"])
        packed_portion = pantry_manager.pack_one_bekal_from_prep()
        if packed_portion is not None:
            step_events["packed_bekal_signature"] = str(packed_portion["meal_signature"])
            return True
    return False


def execute_meal_action(
    requested_meal_mode: str,
    requested_cook_quantities: dict[str, int],
    current_node: LocationNodeConfig,
    step_in_day: int,
    is_stress_burned_out: bool,
    finance_manager: FinanceManager,
    biometrics_state: BiometricsState,
    pantry_manager: PantryManager,
    pantry_config: dict,
    biometrics_config: dict,
    external_foods: dict[str, ExternalFoodSpecConfig],
    step_events: dict[str, object],
) -> str:
    """Menjalankan fase makan & dapur sesuai kapabilitas node (`has_kitchen` & `food_options`)."""
    cooking_gates = pantry_config["cooking_costs_and_gates"]
    step_dynamics = biometrics_config["step_dynamics"]

    has_kitchen = current_node.has_kitchen
    is_morning_step = step_in_day == 0
    executed_mode = requested_meal_mode
    can_eat = biometrics_state.hunger >= float(step_dynamics["min_hunger_to_eat"])

    if not can_eat and executed_mode in (
        "cook_fresh",
        "reheat_or_eat_bekal",
        "budget_external_meal",
        "varied_external_meal",
    ):
        executed_mode = "skip"

    if executed_mode == "cook_fresh":
        energy_cost, min_energy = _get_cooking_energy_requirements(
            cooking_gates, "cook_fresh", is_morning_step
        )
        total_units = sum(requested_cook_quantities.values())
        if (
            has_kitchen
            and not is_stress_burned_out
            and total_units > 0
            and pantry_manager.has_enough_ingredients(requested_cook_quantities)
            and biometrics_state.energy >= min_energy
            and biometrics_state.mood >= float(cooking_gates["cook_fresh_min_mood"])
        ):
            biometrics_state.energy -= energy_cost
            pantry_manager.consume_ingredients_fifo(requested_cook_quantities)
            hunger_delta, energy_gain, mood_delta, meal_signature = (
                pantry_manager.calculate_custom_meal_profile(requested_cook_quantities)
            )
            _apply_eaten_meal(
                biometrics_state,
                pantry_manager,
                hunger_delta,
                energy_gain,
                mood_delta,
                meal_signature,
                step_events,
            )
            if is_morning_step:
                step_events["morning_heavy_cooking"] = True
            return "cook_fresh"
        executed_mode = "skip"

    elif executed_mode == "meal_prep":
        energy_cost, min_energy = _get_cooking_energy_requirements(
            cooking_gates, "meal_prep", is_morning_step
        )
        max_batch_size = int(cooking_gates["meal_prep_batch_size"])
        min_batch_size = int(cooking_gates["meal_prep_min_batch_size"])
        max_possible = pantry_manager.max_portions_possible(requested_cook_quantities)
        actual_batch = min(max_batch_size, max_possible)

        if (
            has_kitchen
            and not is_stress_burned_out
            and actual_batch >= min_batch_size
            and biometrics_state.energy >= min_energy
            and biometrics_state.mood >= float(cooking_gates["meal_prep_min_mood"])
        ):
            batch_energy_scale = 0.65 + 0.35 * (actual_batch / max_batch_size)
            biometrics_state.energy -= energy_cost * batch_energy_scale
            pantry_manager.consume_ingredients_fifo(
                requested_cook_quantities, batch_multiplier=actual_batch
            )
            hunger_delta, energy_gain, mood_delta, meal_signature = (
                pantry_manager.calculate_custom_meal_profile(requested_cook_quantities)
            )
            portion_value_idr = pantry_manager.compute_combo_ingredient_value_idr(
                requested_cook_quantities
            )

            portions_to_store = actual_batch
            if can_eat:
                _apply_eaten_meal(
                    biometrics_state,
                    pantry_manager,
                    hunger_delta,
                    energy_gain,
                    mood_delta,
                    meal_signature,
                    step_events,
                )
                portions_to_store = actual_batch - 1
                step_events["ate_fresh_from_meal_prep"] = True

            pantry_manager.store_meal_prep_batch(
                portions=portions_to_store,
                hunger_delta=hunger_delta,
                energy_delta=energy_gain,
                mood_delta=mood_delta,
                meal_signature=meal_signature,
                portion_value_idr=portion_value_idr,
            )
            step_events["meal_prep_created_signature"] = meal_signature
            step_events["meal_prep_portions_created"] = actual_batch
            if is_morning_step:
                step_events["morning_heavy_cooking"] = True
            return "meal_prep"

        if has_kitchen and not is_stress_burned_out and can_eat and max_possible >= 1:
            fresh_cost, fresh_min_e = _get_cooking_energy_requirements(
                cooking_gates, "cook_fresh", is_morning_step
            )
            if (
                biometrics_state.energy >= fresh_min_e
                and biometrics_state.mood >= float(cooking_gates["cook_fresh_min_mood"])
            ):
                biometrics_state.energy -= fresh_cost
                pantry_manager.consume_ingredients_fifo(requested_cook_quantities)
                hunger_delta, energy_gain, mood_delta, meal_signature = (
                    pantry_manager.calculate_custom_meal_profile(requested_cook_quantities)
                )
                _apply_eaten_meal(
                    biometrics_state,
                    pantry_manager,
                    hunger_delta,
                    energy_gain,
                    mood_delta,
                    meal_signature,
                    step_events,
                )
                return "cook_fresh"
        executed_mode = "skip"

    elif executed_mode == "reheat_or_eat_bekal":
        reheat_mood_delta = float(pantry_config["meal_composition_rules"]["reheat_mood_delta"])
        if has_kitchen:
            if pantry_manager.get_packed_lunch_total_portions() >= 1:
                biometrics_state.energy -= float(cooking_gates["eat_bekal_energy_cost"])
                bekal_portion = pantry_manager.pop_one_packed_lunch_portion(
                    avoid_signature=pantry_manager.last_meal_signature
                )
                assert bekal_portion is not None
                _apply_eaten_meal(
                    biometrics_state,
                    pantry_manager,
                    hunger_delta=float(bekal_portion["hunger_delta"]),
                    energy_gain=float(bekal_portion["energy_delta"]),
                    mood_delta=float(bekal_portion["mood_delta"]) + reheat_mood_delta,
                    meal_signature=str(bekal_portion["meal_signature"]),
                    step_events=step_events,
                )
                return "eat_bekal"

            if (
                pantry_manager.get_meal_prep_total_portions() >= 1
                and biometrics_state.energy >= float(cooking_gates["reheat_prep_min_energy"])
                and biometrics_state.mood >= float(cooking_gates["reheat_prep_min_mood"])
            ):
                biometrics_state.energy -= float(cooking_gates["reheat_prep_energy_cost"])
                reheated_portion = pantry_manager.pop_one_meal_prep_portion_smart_fifo(
                    avoid_signature=pantry_manager.last_meal_signature
                )
                assert reheated_portion is not None
                _apply_eaten_meal(
                    biometrics_state,
                    pantry_manager,
                    hunger_delta=float(reheated_portion["hunger_delta"]),
                    energy_gain=float(reheated_portion["energy_delta"]),
                    mood_delta=float(reheated_portion["mood_delta"]) + reheat_mood_delta,
                    meal_signature=str(reheated_portion["meal_signature"]),
                    step_events=step_events,
                )
                return "reheat_prep"
        else:
            if pantry_manager.get_packed_lunch_total_portions() >= 1:
                biometrics_state.energy -= float(cooking_gates["eat_bekal_energy_cost"])
                bekal_portion = pantry_manager.pop_one_packed_lunch_portion(
                    avoid_signature=pantry_manager.last_meal_signature
                )
                assert bekal_portion is not None
                _apply_eaten_meal(
                    biometrics_state,
                    pantry_manager,
                    hunger_delta=float(bekal_portion["hunger_delta"]),
                    energy_gain=float(bekal_portion["energy_delta"]),
                    mood_delta=float(bekal_portion["mood_delta"]) + reheat_mood_delta,
                    meal_signature=str(bekal_portion["meal_signature"]),
                    step_events=step_events,
                )
                step_events["ate_bekal_outside"] = True
                return "eat_bekal"
        executed_mode = "skip"

    elif executed_mode in ("budget_external_meal", "varied_external_meal"):
        tier = "budget" if executed_mode == "budget_external_meal" else "varied"
        food_spec = select_food_for_node(current_node, tier, external_foods)
        if food_spec is not None and finance_manager.pay(
            float(food_spec.cost_idr), category="external_food"
        ):
            biometrics_state.energy -= float(food_spec.energy_cost)
            _apply_eaten_meal(
                biometrics_state,
                pantry_manager,
                hunger_delta=float(food_spec.hunger),
                energy_gain=float(food_spec.energy),
                mood_delta=float(food_spec.mood),
                meal_signature=food_spec.id,
                step_events=step_events,
            )
            return food_spec.id
        executed_mode = "skip"

    skip_threshold = float(step_dynamics["skip_meal_hunger_threshold"])
    if executed_mode == "skip" and biometrics_state.hunger >= skip_threshold:
        tolerance = float(step_dynamics["skip_meal_tolerance"])
        mood_scale = max(0.3, 1.3 - 0.5 * tolerance)
        biometrics_state.mood += float(step_dynamics["skip_meal_mood_penalty"]) * mood_scale
        biometrics_state.energy += float(step_dynamics["skip_meal_energy_penalty"])
        step_events["skipped_meal_while_hungry"] = True

    return "skip"


def _apply_default_wait_or_rest(
    current_node: LocationNodeConfig,
    is_night: bool,
    biometrics_state: BiometricsState,
    biometrics_config: dict,
    activity_specs: dict[str, ActivitySpecConfig],
) -> tuple[bool, str]:
    if not current_node.can_sleep and not is_night:
        biometrics_state.energy -= float(current_node.wait_energy_cost)
        biometrics_state.mood += float(current_node.wait_mood_delta)
        wait_id = next((aid for aid, s in activity_specs.items() if s.handler == "wait"), "wait")
        return False, wait_id

    step_dynamics = biometrics_config["step_dynamics"]
    craving_cfg = biometrics_config["craving_and_cafe"]
    if is_night:
        biometrics_state.energy += float(step_dynamics["night_unwind_energy_gain"])
        biometrics_state.mood += float(step_dynamics["night_unwind_mood_gain"])
    else:
        biometrics_state.energy += float(step_dynamics["rest_step_energy_gain"])
        if biometrics_state.cafe_craving > float(craving_cfg["high_distraction_threshold"]):
            biometrics_state.mood += float(step_dynamics["rest_step_high_craving_mood_delta"])
        elif biometrics_state.cafe_craving <= float(craving_cfg["distraction_threshold"]):
            biometrics_state.mood += float(step_dynamics["rest_step_mood_gain"])

    rest_id = next((aid for aid, s in activity_specs.items() if s.handler == "rest"), "rest")
    return False, rest_id


def execute_activity_action(
    effective_activity_id: str,
    activity_specs: dict[str, ActivitySpecConfig],
    current_node: LocationNodeConfig,
    world_graph: WorldGraph,
    is_class_scheduled_now: bool,
    is_stress_burned_out: bool,
    step_in_day: int,
    steps_per_day: int,
    current_climate: str,
    is_raining: bool,
    current_day: int,
    requested_grocery_flags: dict[str, bool],
    random_generator: np.random.Generator,
    finance_manager: FinanceManager,
    biometrics_state: BiometricsState,
    pantry_manager: PantryManager,
    finance_config: dict,
    academics_config: dict,
    biometrics_config: dict,
    pantry_config: dict,
    step_events: dict[str, object],
) -> tuple[bool, str]:
    """Menjalankan aktivitas secara generik berbasis handler dan kapabilitas `current_node`."""
    spec = activity_specs[effective_activity_id]
    handler = spec.handler
    is_night = step_in_day == (steps_per_day - 1)
    effective_climate = "rain" if is_raining else current_climate
    valid_targets = set(world_graph.resolve_activity_target_nodes(spec))

    if handler == "attend_schedule":
        if is_class_scheduled_now and current_node.is_class_venue and current_node.id in valid_targets:
            biometrics_state.energy -= float(academics_config["class_attendance_energy_cost"])
            return True, spec.id
        return _apply_default_wait_or_rest(
            current_node, is_night, biometrics_state, biometrics_config, activity_specs
        )

    if handler == "study":
        profile = current_node.study_profile
        if (
            current_node.can_study
            and current_node.id in valid_targets
            and not is_stress_burned_out
            and biometrics_state.academic_stress <= float(profile.max_stress_gate)
        ):
            eff = biometrics_state.get_study_efficiency(profile.affected_by_craving_distraction)
            biometrics_state.energy -= float(profile.energy_cost)
            biometrics_state.academic_stress -= float(profile.stress_reduction) * eff
            biometrics_state.mood += float(profile.mood_delta)
            return False, spec.id
        return _apply_default_wait_or_rest(
            current_node, is_night, biometrics_state, biometrics_config, activity_specs
        )

    if handler == "night_cram":
        if current_node.can_sleep and is_night and not is_stress_burned_out:
            eff = biometrics_state.get_study_efficiency(
                current_node.study_profile.affected_by_craving_distraction
            )
            biometrics_state.energy -= float(academics_config["night_cramming_energy_cost"])
            biometrics_state.academic_stress -= (
                float(academics_config["night_cramming_stress_reduction"]) * eff
            )
            return False, spec.id
        return _apply_default_wait_or_rest(
            current_node, is_night, biometrics_state, biometrics_config, activity_specs
        )

    if handler == "hub_stay":
        if current_node.id in valid_targets and not is_night:
            profile = current_node.study_profile
            eff = biometrics_state.get_study_efficiency(profile.affected_by_craving_distraction)
            biometrics_state.energy -= float(profile.energy_cost)
            biometrics_state.academic_stress -= float(profile.stress_reduction) * eff
            biometrics_state.mood += float(profile.mood_delta)

            if profile.craving_reset:
                biometrics_state.cafe_craving = 0.0
            elif profile.craving_relief > 0.0:
                if profile.daily_diminishing_returns and biometrics_state.hub_visits_today > 0:
                    relief = float(profile.repeat_craving_relief)
                else:
                    relief = float(profile.craving_relief)
                    biometrics_state.mood += float(profile.first_visit_mood_gain)
                floor = float(profile.craving_floor)
                if biometrics_state.cafe_craving > floor and relief > 0.0:
                    biometrics_state.cafe_craving = max(floor, biometrics_state.cafe_craving - relief)

            biometrics_state.hub_visits_today += 1
            biometrics_state.apply_craving_buff(
                duration_steps=int(profile.buff_duration_steps),
                rate_multiplier=float(profile.buff_craving_rate_multiplier),
            )

            mobility_cfg = world_graph.mobility_cfg
            impulse_item = (
                mobility_cfg.impulse_items.get(current_node.impulse_item_id)
                if current_node.impulse_item_id
                else None
            )
            bought, imp_h, imp_m = try_impulse_temptation(
                random_generator=random_generator,
                finance_manager=finance_manager,
                biometrics_state=biometrics_state,
                finance_config=finance_config,
                impulse_item=impulse_item,
                base_temptation_probability=float(current_node.temptation_prob),
                effective_climate=effective_climate,
                current_day=current_day,
            )
            if bought:
                biometrics_state.hunger += imp_h
                biometrics_state.mood += imp_m
                step_events["node_minimarket_impulse"] = True
            return False, spec.id
        return _apply_default_wait_or_rest(
            current_node, is_night, biometrics_state, biometrics_config, activity_specs
        )

    if handler == "leisure_reset":
        if current_node.id in valid_targets and current_node.leisure_spend_idr:
            spend_range = current_node.leisure_spend_idr
            min_spend = float(spend_range[0])
            if finance_manager.can_afford(min_spend):
                sampled_spend = sample_idr_from_range(random_generator, spend_range)
                actual_spend = min(sampled_spend, finance_manager.balance_idr)
                finance_manager.pay(actual_spend, category="cafe")
                biometrics_state.energy -= float(current_node.leisure_energy_cost)
                biometrics_state.hunger += float(current_node.leisure_hunger_delta)
                biometrics_state.mood += float(current_node.leisure_mood_gain)
                biometrics_state.academic_stress -= float(current_node.leisure_stress_reduction)
                biometrics_state.cafe_craving = float(
                    biometrics_config["craving_and_cafe"]["cafe_session"].get("craving_reset_to", 0.0)
                )
                biometrics_state.apply_craving_buff(
                    duration_steps=int(current_node.leisure_buff_duration_steps),
                    rate_multiplier=float(current_node.leisure_buff_craving_rate_multiplier),
                )
                step_events["cafe_spend_idr"] = actual_spend
                return False, spec.id
        return _apply_default_wait_or_rest(
            current_node, is_night, biometrics_state, biometrics_config, activity_specs
        )

    if handler == "wait":
        if current_node.id in valid_targets and not is_night:
            biometrics_state.energy -= float(current_node.wait_energy_cost)
            biometrics_state.mood += float(current_node.wait_mood_delta)
            return False, spec.id
        return _apply_default_wait_or_rest(
            current_node, is_night, biometrics_state, biometrics_config, activity_specs
        )

    if handler == "grocery_shop":
        if current_node.has_grocery and current_node.id in valid_targets:
            ingredients_config = pantry_config["ingredients"]
            total_bundle_cost = sum(
                float(ingredients_config[name]["pack_price_idr"])
                for name in pantry_manager.ingredient_names
                if requested_grocery_flags.get(name, False)
            )
            shopping_config = pantry_config["grocery_shopping"]
            shopping_transport_cost = float(shopping_config["transport_cost_idr"])

            if total_bundle_cost > 0.0 and finance_manager.can_afford(
                total_bundle_cost + shopping_transport_cost
            ):
                purchased_items: list[str] = []
                for ingredient_name in pantry_manager.ingredient_names:
                    if not requested_grocery_flags.get(ingredient_name, False):
                        continue
                    pack_price_idr = float(ingredients_config[ingredient_name]["pack_price_idr"])
                    finance_manager.pay(pack_price_idr, category="grocery")
                    pantry_manager.add_grocery_pack(ingredient_name)
                    purchased_items.append(ingredient_name)

                biometrics_state.energy -= float(shopping_config["energy_cost"])
                biometrics_state.mood += float(shopping_config["mood_delta"])
                if shopping_transport_cost > 0.0:
                    finance_manager.pay(shopping_transport_cost, category="transport")
                step_events["purchased_groceries"] = purchased_items
                return False, spec.id
        return _apply_default_wait_or_rest(
            current_node, is_night, biometrics_state, biometrics_config, activity_specs
        )

    return _apply_default_wait_or_rest(
        current_node, is_night, biometrics_state, biometrics_config, activity_specs
    )


def build_action_masks(
    current_step: int,
    current_node_id: str,
    is_raining: bool,
    steps_per_day: int,
    world_graph: WorldGraph,
    activity_specs: dict[str, ActivitySpecConfig],
    activity_order: list[str],
    route_preferences: list[RoutePreference],
    external_foods: dict[str, ExternalFoodSpecConfig],
    academics: AcademicSchedule,
    finance: FinanceManager,
    biometrics: BiometricsState,
    pantry: PantryManager,
    config: dict,
) -> np.ndarray:
    """Menghasilkan boolean action mask dinamis dari WorldGraph & ActivitySpec."""
    _, _, step_in_day, day_name = academics.get_time_context(current_step)
    step_def = academics.get_step_definition(step_in_day)
    current_climate = str(step_def["climate"])
    current_ojol_tariff = str(step_def["ojol_tariff"])
    is_class_now = academics.is_class_scheduled(day_name, step_in_day)
    is_morning_step = step_in_day == 0
    is_night = step_in_day == (steps_per_day - 1)

    current_node = world_graph.get_node(current_node_id)
    pantry_config = config["pantry_and_cooking"]
    cooking_gates = pantry_config["cooking_costs_and_gates"]
    biometrics_config = config["biometrics_and_psychology"]

    can_eat = biometrics.hunger >= float(biometrics_config["step_dynamics"]["min_hunger_to_eat"])
    is_mood_burned_out = biometrics.is_mood_burned_out()
    is_stress_burned_out = biometrics.is_stress_burned_out()

    # 1. Activity mask (dinamis sesuai jumlah aktivitas)
    cheapest_bundle_price = min(
        pantry.get_grocery_bundle_cost_idr(i) for i in range(len(pantry.grocery_bundles))
    )
    activity_mask: list[bool] = []
    for act_id in activity_order:
        spec = activity_specs[act_id]
        handler = spec.handler
        targets = world_graph.resolve_activity_target_nodes(spec)
        if is_night:
            targets = [nid for nid in targets if not world_graph.get_node(nid).closes_at_night]

        _, cheapest_path = world_graph.choose_nearest_target_node(
            source_node_id=current_node_id,
            candidate_node_ids=targets,
            preference="cheapest",
            climate=current_climate,
            ojol_tariff=current_ojol_tariff,
            is_raining=is_raining,
        )
        can_reach = cheapest_path is not None and (
            is_night or finance.can_afford(cheapest_path.min_fare_idr)
        )
        travel_min_fare = cheapest_path.min_fare_idr if cheapest_path is not None else float("inf")

        if handler == "rest":
            allowed = can_reach
        elif handler == "attend_schedule":
            allowed = is_class_now and (not is_night) and (not is_mood_burned_out) and can_reach
        elif handler == "study":
            target_ok = any(
                biometrics.academic_stress <= float(world_graph.get_node(nid).study_profile.max_stress_gate)
                for nid in targets
            )
            allowed = (not is_mood_burned_out) and (not is_stress_burned_out) and target_ok and can_reach
        elif handler == "hub_stay":
            allowed = (not is_night) and can_reach
        elif handler == "leisure_reset":
            min_leisure = min(
                (float(world_graph.get_node(nid).leisure_spend_idr[0]) for nid in targets if world_graph.get_node(nid).leisure_spend_idr),
                default=0.0,
            )
            allowed = (not is_night) and can_reach and finance.can_afford(travel_min_fare + min_leisure)
        elif handler == "wait":
            allowed = (not is_night) and (current_node_id in targets)
        elif handler == "grocery_shop":
            allowed = (
                (not is_mood_burned_out)
                and can_reach
                and finance.can_afford(travel_min_fare + cheapest_bundle_price)
            )
        elif handler == "night_cram":
            allowed = is_night and (not is_mood_burned_out) and (not is_stress_burned_out) and can_reach
        else:
            allowed = can_reach

        activity_mask.append(bool(allowed))

    if not any(activity_mask):
        activity_mask[0] = True

    # 2. Route preference mask (fastest, cheapest)
    route_mask = [True for _ in route_preferences]

    # 3. Cook combo mask (dinamis mengikuti bahan standar + custom)
    cook_combo_mask = [
        pantry.has_enough_ingredients(pantry.decode_cook_combo_dict(i), batch_multiplier=1)
        for i in range(len(pantry.cook_combos))
    ]
    has_any_cook_combo = any(cook_combo_mask)
    min_batch_size = int(cooking_gates["meal_prep_min_batch_size"])
    has_any_prep_combo = any(
        pantry.max_portions_possible(pantry.decode_cook_combo_dict(i)) >= min_batch_size
        for i in range(len(pantry.cook_combos))
    )
    if not has_any_cook_combo:
        cook_combo_mask[0] = True

    # 4. Meal mode mask (6)
    _, cook_fresh_min_energy = _get_cooking_energy_requirements(
        cooking_gates, "cook_fresh", is_morning_step
    )
    _, meal_prep_min_energy = _get_cooking_energy_requirements(
        cooking_gates, "meal_prep", is_morning_step
    )
    can_cook_location = current_node.has_kitchen or is_night
    has_packed_lunch = pantry.get_packed_lunch_total_portions() >= 1
    has_meal_prep = pantry.get_meal_prep_total_portions() >= 1

    can_reheat_home = can_cook_location and (
        has_packed_lunch
        or (
            has_meal_prep
            and biometrics.energy >= float(cooking_gates["reheat_prep_min_energy"])
            and biometrics.mood >= float(cooking_gates["reheat_prep_min_mood"])
        )
    )
    can_eat_bekal_outside = (not can_cook_location) and has_packed_lunch

    food_node = world_graph.get_node(world_graph.home_node_id) if is_night else current_node
    budget_food = select_food_for_node(food_node, "budget", external_foods)
    varied_food = select_food_for_node(food_node, "varied", external_foods)

    meal_mode_mask = [
        True,
        (
            can_eat
            and can_cook_location
            and (not is_stress_burned_out)
            and has_any_cook_combo
            and biometrics.energy >= cook_fresh_min_energy
            and biometrics.mood >= float(cooking_gates["cook_fresh_min_mood"])
        ),
        (
            can_cook_location
            and (not is_stress_burned_out)
            and has_any_prep_combo
            and biometrics.energy >= meal_prep_min_energy
            and biometrics.mood >= float(cooking_gates["meal_prep_min_mood"])
        ),
        can_eat and (can_reheat_home or can_eat_bekal_outside),
        can_eat and (budget_food is not None) and finance.can_afford(float(budget_food.cost_idr)),
        can_eat and (varied_food is not None) and finance.can_afford(float(varied_food.cost_idr)),
    ]

    # 5. Bring bekal mask (2)
    max_packed = int(cooking_gates["max_packed_lunch_portions"])
    can_bring_bekal = (
        current_node.has_kitchen
        and (not is_night)
        and has_meal_prep
        and pantry.get_packed_lunch_total_portions() < max_packed
        and biometrics.energy >= float(cooking_gates["pack_bekal_min_energy"])
        and biometrics.mood >= float(cooking_gates["pack_bekal_min_mood"])
    )
    bring_bekal_mask = [True, can_bring_bekal]

    # 6. Grocery bundle mask (15)
    grocery_nodes = [n.id for n in world_graph.nodes.values() if n.has_grocery]
    _, grocery_path = world_graph.choose_nearest_target_node(
        source_node_id=current_node_id,
        candidate_node_ids=grocery_nodes,
        preference="cheapest",
        climate=current_climate,
        ojol_tariff=current_ojol_tariff,
        is_raining=is_raining,
    )
    grocery_commute_cost = grocery_path.min_fare_idr if grocery_path is not None else float("inf")
    grocery_bundle_mask = [
        finance.can_afford(pantry.get_grocery_bundle_cost_idr(i) + grocery_commute_cost)
        for i in range(len(pantry.grocery_bundles))
    ]
    if not any(grocery_bundle_mask):
        grocery_bundle_mask[0] = True

    return np.array(
        activity_mask
        + route_mask
        + meal_mode_mask
        + cook_combo_mask
        + bring_bekal_mask
        + grocery_bundle_mask,
        dtype=bool,
    )
