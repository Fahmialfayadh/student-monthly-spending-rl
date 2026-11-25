import numpy as np

from src.env.academics import AcademicSchedule
from src.env.biometrics import BiometricsState
from src.env.finance import FinanceManager
from src.env.mobility import sample_idr_from_range, try_minimarket_impulse
from src.env.pantry import PantryManager


def parse_action_vector(action, action_dimensions: dict[str, list]) -> dict[str, object]:
    """Mengubah vektor integer MultiDiscrete (11 elemen) menjadi dict nama aksi yang jelas."""
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
    ) = [int(value) for value in action]

    return {
        "meal_mode": action_dimensions["meal_mode"][meal_mode_index],
        "cook_quantities": {
            "rice": int(action_dimensions["cook_rice"][cook_rice_index]),
            "egg": int(action_dimensions["cook_egg"][cook_egg_index]),
            "chicken": int(action_dimensions["cook_chicken"][cook_chicken_index]),
            "tempe": int(action_dimensions["cook_tempe"][cook_tempe_index]),
        },
        "mobility": action_dimensions["mobility"][mobility_index],
        "activity": action_dimensions["activity"][activity_index],
        "grocery_flags": {
            "rice": bool(action_dimensions["buy_rice"][buy_rice_index]),
            "egg": bool(action_dimensions["buy_egg"][buy_egg_index]),
            "chicken": bool(action_dimensions["buy_chicken"][buy_chicken_index]),
            "tempe": bool(action_dimensions["buy_tempe"][buy_tempe_index]),
        },
    }


def _get_cooking_energy_requirements(
    cooking_gates: dict, mode_name: str, is_morning_step: bool
) -> tuple[float, float]:
    """
    Mengembalikan (energy_cost, min_energy) untuk cook_fresh atau meal_prep.
    Pada step pagi_awal (step_in_day == 0), memasak menguras tenaga jauh lebih besar.
    """
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


def execute_meal_action(
    requested_meal_mode: str,
    requested_cook_quantities: dict[str, int],
    current_location: str,
    step_in_day: int,
    is_stress_burned_out: bool,
    finance_manager: FinanceManager,
    biometrics_state: BiometricsState,
    pantry_manager: PantryManager,
    pantry_config: dict,
    biometrics_config: dict,
    step_events: dict[str, object],
) -> str:
    """
    Menjalankan fase makan & dapur:
    - cook_fresh / meal_prep (dengan penalti energi ekstra di pagi_awal)
    - pack_bekal (membungkus 1 porsi meal_prep ke tas sebelum berangkat ke kampus)
    - reheat_or_eat_bekal (reheat di apartemen, atau makan bekal saat di kampus)
    - warung_hemat / warung_variasi / craving_snack
    - skip
    """
    cooking_gates = pantry_config["cooking_costs_and_gates"]
    external_food = biometrics_config["external_food_recovery"]
    step_dynamics = biometrics_config["step_dynamics"]

    is_at_apartment = current_location == "apartment"
    is_morning_step = step_in_day == 0
    executed_mode = requested_meal_mode

    if executed_mode == "cook_fresh":
        energy_cost, min_energy = _get_cooking_energy_requirements(
            cooking_gates, "cook_fresh", is_morning_step
        )
        total_units = sum(requested_cook_quantities.values())
        if (
            is_at_apartment
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
        batch_size = int(cooking_gates["meal_prep_batch_size"])
        total_units = sum(requested_cook_quantities.values())
        if (
            is_at_apartment
            and not is_stress_burned_out
            and total_units > 0
            and pantry_manager.has_enough_ingredients(
                requested_cook_quantities, batch_multiplier=batch_size
            )
            and biometrics_state.energy >= min_energy
            and biometrics_state.mood >= float(cooking_gates["meal_prep_min_mood"])
        ):
            biometrics_state.energy -= energy_cost
            pantry_manager.consume_ingredients_fifo(
                requested_cook_quantities, batch_multiplier=batch_size
            )
            hunger_delta, energy_gain, mood_delta, meal_signature = (
                pantry_manager.calculate_custom_meal_profile(requested_cook_quantities)
            )
            pantry_manager.store_meal_prep_batch(
                hunger_delta, energy_gain, mood_delta, meal_signature
            )
            step_events["meal_prep_created_signature"] = meal_signature
            if is_morning_step:
                step_events["morning_heavy_cooking"] = True
            return "meal_prep"
        executed_mode = "skip"

    elif executed_mode == "pack_bekal":
        max_packed = int(cooking_gates["max_packed_lunch_portions"])
        if (
            is_at_apartment
            and pantry_manager.get_meal_prep_total_portions() >= 1
            and pantry_manager.get_packed_lunch_total_portions() < max_packed
            and biometrics_state.energy >= float(cooking_gates["pack_bekal_min_energy"])
            and biometrics_state.mood >= float(cooking_gates["pack_bekal_min_mood"])
        ):
            biometrics_state.energy -= float(cooking_gates["pack_bekal_energy_cost"])
            packed_portion = pantry_manager.pack_one_bekal_from_prep()
            assert packed_portion is not None
            step_events["packed_bekal_signature"] = str(packed_portion["meal_signature"])
            return "pack_bekal"
        executed_mode = "skip"

    elif executed_mode == "reheat_or_eat_bekal":
        reheat_mood_delta = float(pantry_config["meal_composition_rules"]["reheat_mood_delta"])
        if is_at_apartment:
            if (
                pantry_manager.get_meal_prep_total_portions() >= 1
                and biometrics_state.energy >= float(cooking_gates["reheat_prep_min_energy"])
                and biometrics_state.mood >= float(cooking_gates["reheat_prep_min_mood"])
            ):
                biometrics_state.energy -= float(cooking_gates["reheat_prep_energy_cost"])
                reheated_portion = pantry_manager.pop_one_meal_prep_portion_fifo()
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
                bekal_portion = pantry_manager.pop_one_packed_lunch_portion()
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
                step_events["ate_bekal_at_campus"] = True
                return "eat_bekal"
        executed_mode = "skip"

    elif executed_mode in ("warung_hemat", "warung_variasi", "craving_snack"):
        config_key = (
            "craving_food_or_snack" if executed_mode == "craving_snack" else executed_mode
        )
        food_profile = external_food[config_key]
        if finance_manager.pay(float(food_profile["cost_idr"]), category="external_food"):
            _apply_eaten_meal(
                biometrics_state,
                pantry_manager,
                hunger_delta=float(food_profile["hunger"]),
                energy_gain=float(food_profile["energy"]),
                mood_delta=float(food_profile["mood"]),
                meal_signature=executed_mode,
                step_events=step_events,
            )
            return executed_mode
        executed_mode = "skip"

    if executed_mode == "skip" and biometrics_state.hunger >= 50.0:
        biometrics_state.mood += float(step_dynamics["skip_meal_mood_penalty"])
        biometrics_state.energy += float(step_dynamics["skip_meal_energy_penalty"])
        step_events["skipped_meal_while_hungry"] = True

    return "skip"


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


def execute_activity_action(
    effective_activity: str,
    current_location: str,
    waited_until_evening: bool,
    is_class_scheduled_now: bool,
    is_stress_burned_out: bool,
    step_in_day: int,
    steps_per_day: int,
    requested_grocery_flags: dict[str, bool],
    random_generator: np.random.Generator,
    finance_manager: FinanceManager,
    biometrics_state: BiometricsState,
    pantry_manager: PantryManager,
    academics_config: dict,
    mobility_config: dict,
    biometrics_config: dict,
    pantry_config: dict,
    step_events: dict[str, object],
) -> tuple[bool, str, bool]:
    """
    Menjalankan aktivitas utama:
    attend_class, room_study, ccws_stay, cafe_session (termasuk ongkos transport PP),
    wait_at_department, grocery_shop, night_cramming (begadang eksplisit), atau rest_or_sleep.
    """
    study_efficiency = biometrics_state.get_study_efficiency()
    is_night = step_in_day == (steps_per_day - 1)

    if effective_activity == "attend_class":
        if is_class_scheduled_now and current_location == "campus":
            biometrics_state.energy -= float(academics_config["class_attendance_energy_cost"])
            return True, "attend_class", waited_until_evening
        effective_activity = "rest_or_sleep"

    if effective_activity == "room_study":
        max_stress_gate = float(academics_config["room_study_max_stress_gate"])
        if (
            current_location == "apartment"
            and not is_stress_burned_out
            and biometrics_state.academic_stress <= max_stress_gate
        ):
            biometrics_state.energy -= float(academics_config["room_study_energy_cost"])
            biometrics_state.academic_stress -= (
                float(academics_config["room_study_stress_reduction"]) * study_efficiency
            )
            return False, "room_study", waited_until_evening
        effective_activity = "rest_or_sleep"

    if effective_activity == "night_cramming":
        if current_location == "apartment" and is_night and not is_stress_burned_out:
            biometrics_state.energy -= float(academics_config["night_cramming_energy_cost"])
            biometrics_state.academic_stress -= (
                float(academics_config["night_cramming_stress_reduction"]) * study_efficiency
            )
            return False, "night_cramming", waited_until_evening
        effective_activity = "rest_or_sleep"

    if effective_activity == "ccws_stay":
        if current_location == "campus" and not is_night:
            ccws_config = mobility_config["campus_hub_rules"]["ccws_stay"]
            biometrics_state.energy -= float(ccws_config["energy_cost"])
            biometrics_state.academic_stress -= (
                float(ccws_config["stress_reduction_per_step"]) * study_efficiency
            )
            bought_impulse, impulse_hunger, impulse_mood = try_minimarket_impulse(
                random_generator=random_generator,
                finance_manager=finance_manager,
                mobility_config=mobility_config,
                temptation_probability=float(ccws_config["minimarket_temptation_prob"]),
            )
            biometrics_state.hunger += impulse_hunger
            biometrics_state.mood += impulse_mood
            step_events["ccws_minimarket_impulse"] = bought_impulse
            return False, "ccws_stay", waited_until_evening
        effective_activity = "rest_or_sleep"

    if effective_activity == "cafe_session":
        cafe_config = biometrics_config["craving_and_cafe"]["cafe_session"]
        cafe_spend_idr = sample_idr_from_range(
            random_generator,
            [int(cafe_config["cost_min_idr"]), int(cafe_config["cost_max_idr"])],
        )
        transport_range_key = (
            "transport_from_apartment_idr"
            if current_location == "apartment"
            else "transport_from_campus_idr"
        )
        cafe_transport_idr = sample_idr_from_range(
            random_generator, cafe_config[transport_range_key]
        )
        total_cafe_cost_idr = cafe_spend_idr + cafe_transport_idr

        if finance_manager.can_afford(total_cafe_cost_idr):
            finance_manager.pay(cafe_spend_idr, category="cafe")
            finance_manager.pay(cafe_transport_idr, category="transport")
            biometrics_state.energy -= float(cafe_config["energy_cost"])
            biometrics_state.hunger += float(cafe_config.get("hunger_delta", 0.0))
            biometrics_state.mood += float(cafe_config["mood_gain"])
            biometrics_state.academic_stress -= float(cafe_config["stress_reduction"])
            biometrics_state.cafe_craving = float(cafe_config["craving_reset_to"])
            biometrics_state.cafe_buff_remaining_steps = int(cafe_config["buff_duration_steps"])
            step_events["cafe_spend_idr"] = cafe_spend_idr
            step_events["cafe_transport_idr"] = cafe_transport_idr
            return False, "cafe_session", waited_until_evening
        effective_activity = "rest_or_sleep"

    if effective_activity == "wait_at_department":
        if current_location == "campus" and not is_night:
            wait_config = mobility_config["campus_hub_rules"]["wait_at_department_until_evening"]
            biometrics_state.energy -= float(wait_config["energy_cost"])
            biometrics_state.mood += float(wait_config["mood_delta_per_step"])
            return False, "wait_at_department", True
        effective_activity = "rest_or_sleep"

    if effective_activity == "grocery_shop":
        if current_location == "apartment":
            ingredients_config = pantry_config["ingredients"]
            purchased_items: list[str] = []
            for ingredient_name in PantryManager.INGREDIENT_NAMES:
                if not requested_grocery_flags[ingredient_name]:
                    continue
                pack_price_idr = float(ingredients_config[ingredient_name]["pack_price_idr"])
                if finance_manager.pay(pack_price_idr, category="grocery"):
                    pantry_manager.add_grocery_pack(ingredient_name)
                    purchased_items.append(ingredient_name)

            if purchased_items:
                shopping_config = pantry_config["grocery_shopping"]
                biometrics_state.energy -= float(shopping_config["energy_cost"])
                biometrics_state.mood += float(shopping_config["mood_delta"])
                shopping_transport_cost = float(shopping_config["transport_cost_idr"])
                if shopping_transport_cost > 0.0:
                    finance_manager.pay(shopping_transport_cost, category="transport")
                step_events["purchased_groceries"] = purchased_items
                return False, "grocery_shop", waited_until_evening
        effective_activity = "rest_or_sleep"

    # Default: rest_or_sleep (di siang/pagi/sore memberi istirahat; di malam tidur otomatis tetap berjalan)
    step_dynamics = biometrics_config["step_dynamics"]
    biometrics_state.energy += float(step_dynamics["rest_step_energy_gain"])
    biometrics_state.mood += float(step_dynamics["rest_step_mood_gain"])
    return False, "rest_or_sleep", waited_until_evening


def build_action_masks(
    current_step: int,
    current_location: str,
    steps_per_day: int,
    action_dimensions: dict[str, list],
    academics: AcademicSchedule,
    finance: FinanceManager,
    biometrics: BiometricsState,
    pantry: PantryManager,
    config: dict,
) -> np.ndarray:
    """Menghasilkan boolean mask datar (panjang 38) untuk MaskablePPO."""
    _, _, step_in_day, day_name = academics.get_time_context(current_step)
    is_class_now = academics.is_class_scheduled(day_name, step_in_day)
    is_at_apartment = current_location == "apartment"
    is_morning_step = step_in_day == 0
    is_night = step_in_day == (steps_per_day - 1)

    pantry_config = config["pantry_and_cooking"]
    cooking_gates = pantry_config["cooking_costs_and_gates"]
    ingredients_config = pantry_config["ingredients"]
    biometrics_config = config["biometrics_and_psychology"]
    external_food = biometrics_config["external_food_recovery"]
    cafe_config = biometrics_config["craving_and_cafe"]["cafe_session"]
    mobility_config = config["mobility_and_surabaya_climate"]
    academics_config = config["academics"]

    is_mood_burned_out = biometrics.is_mood_burned_out()
    is_stress_burned_out = biometrics.is_stress_burned_out()

    _, cook_fresh_min_energy = _get_cooking_energy_requirements(
        cooking_gates, "cook_fresh", is_morning_step
    )
    _, meal_prep_min_energy = _get_cooking_energy_requirements(
        cooking_gates, "meal_prep", is_morning_step
    )
    batch_size = int(cooking_gates["meal_prep_batch_size"])
    has_enough_for_prep = any(
        pantry.get_ingredient_total_units(name) >= batch_size
        for name in PantryManager.INGREDIENT_NAMES
    )
    max_packed = int(cooking_gates["max_packed_lunch_portions"])

    can_reheat_home = (
        is_at_apartment
        and pantry.get_meal_prep_total_portions() >= 1
        and biometrics.energy >= float(cooking_gates["reheat_prep_min_energy"])
        and biometrics.mood >= float(cooking_gates["reheat_prep_min_mood"])
    )
    can_eat_bekal_campus = (not is_at_apartment) and pantry.get_packed_lunch_total_portions() >= 1

    meal_mode_mask = [
        True,  # skip
        (
            is_at_apartment
            and not is_stress_burned_out
            and pantry.get_total_raw_units() > 0
            and biometrics.energy >= cook_fresh_min_energy
            and biometrics.mood >= float(cooking_gates["cook_fresh_min_mood"])
        ),  # cook_fresh
        (
            is_at_apartment
            and not is_stress_burned_out
            and has_enough_for_prep
            and biometrics.energy >= meal_prep_min_energy
            and biometrics.mood >= float(cooking_gates["meal_prep_min_mood"])
        ),  # meal_prep
        can_reheat_home or can_eat_bekal_campus,  # reheat_or_eat_bekal
        (
            is_at_apartment
            and not is_night
            and pantry.get_meal_prep_total_portions() >= 1
            and pantry.get_packed_lunch_total_portions() < max_packed
            and biometrics.energy >= float(cooking_gates["pack_bekal_min_energy"])
            and biometrics.mood >= float(cooking_gates["pack_bekal_min_mood"])
        ),  # pack_bekal
        finance.can_afford(float(external_food["warung_hemat"]["cost_idr"])),
        finance.can_afford(float(external_food["warung_variasi"]["cost_idr"])),
        finance.can_afford(float(external_food["craving_food_or_snack"]["cost_idr"])),
    ]

    cook_masks: list[bool] = []
    for ingredient_name in PantryManager.INGREDIENT_NAMES:
        stock_units = pantry.get_ingredient_total_units(ingredient_name)
        cook_masks.extend(qty <= stock_units for qty in action_dimensions[f"cook_{ingredient_name}"])

    min_direct_fare = float(mobility_config["routes"]["direct_ojol"]["cost_normal_idr"][0])
    min_transit_fare = float(mobility_config["routes"]["transit_ccws_ojol"]["cost_normal_idr"][0])
    mobility_mask = [
        True,
        finance.can_afford(min_direct_fare),
        finance.can_afford(min_transit_fare),
    ]

    cafe_transport_key = (
        "transport_from_apartment_idr" if is_at_apartment else "transport_from_campus_idr"
    )
    min_cafe_total = float(cafe_config["cost_min_idr"]) + float(cafe_config[cafe_transport_key][0])
    cheapest_pack_price = min(
        float(ingredients_config[name]["pack_price_idr"]) for name in PantryManager.INGREDIENT_NAMES
    )
    can_go_to_campus = (not is_at_apartment) or finance.can_afford(min_transit_fare)

    activity_mask = [
        True,  # rest_or_sleep
        is_class_now and (not is_night) and (not is_mood_burned_out) and can_go_to_campus,  # attend_class
        (
            (not is_mood_burned_out)
            and (not is_stress_burned_out)
            and biometrics.academic_stress <= float(academics_config["room_study_max_stress_gate"])
        ),  # room_study
        (not is_night) and can_go_to_campus,  # ccws_stay
        finance.can_afford(min_cafe_total),  # cafe_session
        (not is_night) and (not is_at_apartment),  # wait_at_department
        (not is_mood_burned_out) and finance.can_afford(cheapest_pack_price),  # grocery_shop
        is_night and (not is_mood_burned_out) and (not is_stress_burned_out),  # night_cramming
    ]

    buy_masks: list[bool] = []
    for ingredient_name in PantryManager.INGREDIENT_NAMES:
        pack_price = float(ingredients_config[ingredient_name]["pack_price_idr"])
        buy_masks.extend([True, finance.can_afford(pack_price)])

    return np.array(
        meal_mode_mask + cook_masks + mobility_mask + activity_mask + buy_masks,
        dtype=bool,
    )
