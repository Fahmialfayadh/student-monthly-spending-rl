import numpy as np

from src.env.academics import AcademicSchedule
from src.env.biometrics import BiometricsState
from src.env.finance import FinanceManager
from src.env.mobility import (
    get_route_fare_bounds,
    sample_idr_from_range,
    try_minimarket_impulse,
)
from src.env.pantry import PantryManager


def parse_action_vector(action, action_dimensions: dict[str, list]) -> dict[str, object]:
    """Mengubah vektor integer MultiDiscrete (6 elemen) menjadi dict nama aksi yang jelas."""
    (
        activity_index,
        route_index,
        meal_mode_index,
        cook_combo_index,
        bring_bekal_index,
        grocery_bundle_index,
    ) = [int(value) for value in action]

    return {
        "activity": action_dimensions["activity"][activity_index],
        "route": action_dimensions["route"][route_index],
        "meal_mode": action_dimensions["meal_mode"][meal_mode_index],
        "cook_combo_index": cook_combo_index,
        "cook_quantities": PantryManager.decode_cook_combo(cook_combo_index),
        "bring_bekal": bool(action_dimensions["bring_bekal"][bring_bekal_index]),
        "grocery_bundle_index": grocery_bundle_index,
        "grocery_flags": PantryManager.decode_grocery_bundle(grocery_bundle_index),
    }


def _get_cooking_energy_requirements(
    cooking_gates: dict, mode_name: str, is_morning_step: bool
) -> tuple[float, float]:
    """
    Mengembalikan (energy_cost, min_energy) untuk cook_fresh atau meal_prep.
    Pada step pagi_awal (step_in_day == 0), memasak menguras tenaga lebih besar.
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


def try_pack_bekal_before_departure(
    bring_bekal: bool,
    biometrics_state: BiometricsState,
    pantry_manager: PantryManager,
    pantry_config: dict,
    step_events: dict[str, object],
) -> bool:
    """
    Membungkus 1 porsi meal_prep dari kulkas ke kotak bekal sebelum berangkat ke kampus
    tanpa mengorbankan slot sarapan/makan pada step yang sama.
    """
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
    Menjalankan fase makan & dapur sesuai lokasi aktual (Apartemen vs Kampus):
    - cook_fresh / meal_prep (kompor + rice cooker; meal_prep mendukung batch parsial >= 2
      dan langsung makan 1 porsi segar bila sedang lapar)
    - reheat_or_eat_bekal (Smart-FIFO; bekal yang terbawa pulang tetap bisa dimakan di apartemen)
    - budget_external_meal (Apartemen: apartemen_warung Rp 15k 1 opsi; Kampus: warung_hemat Rp 13k)
    - varied_external_meal (Apartemen: delivery_food Rp 25k; Kampus: warung_variasi Rp 18k)
    - skip
    """
    cooking_gates = pantry_config["cooking_costs_and_gates"]
    external_food = biometrics_config["external_food_recovery"]
    step_dynamics = biometrics_config["step_dynamics"]

    is_at_apartment = current_location == "apartment"
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
        max_batch_size = int(cooking_gates["meal_prep_batch_size"])
        min_batch_size = int(cooking_gates["meal_prep_min_batch_size"])
        max_possible = pantry_manager.max_portions_possible(requested_cook_quantities)
        actual_batch = min(max_batch_size, max_possible)

        if (
            is_at_apartment
            and not is_stress_burned_out
            and actual_batch >= min_batch_size
            and biometrics_state.energy >= min_energy
            and biometrics_state.mood >= float(cooking_gates["meal_prep_min_mood"])
        ):
            # Skala ringan biaya energi bila batch parsial
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

            # Jika sedang lapar saat selesai meal prep, 1 porsi langsung dimakan segar
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

        # Fallback natural: jika bahan hanya cukup 1 porsi dan sedang lapar, masak segar 1 porsi
        if (
            is_at_apartment
            and not is_stress_burned_out
            and can_eat
            and max_possible >= 1
        ):
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
        if is_at_apartment:
            # Bekal sisa yang terbawa pulang tetap bisa dimakan di apartemen sebelum basi
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
                step_events["ate_bekal_at_campus"] = True
                return "eat_bekal"
        executed_mode = "skip"

    elif executed_mode in ("budget_external_meal", "varied_external_meal"):
        if is_at_apartment:
            food_key = (
                "apartemen_warung" if executed_mode == "budget_external_meal" else "delivery_food"
            )
        else:
            food_key = (
                "warung_hemat" if executed_mode == "budget_external_meal" else "warung_variasi"
            )

        food_profile = external_food[food_key]
        if finance_manager.pay(float(food_profile["cost_idr"]), category="external_food"):
            biometrics_state.energy -= float(food_profile.get("energy_cost", 0.0))
            _apply_eaten_meal(
                biometrics_state,
                pantry_manager,
                hunger_delta=float(food_profile["hunger"]),
                energy_gain=float(food_profile["energy"]),
                mood_delta=float(food_profile["mood"]),
                meal_signature=food_key,
                step_events=step_events,
            )
            return food_key
        executed_mode = "skip"

    skip_threshold = float(step_dynamics["skip_meal_hunger_threshold"])
    if executed_mode == "skip" and biometrics_state.hunger >= skip_threshold:
        tolerance = float(step_dynamics["skip_meal_tolerance"])
        mood_scale = max(0.3, 1.3 - 0.5 * tolerance)
        biometrics_state.mood += float(step_dynamics["skip_meal_mood_penalty"]) * mood_scale
        biometrics_state.energy += float(step_dynamics["skip_meal_energy_penalty"])
        step_events["skipped_meal_while_hungry"] = True

    return "skip"


def execute_activity_action(
    effective_activity: str,
    current_location: str,
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
    mobility_config: dict,
    biometrics_config: dict,
    pantry_config: dict,
    step_events: dict[str, object],
) -> tuple[bool, str, str]:
    """
    Menjalankan aktivitas utama sesuai lokasi dan mengembalikan
    (attended_class, executed_activity, updated_location).
    """
    room_study_efficiency = biometrics_state.get_room_study_efficiency()
    is_night = step_in_day == (steps_per_day - 1)
    effective_climate = "rain" if is_raining else current_climate
    updated_location = current_location

    if effective_activity == "attend_class":
        if is_class_scheduled_now and current_location == "campus":
            biometrics_state.energy -= float(academics_config["class_attendance_energy_cost"])
            return True, "attend_class", updated_location
        effective_activity = "wait_at_department" if current_location == "campus" else "rest_or_sleep"

    if effective_activity == "room_study":
        max_stress_gate = float(academics_config["room_study_max_stress_gate"])
        if (
            current_location == "apartment"
            and not is_stress_burned_out
            and biometrics_state.academic_stress <= max_stress_gate
        ):
            biometrics_state.energy -= float(academics_config["room_study_energy_cost"])
            biometrics_state.academic_stress -= (
                float(academics_config["room_study_stress_reduction"]) * room_study_efficiency
            )
            return False, "room_study", updated_location
        effective_activity = "wait_at_department" if current_location == "campus" else "rest_or_sleep"

    if effective_activity == "night_cramming":
        if current_location == "apartment" and is_night and not is_stress_burned_out:
            biometrics_state.energy -= float(academics_config["night_cramming_energy_cost"])
            biometrics_state.academic_stress -= (
                float(academics_config["night_cramming_stress_reduction"]) * room_study_efficiency
            )
            return False, "night_cramming", updated_location
        effective_activity = "rest_or_sleep"

    if effective_activity == "ccws_stay":
        if current_location == "campus" and not is_night:
            ccws_config = mobility_config["campus_hub_rules"]["ccws_stay"]
            biometrics_state.energy -= float(ccws_config["energy_cost"])
            biometrics_state.academic_stress -= float(ccws_config["stress_reduction_per_step"])

            # CCWS meredakan sebagian craving dengan batas bawah (floor) & diminishing returns harian
            craving_floor = float(ccws_config["craving_floor"])
            if biometrics_state.ccws_visits_today == 0:
                craving_reduction = float(ccws_config["craving_reduction_first_visit"])
                biometrics_state.mood += float(ccws_config.get("mood_gain_first_visit", 0.0))
            else:
                craving_reduction = float(ccws_config["craving_reduction_repeat_visit"])
            if biometrics_state.cafe_craving > craving_floor:
                biometrics_state.cafe_craving = max(
                    craving_floor, biometrics_state.cafe_craving - craving_reduction
                )
            biometrics_state.ccws_visits_today += 1

            biometrics_state.apply_craving_buff(
                duration_steps=int(ccws_config["buff_duration_steps"]),
                rate_multiplier=float(ccws_config["buff_craving_rate_multiplier"]),
            )
            bought_impulse, impulse_hunger, impulse_mood = try_minimarket_impulse(
                random_generator=random_generator,
                finance_manager=finance_manager,
                biometrics_state=biometrics_state,
                finance_config=finance_config,
                mobility_config=mobility_config,
                base_temptation_probability=float(ccws_config["minimarket_base_temptation_prob"]),
                effective_climate=effective_climate,
                current_day=current_day,
            )
            biometrics_state.hunger += impulse_hunger
            biometrics_state.mood += impulse_mood
            step_events["ccws_minimarket_impulse"] = bought_impulse
            return False, "ccws_stay", updated_location
        effective_activity = "rest_or_sleep"

    if effective_activity == "cafe_session":
        cafe_config = biometrics_config["craving_and_cafe"]["cafe_session"]
        rain_surge = float(mobility_config["rain_ojol_surge_multiplier"]) if is_raining else 1.0
        # Jika dari kampus saat malam, pulang langsung ke apartemen setelah dari kafe (1x transport)
        if current_location == "apartment" or is_night:
            transport_range = cafe_config["transport_from_apartment_idr"]
            destination_after_cafe = "apartment"
        else:
            transport_range = cafe_config["transport_from_campus_idr"]
            destination_after_cafe = "campus"

        min_transport = float(round(int(transport_range[0]) * rain_surge))
        min_cafe_spend = float(cafe_config["cost_min_idr"])
        min_total_cost = min_cafe_spend + min_transport

        if finance_manager.can_afford(min_total_cost):
            sampled_transport = sample_idr_from_range(
                random_generator, transport_range, multiplier=rain_surge
            )
            cafe_transport_idr = min(
                sampled_transport, max(min_transport, finance_manager.balance_idr - min_cafe_spend)
            )
            sampled_spend = sample_idr_from_range(
                random_generator,
                [int(cafe_config["cost_min_idr"]), int(cafe_config["cost_max_idr"])],
            )
            cafe_spend_idr = min(
                sampled_spend, max(min_cafe_spend, finance_manager.balance_idr - cafe_transport_idr)
            )
            finance_manager.pay(cafe_spend_idr, category="cafe")
            finance_manager.pay(cafe_transport_idr, category="transport")
            biometrics_state.energy -= float(cafe_config["energy_cost"])
            biometrics_state.hunger += float(cafe_config.get("hunger_delta", 0.0))
            biometrics_state.mood += float(cafe_config["mood_gain"])
            biometrics_state.academic_stress -= float(cafe_config["stress_reduction"])
            biometrics_state.cafe_craving = float(cafe_config["craving_reset_to"])
            biometrics_state.apply_craving_buff(
                duration_steps=int(cafe_config["buff_duration_steps"]),
                rate_multiplier=float(cafe_config["buff_craving_rate_multiplier"]),
            )
            step_events["cafe_spend_idr"] = cafe_spend_idr
            step_events["cafe_transport_idr"] = cafe_transport_idr
            updated_location = destination_after_cafe
            return False, "cafe_session", updated_location

        effective_activity = "wait_at_department" if current_location == "campus" and not is_night else "rest_or_sleep"

    if effective_activity == "wait_at_department":
        if current_location == "campus" and not is_night:
            wait_config = mobility_config["campus_hub_rules"]["wait_at_department"]
            biometrics_state.energy -= float(wait_config["energy_cost"])
            biometrics_state.mood += float(wait_config["mood_delta_per_step"])
            return False, "wait_at_department", updated_location
        effective_activity = "rest_or_sleep"

    if effective_activity == "grocery_shop":
        if current_location == "apartment":
            ingredients_config = pantry_config["ingredients"]
            total_bundle_cost = sum(
                float(ingredients_config[name]["pack_price_idr"])
                for name in PantryManager.INGREDIENT_NAMES
                if requested_grocery_flags[name]
            )
            shopping_config = pantry_config["grocery_shopping"]
            shopping_transport_cost = float(shopping_config["transport_cost_idr"])

            if total_bundle_cost > 0.0 and finance_manager.can_afford(
                total_bundle_cost + shopping_transport_cost
            ):
                purchased_items: list[str] = []
                for ingredient_name in PantryManager.INGREDIENT_NAMES:
                    if not requested_grocery_flags[ingredient_name]:
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
                return False, "grocery_shop", updated_location
        effective_activity = "wait_at_department" if current_location == "campus" and not is_night else "rest_or_sleep"

    # Default aman: di kampus siang tidak boleh tidur -> wait_at_department
    if current_location == "campus" and not is_night:
        wait_config = mobility_config["campus_hub_rules"]["wait_at_department"]
        biometrics_state.energy -= float(wait_config["energy_cost"])
        biometrics_state.mood += float(wait_config["mood_delta_per_step"])
        return False, "wait_at_department", updated_location

    step_dynamics = biometrics_config["step_dynamics"]
    craving_cfg = biometrics_config["craving_and_cafe"]
    if is_night:
        # Tidak didobel dengan rest_step penuh karena tidur malam otomatis berjalan di akhir step
        biometrics_state.energy += float(step_dynamics["night_unwind_energy_gain"])
        biometrics_state.mood += float(step_dynamics["night_unwind_mood_gain"])
    else:
        biometrics_state.energy += float(step_dynamics["rest_step_energy_gain"])
        # Rebahan di kamar saat sumpek/craving tinggi memulihkan energi, bukan mood
        if biometrics_state.cafe_craving > float(craving_cfg["high_distraction_threshold"]):
            biometrics_state.mood += float(step_dynamics["rest_step_high_craving_mood_delta"])
        elif biometrics_state.cafe_craving <= float(craving_cfg["distraction_threshold"]):
            biometrics_state.mood += float(step_dynamics["rest_step_mood_gain"])
    return False, "rest_or_sleep", updated_location


def build_action_masks(
    current_step: int,
    current_location: str,
    is_raining: bool,
    steps_per_day: int,
    academics: AcademicSchedule,
    finance: FinanceManager,
    biometrics: BiometricsState,
    pantry: PantryManager,
    config: dict,
) -> np.ndarray:
    """
    Menghasilkan boolean mask datar (panjang 51) untuk 6 head terpadu MaskablePPO:
    activity (8) + route (2) + meal_mode (6) + cook_combo (18) + bring_bekal (2) + grocery_bundle (15).
    Setiap head dijamin memiliki minimal satu opsi True.
    """
    _, _, step_in_day, day_name = academics.get_time_context(current_step)
    step_def = academics.get_step_definition(step_in_day)
    current_ojol_tariff = str(step_def["ojol_tariff"])
    is_class_now = academics.is_class_scheduled(day_name, step_in_day)
    is_at_apartment = current_location == "apartment"
    is_morning_step = step_in_day == 0
    is_night = step_in_day == (steps_per_day - 1)

    pantry_config = config["pantry_and_cooking"]
    cooking_gates = pantry_config["cooking_costs_and_gates"]
    biometrics_config = config["biometrics_and_psychology"]
    external_food = biometrics_config["external_food_recovery"]
    cafe_config = biometrics_config["craving_and_cafe"]["cafe_session"]
    mobility_config = config["mobility_and_surabaya_climate"]
    academics_config = config["academics"]

    can_eat = biometrics.hunger >= float(biometrics_config["step_dynamics"]["min_hunger_to_eat"])
    is_mood_burned_out = biometrics.is_mood_burned_out()
    is_stress_burned_out = biometrics.is_stress_burned_out()

    min_direct_fare, _ = get_route_fare_bounds(
        mobility_config, "direct_ojol", current_ojol_tariff, is_raining
    )
    min_transit_fare, _ = get_route_fare_bounds(
        mobility_config, "transit_ccws_ojol", current_ojol_tariff, is_raining
    )
    can_commute_campus = (not is_at_apartment) or finance.can_afford(min_transit_fare)
    can_reach_apartment = is_at_apartment or is_night or finance.can_afford(min_transit_fare)

    # 1. Activity mask (8)
    rain_surge = float(mobility_config["rain_ojol_surge_multiplier"]) if is_raining else 1.0
    cafe_transport_key = (
        "transport_from_apartment_idr"
        if (is_at_apartment or is_night)
        else "transport_from_campus_idr"
    )
    min_cafe_total = float(cafe_config["cost_min_idr"]) + float(
        round(int(cafe_config[cafe_transport_key][0]) * rain_surge)
    )
    cheapest_bundle_price = min(
        pantry.get_grocery_bundle_cost_idr(i) for i in range(len(PantryManager.GROCERY_BUNDLES))
    )
    grocery_commute_cost = 0.0 if (is_at_apartment or is_night) else min_transit_fare

    activity_mask = [
        can_reach_apartment,  # 0: rest_or_sleep
        is_class_now and (not is_night) and (not is_mood_burned_out) and can_commute_campus,  # 1: attend_class
        (
            can_reach_apartment
            and (not is_mood_burned_out)
            and (not is_stress_burned_out)
            and biometrics.academic_stress <= float(academics_config["room_study_max_stress_gate"])
        ),  # 2: room_study
        (not is_night) and can_commute_campus,  # 3: ccws_stay (boleh juga akhir pekan)
        finance.can_afford(min_cafe_total),  # 4: cafe_session
        (not is_night) and (not is_at_apartment),  # 5: wait_at_department
        (
            can_reach_apartment
            and (not is_mood_burned_out)
            and finance.can_afford(cheapest_bundle_price + grocery_commute_cost)
        ),  # 6: grocery_shop
        is_night and (not is_mood_burned_out) and (not is_stress_burned_out),  # 7: night_cramming
    ]

    # 2. Route mask (2: direct_ojol, transit_ccws_ojol)
    can_direct = finance.can_afford(min_direct_fare)
    can_transit = finance.can_afford(min_transit_fare)
    route_mask = [can_direct, can_transit]
    if not any(route_mask):
        route_mask = [False, True]

    # 3. Cook combo mask (18)
    cook_combo_mask = [
        pantry.has_enough_ingredients(PantryManager.decode_cook_combo(i), batch_multiplier=1)
        for i in range(len(PantryManager.COOK_COMBOS))
    ]
    has_any_cook_combo = any(cook_combo_mask)
    min_batch_size = int(cooking_gates["meal_prep_min_batch_size"])
    has_any_prep_combo = any(
        pantry.max_portions_possible(PantryManager.decode_cook_combo(i)) >= min_batch_size
        for i in range(len(PantryManager.COOK_COMBOS))
    )
    if not has_any_cook_combo:
        cook_combo_mask[0] = True  # dummy aman (cook_fresh & meal_prep sudah False)

    # 4. Meal mode mask (6)
    _, cook_fresh_min_energy = _get_cooking_energy_requirements(
        cooking_gates, "cook_fresh", is_morning_step
    )
    _, meal_prep_min_energy = _get_cooking_energy_requirements(
        cooking_gates, "meal_prep", is_morning_step
    )
    can_cook_location = is_at_apartment or is_night
    has_packed_lunch = pantry.get_packed_lunch_total_portions() >= 1
    has_meal_prep = pantry.get_meal_prep_total_portions() >= 1

    can_reheat_home = (
        can_cook_location
        and (
            has_packed_lunch
            or (
                has_meal_prep
                and biometrics.energy >= float(cooking_gates["reheat_prep_min_energy"])
                and biometrics.mood >= float(cooking_gates["reheat_prep_min_mood"])
            )
        )
    )
    can_eat_bekal_campus = (not is_at_apartment and not is_night) and has_packed_lunch

    if is_at_apartment or is_night:
        budget_meal_price = float(external_food["apartemen_warung"]["cost_idr"])
        varied_meal_price = float(external_food["delivery_food"]["cost_idr"])
    else:
        budget_meal_price = float(external_food["warung_hemat"]["cost_idr"])
        varied_meal_price = float(external_food["warung_variasi"]["cost_idr"])

    meal_mode_mask = [
        True,  # 0: skip
        (
            can_eat
            and can_cook_location
            and (not is_stress_burned_out)
            and has_any_cook_combo
            and biometrics.energy >= cook_fresh_min_energy
            and biometrics.mood >= float(cooking_gates["cook_fresh_min_mood"])
        ),  # 1: cook_fresh
        (
            can_cook_location
            and (not is_stress_burned_out)
            and has_any_prep_combo
            and biometrics.energy >= meal_prep_min_energy
            and biometrics.mood >= float(cooking_gates["meal_prep_min_mood"])
        ),  # 2: meal_prep
        can_eat and (can_reheat_home or can_eat_bekal_campus),  # 3: reheat_or_eat_bekal
        can_eat and finance.can_afford(budget_meal_price),  # 4: budget_external_meal
        can_eat and finance.can_afford(varied_meal_price),  # 5: varied_external_meal
    ]

    # 5. Bring bekal mask (2: [0, 1])
    max_packed = int(cooking_gates["max_packed_lunch_portions"])
    can_bring_bekal = (
        is_at_apartment
        and (not is_night)
        and has_meal_prep
        and pantry.get_packed_lunch_total_portions() < max_packed
        and biometrics.energy >= float(cooking_gates["pack_bekal_min_energy"])
        and biometrics.mood >= float(cooking_gates["pack_bekal_min_mood"])
    )
    bring_bekal_mask = [True, can_bring_bekal]

    # 6. Grocery bundle mask (15)
    grocery_bundle_mask = [
        finance.can_afford(pantry.get_grocery_bundle_cost_idr(i) + grocery_commute_cost)
        for i in range(len(PantryManager.GROCERY_BUNDLES))
    ]
    if not any(grocery_bundle_mask):
        grocery_bundle_mask[0] = True  # dummy aman (grocery_shop sudah False)

    return np.array(
        activity_mask
        + route_mask
        + meal_mode_mask
        + cook_combo_mask
        + bring_bekal_mask
        + grocery_bundle_mask,
        dtype=bool,
    )
