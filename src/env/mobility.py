import numpy as np

from src.env.biometrics import BiometricsState
from src.env.finance import FinanceManager

VALID_COMMUTE_ROUTES = ("direct_ojol", "transit_ccws_ojol")


def sample_idr_from_range(
    random_generator: np.random.Generator,
    price_range: list[int] | tuple[int, int],
    multiplier: float = 1.0,
) -> float:
    minimum_price = int(round(int(price_range[0]) * multiplier))
    maximum_price = int(round(int(price_range[1]) * multiplier))
    if minimum_price >= maximum_price:
        return float(minimum_price)
    return float(random_generator.integers(minimum_price, maximum_price + 1))


def get_route_fare_bounds(
    mobility_config: dict,
    route_name: str,
    current_ojol_tariff: str,
    is_raining: bool,
) -> tuple[float, float]:
    route_profile = mobility_config["routes"][route_name]
    tariff_key = "cost_peak_idr" if current_ojol_tariff == "peak" else "cost_normal_idr"
    surge = float(mobility_config["rain_ojol_surge_multiplier"]) if is_raining else 1.0
    low = float(round(int(route_profile[tariff_key][0]) * surge))
    high = float(round(int(route_profile[tariff_key][1]) * surge))
    return low, high


def apply_ccws_department_walk(
    biometrics_state: BiometricsState,
    mobility_config: dict,
    current_climate: str,
    is_raining: bool,
    step_events: dict[str, object],
) -> None:
    """
    Menerapkan satu config tunggal beban jalan kaki 1.5 km (CCWS <-> Departemen Informatika)
    baik sebagai bagian dari transit_ccws_ojol maupun saat berpindah antar gedung di kampus.
    """
    walk_cfg = mobility_config["ccws_department_walk_1_5km"]
    heat_sensitivity = float(mobility_config["heat_sensitivity"])

    if is_raining:
        energy_cost = float(walk_cfg["energy_cost_cool"]) + float(walk_cfg["extra_energy_cost_rain"])
        mood_delta = float(walk_cfg["mood_delta_cool"]) + float(walk_cfg["extra_mood_delta_rain"])
    elif current_climate == "midday":
        energy_cost = float(walk_cfg["energy_cost_midday"]) * heat_sensitivity
        mood_delta = float(walk_cfg["mood_delta_midday"]) * heat_sensitivity
    else:
        energy_cost = float(walk_cfg["energy_cost_cool"])
        mood_delta = float(walk_cfg["mood_delta_cool"])

    biometrics_state.energy -= energy_cost
    biometrics_state.mood += mood_delta
    step_events["walked_1_5km_ccws_dept"] = True


def compute_contextual_minimarket_probability(
    base_probability: float,
    hunger: float,
    effective_climate: str,
    current_day: int,
    finance_manager: FinanceManager,
    finance_config: dict,
    mobility_config: dict,
) -> float:
    """
    Menghitung peluang jajan impulsif minimarket secara kontekstual:
    dipengaruhi rasa lapar, cuaca panas terik, impulsivitas akhir bulan,
    dan kontrol diri saat saldo mulai menipis.
    """
    impulse_cfg = mobility_config["campus_hub_rules"]["minimarket_impulse_item"]
    probability = float(base_probability)

    if hunger > 45.0:
        probability += float(impulse_cfg["hungry_prob_bonus"])
    if effective_climate == "midday":
        probability += float(impulse_cfg["midday_prob_bonus"])

    late_month_cfg = finance_config["late_month_impulsivity"]
    if current_day >= int(late_month_cfg["start_day"]):
        probability += float(impulse_cfg["late_month_prob_bonus"]) * float(
            late_month_cfg["weight"]
        )

    if finance_manager.is_balance_low():
        probability *= float(impulse_cfg["low_balance_prob_multiplier"])

    return float(np.clip(probability, 0.0, 0.90))


def try_minimarket_impulse(
    random_generator: np.random.Generator,
    finance_manager: FinanceManager,
    biometrics_state: BiometricsState,
    finance_config: dict,
    mobility_config: dict,
    base_temptation_probability: float,
    effective_climate: str,
    current_day: int,
) -> tuple[bool, float, float]:
    """
    Mengevaluasi peluang jajan impulsif di minimarket area CCWS secara kontekstual.
    Mengembalikan (jadi_jajan, hunger_delta, mood_delta).
    """
    probability = compute_contextual_minimarket_probability(
        base_probability=base_temptation_probability,
        hunger=biometrics_state.hunger,
        effective_climate=effective_climate,
        current_day=current_day,
        finance_manager=finance_manager,
        finance_config=finance_config,
        mobility_config=mobility_config,
    )
    if random_generator.random() >= probability:
        return False, 0.0, 0.0

    impulse_config = mobility_config["campus_hub_rules"]["minimarket_impulse_item"]
    impulse_cost_idr = sample_idr_from_range(random_generator, impulse_config["spend_idr"])
    if finance_manager.pay(impulse_cost_idr, category="minimarket_impulse"):
        return (
            True,
            float(impulse_config["hunger_delta"]),
            float(impulse_config["mood_delta"]),
        )
    return False, 0.0, 0.0


def execute_commute_trip(
    random_generator: np.random.Generator,
    finance_manager: FinanceManager,
    biometrics_state: BiometricsState,
    finance_config: dict,
    mobility_config: dict,
    rewards_config: dict,
    requested_route_name: str,
    destination_location: str,
    current_climate: str,
    current_ojol_tariff: str,
    is_raining: bool,
    current_day: int,
    force_return_if_broke: bool = False,
) -> dict[str, object]:
    """
    Menjalankan bagian naik ojol (Apartemen <-> Departemen atau Apartemen <-> CCWS).
    Beban jalan kaki 1.5 km CCWS <-> Departemen ditangani terpisah melalui satu config
    tunggal `ccws_department_walk_1_5km` agar tidak ganda maupun hilang.
    """
    effective_climate = "rain" if is_raining else current_climate
    routes_config = mobility_config["routes"]
    selected_route_name = (
        requested_route_name if requested_route_name in VALID_COMMUTE_ROUTES else "transit_ccws_ojol"
    )

    def _try_pay_route(route_name: str) -> bool:
        min_fare, max_fare = get_route_fare_bounds(
            mobility_config, route_name, current_ojol_tariff, is_raining
        )
        if not finance_manager.can_afford(min_fare):
            return False
        sampled_fare = sample_idr_from_range(random_generator, (int(min_fare), int(max_fare)))
        actual_fare = min(sampled_fare, finance_manager.balance_idr)
        return finance_manager.pay(actual_fare, category="transport")

    payment_succeeded = _try_pay_route(selected_route_name)

    if not payment_succeeded and selected_route_name == "direct_ojol":
        selected_route_name = "transit_ccws_ojol"
        payment_succeeded = _try_pay_route(selected_route_name)

    if not payment_succeeded:
        if force_return_if_broke and destination_location == "apartment":
            finance_manager.drain_remaining_for_emergency(category="transport")
            emergency_reward_penalty = float(
                rewards_config["critical_state_penalties"]["emergency_return_penalty"]
            )
            return {
                "trip_succeeded": True,
                "executed_route": "emergency_return",
                "destination_location": "apartment",
                "energy_delta": -20.0,
                "mood_delta": -15.0,
                "hunger_delta": 0.0,
                "reward_penalty": emergency_reward_penalty,
                "bought_minimarket_impulse": False,
            }

        return {
            "trip_succeeded": False,
            "executed_route": "stay",
            "destination_location": "apartment" if destination_location == "campus" else "campus",
            "energy_delta": 0.0,
            "mood_delta": 0.0,
            "hunger_delta": 0.0,
            "reward_penalty": 0.0,
            "bought_minimarket_impulse": False,
        }

    route_profile = routes_config[selected_route_name]
    heat_sensitivity = float(mobility_config["heat_sensitivity"])
    energy_delta = -float(route_profile["energy_cost"])

    if effective_climate == "rain":
        mood_delta = float(route_profile["mood_delta_rain"])
    elif effective_climate == "midday":
        mood_delta = float(route_profile["mood_delta_midday"]) * heat_sensitivity
    else:
        mood_delta = float(route_profile["mood_delta_cool"])

    hunger_delta = 0.0
    bought_minimarket_impulse = False
    if selected_route_name == "transit_ccws_ojol":
        bought_minimarket_impulse, impulse_hunger, impulse_mood = try_minimarket_impulse(
            random_generator=random_generator,
            finance_manager=finance_manager,
            biometrics_state=biometrics_state,
            finance_config=finance_config,
            mobility_config=mobility_config,
            base_temptation_probability=float(route_profile["minimarket_base_temptation_prob"]),
            effective_climate=effective_climate,
            current_day=current_day,
        )
        hunger_delta += impulse_hunger
        mood_delta += impulse_mood

    return {
        "trip_succeeded": True,
        "executed_route": selected_route_name,
        "destination_location": destination_location,
        "energy_delta": energy_delta,
        "mood_delta": mood_delta,
        "hunger_delta": hunger_delta,
        "reward_penalty": 0.0,
        "bought_minimarket_impulse": bought_minimarket_impulse,
    }


def resolve_mobility_phase(
    current_location: str,
    current_campus_sub_location: str | None,
    effective_activity: str,
    requested_route: str,
    is_class_scheduled_now: bool,
    step_in_day: int,
    steps_per_day: int,
    current_climate: str,
    current_ojol_tariff: str,
    is_raining: bool,
    current_day: int,
    random_generator: np.random.Generator,
    finance_manager: FinanceManager,
    biometrics_state: BiometricsState,
    finance_config: dict,
    mobility_config: dict,
    rewards_config: dict,
    step_events: dict[str, object],
) -> tuple[str, str | None, str, float]:
    """
    Menentukan perpindahan apartemen <-> kampus dan antar sub-lokasi kampus (ccws <-> department):
    - direct_ojol mendarat di / berangkat dari 'department'.
    - transit_ccws_ojol mendarat di / berangkat dari 'ccws'.
    - Setiap perpindahan 'ccws' <-> 'department' (baik saat transit maupun pindah aktivitas di kampus)
      memakai satu config tunggal `ccws_department_walk_1_5km` (Rp 0, beban jalan kaki 1.5 km).
    """
    campus_activities = {"attend_class", "ccws_stay", "wait_at_department"}
    apartment_only_activities = {"room_study", "grocery_shop", "rest_or_sleep", "night_cramming"}
    is_night = step_in_day == (steps_per_day - 1)

    if is_night and effective_activity in campus_activities:
        effective_activity = "rest_or_sleep"

    if effective_activity == "attend_class" and not is_class_scheduled_now:
        effective_activity = (
            "rest_or_sleep" if current_location == "apartment" else "wait_at_department"
        )

    if effective_activity == "night_cramming" and not is_night:
        effective_activity = "room_study"

    executed_route = "stay"
    mobility_reward_penalty = 0.0
    updated_location = current_location
    updated_sub_location = current_campus_sub_location

    target_sub_location = (
        "ccws"
        if effective_activity == "ccws_stay"
        else ("department" if effective_activity in ("attend_class", "wait_at_department") else None)
    )

    if effective_activity in campus_activities and current_location == "apartment":
        trip_result = execute_commute_trip(
            random_generator=random_generator,
            finance_manager=finance_manager,
            biometrics_state=biometrics_state,
            finance_config=finance_config,
            mobility_config=mobility_config,
            rewards_config=rewards_config,
            requested_route_name=requested_route,
            destination_location="campus",
            current_climate=current_climate,
            current_ojol_tariff=current_ojol_tariff,
            is_raining=is_raining,
            current_day=current_day,
            force_return_if_broke=False,
        )
        if bool(trip_result["trip_succeeded"]):
            updated_location, executed_route = _apply_trip_effects(
                trip_result, biometrics_state, step_events
            )
            mobility_reward_penalty += float(trip_result["reward_penalty"])
            landed_sub = "department" if executed_route == "direct_ojol" else "ccws"
            if target_sub_location is not None and landed_sub != target_sub_location:
                apply_ccws_department_walk(
                    biometrics_state, mobility_config, current_climate, is_raining, step_events
                )
            updated_sub_location = target_sub_location or landed_sub
        else:
            effective_activity = "rest_or_sleep"
            step_events["campus_activity_cancelled_insufficient_funds"] = True

    elif effective_activity in campus_activities and current_location == "campus":
        current_sub = updated_sub_location or "department"
        if target_sub_location is not None and current_sub != target_sub_location:
            apply_ccws_department_walk(
                biometrics_state, mobility_config, current_climate, is_raining, step_events
            )
            updated_sub_location = target_sub_location

    elif effective_activity in apartment_only_activities and current_location == "campus":
        current_sub = updated_sub_location or "department"
        # Jika minta pulang via transit_ccws_ojol dari departemen, jalan kaki 1.5 km ke CCWS dulu
        if requested_route == "transit_ccws_ojol" and current_sub == "department":
            apply_ccws_department_walk(
                biometrics_state, mobility_config, current_climate, is_raining, step_events
            )
            current_sub = "ccws"

        trip_result = execute_commute_trip(
            random_generator=random_generator,
            finance_manager=finance_manager,
            biometrics_state=biometrics_state,
            finance_config=finance_config,
            mobility_config=mobility_config,
            rewards_config=rewards_config,
            requested_route_name=requested_route,
            destination_location="apartment",
            current_climate=current_climate,
            current_ojol_tariff=current_ojol_tariff,
            is_raining=is_raining,
            current_day=current_day,
            force_return_if_broke=is_night,
        )
        if bool(trip_result["trip_succeeded"]):
            updated_location, executed_route = _apply_trip_effects(
                trip_result, biometrics_state, step_events
            )
            mobility_reward_penalty += float(trip_result["reward_penalty"])
            updated_sub_location = None
        else:
            effective_activity = "wait_at_department"
            updated_sub_location = current_sub
            step_events["return_cancelled_insufficient_funds"] = True

    step_events["executed_route"] = executed_route
    return updated_location, updated_sub_location, effective_activity, mobility_reward_penalty


def _apply_trip_effects(
    trip_result: dict[str, object],
    biometrics_state: BiometricsState,
    step_events: dict[str, object],
) -> tuple[str, str]:
    biometrics_state.energy += float(trip_result["energy_delta"])
    biometrics_state.mood += float(trip_result["mood_delta"])
    biometrics_state.hunger += float(trip_result["hunger_delta"])
    step_events["transit_minimarket_impulse"] = bool(trip_result["bought_minimarket_impulse"])
    return str(trip_result["destination_location"]), str(trip_result["executed_route"])
