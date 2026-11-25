import numpy as np

from src.env.biometrics import BiometricsState
from src.env.finance import FinanceManager

VALID_COMMUTE_ROUTES = ("direct_ojol", "transit_ccws_ojol")


def sample_idr_from_range(
    random_generator: np.random.Generator, price_range: list[int] | tuple[int, int]
) -> float:
    minimum_price = int(price_range[0])
    maximum_price = int(price_range[1])
    if minimum_price >= maximum_price:
        return float(minimum_price)
    return float(random_generator.integers(minimum_price, maximum_price + 1))


def try_minimarket_impulse(
    random_generator: np.random.Generator,
    finance_manager: FinanceManager,
    mobility_config: dict,
    temptation_probability: float,
) -> tuple[bool, float, float]:
    """
    Mengevaluasi peluang jajan impulsif di minimarket area CCWS.
    Mengembalikan (jadi_jajan, hunger_delta, mood_delta).
    """
    if random_generator.random() >= temptation_probability:
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
    mobility_config: dict,
    rewards_config: dict,
    requested_route_name: str,
    destination_location: str,
    current_climate: str,
    current_ojol_tariff: str,
    waited_until_evening: bool,
    force_return_if_broke: bool = False,
) -> dict[str, object]:
    """
    Menjalankan satu perjalanan apartemen <-> kampus menggunakan direct_ojol atau
    transit_ccws_ojol (tanpa full_walk) dari saldo operasional tunggal.
    """
    effective_climate = current_climate
    effective_tariff = current_ojol_tariff
    consumed_evening_wait = False

    wait_rules = mobility_config["campus_hub_rules"]["wait_at_department_until_evening"]
    if waited_until_evening and destination_location == "apartment":
        if bool(wait_rules["avoids_midday_heat"]):
            effective_climate = "cool"
        if bool(wait_rules["avoids_peak_ojol"]):
            effective_tariff = "normal"
        consumed_evening_wait = True

    routes_config = mobility_config["routes"]
    selected_route_name = requested_route_name
    route_profile = routes_config[selected_route_name]
    tariff_key = "cost_peak_idr" if effective_tariff == "peak" else "cost_normal_idr"
    fare_idr = sample_idr_from_range(random_generator, route_profile[tariff_key])

    payment_succeeded = finance_manager.pay(fare_idr, category="transport")

    # Jika saldo tidak cukup untuk direct_ojol, coba turun ke transit_ccws_ojol
    if not payment_succeeded and selected_route_name == "direct_ojol":
        selected_route_name = "transit_ccws_ojol"
        route_profile = routes_config[selected_route_name]
        fare_idr = sample_idr_from_range(random_generator, route_profile[tariff_key])
        payment_succeeded = finance_manager.pay(fare_idr, category="transport")

    energy_delta = 0.0
    mood_delta = 0.0
    hunger_delta = 0.0
    emergency_reward_penalty = 0.0
    bought_minimarket_impulse = False

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
                "consumed_evening_wait": consumed_evening_wait,
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
            "consumed_evening_wait": False,
        }

    mood_key = "mood_delta_midday" if effective_climate == "midday" else "mood_delta_cool"
    if selected_route_name == "direct_ojol":
        energy_delta -= float(route_profile["energy_cost"])
        mood_delta += float(route_profile[mood_key])
    else:
        energy_key = "energy_cost_midday" if effective_climate == "midday" else "energy_cost_cool"
        energy_delta -= float(route_profile[energy_key])
        mood_delta += float(route_profile[mood_key])

        temptation_probability = float(route_profile["minimarket_temptation_prob"])
        bought_minimarket_impulse, impulse_hunger, impulse_mood = try_minimarket_impulse(
            random_generator=random_generator,
            finance_manager=finance_manager,
            mobility_config=mobility_config,
            temptation_probability=temptation_probability,
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
        "reward_penalty": emergency_reward_penalty,
        "bought_minimarket_impulse": bought_minimarket_impulse,
        "consumed_evening_wait": consumed_evening_wait,
    }


def resolve_mobility_phase(
    current_location: str,
    waited_until_evening: bool,
    effective_activity: str,
    requested_mobility: str,
    is_class_scheduled_now: bool,
    step_in_day: int,
    steps_per_day: int,
    current_climate: str,
    current_ojol_tariff: str,
    random_generator: np.random.Generator,
    finance_manager: FinanceManager,
    biometrics_state: BiometricsState,
    mobility_config: dict,
    rewards_config: dict,
    step_events: dict[str, object],
) -> tuple[str, bool, str, float]:
    """
    Menentukan perpindahan apartemen <-> kampus dengan aturan realistis:
    1. Jam Malam (step terakhir): kampus tutup, agen di kampus wajib pulang ke apartemen.
    2. attend_class hanya valid jika benar-benar ada jadwal kuliah pada step ini.
    3. Perjalanan apartemen <-> kampus hanya bisa memakai direct_ojol atau transit_ccws_ojol.
    4. Di kampus siang/sore tanpa pulang tidak boleh rest_or_sleep/room_study/grocery_shop/night_cramming.
    """
    campus_activities = {"attend_class", "ccws_stay", "wait_at_department"}
    apartment_only_activities = {"room_study", "grocery_shop", "rest_or_sleep", "night_cramming"}
    is_night = step_in_day == (steps_per_day - 1)

    if is_night and effective_activity in campus_activities:
        effective_activity = "rest_or_sleep"

    if effective_activity == "attend_class" and not is_class_scheduled_now:
        effective_activity = "rest_or_sleep"

    if effective_activity == "night_cramming" and not is_night:
        effective_activity = "room_study"

    requires_campus_location = effective_activity in campus_activities
    executed_route = "stay"
    mobility_reward_penalty = 0.0
    updated_location = current_location
    updated_waited_until_evening = waited_until_evening

    if requires_campus_location and current_location == "apartment":
        if requested_mobility not in VALID_COMMUTE_ROUTES:
            effective_activity = "rest_or_sleep"
            step_events["campus_activity_cancelled_no_transport"] = True
        else:
            trip_result = execute_commute_trip(
                random_generator=random_generator,
                finance_manager=finance_manager,
                mobility_config=mobility_config,
                rewards_config=rewards_config,
                requested_route_name=requested_mobility,
                destination_location="campus",
                current_climate=current_climate,
                current_ojol_tariff=current_ojol_tariff,
                waited_until_evening=waited_until_evening,
                force_return_if_broke=False,
            )
            if bool(trip_result["trip_succeeded"]):
                updated_location, updated_waited_until_evening, executed_route = (
                    _apply_trip_effects(
                        trip_result, biometrics_state, waited_until_evening, step_events
                    )
                )
                mobility_reward_penalty += float(trip_result["reward_penalty"])
            else:
                effective_activity = "rest_or_sleep"
                step_events["campus_activity_cancelled_insufficient_funds"] = True

    elif not requires_campus_location and current_location == "campus":
        should_return_home = (requested_mobility in VALID_COMMUTE_ROUTES) or is_night
        if should_return_home:
            return_route = (
                requested_mobility
                if requested_mobility in VALID_COMMUTE_ROUTES
                else "transit_ccws_ojol"
            )
            trip_result = execute_commute_trip(
                random_generator=random_generator,
                finance_manager=finance_manager,
                mobility_config=mobility_config,
                rewards_config=rewards_config,
                requested_route_name=return_route,
                destination_location="apartment",
                current_climate=current_climate,
                current_ojol_tariff=current_ojol_tariff,
                waited_until_evening=waited_until_evening,
                force_return_if_broke=is_night,
            )
            if bool(trip_result["trip_succeeded"]):
                updated_location, updated_waited_until_evening, executed_route = (
                    _apply_trip_effects(
                        trip_result, biometrics_state, waited_until_evening, step_events
                    )
                )
                mobility_reward_penalty += float(trip_result["reward_penalty"])
            elif effective_activity in apartment_only_activities:
                effective_activity = "wait_at_department"
                step_events["return_cancelled_insufficient_funds"] = True
        elif effective_activity in apartment_only_activities:
            effective_activity = "wait_at_department"

    step_events["executed_route"] = executed_route
    return (
        updated_location,
        updated_waited_until_evening,
        effective_activity,
        mobility_reward_penalty,
    )


def _apply_trip_effects(
    trip_result: dict[str, object],
    biometrics_state: BiometricsState,
    waited_until_evening: bool,
    step_events: dict[str, object],
) -> tuple[str, bool, str]:
    biometrics_state.energy += float(trip_result["energy_delta"])
    biometrics_state.mood += float(trip_result["mood_delta"])
    biometrics_state.hunger += float(trip_result["hunger_delta"])
    updated_waited = False if bool(trip_result["consumed_evening_wait"]) else waited_until_evening
    step_events["transit_minimarket_impulse"] = bool(trip_result["bought_minimarket_impulse"])
    return (
        str(trip_result["destination_location"]),
        updated_waited,
        str(trip_result["executed_route"]),
    )
