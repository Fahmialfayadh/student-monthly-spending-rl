from __future__ import annotations

import numpy as np

from src.env.biometrics import BiometricsState
from src.env.finance import FinanceManager
from src.env.pricing import (
    compute_edge_fare_bounds,
    compute_edge_physical_deltas,
    sample_idr_from_range,
)
from src.env.schema import ActivitySpecConfig, ImpulseItemConfig
from src.env.world_graph import PlannedPath, RoutePreference, WorldGraph


def compute_contextual_impulse_probability(
    base_probability: float,
    hunger: float,
    effective_climate: str,
    current_day: int,
    finance_manager: FinanceManager,
    finance_config: dict,
    impulse_item: ImpulseItemConfig,
) -> float:
    """Peluang jajan impulsif kontekstual (lapar, panas midday, akhir bulan, saldo rendah)."""
    probability = float(base_probability)
    if hunger > 45.0:
        probability += float(impulse_item.hungry_prob_bonus)
    if effective_climate == "midday":
        probability += float(impulse_item.midday_prob_bonus)

    late_month_cfg = finance_config["late_month_impulsivity"]
    if current_day >= int(late_month_cfg["start_day"]):
        probability += float(impulse_item.late_month_prob_bonus) * float(
            late_month_cfg["weight"]
        )

    if finance_manager.is_balance_low():
        probability *= float(impulse_item.low_balance_prob_multiplier)

    return float(np.clip(probability, 0.0, 0.90))


def try_impulse_temptation(
    random_generator: np.random.Generator,
    finance_manager: FinanceManager,
    biometrics_state: BiometricsState,
    finance_config: dict,
    impulse_item: ImpulseItemConfig | None,
    base_temptation_probability: float,
    effective_climate: str,
    current_day: int,
) -> tuple[bool, float, float]:
    """Mengevaluasi peluang jajan impulsif pada sebuah node atau edge."""
    if impulse_item is None or base_temptation_probability <= 0.0:
        return False, 0.0, 0.0

    probability = compute_contextual_impulse_probability(
        base_probability=base_temptation_probability,
        hunger=biometrics_state.hunger,
        effective_climate=effective_climate,
        current_day=current_day,
        finance_manager=finance_manager,
        finance_config=finance_config,
        impulse_item=impulse_item,
    )
    if random_generator.random() >= probability:
        return False, 0.0, 0.0

    impulse_cost_idr = sample_idr_from_range(random_generator, impulse_item.spend_idr)
    if finance_manager.pay(impulse_cost_idr, category="minimarket_impulse"):
        return True, float(impulse_item.hunger_delta), float(impulse_item.mood_delta)
    return False, 0.0, 0.0


def execute_graph_path(
    path: PlannedPath,
    random_generator: np.random.Generator,
    finance_manager: FinanceManager,
    biometrics_state: BiometricsState,
    world_graph: WorldGraph,
    finance_config: dict,
    current_climate: str,
    current_ojol_tariff: str,
    is_raining: bool,
    current_day: int,
    step_events: dict[str, object],
) -> bool:
    """
    Mengeksekusi perjalanan menyusuri setiap edge pada PlannedPath:
    membayar tarif sesuai model tarif edge, menerapkan beban fisik berbasis distance_km,
    serta mengevaluasi godaan jajan pada edge yang dilalui.
    """
    if not path.edges:
        return True
    if not finance_manager.can_afford(path.min_fare_idr):
        return False

    mobility_cfg = world_graph.mobility_cfg
    effective_climate = "rain" if is_raining else current_climate
    remaining_min_after = path.min_fare_idr
    traversed_edges: list[str] = []

    for edge in path.edges:
        low_fare, high_fare = compute_edge_fare_bounds(edge, current_ojol_tariff, is_raining)
        remaining_min_after = max(0.0, remaining_min_after - low_fare)
        if low_fare > 0.0:
            max_allowed = max(low_fare, finance_manager.balance_idr - remaining_min_after)
            sampled_fare = sample_idr_from_range(random_generator, (int(low_fare), int(high_fare)))
            actual_fare = min(sampled_fare, max_allowed)
            if not finance_manager.pay(actual_fare, category="transport"):
                return False

        energy_delta, mood_delta = compute_edge_physical_deltas(
            edge=edge,
            walk_physics=mobility_cfg.walk_physics,
            motorized_physics=mobility_cfg.motorized_physics,
            heat_sensitivity=float(mobility_cfg.heat_sensitivity),
            climate=current_climate,
            is_raining=is_raining,
        )
        biometrics_state.energy += energy_delta
        biometrics_state.mood += mood_delta
        traversed_edges.append(edge.id)

        if edge.mode == "walk":
            step_events["walked_edge"] = edge.id

        impulse_item = (
            mobility_cfg.impulse_items.get(edge.impulse_item_id)
            if edge.impulse_item_id
            else None
        )
        bought, imp_h, imp_m = try_impulse_temptation(
            random_generator=random_generator,
            finance_manager=finance_manager,
            biometrics_state=biometrics_state,
            finance_config=finance_config,
            impulse_item=impulse_item,
            base_temptation_probability=float(edge.temptation_prob),
            effective_climate=effective_climate,
            current_day=current_day,
        )
        if bought:
            biometrics_state.hunger += imp_h
            biometrics_state.mood += imp_m
            step_events["transit_minimarket_impulse"] = True

    step_events["traversed_edges"] = traversed_edges
    step_events["travel_distance_km"] = path.total_distance_km
    return True


def _pick_fallback_stay_activity(
    current_node_id: str,
    world_graph: WorldGraph,
    activity_specs: dict[str, ActivitySpecConfig],
    is_night: bool,
) -> str:
    node = world_graph.get_node(current_node_id)
    if node.can_sleep:
        for act_id, spec in activity_specs.items():
            if spec.handler == "rest":
                return act_id
    if not is_night:
        for act_id, spec in activity_specs.items():
            if spec.handler == "wait" and current_node_id in world_graph.resolve_activity_target_nodes(spec):
                return act_id
    for act_id, spec in activity_specs.items():
        if spec.handler == "rest":
            return act_id
    return next(iter(activity_specs))


def resolve_graph_mobility(
    current_node_id: str,
    effective_activity_id: str,
    activity_specs: dict[str, ActivitySpecConfig],
    world_graph: WorldGraph,
    requested_route_pref: RoutePreference,
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
    rewards_config: dict,
    step_events: dict[str, object],
) -> tuple[str, str, float]:
    """
    Menentukan perpindahan antar node di WorldGraph untuk menjalankan aktivitas tujuan:
    - Mematuhi jam malam (node dengan `closes_at_night=True` wajib pulang ke node `can_sleep`).
    - Memilih node target & rute terpendek (`fastest` -> fallback `cheapest` jika saldo tipis).
    - Jika jam malam gagal pulang karena saldo habis, mengeksekusi `emergency_return`.
    """
    is_night = step_in_day == (steps_per_day - 1)
    spec = activity_specs[effective_activity_id]

    if spec.handler == "attend_schedule" and not is_class_scheduled_now:
        effective_activity_id = _pick_fallback_stay_activity(
            current_node_id, world_graph, activity_specs, is_night
        )
        spec = activity_specs[effective_activity_id]

    if spec.handler == "night_cram" and not is_night:
        study_id = next(
            (aid for aid, s in activity_specs.items() if s.handler == "study"),
            None,
        )
        if study_id is not None:
            effective_activity_id = study_id
            spec = activity_specs[effective_activity_id]

    candidate_nodes = world_graph.resolve_activity_target_nodes(spec)
    if is_night:
        # Di step terakhir, hanya boleh ke node yang tidak tutup malam (node `can_sleep`)
        night_safe_candidates = [
            nid for nid in candidate_nodes if not world_graph.get_node(nid).closes_at_night
        ]
        if not night_safe_candidates:
            effective_activity_id = _pick_fallback_stay_activity(
                world_graph.home_node_id, world_graph, activity_specs, is_night=True
            )
            spec = activity_specs[effective_activity_id]
            candidate_nodes = world_graph.resolve_activity_target_nodes(spec)
        else:
            candidate_nodes = night_safe_candidates

    # Coba rute sesuai preferensi agen; bila tidak cukup saldo, fallback ke "cheapest"
    target_node, planned_path = world_graph.choose_nearest_target_node(
        source_node_id=current_node_id,
        candidate_node_ids=candidate_nodes,
        preference=requested_route_pref,
        climate=current_climate,
        ojol_tariff=current_ojol_tariff,
        is_raining=is_raining,
        max_budget_idr=finance_manager.balance_idr,
    )
    if planned_path is None and requested_route_pref != "cheapest":
        target_node, planned_path = world_graph.choose_nearest_target_node(
            source_node_id=current_node_id,
            candidate_node_ids=candidate_nodes,
            preference="cheapest",
            climate=current_climate,
            ojol_tariff=current_ojol_tariff,
            is_raining=is_raining,
            max_budget_idr=finance_manager.balance_idr,
        )

    mobility_reward_penalty = 0.0
    updated_node_id = current_node_id
    executed_route = "stay"

    if target_node is not None and planned_path is not None:
        if planned_path.edges:
            succeeded = execute_graph_path(
                path=planned_path,
                random_generator=random_generator,
                finance_manager=finance_manager,
                biometrics_state=biometrics_state,
                world_graph=world_graph,
                finance_config=finance_config,
                current_climate=current_climate,
                current_ojol_tariff=current_ojol_tariff,
                is_raining=is_raining,
                current_day=current_day,
                step_events=step_events,
            )
            if succeeded:
                updated_node_id = target_node
                executed_route = "->".join(e.id for e in planned_path.edges)
            else:
                effective_activity_id = _pick_fallback_stay_activity(
                    current_node_id, world_graph, activity_specs, is_night
                )
        else:
            updated_node_id = target_node
    else:
        # Tidak cukup dana untuk berpindah
        current_node = world_graph.get_node(current_node_id)
        if is_night and current_node.closes_at_night:
            finance_manager.drain_remaining_for_emergency(category="transport")
            biometrics_state.energy -= 20.0
            biometrics_state.mood -= 15.0
            mobility_reward_penalty += float(
                rewards_config["critical_state_penalties"]["emergency_return_penalty"]
            )
            updated_node_id = world_graph.home_node_id
            executed_route = "emergency_return"
        else:
            effective_activity_id = _pick_fallback_stay_activity(
                current_node_id, world_graph, activity_specs, is_night
            )
            step_events["activity_cancelled_insufficient_funds"] = True

    step_events["executed_route"] = executed_route
    return updated_node_id, effective_activity_id, mobility_reward_penalty
