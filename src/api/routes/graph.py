from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.dependencies import resolve_world_config
from src.api.schemas import RoutePreviewRequest
from src.env import WorldGraph
from src.env.pricing import compute_edge_fare_bounds, compute_edge_physical_deltas

router = APIRouter(prefix="/graph", tags=["graph"])


@router.post("/preview-route")
def preview_route(req: RoutePreviewRequest) -> dict[str, Any]:
    """Menghitung rute Dijkstra (fastest/cheapest), jarak tempuh, estimasi tarif, dan beban fisik."""
    cfg = resolve_world_config(req.config, req.preset_name, req.custom_config_name)
    try:
        graph = WorldGraph(cfg)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if req.source_node_id not in graph.nodes or req.target_node_id not in graph.nodes:
        raise HTTPException(
            status_code=404,
            detail="source_node_id atau target_node_id tidak terdaftar di graf.",
        )

    path = graph.find_best_path(
        source_node_id=req.source_node_id,
        target_node_id=req.target_node_id,
        preference=req.preference,
        climate=req.climate,
        ojol_tariff=req.ojol_tariff,
        is_raining=req.is_raining,
    )
    if path is None:
        raise HTTPException(
            status_code=422,
            detail=f"Tidak ada rute dari '{req.source_node_id}' ke '{req.target_node_id}'.",
        )

    edge_breakdown: list[dict[str, Any]] = []
    for edge in path.edges:
        fare_bounds = compute_edge_fare_bounds(
            edge=edge,
            ojol_tariff=req.ojol_tariff,
            is_raining=req.is_raining,
        )
        phys = compute_edge_physical_deltas(
            edge=edge,
            walk_physics=cfg.mobility_and_climate.walk_physics,
            motorized_physics=cfg.mobility_and_climate.motorized_physics,
            heat_sensitivity=cfg.mobility_and_climate.heat_sensitivity,
            climate=req.climate,
            is_raining=req.is_raining,
        )
        edge_breakdown.append(
            {
                "edge_id": edge.id,
                "label": edge.label or edge.id,
                "mode": edge.mode,
                "pricing_model": edge.pricing.pricing_model,
                "distance_km": float(edge.distance_km),
                "min_fare_idr": float(fare_bounds[0]),
                "max_fare_idr": float(fare_bounds[1]),
                "energy_cost": round(-float(phys[0]), 4),
                "mood_delta": round(float(phys[1]), 4),
            }
        )

    return {
        "source_node_id": path.source_node_id,
        "target_node_id": path.target_node_id,
        "preference": req.preference,
        "edge_ids": [edge.id for edge in path.edges],
        "total_distance_km": float(path.total_distance_km),
        "walk_distance_km": float(path.walk_distance_km),
        "min_fare_idr": float(path.min_fare_idr),
        "max_fare_idr": float(path.max_fare_idr),
        "total_energy_cost": float(path.total_energy_cost),
        "total_mood_delta": float(path.total_mood_delta),
        "edges": edge_breakdown,
    }
