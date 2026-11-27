from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from src.env import (
    RIDE_HAILING_ZONE_PRESETS,
    StudentLifeEnv,
    WorldConfig,
    WorldGraph,
    export_world_json_schema,
    load_world_config,
)
from src.env.pricing import compute_edge_fare_bounds, compute_edge_physical_deltas

PRESETS_DIR = Path(__file__).resolve().parents[2] / "configs" / "presets"

_ACTIVITY_HANDLERS = (
    "rest",
    "attend_schedule",
    "study",
    "hub_stay",
    "leisure_reset",
    "wait",
    "grocery_shop",
    "night_cram",
)


class RoutePreviewRequest(BaseModel):
    """Payload permintaan simulasi biaya & beban fisik antar-node pada graf."""

    config: dict[str, Any] | None = None
    preset_name: str | None = "surabaya_its"
    source_node_id: str
    target_node_id: str
    preference: Literal["fastest", "cheapest"] = "fastest"
    climate: str = "cool"
    ojol_tariff: str = "normal"
    is_raining: bool = False


class SimCreateRequest(BaseModel):
    """Payload pembuatan sesi interaktif sandbox environment untuk UI."""

    config: dict[str, Any] | None = None
    preset_name: str | None = "surabaya_its"
    seed: int = 42
    exploring_starts_pantry: bool | None = None


class SimStepRequest(BaseModel):
    """Payload satu langkah aksi (6 head MultiDiscrete) pada sesi sandbox."""

    action: list[int] = Field(min_length=6, max_length=6)
    validate_mask: bool = True


def _resolve_world_config(
    config_payload: dict[str, Any] | None,
    preset_name: str | None,
) -> WorldConfig:
    if config_payload is not None:
        return WorldConfig.model_validate(config_payload)
    target_preset = preset_name or "surabaya_its"
    preset_path = PRESETS_DIR / f"{target_preset}.yaml"
    if not preset_path.exists():
        raise HTTPException(status_code=404, detail=f"Preset '{target_preset}' tidak ditemukan.")
    return load_world_config(preset_path)


def _split_action_masks(env: StudentLifeEnv) -> dict[str, list[bool]]:
    flat_mask = env.action_masks()
    head_names = ("activity", "route", "meal_mode", "cook_combo", "bring_bekal", "grocery_bundle")
    grouped: dict[str, list[bool]] = {}
    cursor = 0
    for head, dim_len in zip(head_names, env.action_space.nvec):
        n = int(dim_len)
        grouped[head] = [bool(x) for x in flat_mask[cursor : cursor + n]]
        cursor += n
    return grouped


def _obs_to_dict(env: StudentLifeEnv, obs: np.ndarray) -> dict[str, float]:
    return {
        name: round(float(val), 6)
        for name, val in zip(env.observation_feature_names, obs.tolist())
    }


def _build_session_snapshot(
    session_id: str,
    env: StudentLifeEnv,
    obs: np.ndarray,
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "observation": _obs_to_dict(env, obs),
        "action_nvec": [int(x) for x in env.action_space.nvec.tolist()],
        "action_labels": env.get_action_labels(),
        "action_masks": _split_action_masks(env),
        "graph_state": env.get_graph_state(),
        "info": info,
    }


def create_app() -> FastAPI:
    """Factory FastAPI untuk manajemen konfigurasi graf, pricing, validasi, & sandbox RL."""
    application = FastAPI(
        title="Student Survival Spatial-Graph RL API",
        version="2.0.0",
        description=(
            "Backend API untuk mendesain graf spasial lokasi, tarif transport berbasis jarak, "
            "validasi skema WorldConfig, dan sandbox simulasi StudentLifeEnv."
        ),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sessions: dict[str, StudentLifeEnv] = {}

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "2.0.0"}

    @application.get("/schema")
    def get_schema() -> dict[str, Any]:
        return {
            "schema_version": "2.0",
            "json_schema": export_world_json_schema(),
            "ride_hailing_zone_presets": RIDE_HAILING_ZONE_PRESETS,
            "activity_handlers": list(_ACTIVITY_HANDLERS),
            "pricing_models": ["free", "manual_range", "per_km_custom", "ride_hailing_zone"],
            "route_preferences": list(StudentLifeEnv.ROUTE_PREFERENCES),
            "meal_modes": list(StudentLifeEnv.MEAL_MODES),
        }

    @application.get("/presets")
    def list_presets() -> dict[str, list[dict[str, Any]]]:
        items: list[dict[str, Any]] = []
        if PRESETS_DIR.exists():
            for yaml_file in sorted(PRESETS_DIR.glob("*.yaml")):
                cfg = load_world_config(yaml_file)
                items.append(
                    {
                        "name": yaml_file.stem,
                        "preset_name": cfg.preset_name,
                        "schema_version": cfg.schema_version,
                        "home_node_id": cfg.world_graph.home_node_id,
                        "num_nodes": len(cfg.world_graph.locations),
                        "num_edges": len(cfg.world_graph.edges),
                        "num_activities": len(cfg.activities),
                        "num_external_foods": len(cfg.external_foods),
                    }
                )
        return {"presets": items}

    @application.get("/presets/{name}")
    def get_preset(name: str) -> dict[str, Any]:
        preset_path = PRESETS_DIR / f"{name}.yaml"
        if not preset_path.exists():
            raise HTTPException(status_code=404, detail=f"Preset '{name}' tidak ditemukan.")
        cfg = load_world_config(preset_path)
        return {"name": name, "config": cfg.model_dump(mode="json")}

    @application.post("/config/validate")
    def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            cfg = WorldConfig.model_validate(payload)
            graph = WorldGraph(cfg)
            env = StudentLifeEnv(yaml_path=cfg.model_dump())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "valid": True,
            "preset_name": cfg.preset_name,
            "observation_dim": int(env.observation_space.shape[0]),
            "observation_features": list(env.observation_feature_names),
            "action_nvec": [int(x) for x in env.action_space.nvec.tolist()],
            "action_labels": env.get_action_labels(),
            "graph_summary": {
                "home_node_id": graph.home_node_id,
                "nodes": [n.id for n in cfg.world_graph.locations],
                "edges": [e.id for e in cfg.world_graph.edges],
            },
        }

    @application.post("/graph/preview-route")
    def preview_route(req: RoutePreviewRequest) -> dict[str, Any]:
        try:
            cfg = _resolve_world_config(req.config, req.preset_name)
            graph = WorldGraph(cfg)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc
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
            "min_fare_idr": float(path.min_fare_idr),
            "max_fare_idr": float(path.max_fare_idr),
            "total_energy_cost": float(path.total_energy_cost),
            "total_mood_delta": float(path.total_mood_delta),
            "edges": edge_breakdown,
        }

    @application.post("/sim/sessions")
    def create_sim_session(req: SimCreateRequest) -> dict[str, Any]:
        try:
            cfg = _resolve_world_config(req.config, req.preset_name)
            env = StudentLifeEnv(
                yaml_path=cfg.model_dump(),
                exploring_starts_pantry=req.exploring_starts_pantry,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        obs, info = env.reset(seed=req.seed)
        session_id = uuid4().hex[:12]
        sessions[session_id] = env
        return _build_session_snapshot(session_id, env, obs, info)

    @application.post("/sim/sessions/{session_id}/step")
    def step_sim_session(session_id: str, req: SimStepRequest) -> dict[str, Any]:
        env = sessions.get(session_id)
        if env is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' tidak ditemukan.")

        nvec = [int(x) for x in env.action_space.nvec.tolist()]
        for head_idx, (act_val, limit) in enumerate(zip(req.action, nvec)):
            if act_val < 0 or act_val >= limit:
                raise HTTPException(
                    status_code=422,
                    detail=f"Action head {head_idx} di luar rentang [0, {limit - 1}].",
                )

        if req.validate_mask:
            masks = env.action_masks()
            cursor = 0
            for head_idx, (act_val, limit) in enumerate(zip(req.action, nvec)):
                if not bool(masks[cursor + act_val]):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Action head {head_idx}={act_val} diblokir oleh action mask.",
                    )
                cursor += limit

        obs, reward, terminated, truncated, info = env.step(req.action)
        snapshot = _build_session_snapshot(session_id, env, obs, info)
        snapshot.update(
            {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        return snapshot

    @application.delete("/sim/sessions/{session_id}")
    def delete_sim_session(session_id: str) -> dict[str, Any]:
        env = sessions.pop(session_id, None)
        if env is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' tidak ditemukan.")
        env.close()
        return {"deleted": True, "session_id": session_id}

    return application


app = create_app()
