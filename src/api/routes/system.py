from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from src.api.dependencies import ACTIVITY_HANDLERS
from src.env import (
    RIDE_HAILING_ZONE_PRESETS,
    StudentLifeEnv,
    TrainingProfileConfig,
    export_training_json_schema,
    export_world_json_schema,
)

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    """Cek kesehatan layanan API."""
    return {"status": "ok", "version": "2.0.0"}


@router.get("/schema")
def get_schema() -> dict[str, Any]:
    """Ekspor JSON Schema WorldConfig, TrainingProfileConfig, preset zona, dan enum aksi."""
    return {
        "schema_version": "2.0",
        "json_schema": export_world_json_schema(),
        "training_json_schema": export_training_json_schema(),
        "default_training_profile": TrainingProfileConfig().model_dump(mode="json"),
        "ride_hailing_zone_presets": RIDE_HAILING_ZONE_PRESETS,
        "activity_handlers": list(ACTIVITY_HANDLERS),
        "pricing_models": ["free", "manual_range", "per_km_custom", "ride_hailing_zone"],
        "walk_preferences": ["prefer_walk", "balanced", "avoid_walk"],
        "route_preferences": list(StudentLifeEnv.ROUTE_PREFERENCES),
        "meal_modes": list(StudentLifeEnv.MEAL_MODES),
    }
