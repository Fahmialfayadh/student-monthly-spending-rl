from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from fastapi import HTTPException
from pydantic import ValidationError

from src.env import StudentLifeEnv, WorldConfig, load_world_config

ROOT_DIR = Path(__file__).resolve().parents[2]
PRESETS_DIR = ROOT_DIR / "configs" / "presets"
CUSTOM_CONFIGS_DIR = ROOT_DIR / "configs" / "custom"
ACTIVE_ENV_CONFIG_PATH = ROOT_DIR / "env_config.yaml"

_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

ConfigScope = Literal["preset", "custom", "active"]

ACTIVITY_HANDLERS: tuple[str, ...] = (
    "rest",
    "attend_schedule",
    "study",
    "hub_stay",
    "leisure_reset",
    "wait",
    "grocery_shop",
    "night_cram",
)


def validation_error_http(exc: ValidationError) -> HTTPException:
    """Konversi Pydantic ValidationError ke HTTP 422 yang selalu JSON-serializable."""
    return HTTPException(status_code=422, detail=json.loads(exc.json()))


def validate_config_slug(name: str) -> str:
    """Memvalidasi nama file config agar aman (tanpa path traversal)."""
    cleaned = name.strip()
    if not _SLUG_RE.match(cleaned):
        raise HTTPException(
            status_code=422,
            detail=(
                "Nama config harus alfanumerik (boleh '-', '_'), diawali huruf/angka, "
                "panjang 1-64 karakter."
            ),
        )
    return cleaned


def compute_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_canonical_world_sha256(cfg: WorldConfig) -> str:
    """Hash kanonik YAML yang identik dengan file sementara saat training."""
    canonical = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    return compute_text_sha256(canonical)


def resolve_scoped_config_path(scope: str, name: str) -> Path:
    """Mengembalikan path file YAML lokal berdasarkan scope ('preset', 'custom', 'active')."""
    if scope == "active":
        if name not in ("env_config", "default"):
            raise HTTPException(
                status_code=404,
                detail="Untuk scope 'active', gunakan nama 'env_config'.",
            )
        return ACTIVE_ENV_CONFIG_PATH
    slug = validate_config_slug(name)
    if scope == "preset":
        return PRESETS_DIR / f"{slug}.yaml"
    if scope == "custom":
        return CUSTOM_CONFIGS_DIR / f"{slug}.yaml"
    raise HTTPException(
        status_code=400,
        detail=f"Scope '{scope}' tidak dikenal. Gunakan 'preset', 'custom', atau 'active'.",
    )


def resolve_world_config(
    config_payload: dict[str, Any] | None,
    preset_name: str | None,
    custom_config_name: str | None = None,
) -> WorldConfig:
    """Memuat WorldConfig dari payload JSON inline, custom YAML lokal, atau preset YAML."""
    if config_payload is not None:
        try:
            return WorldConfig.model_validate(config_payload)
        except ValidationError as exc:
            raise validation_error_http(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if custom_config_name:
        custom_slug = validate_config_slug(custom_config_name)
        custom_path = CUSTOM_CONFIGS_DIR / f"{custom_slug}.yaml"
        if not custom_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Custom config '{custom_slug}' tidak ditemukan di {custom_path.name}.",
            )
        return load_world_config(custom_path)

    target_preset = preset_name or "standard_student"
    preset_slug = validate_config_slug(target_preset)
    preset_path = PRESETS_DIR / f"{preset_slug}.yaml"
    if not preset_path.exists():
        raise HTTPException(status_code=404, detail=f"Preset '{preset_slug}' tidak ditemukan.")
    return load_world_config(preset_path)


def split_action_masks(env: StudentLifeEnv) -> dict[str, list[bool]]:
    """Memecah flat binary action mask menjadi dictionary per head MultiDiscrete."""
    flat_mask = env.action_masks()
    head_names = ("activity", "route", "meal_mode", "cook_combo", "bring_bekal", "grocery_bundle")
    grouped: dict[str, list[bool]] = {}
    cursor = 0
    for head, dim_len in zip(head_names, env.action_space.nvec):
        n = int(dim_len)
        grouped[head] = [bool(x) for x in flat_mask[cursor : cursor + n]]
        cursor += n
    return grouped


def obs_to_dict(env: StudentLifeEnv, obs: np.ndarray) -> dict[str, float]:
    """Mengonversi vektor observasi float32 ke dictionary bernilai float berlabel."""
    return {
        name: round(float(val), 6)
        for name, val in zip(env.observation_feature_names, obs.tolist())
    }


def build_session_snapshot(
    session_id: str,
    env: StudentLifeEnv,
    obs: np.ndarray,
    info: dict[str, Any],
) -> dict[str, Any]:
    """Menyusun payload state lengkap sesi sandbox untuk konsumsi UI."""
    return {
        "session_id": session_id,
        "observation": obs_to_dict(env, obs),
        "action_nvec": [int(x) for x in env.action_space.nvec.tolist()],
        "action_labels": env.get_action_labels(),
        "action_masks": split_action_masks(env),
        "graph_state": env.get_graph_state(),
        "info": info,
    }
