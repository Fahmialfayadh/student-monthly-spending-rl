from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from src.api.dependencies import (
    ACTIVE_ENV_CONFIG_PATH,
    CUSTOM_CONFIGS_DIR,
    PRESETS_DIR,
    ROOT_DIR,
    compute_text_sha256,
    resolve_scoped_config_path,
    validate_config_slug,
    validation_error_http,
)
from src.api.schemas import ImportYamlRequest, SaveConfigRequest
from src.env import StudentLifeEnv, WorldConfig, WorldGraph, load_world_config

router = APIRouter(tags=["configs"])


def _summarize_yaml_file(yaml_file: Path, scope: str, name: str | None = None) -> dict[str, Any]:
    raw_text = yaml_file.read_text(encoding="utf-8")
    cfg = load_world_config(yaml_file)
    stat = yaml_file.stat()
    try:
        rel_path = str(yaml_file.relative_to(ROOT_DIR))
    except ValueError:
        rel_path = str(yaml_file)
    return {
        "scope": scope,
        "name": name or yaml_file.stem,
        "preset_name": cfg.preset_name,
        "schema_version": cfg.schema_version,
        "path": rel_path,
        "sha256": compute_text_sha256(raw_text),
        "updated_at_epoch": float(stat.st_mtime),
        "editable": scope == "custom",
        "home_node_id": cfg.world_graph.home_node_id,
        "num_nodes": len(cfg.world_graph.locations),
        "num_edges": len(cfg.world_graph.edges),
        "num_activities": len(cfg.activities),
        "num_external_foods": len(cfg.external_foods),
    }


def _persist_validated_config(
    slug: str,
    cfg: WorldConfig,
    overwrite: bool,
    expected_sha256: str | None,
    set_active_env_config: bool,
) -> dict[str, Any]:
    try:
        graph = WorldGraph(cfg)
        env = StudentLifeEnv(yaml_path=cfg.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    CUSTOM_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = CUSTOM_CONFIGS_DIR / f"{slug}.yaml"

    if target_path.exists():
        if not overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"Config lokal '{slug}' sudah ada dan overwrite=false.",
            )
        if expected_sha256 is not None:
            current_sha = compute_text_sha256(target_path.read_text(encoding="utf-8"))
            if current_sha != expected_sha256:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"File lokal '{target_path.name}' sudah berubah di disk "
                        "(sha256 mismatch). Muat ulang sebelum menyimpan."
                    ),
                )

    payload_dict = cfg.model_dump(mode="json")
    yaml_text = yaml.safe_dump(payload_dict, sort_keys=False, allow_unicode=True)
    target_path.write_text(yaml_text, encoding="utf-8")

    active_synced = False
    if set_active_env_config:
        ACTIVE_ENV_CONFIG_PATH.write_text(yaml_text, encoding="utf-8")
        active_synced = True

    new_sha256 = compute_text_sha256(yaml_text)
    return {
        "saved": True,
        "scope": "custom",
        "name": slug,
        "path": str(target_path.relative_to(ROOT_DIR)),
        "sha256": new_sha256,
        "active_env_config_synced": active_synced,
        "observation_dim": int(env.observation_space.shape[0]),
        "action_nvec": [int(x) for x in env.action_space.nvec.tolist()],
        "graph_summary": {
            "home_node_id": graph.home_node_id,
            "nodes": [n.id for n in cfg.world_graph.locations],
            "edges": [e.id for e in cfg.world_graph.edges],
        },
        "yaml_text": yaml_text,
    }


@router.get("/presets")
def list_presets() -> dict[str, list[dict[str, Any]]]:
    """Menampilkan daftar preset bawaan di configs/presets."""
    items: list[dict[str, Any]] = []
    if PRESETS_DIR.exists():
        for yaml_file in sorted(PRESETS_DIR.glob("*.yaml")):
            items.append(_summarize_yaml_file(yaml_file, scope="preset"))
    return {"presets": items}


@router.get("/presets/{name}")
def get_preset(name: str) -> dict[str, Any]:
    """Mengambil isi WorldConfig dari preset berdasarkan nama file."""
    slug = validate_config_slug(name)
    preset_path = PRESETS_DIR / f"{slug}.yaml"
    if not preset_path.exists():
        raise HTTPException(status_code=404, detail=f"Preset '{slug}' tidak ditemukan.")
    raw_text = preset_path.read_text(encoding="utf-8")
    cfg = load_world_config(preset_path)
    return {
        "scope": "preset",
        "name": slug,
        "sha256": compute_text_sha256(raw_text),
        "config": cfg.model_dump(mode="json"),
    }


@router.get("/configs")
def list_local_configs() -> dict[str, list[dict[str, Any]]]:
    """Mendaftar seluruh config lokal: active (env_config.yaml), preset bawaan, dan custom user."""
    configs: list[dict[str, Any]] = []
    if ACTIVE_ENV_CONFIG_PATH.exists():
        configs.append(_summarize_yaml_file(ACTIVE_ENV_CONFIG_PATH, scope="active", name="env_config"))
    if PRESETS_DIR.exists():
        for yaml_file in sorted(PRESETS_DIR.glob("*.yaml")):
            configs.append(_summarize_yaml_file(yaml_file, scope="preset"))
    if CUSTOM_CONFIGS_DIR.exists():
        for yaml_file in sorted(CUSTOM_CONFIGS_DIR.glob("*.yaml")):
            configs.append(_summarize_yaml_file(yaml_file, scope="custom"))
    return {"configs": configs}


@router.get("/configs/{scope}/{name}")
def get_local_config(scope: str, name: str) -> dict[str, Any]:
    """Membaca isi file YAML lokal beserta sha256 untuk sinkronisasi 2-arah UI <-> disk."""
    target_path = resolve_scoped_config_path(scope, name)
    if not target_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Config '{scope}/{name}' tidak ditemukan di disk.",
        )
    raw_text = target_path.read_text(encoding="utf-8")
    cfg = load_world_config(target_path)
    return {
        "scope": scope,
        "name": "env_config" if scope == "active" else target_path.stem,
        "path": str(target_path.relative_to(ROOT_DIR)),
        "sha256": compute_text_sha256(raw_text),
        "updated_at_epoch": float(target_path.stat().st_mtime),
        "config": cfg.model_dump(mode="json"),
        "yaml_text": raw_text,
    }


@router.post("/configs/save")
def save_local_config(req: SaveConfigRequest) -> dict[str, Any]:
    """Memvalidasi payload WorldConfig UI lalu menyimpannya ke configs/custom/{name}.yaml."""
    slug = validate_config_slug(req.name)
    payload = {**req.config, "preset_name": req.config.get("preset_name") or slug}
    try:
        cfg = WorldConfig.model_validate(payload)
    except ValidationError as exc:
        raise validation_error_http(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _persist_validated_config(
        slug=slug,
        cfg=cfg,
        overwrite=req.overwrite,
        expected_sha256=req.expected_sha256,
        set_active_env_config=req.set_active_env_config,
    )


@router.post("/configs/import-yaml")
def import_yaml_config(req: ImportYamlRequest) -> dict[str, Any]:
    """Mengimpor string YAML mentah, memvalidasi skema, lalu menyimpannya ke configs/custom/{name}.yaml."""
    slug = validate_config_slug(req.name)
    try:
        raw = yaml.safe_load(req.yaml_content)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"YAML tidak valid: {exc}") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="Isi YAML harus berupa mapping/object di root.")

    payload = {**raw, "preset_name": raw.get("preset_name") or slug}
    try:
        cfg = WorldConfig.model_validate(payload)
    except ValidationError as exc:
        raise validation_error_http(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _persist_validated_config(
        slug=slug,
        cfg=cfg,
        overwrite=req.overwrite,
        expected_sha256=None,
        set_active_env_config=req.set_active_env_config,
    )


@router.delete("/configs/custom/{name}")
def delete_custom_config(name: str) -> dict[str, Any]:
    """Menghapus file custom config milik user di configs/custom/{name}.yaml."""
    slug = validate_config_slug(name)
    target_path = CUSTOM_CONFIGS_DIR / f"{slug}.yaml"
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Custom config '{slug}' tidak ditemukan.")
    target_path.unlink()
    return {"deleted": True, "scope": "custom", "name": slug}


@router.post("/configs/render-yaml")
def render_yaml_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Memvalidasi payload WorldConfig lalu mengembalikan teks YAML tanpa menulis ke disk."""
    try:
        cfg = WorldConfig.model_validate(payload)
        WorldGraph(cfg)
    except ValidationError as exc:
        raise validation_error_http(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    yaml_text = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    return {
        "valid": True,
        "sha256": compute_text_sha256(yaml_text),
        "yaml_text": yaml_text,
    }


@router.post("/config/validate")
def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Memvalidasi payload WorldConfig UI, konektivitas WorldGraph, dan dimensi env."""
    try:
        cfg = WorldConfig.model_validate(payload)
        graph = WorldGraph(cfg)
        env = StudentLifeEnv(yaml_path=cfg.model_dump())
    except ValidationError as exc:
        raise validation_error_http(exc) from exc
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
