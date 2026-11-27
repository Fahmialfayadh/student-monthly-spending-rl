from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RoutePreviewRequest(BaseModel):
    """Payload permintaan simulasi biaya & beban fisik antar-node pada graf."""

    config: dict[str, Any] | None = None
    preset_name: str | None = "surabaya_its"
    custom_config_name: str | None = None
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
    custom_config_name: str | None = None
    seed: int = 42
    exploring_starts_pantry: bool | None = None


class SimStepRequest(BaseModel):
    """Payload satu langkah aksi (6 head MultiDiscrete) pada sesi sandbox."""

    action: list[int] = Field(min_length=6, max_length=6)
    validate_mask: bool = True


class SaveConfigRequest(BaseModel):
    """Payload untuk menyimpan konfigurasi WorldConfig ke file YAML lokal."""

    name: str = Field(min_length=1, max_length=64)
    config: dict[str, Any]
    overwrite: bool = True
    expected_sha256: str | None = None
    set_active_env_config: bool = False


class ImportYamlRequest(BaseModel):
    """Payload untuk mengimpor string YAML mentah menjadi config lokal bernama."""

    name: str = Field(min_length=1, max_length=64)
    yaml_content: str = Field(min_length=1)
    overwrite: bool = True
    set_active_env_config: bool = False
