from __future__ import annotations

from typing import Any

import numpy as np

# Preset tarif resmi angkutan sewa khusus / ojek online Indonesia (dapat di-override user di config)
RIDE_HAILING_ZONE_PRESETS: dict[str, dict[str, Any]] = {
    "goride_zona_1": {
        "label": "GoRide Zona I (Sumatera, Jawa non-Jabodetabek, Bali)",
        "min_distance_km": 4.0,
        "min_fare_idr": [7000, 10000],
        "per_km_idr": [1850, 2300],
        "app_fee_idr": [1500, 2000],
        "peak_multiplier": 1.20,
        "rain_surge_multiplier": 1.25,
    },
    "goride_zona_2": {
        "label": "GoRide Zona II (Jabodetabek)",
        "min_distance_km": 4.0,
        "min_fare_idr": [9000, 10500],
        "per_km_idr": [2250, 2700],
        "app_fee_idr": [1500, 2000],
        "peak_multiplier": 1.25,
        "rain_surge_multiplier": 1.25,
    },
    "goride_zona_3": {
        "label": "GoRide Zona III (Kalimantan, Sulawesi, Nusra, Maluku, Papua)",
        "min_distance_km": 4.0,
        "min_fare_idr": [10200, 11200],
        "per_km_idr": [2100, 2600],
        "app_fee_idr": [1500, 2000],
        "peak_multiplier": 1.20,
        "rain_surge_multiplier": 1.25,
    },
    "gocar": {
        "label": "GoCar (Mobil)",
        "min_distance_km": 1.0,
        "min_fare_idr": [10000, 15000],
        "per_km_idr": [3500, 6000],
        "app_fee_idr": [2000, 3000],
        "peak_multiplier": 1.25,
        "rain_surge_multiplier": 1.30,
    },
}


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


def compute_edge_fare_bounds(
    edge: Any,
    ojol_tariff: str,
    is_raining: bool,
) -> tuple[float, float]:
    """
    Menghitung batas bawah dan batas atas tarif perjalanan (IDR) pada sebuah edge
    berdasarkan model tarifnya:
    - free: Rp 0 (mis. jalan kaki)
    - manual_range: range harga langsung [min, max] normal/peak
    - per_km_custom: base_fare + distance_km * cost_per_km_idr
    - ride_hailing_zone: preset GoRide Zona I/II/III atau GoCar + override opsional user
    """
    pricing = edge.pricing
    model = pricing.pricing_model
    if model == "free":
        return 0.0, 0.0

    is_peak = ojol_tariff == "peak"

    if model == "manual_range":
        raw_range = pricing.cost_peak_idr if (is_peak and pricing.cost_peak_idr) else pricing.cost_normal_idr
        surge = float(pricing.rain_surge_multiplier) if is_raining else 1.0
        return (
            float(round(int(raw_range[0]) * surge)),
            float(round(int(raw_range[1]) * surge)),
        )

    if model == "per_km_custom":
        base_r = pricing.base_fare_idr or [0, 0]
        km_r = pricing.per_km_idr or [0, 0]
        dist = float(edge.distance_km)
        mult = (float(pricing.peak_multiplier) if is_peak else 1.0) * (
            float(pricing.rain_surge_multiplier) if is_raining else 1.0
        )
        low = (int(base_r[0]) + dist * int(km_r[0])) * mult
        high = (int(base_r[1]) + dist * int(km_r[1])) * mult
        return float(round(low)), float(round(high))

    if model == "ride_hailing_zone":
        preset_key = pricing.zone_preset or "goride_zona_1"
        preset = RIDE_HAILING_ZONE_PRESETS[preset_key]
        min_km = float(
            pricing.min_distance_km
            if pricing.min_distance_km is not None
            else preset["min_distance_km"]
        )
        min_fare = pricing.min_fare_idr or preset["min_fare_idr"]
        per_km = pricing.per_km_idr or preset["per_km_idr"]
        app_fee = pricing.app_fee_idr if pricing.app_fee_idr is not None else preset["app_fee_idr"]
        peak_mult = (
            float(pricing.peak_multiplier)
            if pricing.peak_multiplier is not None
            else float(preset["peak_multiplier"])
        )
        rain_mult = (
            float(pricing.rain_surge_multiplier)
            if pricing.rain_surge_multiplier is not None
            else float(preset["rain_surge_multiplier"])
        )

        extra_km = max(0.0, float(edge.distance_km) - min_km)
        dynamic_mult = (peak_mult if is_peak else 1.0) * (rain_mult if is_raining else 1.0)
        low = (int(min_fare[0]) + extra_km * int(per_km[0])) * dynamic_mult + int(app_fee[0])
        high = (int(min_fare[1]) + extra_km * int(per_km[1])) * dynamic_mult + int(app_fee[1])
        return float(round(low)), float(round(high))

    raise ValueError(f"Model tarif tidak dikenal: {model}")


def compute_edge_physical_deltas(
    edge: Any,
    walk_physics: Any,
    motorized_physics: Any,
    heat_sensitivity: float,
    climate: str,
    is_raining: bool,
) -> tuple[float, float]:
    """
    Menghitung (energy_delta, mood_delta) untuk melewati sebuah edge.
    Jika user tidak mengisi override eksplisit pada edge, nilai diturunkan
    secara proporsional dari `edge.distance_km` dan physics mode (`walk` vs motorized).
    """
    dist = float(edge.distance_km)
    is_walk = edge.mode == "walk"

    if is_raining:
        if edge.energy_cost_rain is not None:
            energy_cost = float(edge.energy_cost_rain)
        elif is_walk:
            energy_cost = dist * (
                float(walk_physics.energy_cost_per_km_cool)
                + float(walk_physics.rain_extra_energy_per_km)
            )
        else:
            energy_cost = float(
                edge.energy_cost_cool
                if edge.energy_cost_cool is not None
                else motorized_physics.default_energy_cost
            )

        if edge.mood_delta_rain is not None:
            mood_delta = float(edge.mood_delta_rain)
        elif is_walk:
            mood_delta = dist * (
                float(walk_physics.mood_delta_per_km_cool)
                + float(walk_physics.rain_extra_mood_per_km)
            )
        else:
            mood_delta = float(motorized_physics.mood_delta_rain)
        return -energy_cost, mood_delta

    if climate == "midday":
        if edge.energy_cost_midday is not None:
            energy_cost = float(edge.energy_cost_midday) * (heat_sensitivity if is_walk else 1.0)
        elif is_walk:
            energy_cost = dist * float(walk_physics.energy_cost_per_km_midday) * heat_sensitivity
        else:
            energy_cost = float(
                edge.energy_cost_cool
                if edge.energy_cost_cool is not None
                else motorized_physics.default_energy_cost
            )

        if edge.mood_delta_midday is not None:
            mood_delta = float(edge.mood_delta_midday) * heat_sensitivity
        elif is_walk:
            mood_delta = dist * float(walk_physics.mood_delta_per_km_midday) * heat_sensitivity
        else:
            mood_delta = float(motorized_physics.mood_delta_midday) * heat_sensitivity
        return -energy_cost, mood_delta

    # Cool / normal climate
    if edge.energy_cost_cool is not None:
        energy_cost = float(edge.energy_cost_cool)
    elif is_walk:
        energy_cost = dist * float(walk_physics.energy_cost_per_km_cool)
    else:
        energy_cost = float(motorized_physics.default_energy_cost)

    if edge.mood_delta_cool is not None:
        mood_delta = float(edge.mood_delta_cool)
    elif is_walk:
        mood_delta = dist * float(walk_physics.mood_delta_per_km_cool)
    else:
        mood_delta = float(motorized_physics.mood_delta_cool)

    return -energy_cost, mood_delta
