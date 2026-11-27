# Student Survival Spatial-Graph RL Framework

A configurable Reinforcement Learning (`MaskablePPO` + Behavior Cloning) environment and FastAPI backend for simulating student monthly budgeting, spatial commuting decisions, pantry/cooking management, biometrics/stress dynamics, and academic schedules.

## Architecture Overview

Every location, route, activity, external food option, and transport pricing rule is defined in a validated `WorldConfig` (`schema_version: "2.0"`) rather than hardcoded string checks:

- **Spatial Graph & Dijkstra Routing (`src/env/world_graph.py`)**:
  Locations (`LocationNodeConfig`) define capabilities (`can_sleep`, `has_kitchen`, `has_grocery`, `is_class_venue`, `can_study`, `closes_at_night`), ambient effects (`ambient_mood_delta`, `ambient_stress_delta`, `craving_gain_per_step`), `study_profile`, leisure parameters, and `ui_position`. `WorldGraph` computes optimal paths for `route` preferences `"fastest"` and `"cheapest"`.
- **Distance & Transport Pricing (`src/env/pricing.py`)**:
  Edges carry `distance_km` (driving walk fatigue/mood physics via `walk_physics`) and support 4 pricing models:
  - `free` (e.g. walking, such as the `1.5 km` transit walk)
  - `manual_range` (fixed A→B public transit or user-specified min/max fare range)
  - `per_km_custom` (custom base fare + per-km rate + app fee)
  - `ride_hailing_zone` (GoRide Zona I / II / III & GoCar tariff presets with editable overrides and rush/rain surge)
- **First-Class `cafe` Node & Handler-Driven Actions (`src/env/actions.py`, `src/env/biometrics.py`, `src/env/lifeenv.py`)**:
  Locations such as `cafe` are real graph nodes connected by real edges, supporting study profiles, craving relief, buff durations, mood/stress effects, and curfew rules.
- **RL Training & Evaluation (`src/agents/`)**:
  `StudentPPOAgent`, `BehaviorCloningPretrainer`, `StudentEvaluator`, and `ExperimentTrainer` read observations, action spaces, and activity handlers dynamically so training works on any valid graph config.
- **FastAPI Backend (`src/api/app.py`)**:
  Exposes JSON Schema generation, preset listing, graph validation, route/fare previewing, and interactive step-by-step sandbox sessions for UI graph builders.

## Quickstart

### 1. Run FastAPI Server

```bash
PYTHONPATH=. /cache/ml/.venv/bin/uvicorn src.api.app:app --reload --port 8000
```

Key endpoints:
- `GET /health` — health check
- `GET /schema` — full `WorldConfig` JSON Schema + GoRide/GoCar zone presets & handler enums
- `GET /presets` & `GET /presets/{name}` — list or load YAML presets (`surabaya_its`, `minimal_dorm`)
- `POST /config/validate` — validate custom graph/config JSON and inspect resulting observation/action dimensions
- `POST /graph/preview-route` — preview `"fastest"` / `"cheapest"` Dijkstra path, `distance_km`, IDR fare range, and energy/mood cost
- `POST /sim/sessions`, `POST /sim/sessions/{id}/step`, `DELETE /sim/sessions/{id}` — interactive environment sandbox

### 2. Run Tests & Verification

```bash
PYTHONPATH=. /cache/ml/.venv/bin/pytest -v tests/test_framework.py
PYTHONPATH=. /cache/ml/.venv/bin/python scripts/tmp/quick_check.py
```

### 3. Run RL Training

- Presets live in `configs/presets/surabaya_its.yaml` (mirrored to `env_config.yaml`) and `configs/presets/minimal_dorm.yaml`.
- Execute `main.ipynb` or use `StudentPPOAgent` directly with any preset or custom YAML path.
