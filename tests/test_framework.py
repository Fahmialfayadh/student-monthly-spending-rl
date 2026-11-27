"""Comprehensive tests for Spatial-Graph StudentLifeEnv, pricing, agents, and FastAPI."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.agents import StudentPPOAgent
from src.api.app import app
from src.env import (
    RIDE_HAILING_ZONE_PRESETS,
    StudentLifeEnv,
    WorldConfig,
    WorldGraph,
    export_world_json_schema,
    load_world_config,
)
from src.env.pantry import PantryManager
from src.env.pricing import compute_edge_fare_bounds, compute_edge_physical_deltas
from src.env.schema import EdgePricingConfig, EdgeRouteConfig, MobilityAndClimateConfig


SURABAYA_YAML = "configs/presets/surabaya_its.yaml"
MINIMAL_YAML = "configs/presets/minimal_dorm.yaml"


def _make_edge(
    pricing: EdgePricingConfig,
    distance_km: float = 5.0,
    mode: str = "ojol",
) -> EdgeRouteConfig:
    return EdgeRouteConfig(
        id="test_edge",
        source="a",
        target="b",
        mode=mode,
        distance_km=distance_km,
        pricing=pricing,
    )


def test_schema_export_and_validation() -> None:
    schema = export_world_json_schema()
    assert "$defs" in schema
    for required_section in (
        "environment",
        "finance",
        "pantry_and_cooking",
        "biometrics_and_psychology",
        "mobility_and_climate",
        "world_graph",
        "activities",
        "external_foods",
        "academics",
        "termination",
        "rewards",
    ):
        assert required_section in schema["properties"]

    raw = load_world_config(SURABAYA_YAML).model_dump()
    bad_home = copy.deepcopy(raw)
    bad_home["world_graph"]["home_node_id"] = "cafe"  # cafe has can_sleep=False
    with pytest.raises(Exception):
        WorldConfig.model_validate(bad_home)

    bad_food = copy.deepcopy(raw)
    bad_food["world_graph"]["locations"][0]["food_options"].append("nonexistent_food")
    with pytest.raises(Exception):
        WorldConfig.model_validate(bad_food)


def test_distance_and_zone_pricing_models() -> None:
    # 1. Free walk
    walk_edge = _make_edge(EdgePricingConfig(pricing_model="free"), distance_km=1.5, mode="walk")
    assert compute_edge_fare_bounds(walk_edge, "normal", False) == (0.0, 0.0)
    mob = MobilityAndClimateConfig()
    e_walk, m_walk = compute_edge_physical_deltas(
        walk_edge,
        mob.walk_physics,
        mob.motorized_physics,
        mob.heat_sensitivity,
        "midday",
        False,
    )
    # 1.5 km * walk_physics defaults * heat_sensitivity (0.95)
    assert np.isclose(e_walk, -(1.5 * 10.0 * mob.heat_sensitivity))
    assert np.isclose(m_walk, 1.5 * -8.0 * mob.heat_sensitivity)

    # 2. Manual range (fixed transit / O-D fare)
    manual_edge = _make_edge(
        EdgePricingConfig(
            pricing_model="manual_range",
            cost_normal_idr=[4000, 6000],
            cost_peak_idr=[5000, 7000],
        ),
        distance_km=2.0,
        mode="public_transit",
    )
    assert compute_edge_fare_bounds(manual_edge, "normal", False) == (4000.0, 6000.0)
    assert compute_edge_fare_bounds(manual_edge, "peak", False) == (5000.0, 7000.0)

    # 3. Custom per-km (base + distance * per_km)
    custom_edge = _make_edge(
        EdgePricingConfig(
            pricing_model="per_km_custom",
            base_fare_idr=[5000, 7000],
            per_km_idr=[2000, 3000],
        ),
        distance_km=5.0,
    )
    # 5 km: min = 5000 + 5*2000 = 15000; max = 7000 + 5*3000 = 22000
    assert compute_edge_fare_bounds(custom_edge, "normal", False) == (15000.0, 22000.0)

    # 4. GoRide Zona I / II / III and GoCar presets
    assert set(RIDE_HAILING_ZONE_PRESETS.keys()) == {
        "goride_zona_1",
        "goride_zona_2",
        "goride_zona_3",
        "gocar",
    }
    zona1_edge = _make_edge(
        EdgePricingConfig(pricing_model="ride_hailing_zone", zone_preset="goride_zona_1"),
        distance_km=6.0,
        mode="goride",
    )
    f_min, f_max = compute_edge_fare_bounds(zona1_edge, "normal", False)
    # Zona 1: min [7000, 10000] + 2km * [1850, 2300] + app [1500, 2000] = [12200, 16600]
    assert np.isclose(f_min, 12200.0)
    assert np.isclose(f_max, 16600.0)

    gocar_edge = _make_edge(
        EdgePricingConfig(pricing_model="ride_hailing_zone", zone_preset="gocar"),
        distance_km=6.0,
        mode="gocar",
    )
    gc_min, gc_max = compute_edge_fare_bounds(gocar_edge, "normal", False)
    assert gc_min > f_min and gc_max > f_max

    # Peak + Rain multiplier increases fare
    f_rain_min, _ = compute_edge_fare_bounds(zona1_edge, "peak", True)
    assert f_rain_min > f_min


def test_surabaya_graph_routing_and_1_5km_walk_parity() -> None:
    env = StudentLifeEnv(SURABAYA_YAML)
    obs, _ = env.reset(seed=1)

    assert env.balance_idr == 1_465_000.0
    assert obs.shape == (56,)
    assert env.action_space.nvec.tolist() == [8, 2, 6, 18, 2, 15]

    # Dijkstra fastest vs cheapest from apartment -> department
    fastest = env.world_graph.find_best_path("apartment", "department", "fastest", "cool", "normal", False)
    cheapest = env.world_graph.find_best_path("apartment", "department", "cheapest", "cool", "normal", False)
    assert fastest is not None and cheapest is not None
    assert [e.id for e in fastest.edges] == ["direct_ojol"]
    assert [e.id for e in cheapest.edges] == ["transit_ccws_ojol", "walk_ccws_department_1_5km"]
    assert np.isclose(cheapest.total_distance_km, 3.7)
    assert np.isclose(cheapest.min_fare_idr, 7000.0)

    # Execute attend_class via cheapest from apartment at step 3 (Monday 13-17 has class in schedule)
    env.reset(seed=7)
    env.current_step = 3
    attend_idx = env.activity_names.index("attend_class")
    cheapest_idx = env.route_names.index("cheapest")
    _, _, _, _, info = env.step([attend_idx, cheapest_idx, 0, 0, 0, 0])
    events = info["step_events"]
    assert info["current_location"] == "department"
    assert events["executed_route"] == "transit_ccws_ojol->walk_ccws_department_1_5km"
    assert events["walked_edge"] == "walk_ccws_department_1_5km"
    assert np.isclose(events["travel_distance_km"], 3.7)


def test_cafe_as_real_node_capabilities_and_curfew() -> None:
    env = StudentLifeEnv(SURABAYA_YAML)
    env.reset(seed=42)
    cafe_node = env.world_graph.get_node("cafe")
    assert cafe_node.can_study is True
    assert cafe_node.closes_at_night is True
    assert cafe_node.leisure_mood_gain > 0
    assert cafe_node.study_profile.mood_delta > 0

    cafe_idx = env.activity_names.index("cafe_session")
    fastest_idx = env.route_names.index("fastest")
    _, _, _, _, info = env.step([cafe_idx, fastest_idx, 0, 0, 0, 0])
    assert info["current_location"] == "cafe"
    assert info["step_events"]["executed_activity"] == "cafe_session"
    assert info["step_events"]["cafe_spend_idr"] >= 25_000


def test_minimal_dorm_custom_topology_and_ppo_smoke() -> None:
    env = StudentLifeEnv(MINIMAL_YAML)
    assert env.observation_space.shape == (52,)
    assert env.action_space.nvec.tolist() == [7, 2, 6, 18, 2, 15]

    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        done = False
        while not done:
            mask = env.action_masks()
            action = []
            offset = 0
            for d in env.action_space.nvec:
                valid = np.flatnonzero(mask[offset : offset + d])
                assert len(valid) > 0
                action.append(int(rng.choice(valid)))
                offset += int(d)
            obs, reward, done, _, _ = env.step(action)
            assert np.all(np.isfinite(obs)) and np.isfinite(reward)

    with tempfile.TemporaryDirectory() as tmpdir:
        agent = StudentPPOAgent(
            yaml_path=MINIMAL_YAML,
            num_environments=2,
            features_dim=32,
            hidden_dim=32,
            policy_hidden_layers=[32],
            value_hidden_layers=[32],
            rollout_steps_per_env=64,
            batch_size=32,
            num_epochs=2,
            seed=42,
            verbose=0,
        )
        bc = agent.pretrain_bc(num_episodes=4, epochs=2, batch_size=64, verbose=False)
        assert bc["num_transitions"] > 0
        run_info = agent.train_with_logging(
            total_timesteps=256,
            eval_interval_steps=256,
            eval_episodes=2,
            final_eval_episodes=2,
            runs_root_dir=tmpdir,
            experiment_name="pytest_minimal",
        )
        reloaded = StudentPPOAgent.load_from_run(Path(run_info["run_dir"]), device="cpu")
        summary, ep_df, trace_df = reloaded.evaluate_detailed(
            num_episodes=2, eval_seed=999, record_first_episode_trace=True
        )
        assert np.isfinite(summary["mean_episode_reward"])
        assert len(trace_df) > 0
        assert "library_study_count" in ep_df.columns
        agent.training_env.close()
        reloaded.training_env.close()


def test_fastapi_endpoints_and_sandbox_flow() -> None:
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_schema = client.get("/schema")
    assert r_schema.status_code == 200
    assert "ride_hailing_zone_presets" in r_schema.json()
    assert "goride_zona_1" in r_schema.json()["ride_hailing_zone_presets"]

    r_presets = client.get("/presets")
    assert r_presets.status_code == 200
    names = {item["name"] for item in r_presets.json()["presets"]}
    assert {"surabaya_its", "minimal_dorm"}.issubset(names)

    r_cfg = client.get("/presets/surabaya_its")
    assert r_cfg.status_code == 200
    cfg_payload = r_cfg.json()["config"]

    r_val = client.post("/config/validate", json=cfg_payload)
    assert r_val.status_code == 200
    assert r_val.json()["valid"] is True
    assert r_val.json()["observation_dim"] == 56

    bad_payload = copy.deepcopy(cfg_payload)
    bad_payload["world_graph"]["home_node_id"] = "missing_node"
    r_bad = client.post("/config/validate", json=bad_payload)
    assert r_bad.status_code == 422

    r_route = client.post(
        "/graph/preview-route",
        json={
            "preset_name": "surabaya_its",
            "source_node_id": "apartment",
            "target_node_id": "cafe",
            "preference": "fastest",
            "climate": "midday",
            "ojol_tariff": "normal",
            "is_raining": False,
        },
    )
    assert r_route.status_code == 200
    assert r_route.json()["edge_ids"] == ["ojol_apartment_cafe"]
    assert r_route.json()["min_fare_idr"] == 8500.0

    r_sess = client.post("/sim/sessions", json={"preset_name": "minimal_dorm", "seed": 11})
    assert r_sess.status_code == 200
    sess_data = r_sess.json()
    sid = sess_data["session_id"]

    # Pick first valid masked action across 6 heads
    masks = sess_data["action_masks"]
    action = [
        masks["activity"].index(True),
        masks["route"].index(True),
        masks["meal_mode"].index(True),
        masks["cook_combo"].index(True),
        masks["bring_bekal"].index(True),
        masks["grocery_bundle"].index(True),
    ]
    r_step = client.post(f"/sim/sessions/{sid}/step", json={"action": action})
    assert r_step.status_code == 200
    assert "reward" in r_step.json()

    r_del = client.delete(f"/sim/sessions/{sid}")
    assert r_del.status_code == 200
    assert r_del.json()["deleted"] is True


def test_local_config_save_load_sync_and_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api.dependencies as api_deps
    import src.api.routes.configs as cfg_routes

    custom_dir = tmp_path / "custom_configs"
    active_cfg = tmp_path / "env_config.yaml"
    monkeypatch.setattr(api_deps, "CUSTOM_CONFIGS_DIR", custom_dir)
    monkeypatch.setattr(api_deps, "ACTIVE_ENV_CONFIG_PATH", active_cfg)
    monkeypatch.setattr(cfg_routes, "CUSTOM_CONFIGS_DIR", custom_dir)
    monkeypatch.setattr(cfg_routes, "ACTIVE_ENV_CONFIG_PATH", active_cfg)
    monkeypatch.setattr(cfg_routes, "ROOT_DIR", tmp_path)

    client = TestClient(app)
    base_cfg = client.get("/presets/minimal_dorm").json()["config"]
    edited = copy.deepcopy(base_cfg)
    edited["finance"]["monthly_budget"] = 1_800_000.0

    # 1. Save named local custom config + sync to active env_config.yaml
    r_save = client.post(
        "/configs/save",
        json={
            "name": "campus_experiment_v1",
            "config": edited,
            "set_active_env_config": True,
        },
    )
    assert r_save.status_code == 200
    save_body = r_save.json()
    assert save_body["saved"] is True
    assert (custom_dir / "campus_experiment_v1.yaml").exists()
    assert active_cfg.exists()
    sha1 = save_body["sha256"]

    # 2. List & read back from local YAML
    r_list = client.get("/configs")
    assert r_list.status_code == 200
    listed_scopes = {(c["scope"], c["name"]) for c in r_list.json()["configs"]}
    assert ("custom", "campus_experiment_v1") in listed_scopes
    assert ("active", "env_config") in listed_scopes

    r_get = client.get("/configs/custom/campus_experiment_v1")
    assert r_get.status_code == 200
    assert r_get.json()["sha256"] == sha1
    assert r_get.json()["config"]["finance"]["monthly_budget"] == 1_800_000.0

    # 3. Optimistic sha256 check catches external file edit conflict (409)
    edited2 = copy.deepcopy(edited)
    edited2["finance"]["monthly_budget"] = 1_900_000.0
    r_conflict = client.post(
        "/configs/save",
        json={
            "name": "campus_experiment_v1",
            "config": edited2,
            "expected_sha256": "stale_hash",
        },
    )
    assert r_conflict.status_code == 409

    # 4. Run sandbox simulation and route preview directly from saved custom config
    r_route = client.post(
        "/graph/preview-route",
        json={
            "custom_config_name": "campus_experiment_v1",
            "source_node_id": "dorm",
            "target_node_id": "campus_library",
            "preference": "cheapest",
        },
    )
    assert r_route.status_code == 200
    assert r_route.json()["edge_ids"] == ["walk_dorm_campus"]

    r_sess = client.post(
        "/sim/sessions",
        json={"custom_config_name": "campus_experiment_v1", "seed": 5},
    )
    assert r_sess.status_code == 200
    sid = r_sess.json()["session_id"]
    client.delete(f"/sim/sessions/{sid}")

    # 5. Import raw YAML text & delete cleanly
    yaml_text = r_get.json()["yaml_text"]
    r_imp = client.post(
        "/configs/import-yaml",
        json={"name": "imported_copy", "yaml_content": yaml_text},
    )
    assert r_imp.status_code == 200
    assert (custom_dir / "imported_copy.yaml").exists()

    assert client.delete("/configs/custom/campus_experiment_v1").status_code == 200
    assert client.delete("/configs/custom/imported_copy").status_code == 200
    assert not (custom_dir / "campus_experiment_v1.yaml").exists()


def test_user_walk_preference_and_distance_limit() -> None:
    raw = load_world_config(SURABAYA_YAML).model_dump()

    # 1. User limits walking to 1.0 km -> 1.5 km walk edge is skipped; cheapest falls back to direct motorized
    limited = copy.deepcopy(raw)
    limited["mobility_and_climate"]["walk_physics"]["max_walk_distance_km"] = 1.0
    graph_limited = WorldGraph(WorldConfig.model_validate(limited))
    p_limited = graph_limited.find_best_path(
        "apartment", "department", preference="cheapest", climate="cool", ojol_tariff="normal", is_raining=False
    )
    assert p_limited is not None
    assert [e.id for e in p_limited.edges] == ["direct_ojol"]
    assert p_limited.walk_distance_km == 0.0

    # 2. User allows 2.0 km when dry, but max 0.5 km when raining -> rain blocks 1.5 km walk
    rain_limited = copy.deepcopy(raw)
    rain_limited["mobility_and_climate"]["walk_physics"]["max_walk_distance_km"] = 2.0
    rain_limited["mobility_and_climate"]["walk_physics"]["max_walk_distance_rain_km"] = 0.5
    graph_rain = WorldGraph(WorldConfig.model_validate(rain_limited))
    p_dry = graph_rain.find_best_path(
        "apartment", "department", preference="cheapest", is_raining=False
    )
    p_wet = graph_rain.find_best_path(
        "apartment", "department", preference="cheapest", is_raining=True
    )
    assert p_dry is not None and [e.id for e in p_dry.edges] == [
        "transit_ccws_ojol",
        "walk_ccws_department_1_5km",
    ]
    assert p_wet is not None and [e.id for e in p_wet.edges] == ["direct_ojol"]

    # 3. minimal_dorm: prefer_walk vs avoid_walk changes route even on same preference
    dorm_raw = load_world_config(MINIMAL_YAML).model_dump()
    prefer_cfg = copy.deepcopy(dorm_raw)
    prefer_cfg["mobility_and_climate"]["walk_physics"]["walk_preference"] = "prefer_walk"
    prefer_cfg["mobility_and_climate"]["walk_physics"]["prefer_walk_score_multiplier"] = 0.05
    graph_pref = WorldGraph(WorldConfig.model_validate(prefer_cfg))
    p_pref_fast = graph_pref.find_best_path("dorm", "campus_library", preference="fastest")
    assert p_pref_fast is not None and [e.id for e in p_pref_fast.edges] == ["walk_dorm_campus"]

    avoid_cfg = copy.deepcopy(dorm_raw)
    avoid_cfg["mobility_and_climate"]["walk_physics"]["walk_preference"] = "avoid_walk"
    graph_avoid = WorldGraph(WorldConfig.model_validate(avoid_cfg))
    p_avoid_cheap = graph_avoid.find_best_path("dorm", "campus_library", preference="cheapest")
    assert p_avoid_cheap is not None and [e.id for e in p_avoid_cheap.edges] == ["bus_dorm_campus"]


def test_typed_schema_training_api_runs_and_autostep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time
    import src.api.routes.runs as runs_routes
    import src.api.routes.simulation as sim_routes
    import src.api.routes.training as train_routes

    monkeypatch.setattr(runs_routes, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(sim_routes, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(train_routes, "ROOT_DIR", tmp_path)

    client = TestClient(app)

    # 1. Verify /schema exposes training_json_schema and typed sub-models
    schema_resp = client.get("/schema").json()
    assert "training_json_schema" in schema_resp
    assert "default_training_profile" in schema_resp
    assert "walk_preferences" in schema_resp
    world_defs = schema_resp["json_schema"]["$defs"]
    for typed_submodel in (
        "StepDynamicsConfig",
        "CookingCostsAndGatesConfig",
        "WellbeingWeightsConfig",
        "BudgetPotentialShapingConfig",
        "WeeklyScheduleConfig",
    ):
        assert typed_submodel in world_defs

    # Missing 1 of 4 standard pantry ingredients must fail schema validation (422)
    bad_pantry = client.get("/presets/minimal_dorm").json()["config"]
    del bad_pantry["pantry_and_cooking"]["ingredients"]["tempe"]
    assert client.post("/config/validate", json=bad_pantry).status_code == 422

    # 2. Training validation: matching gamma passes, mismatched gamma fails 422
    r_ok = client.post(
        "/training/validate",
        json={"preset_name": "minimal_dorm", "training_profile": {}},
    )
    assert r_ok.status_code == 200
    assert r_ok.json()["valid"] is True

    r_bad_gamma = client.post(
        "/training/validate",
        json={
            "preset_name": "minimal_dorm",
            "training_profile": {"ppo": {"discount_factor_gamma": 0.90}},
        },
    )
    assert r_bad_gamma.status_code == 422

    # 3. Run mini background training job, poll until done, inspect /runs, and auto-step with model
    mini_profile = {
        "experiment_name": "api_smoke",
        "runs_root_dir": "runs",
        "ppo": {
            "num_environments": 2,
            "features_dim": 32,
            "hidden_dim": 32,
            "policy_hidden_layers": [32],
            "value_hidden_layers": [32],
            "rollout_steps_per_env": 64,
            "batch_size": 32,
            "num_epochs": 2,
            "discount_factor_gamma": 0.995,
            "device": "cpu",
        },
        "bc": {"enabled": True, "num_episodes": 2, "epochs": 2, "batch_size": 64},
        "schedule": {
            "total_timesteps": 128,
            "eval_interval_steps": 128,
            "eval_episodes": 1,
            "final_eval_episodes": 1,
            "survival_phase_timesteps": 64,
        },
    }
    r_start = client.post(
        "/training/start",
        json={"preset_name": "minimal_dorm", "training_profile": mini_profile},
    )
    assert r_start.status_code == 200
    job_id = r_start.json()["job_id"]

    job_data = None
    for _ in range(60):
        job_data = client.get(f"/training/jobs/{job_id}").json()
        if job_data["status"] in ("completed", "failed"):
            break
        time.sleep(0.25)
    assert job_data is not None and job_data["status"] == "completed", job_data

    runs_list = client.get("/runs").json()["runs"]
    assert len(runs_list) == 1
    run_id = runs_list[0]["run_id"]
    run_detail = client.get(f"/runs/{run_id}").json()
    assert run_detail["final_summary"] is not None
    run_trace = client.get(f"/runs/{run_id}/trace").json()
    assert len(run_trace["steps"]) > 0

    # 4. Simulation session GET, reset, heuristic auto-step, and trained model auto-step
    sid = client.post("/sim/sessions", json={"preset_name": "minimal_dorm", "seed": 7}).json()[
        "session_id"
    ]
    assert client.get(f"/sim/sessions/{sid}").status_code == 200
    r_h = client.post(f"/sim/sessions/{sid}/auto-step", json={"mode": "heuristic"})
    assert r_h.status_code == 200 and len(r_h.json()["action_taken"]) == 6

    r_m = client.post(
        f"/sim/sessions/{sid}/auto-step",
        json={"mode": "model", "run_id": run_id, "checkpoint": "best"},
    )
    assert r_m.status_code == 200 and len(r_m.json()["action_taken"]) == 6

    r_rst = client.post(f"/sim/sessions/{sid}/reset", json={"seed": 99})
    assert r_rst.status_code == 200 and r_rst.json()["graph_state"]["current_step"] == 0
    client.delete(f"/sim/sessions/{sid}")
