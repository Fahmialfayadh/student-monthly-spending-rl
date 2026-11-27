from __future__ import annotations

from collections import OrderedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import (
    configs_router,
    graph_router,
    runs_router,
    simulation_router,
    system_router,
    training_router,
)


def create_app() -> FastAPI:
    """Factory FastAPI yang mendaftarkan middleware dan seluruh router modular."""
    application = FastAPI(
        title="Student Survival Spatial-Graph RL API",
        version="2.0.0",
        description=(
            "Backend API untuk mendesain graf spasial lokasi, tarif transport berbasis jarak, "
            "validasi skema WorldConfig & TrainingProfileConfig, training PPO, dan sandbox simulasi."
        ),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.state.sim_sessions = OrderedDict()
    application.state.loaded_agents = {}
    application.state.training_jobs = {}

    application.include_router(system_router)
    application.include_router(configs_router)
    application.include_router(graph_router)
    application.include_router(simulation_router)
    application.include_router(training_router)
    application.include_router(runs_router)

    return application


app = create_app()
