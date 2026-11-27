from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import configs_router, graph_router, simulation_router, system_router


def create_app() -> FastAPI:
    """Factory FastAPI yang mendaftarkan middleware dan seluruh router modular."""
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
    application.state.sim_sessions = {}

    application.include_router(system_router)
    application.include_router(configs_router)
    application.include_router(graph_router)
    application.include_router(simulation_router)

    return application


app = create_app()
