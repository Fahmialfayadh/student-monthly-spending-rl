from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    if frontend_dir.exists():
        for subdir in ("css", "js", "partials"):
            subpath = frontend_dir / subdir
            if subpath.exists():
                application.mount(f"/{subdir}", StaticFiles(directory=str(subpath)), name=subdir)

        @application.get("/", include_in_schema=False)
        def serve_index() -> FileResponse:
            return FileResponse(str(frontend_dir / "index.html"))

    return application


app = create_app()
