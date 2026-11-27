from src.api.routes.configs import router as configs_router
from src.api.routes.graph import router as graph_router
from src.api.routes.runs import router as runs_router
from src.api.routes.simulation import router as simulation_router
from src.api.routes.system import router as system_router
from src.api.routes.training import router as training_router

__all__ = [
    "configs_router",
    "graph_router",
    "runs_router",
    "simulation_router",
    "system_router",
    "training_router",
]
