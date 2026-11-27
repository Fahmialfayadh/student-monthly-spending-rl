from src.api.routes.configs import router as configs_router
from src.api.routes.graph import router as graph_router
from src.api.routes.simulation import router as simulation_router
from src.api.routes.system import router as system_router

__all__ = [
    "configs_router",
    "graph_router",
    "simulation_router",
    "system_router",
]
