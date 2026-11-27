from __future__ import annotations

from collections import OrderedDict
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from src.agents import StudentPPOAgent
from src.agents.pretrain import default_balanced_expert_policy
from src.api.dependencies import ROOT_DIR, build_session_snapshot, resolve_world_config
from src.api.schemas import (
    SimAutoStepRequest,
    SimCreateRequest,
    SimResetRequest,
    SimStepRequest,
)
from src.env import StudentLifeEnv

router = APIRouter(prefix="/sim", tags=["simulation"])

MAX_SIM_SESSIONS = 32


def _get_sessions(request: Request) -> OrderedDict[str, StudentLifeEnv]:
    sessions = getattr(request.app.state, "sim_sessions", None)
    if not isinstance(sessions, OrderedDict):
        sessions = OrderedDict()
        request.app.state.sim_sessions = sessions
    return sessions


def _get_agent_cache(request: Request) -> dict[tuple[str, str], StudentPPOAgent]:
    cache = getattr(request.app.state, "loaded_agents", None)
    if cache is None:
        cache = {}
        request.app.state.loaded_agents = cache
    return cache


def _require_session(request: Request, session_id: str) -> StudentLifeEnv:
    sessions = _get_sessions(request)
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' tidak ditemukan.")
    sessions.move_to_end(session_id)
    return env


def _build_initial_info(env: StudentLifeEnv) -> dict[str, Any]:
    return {
        "step": env.current_step,
        "current_location": env.current_node_id,
        "balance_idr": float(env.balance_idr),
        "pantry_totals": env.pantry.get_stock_summary(),
        "step_events": {},
    }


@router.post("/sessions")
def create_sim_session(req: SimCreateRequest, request: Request) -> dict[str, Any]:
    """Membuat sesi sandbox StudentLifeEnv baru dari preset, custom config lokal, atau payload JSON."""
    cfg = resolve_world_config(req.config, req.preset_name, req.custom_config_name)
    try:
        env = StudentLifeEnv(
            yaml_path=cfg.model_dump(),
            exploring_starts_pantry=req.exploring_starts_pantry,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    sessions = _get_sessions(request)
    while len(sessions) >= MAX_SIM_SESSIONS:
        _, oldest_env = sessions.popitem(last=False)
        oldest_env.close()

    obs, info = env.reset(seed=req.seed)
    session_id = uuid4().hex[:12]
    sessions[session_id] = env
    return build_session_snapshot(session_id, env, obs, info)


@router.get("/sessions/{session_id}")
def get_sim_session(session_id: str, request: Request) -> dict[str, Any]:
    """Mengambil state terkini dari sesi sandbox aktif tanpa melangkah."""
    env = _require_session(request, session_id)
    obs = env._get_observation()
    return build_session_snapshot(session_id, env, obs, _build_initial_info(env))


@router.post("/sessions/{session_id}/reset")
def reset_sim_session(
    session_id: str,
    request: Request,
    req: SimResetRequest | None = None,
) -> dict[str, Any]:
    """Mereset sesi sandbox aktif ke step 0 dengan seed opsional."""
    env = _require_session(request, session_id)
    obs, info = env.reset(seed=req.seed if req else None)
    return build_session_snapshot(session_id, env, obs, info)


@router.post("/sessions/{session_id}/step")
def step_sim_session(
    session_id: str, req: SimStepRequest, request: Request
) -> dict[str, Any]:
    """Mengeksekusi satu step aksi manual pada sesi sandbox aktif."""
    env = _require_session(request, session_id)

    nvec = [int(x) for x in env.action_space.nvec.tolist()]
    for head_idx, (act_val, limit) in enumerate(zip(req.action, nvec)):
        if act_val < 0 or act_val >= limit:
            raise HTTPException(
                status_code=422,
                detail=f"Action head {head_idx} di luar rentang [0, {limit - 1}].",
            )

    if req.validate_mask:
        masks = env.action_masks()
        cursor = 0
        for head_idx, (act_val, limit) in enumerate(zip(req.action, nvec)):
            if not bool(masks[cursor + act_val]):
                raise HTTPException(
                    status_code=422,
                    detail=f"Action head {head_idx}={act_val} diblokir oleh action mask.",
                )
            cursor += limit

    obs, reward, terminated, truncated, info = env.step(req.action)
    snapshot = build_session_snapshot(session_id, env, obs, info)
    snapshot.update(
        {
            "action_taken": list(req.action),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
    )
    return snapshot


@router.post("/sessions/{session_id}/auto-step")
def auto_step_sim_session(
    session_id: str, req: SimAutoStepRequest, request: Request
) -> dict[str, Any]:
    """Mengeksekusi 1 step otomatis memakai heuristic expert atau checkpoint model terlatih."""
    env = _require_session(request, session_id)

    if req.mode == "heuristic":
        action_arr = default_balanced_expert_policy(env)
    else:
        if not req.run_id:
            raise HTTPException(
                status_code=422,
                detail="run_id wajib diisi saat mode='model'.",
            )
        run_dir = ROOT_DIR / "runs" / req.run_id
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run '{req.run_id}' tidak ditemukan.")
        cache_key = (req.run_id, req.checkpoint)
        agent_cache = _get_agent_cache(request)
        agent = agent_cache.get(cache_key)
        if agent is None:
            agent = StudentPPOAgent.load_from_run(
                run_dir, checkpoint=req.checkpoint, device="cpu"
            )
            agent_cache[cache_key] = agent

        raw_obs = env._get_observation()
        norm_obs = agent.training_env.normalize_obs(raw_obs[None, :])[0]
        masks = env.action_masks() if agent.use_action_masking else None
        action_arr, _ = agent.predict(
            normalized_observation=norm_obs,
            deterministic=req.deterministic,
            action_masks=masks,
        )

    action_list = [int(x) for x in action_arr.tolist()]
    obs, reward, terminated, truncated, info = env.step(action_list)
    snapshot = build_session_snapshot(session_id, env, obs, info)
    snapshot.update(
        {
            "action_taken": action_list,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
    )
    return snapshot


@router.delete("/sessions/{session_id}")
def delete_sim_session(session_id: str, request: Request) -> dict[str, Any]:
    """Menutup dan menghapus sesi sandbox aktif."""
    env = _get_sessions(request).pop(session_id, None)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' tidak ditemukan.")
    env.close()
    return {"deleted": True, "session_id": session_id}
