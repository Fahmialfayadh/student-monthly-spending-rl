from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from src.api.dependencies import build_session_snapshot, resolve_world_config
from src.api.schemas import SimCreateRequest, SimStepRequest
from src.env import StudentLifeEnv

router = APIRouter(prefix="/sim", tags=["simulation"])


def _get_sessions(request: Request) -> dict[str, StudentLifeEnv]:
    sessions = getattr(request.app.state, "sim_sessions", None)
    if sessions is None:
        sessions = {}
        request.app.state.sim_sessions = sessions
    return sessions


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

    obs, info = env.reset(seed=req.seed)
    session_id = uuid4().hex[:12]
    _get_sessions(request)[session_id] = env
    return build_session_snapshot(session_id, env, obs, info)


@router.post("/sessions/{session_id}/step")
def step_sim_session(
    session_id: str, req: SimStepRequest, request: Request
) -> dict[str, Any]:
    """Mengeksekusi satu step aksi pada sesi sandbox aktif."""
    env = _get_sessions(request).get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' tidak ditemukan.")

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
