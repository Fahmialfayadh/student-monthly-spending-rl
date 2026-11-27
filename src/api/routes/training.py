from __future__ import annotations

import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from src.agents import StudentPPOAgent
from src.api.dependencies import ROOT_DIR, resolve_world_config, validation_error_http
from src.api.schemas import TrainingStartRequest, TrainingValidateRequest
from src.env import TrainingProfileConfig, validate_training_against_world

router = APIRouter(prefix="/training", tags=["training"])


def _get_jobs(request: Request) -> dict[str, dict[str, Any]]:
    jobs = getattr(request.app.state, "training_jobs", None)
    if jobs is None:
        jobs = {}
        request.app.state.training_jobs = jobs
    return jobs


def _parse_training_profile(payload: dict[str, Any]) -> TrainingProfileConfig:
    try:
        return TrainingProfileConfig.model_validate(payload)
    except ValidationError as exc:
        raise validation_error_http(exc) from exc


@router.post("/validate")
def validate_training(req: TrainingValidateRequest) -> dict[str, Any]:
    """Memvalidasi hyperparameter TrainingProfileConfig dan kesesuaian gamma dengan WorldConfig."""
    profile = _parse_training_profile(req.training_profile)
    world_cfg = resolve_world_config(req.config, req.preset_name, req.custom_config_name)
    try:
        warnings = validate_training_against_world(profile, world_cfg)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "valid": True,
        "warnings": warnings,
        "training_profile": profile.model_dump(mode="json"),
    }


@router.post("/start")
def start_training(req: TrainingStartRequest, request: Request) -> dict[str, Any]:
    """Menjalankan job training RL (BC opsional + MaskablePPO) secara asynchronous di background."""
    profile = _parse_training_profile(req.training_profile)
    world_cfg = resolve_world_config(req.config, req.preset_name, req.custom_config_name)
    try:
        warnings = validate_training_against_world(profile, world_cfg)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job_id = uuid4().hex[:12]
    jobs = _get_jobs(request)
    job_record: dict[str, Any] = {
        "job_id": job_id,
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "experiment_name": profile.experiment_name,
        "warnings": warnings,
        "run_dir": None,
        "bc_summary": None,
        "final_summary": None,
        "error": None,
    }
    jobs[job_id] = job_record

    runs_root = (
        Path(profile.runs_root_dir)
        if Path(profile.runs_root_dir).is_absolute()
        else ROOT_DIR / profile.runs_root_dir
    )

    def _run_job() -> None:
        try:
            with tempfile.TemporaryDirectory() as tmp_cfg_dir:
                cfg_file = Path(tmp_cfg_dir) / "world_config.yaml"
                cfg_file.write_text(
                    yaml.safe_dump(world_cfg.model_dump(mode="json"), sort_keys=False),
                    encoding="utf-8",
                )
                ppo = profile.ppo
                agent = StudentPPOAgent(
                    yaml_path=str(cfg_file),
                    num_environments=ppo.num_environments,
                    features_dim=ppo.features_dim,
                    hidden_dim=ppo.hidden_dim,
                    policy_hidden_layers=ppo.policy_hidden_layers,
                    value_hidden_layers=ppo.value_hidden_layers,
                    learning_rate=ppo.learning_rate,
                    rollout_steps_per_env=ppo.rollout_steps_per_env,
                    batch_size=ppo.batch_size,
                    num_epochs=ppo.num_epochs,
                    discount_factor_gamma=ppo.discount_factor_gamma,
                    gae_lambda=ppo.gae_lambda,
                    clip_range=ppo.clip_range,
                    entropy_coefficient=ppo.entropy_coefficient,
                    target_kl=ppo.target_kl,
                    use_action_masking=ppo.use_action_masking,
                    share_features_extractor=ppo.share_features_extractor,
                    seed=ppo.seed,
                    device=ppo.device,
                    verbose=0,
                )
                if profile.bc.enabled:
                    bc_res = agent.pretrain_bc(
                        num_episodes=profile.bc.num_episodes,
                        epochs=profile.bc.epochs,
                        batch_size=profile.bc.batch_size,
                        learning_rate=profile.bc.learning_rate,
                        entropy_coef=profile.bc.entropy_coef,
                        vf_coef=profile.bc.vf_coef,
                        demo_seed=profile.bc.demo_seed,
                        verbose=False,
                    )
                    job_record["bc_summary"] = {
                        k: v for k, v in bc_res.items() if k != "history"
                    }

                sched = profile.schedule
                run_res = agent.train_with_logging(
                    total_timesteps=sched.total_timesteps,
                    eval_interval_steps=sched.eval_interval_steps,
                    eval_episodes=sched.eval_episodes,
                    final_eval_episodes=sched.final_eval_episodes,
                    eval_seed=sched.eval_seed,
                    survival_phase_timesteps=sched.survival_phase_timesteps,
                    experiment_name=profile.experiment_name,
                    runs_root_dir=str(runs_root),
                    progress_bar=False,
                )
                agent.training_env.close()
                job_record["run_dir"] = str(run_res["run_dir"])
                job_record["final_summary"] = run_res["final_summary"]
                job_record["status"] = "completed"
        except Exception as exc:
            job_record["error"] = str(exc)
            job_record["status"] = "failed"
        finally:
            job_record["finished_at"] = datetime.now().isoformat(timespec="seconds")

    thread = threading.Thread(target=_run_job, daemon=True)
    thread.start()
    return job_record


@router.get("/jobs")
def list_training_jobs(request: Request) -> dict[str, list[dict[str, Any]]]:
    """Menampilkan daftar seluruh background training job pada sesi server ini."""
    return {"jobs": list(_get_jobs(request).values())}


@router.get("/jobs/{job_id}")
def get_training_job(job_id: str, request: Request) -> dict[str, Any]:
    """Mengambil status terkini sebuah background training job."""
    job = _get_jobs(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' tidak ditemukan.")
    return job
