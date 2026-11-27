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


def _get_stop_events(request: Request) -> dict[str, threading.Event]:
    events = getattr(request.app.state, "training_stop_events", None)
    if events is None:
        events = {}
        request.app.state.training_stop_events = events
    return events


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
    """Menjalankan job training RL (BC opsional + MaskablePPO) di background dengan progress & stop support."""
    profile = _parse_training_profile(req.training_profile)
    world_cfg = resolve_world_config(req.config, req.preset_name, req.custom_config_name)
    try:
        warnings = validate_training_against_world(profile, world_cfg)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job_id = uuid4().hex[:12]
    jobs = _get_jobs(request)
    stop_events = _get_stop_events(request)
    stop_event = threading.Event()
    stop_events[job_id] = stop_event

    total_ts = int(profile.schedule.total_timesteps)
    job_record: dict[str, Any] = {
        "job_id": job_id,
        "status": "running",
        "phase": "bc_pretrain" if profile.bc.enabled else "survive",
        "current_step": 0,
        "total_timesteps": total_ts,
        "progress_pct": 0.0,
        "stop_requested": False,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "experiment_name": profile.experiment_name,
        "warnings": warnings,
        "logs": [],
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

    def _on_log(message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        job_record["logs"].append(f"[{stamp}] {message}")
        if len(job_record["logs"]) > 300:
            del job_record["logs"][: len(job_record["logs"]) - 300]

    def _on_progress(current_step: int, total_steps: int, phase: str) -> None:
        job_record["current_step"] = int(current_step)
        job_record["total_timesteps"] = int(total_steps)
        job_record["progress_pct"] = round(min(100.0, (current_step / max(1, total_steps)) * 100.0), 1)
        job_record["phase"] = phase

    def _run_job() -> None:
        try:
            _on_log(
                f"Job {job_id} dimulai ({profile.experiment_name} | "
                f"{total_ts:,} steps | envs={profile.ppo.num_environments} | device={profile.ppo.device})."
            )
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
                if profile.bc.enabled and not stop_event.is_set():
                    job_record["phase"] = "bc_pretrain"
                    bc_res = agent.pretrain_bc(
                        num_episodes=profile.bc.num_episodes,
                        epochs=profile.bc.epochs,
                        batch_size=profile.bc.batch_size,
                        learning_rate=profile.bc.learning_rate,
                        entropy_coef=profile.bc.entropy_coef,
                        vf_coef=profile.bc.vf_coef,
                        demo_seed=profile.bc.demo_seed,
                        verbose=False,
                        log_callback=_on_log,
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
                    stop_event=stop_event,
                    progress_callback=_on_progress,
                    log_callback=_on_log,
                )
                agent.training_env.close()
                job_record["run_dir"] = str(run_res["run_dir"])
                job_record["final_summary"] = run_res["final_summary"]
                fs = run_res["final_summary"]
                _on_log(
                    f"[Selesai] Complete={fs['completion_rate'] * 100:.0f}% | "
                    f"Hari={fs['mean_days_survived']:.1f} | "
                    f"Reward={fs['mean_episode_reward']:.2f} | "
                    f"Sisa=Rp {fs['mean_final_balance_idr']:,.0f} | "
                    f"Disimpan di {Path(run_res['run_dir']).name}"
                )
                if stop_event.is_set():
                    job_record["status"] = "stopped"
                else:
                    job_record["current_step"] = total_ts
                    job_record["progress_pct"] = 100.0
                    job_record["status"] = "completed"
        except Exception as exc:
            _on_log(f"[ERROR] {exc}")
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


@router.post("/jobs/{job_id}/stop")
def stop_training_job(job_id: str, request: Request) -> dict[str, Any]:
    """Mengirim sinyal berhenti aman ke job training RL yang sedang berjalan."""
    job = _get_jobs(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' tidak ditemukan.")
    ev = _get_stop_events(request).get(job_id)
    if ev is not None and job["status"] == "running":
        ev.set()
        job["stop_requested"] = True
        stamp = datetime.now().strftime("%H:%M:%S")
        job.setdefault("logs", []).append(
            f"[{stamp}] [Stop] Permintaan berhenti diterima; menyimpan checkpoint aman..."
        )
    return job
