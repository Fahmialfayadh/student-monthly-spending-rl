from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.dependencies import ROOT_DIR

router = APIRouter(prefix="/runs", tags=["runs"])


def _runs_root() -> Path:
    return ROOT_DIR / "runs"


@router.get("")
def list_runs() -> dict[str, list[dict[str, Any]]]:
    """Menampilkan daftar seluruh experiment run di direktori runs/."""
    root = _runs_root()
    if not root.exists():
        return {"runs": []}

    items: list[dict[str, Any]] = []
    for run_dir in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        train_cfg_path = run_dir / "train_config.json"
        if not train_cfg_path.exists():
            continue
        train_cfg = json.loads(train_cfg_path.read_text(encoding="utf-8"))
        summary_path = run_dir / "final_eval_summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else None
        )
        items.append(
            {
                "run_id": run_dir.name,
                "experiment_name": train_cfg.get("experiment_name"),
                "timestamp": train_cfg.get("timestamp"),
                "total_timesteps": train_cfg.get("total_timesteps"),
                "has_best_checkpoint": (run_dir / "best_model.zip").exists(),
                "has_final_checkpoint": (run_dir / "final_model.zip").exists(),
                "final_summary": summary,
            }
        )
    return {"runs": items}


@router.get("/{run_id}")
def get_run_detail(run_id: str) -> dict[str, Any]:
    """Mengambil metadata hyperparameter, eval_history, dan final_summary dari sebuah run."""
    run_dir = _runs_root() / run_id
    train_cfg_path = run_dir / "train_config.json"
    if not run_dir.is_dir() or not train_cfg_path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' tidak ditemukan.")

    train_cfg = json.loads(train_cfg_path.read_text(encoding="utf-8"))
    summary_path = run_dir / "final_eval_summary.json"
    eval_history_path = run_dir / "eval_history.csv"

    return {
        "run_id": run_id,
        "train_config": train_cfg,
        "final_summary": (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else None
        ),
        "eval_history": (
            pd.read_csv(eval_history_path).to_dict("records")
            if eval_history_path.exists()
            else []
        ),
    }


@router.get("/{run_id}/trace")
def get_run_trace(
    run_id: str,
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict[str, Any]:
    """Mengambil trace langkah episode terbaik (best_episode_trace.csv) untuk replay di UI."""
    run_dir = _runs_root() / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' tidak ditemukan.")

    trace_path = run_dir / "best_episode_trace.csv"
    if not trace_path.exists():
        trace_path = run_dir / "manual_best_episode_trace.csv"
    if not trace_path.exists():
        return {"run_id": run_id, "steps": []}

    df = pd.read_csv(trace_path).head(limit).astype(object).fillna("")
    return {"run_id": run_id, "steps": df.to_dict("records")}


@router.delete("/{run_id}")
def delete_run(run_id: str) -> dict[str, Any]:
    """Menghapus folder experiment run dan baris terkait di experiments_summary.csv."""
    if "/" in run_id or "\\" in run_id or run_id in ("", ".", ".."):
        raise HTTPException(status_code=400, detail="run_id tidak valid.")
    root = _runs_root().resolve()
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' tidak ditemukan.")

    shutil.rmtree(run_dir)

    master_summary = root / "experiments_summary.csv"
    if master_summary.exists():
        try:
            df = pd.read_csv(master_summary)
            if "run_id" in df.columns:
                df[df["run_id"] != run_id].to_csv(master_summary, index=False)
        except Exception:
            pass

    return {"deleted": True, "run_id": run_id}
