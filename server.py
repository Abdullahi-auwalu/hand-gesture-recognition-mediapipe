#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, glob, json, math, statistics, io, csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DEFAULT_LOG_DIR = os.path.join(os.getcwd(), "logs")
LOG_DIR = os.environ.get("GESTURE_LOG_DIR", DEFAULT_LOG_DIR)

# recordings live in LOG_DIR/videos by default (override with env)
DEFAULT_VIDEO_DIR = os.path.join(LOG_DIR, "videos")
VIDEO_DIR = os.environ.get("GESTURE_VIDEO_DIR", DEFAULT_VIDEO_DIR)

app = FastAPI(title="Gesture Logs Service", version="0.3.0")

# CORS (Next.js dev, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_logs_dir() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR

def _ensure_video_dir() -> str:
    os.makedirs(VIDEO_DIR, exist_ok=True)
    return VIDEO_DIR

def _list_runs() -> List[Dict[str, Any]]:
    _ensure_logs_dir()
    runs: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "run_*.jsonl"))):
        fn = os.path.basename(path)
        run_id = fn.replace("run_", "").replace(".jsonl", "")
        try:
            t = datetime.strptime(run_id, "%Y%m%d-%H%M%S")
            ts = t.isoformat()
        except Exception:
            ts = None
        size = os.path.getsize(path)
        runs.append({"id": run_id, "file": fn, "bytes": size, "started_at": ts})
    runs.sort(key=lambda r: r["id"], reverse=True)  # newest first
    return runs

def _path_for(run_id: str) -> str:
    return os.path.join(LOG_DIR, f"run_{run_id}.jsonl")

def _parse_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                pass  # skip malformed lines
    return rows

def _only_frames(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("type") == "frame"]

def _percentiles(values: List[float], ps: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {f"p{int(p*100)}": None for p in ps}
    vals = sorted(values)
    out: Dict[str, Optional[float]] = {}
    for p in ps:
        if len(vals) == 1:
            out[f"p{int(p*100)}"] = vals[0]
        else:
            k = (len(vals) - 1) * p
            f = math.floor(k); c = math.ceil(k)
            out[f"p{int(p*100)}"] = vals[int(k)] if f == c else vals[f] + (vals[c] - vals[f]) * (k - f)
    return out

def _row_pred_label(r: Dict[str, Any]) -> str:
    # prefer top-level pred_label; fallback to nested hand.label; else Unknown
    return r.get("pred_label") or (r.get("hand") or {}).get("label") or "Unknown"

def _confusion(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    mat: Dict[str, Dict[str, int]] = {}
    for r in _only_frames(rows):
        gt = r.get("gt_label")
        if not gt:
            continue
        pred = _row_pred_label(r)
        mat.setdefault(gt, {})
        mat[gt][pred] = mat[gt].get(pred, 0) + 1
    return mat

def _class_hist(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for r in _only_frames(rows):
        pred = _row_pred_label(r)
        hist[pred] = hist.get(pred, 0) + 1
    return hist

def _latencies(rows: List[Dict[str, Any]]) -> List[float]:
    vals: List[float] = []
    for r in _only_frames(rows):
        v = r.get("classifier_ms") or r.get("latency_ms") or r.get("total_ms")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals

def _paged(items: List[Any], offset: int, limit: int) -> Tuple[List[Any], int]:
    total = len(items)
    if offset < 0: offset = 0
    if limit <= 0: limit = 50
    return items[offset: offset + limit], total

# ---- video helpers ----
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

def _safe_video_path(filename: str) -> str:
    # prevent path traversal
    base = os.path.basename(filename)
    p = os.path.abspath(os.path.join(VIDEO_DIR, base))
    if not p.startswith(os.path.abspath(VIDEO_DIR) + os.sep) and p != os.path.abspath(VIDEO_DIR):
        raise HTTPException(400, "Invalid path")
    return p

def _list_videos() -> List[Dict[str, Any]]:
    _ensure_video_dir()
    items: List[Dict[str, Any]] = []
    for path in glob.glob(os.path.join(VIDEO_DIR, "*")):
        ext = os.path.splitext(path)[1].lower()
        if ext not in VIDEO_EXTS or not os.path.isfile(path):
            continue
        st = os.stat(path)
        items.append({
            "file": os.path.basename(path),
            "bytes": st.st_size,
            "created_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "url": f"/videos/{os.path.basename(path)}",
        })
    # newest first by mtime
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class Run(BaseModel):
    id: str
    file: str
    bytes: int
    started_at: Optional[str]

class Metrics(BaseModel):
    run_id: str
    total_frames: int
    class_histogram: Dict[str, int]
    confusion: Dict[str, Dict[str, int]]
    latency: Dict[str, Optional[float]]  # mean, p50, p90, p99
    labels_seen: List[str]

class PagedEvents(BaseModel):
    run_id: str
    offset: int
    limit: int
    total: int
    events: List[Dict[str, Any]]

class Video(BaseModel):
    file: str
    bytes: int
    created_at: str
    url: str  # /videos/<file>

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/health", include_in_schema=False)
def health():
    runs = _list_runs()
    vids = _list_videos()
    return {
        "ok": True,
        "log_dir": LOG_DIR,
        "video_dir": VIDEO_DIR,
        "runs": len(runs),
        "videos": len(vids),
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/runs", response_model=List[Run])
def list_runs():
    return _list_runs()

@app.get("/runs/latest/metrics", response_model=Metrics)
def latest_metrics():
    runs = _list_runs()
    if not runs:
        raise HTTPException(404, "No runs found")
    return metrics(runs[0]["id"])

@app.get("/runs/{run_id}/raw")
def raw(run_id: str):
    path = _path_for(run_id)
    if not os.path.exists(path):
        raise HTTPException(404, f"run {run_id} not found")
    with open(path, "r", encoding="utf-8") as f:
        return {"id": run_id, "content": f.read()}

@app.get("/runs/{run_id}/metrics", response_model=Metrics)
def metrics(run_id: str):
    path = _path_for(run_id)
    if not os.path.exists(path):
        raise HTTPException(404, f"run {run_id} not found")
    rows = _parse_jsonl(path)
    frames = _only_frames(rows)

    # histogram / confusion / latency from frame rows only
    hist = _class_hist(rows)
    conf = _confusion(rows)
    lats = _latencies(rows)

    latency = {
        "mean": (statistics.fmean(lats) if lats else None),
        **_percentiles(lats, [0.50, 0.90, 0.99]),
    }

    # label order: prefer meta.labels if available
    meta = next((r for r in rows if r.get("type") == "meta"), {})
    labels_from_meta = meta.get("labels") or []
    if labels_from_meta:
        labels_seen = [lbl for lbl in labels_from_meta if lbl in hist]
        extras = sorted([k for k in hist.keys() if k not in labels_seen])
        labels_seen.extend(extras)
    else:
        labels_seen = sorted(hist.keys())

    return Metrics(
        run_id=run_id,
        total_frames=len(frames),
        class_histogram=hist,
        confusion=conf,
        latency=latency,
        labels_seen=labels_seen,
    )

@app.get("/runs/{run_id}/events", response_model=PagedEvents)
def events(
    run_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
):
    path = _path_for(run_id)
    if not os.path.exists(path):
        raise HTTPException(404, f"run {run_id} not found")
    rows = _parse_jsonl(path)
    page, total = _paged(rows, offset, limit)
    return PagedEvents(run_id=run_id, offset=offset, limit=limit, total=total, events=page)

@app.get("/runs/{run_id}/export.csv")
def export_csv(run_id: str):
    path = _path_for(run_id)
    if not os.path.exists(path):
        raise HTTPException(404, f"run {run_id} not found")
    rows = _parse_jsonl(path)
    if not rows:
        return Response(content="", media_type="text/csv")

    # Collect union of keys to form header
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buf.getvalue().encode("utf-8")
    headers = {"Content-Disposition": f'attachment; filename="run_{run_id}.csv"'}
    return Response(content=csv_bytes, media_type="text/csv", headers=headers)

# ----------------- Video endpoints -----------------
@app.get("/videos", response_model=List[Video])
def list_videos():
    return [Video(**v) for v in _list_videos()]

@app.get("/videos/{filename}")
def get_video(filename: str):
    _ensure_video_dir()
    path = _safe_video_path(filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Video not found")
    # Let Starlette handle Range requests for streaming
    # Basic type guess; browsers rely mostly on extension
    return FileResponse(path, media_type="video/mp4")

# ---------------------------------------------------------------------
# How to run:
#   uvicorn server:app --reload
# ---------------------------------------------------------------------

