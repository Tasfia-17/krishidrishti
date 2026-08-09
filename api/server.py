"""
KrishiDrishti FastAPI Server
Repurposed from ScreenMind's screenmind/api/server.py

Endpoints:
  POST /api/diagnose        — Upload crop photo → get diagnosis
  GET  /api/history         — Diagnosis history list
  GET  /api/history/{id}    — Single diagnosis detail
  GET  /api/stats           — Aggregate stats
  GET  /api/search          — Search history
  DELETE /api/history/{id}  — Delete a record
  GET  /api/health          — Health check
  GET  /                    — Serve frontend UI
"""
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from gemma_engine.crop_analyzer import CropAnalyzer
from gemma_engine import llm_client
from storage.database import get_database
from storage.models import HealthResponse

logger = logging.getLogger("krishidrishti.api.server")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KrishiDrishti",
    description="AI Crop Doctor powered by Gemma 4 — 100% local, 100% private",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────
analyzer = CropAnalyzer()

# Static files (frontend)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Image upload storage
UPLOAD_DIR = Path.home() / ".krishidrishti" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    db = get_database()
    await db.initialize()
    gemma_ok = llm_client.is_available()
    logger.info(
        f"KrishiDrishti started — Gemma 4: {'✓ connected' if gemma_ok else '✗ not available'}"
    )


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check — reports whether Gemma 4 llama-server is reachable."""
    gemma_ok = llm_client.is_available()
    status_info = llm_client.get_server_status()
    return HealthResponse(
        status="ok" if gemma_ok else "degraded",
        gemma_available=gemma_ok,
        server=status_info.get("server", settings.llama_server_host),
    )


# ── Core Diagnosis Endpoint ───────────────────────────────────────────────────
@app.post("/api/diagnose")
async def diagnose_crop(
    image: UploadFile = File(..., description="Crop/plant photo (JPEG, PNG, WebP)"),
    farmer_note: Optional[str] = Form(
        default=None,
        description="Optional: Farmer's own description of the problem"
    ),
    mode: Optional[str] = Form(
        default=None,
        description="Analysis mode: fast / balanced / accurate"
    ),
):
    """
    Upload a crop photo and get an AI-powered disease diagnosis.

    Gemma 4 analyzes the image and returns:
    - Disease identification (English + Hindi)
    - Severity assessment
    - Treatment recommendations (English + Hindi)
    - Prevention advice
    - Confidence score

    Powered by Gemma 4 running 100% locally via llama-server.
    """
    # Validate file type
    if image.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {image.content_type}. Use JPEG, PNG, or WebP."
        )

    # Read image bytes
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")
    if len(image_bytes) > settings.max_upload_size:
        max_mb = settings.max_upload_size / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Maximum size: {max_mb:.0f}MB"
        )

    # Validate mode
    valid_modes = ("fast", "balanced", "accurate", None)
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Choose: fast, balanced, accurate"
        )

    # Save uploaded image to disk
    ts = int(time.time() * 1000)
    ext = Path(image.filename or "crop.jpg").suffix or ".jpg"
    save_path = UPLOAD_DIR / f"crop_{ts}{ext}"
    save_path.write_bytes(image_bytes)

    # Run Gemma 4 analysis
    start = time.time()
    try:
        record = analyzer.analyze_from_bytes(
            image_bytes=image_bytes,
            farmer_note=farmer_note,
            mode=mode,
        )
    except ConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Gemma 4 not available: {str(e)}. Start llama-server first."
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)[:200]}")

    elapsed = time.time() - start

    # Save to history database
    db = get_database()
    entry_id = await db.save_diagnosis(
        record=record,
        image_path=str(save_path),
        farmer_note=farmer_note,
        mode=mode or settings.analysis_mode,
    )

    # Build response
    response = record.model_dump()
    response["diagnosis_id"] = entry_id
    response["analysis_time_seconds"] = round(elapsed, 1)
    response["mode_used"] = mode or settings.analysis_mode
    response["image_saved"] = str(save_path)

    return JSONResponse(content=response)


# ── History Endpoints ─────────────────────────────────────────────────────────
@app.get("/api/history")
async def get_history(
    limit: int = 20,
    offset: int = 0,
    crop: Optional[str] = None,
):
    """
    Retrieve diagnosis history (newest first).
    Optionally filter by crop name.
    """
    limit = min(limit, 100)  # Cap at 100
    db = get_database()
    entries = await db.get_history(limit=limit, offset=offset, crop_filter=crop)

    return {
        "total": len(entries),
        "offset": offset,
        "limit": limit,
        "entries": [_entry_to_dict(e) for e in entries],
    }


@app.get("/api/history/{entry_id}")
async def get_diagnosis(entry_id: int):
    """Get a single diagnosis record by ID."""
    db = get_database()
    entry = await db.get_by_id(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Diagnosis #{entry_id} not found")
    return _entry_to_dict(entry)


@app.delete("/api/history/{entry_id}")
async def delete_diagnosis(entry_id: int):
    """Delete a diagnosis record."""
    db = get_database()
    deleted = await db.delete_diagnosis(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Diagnosis #{entry_id} not found")
    return {"deleted": True, "id": entry_id}


# ── Search ────────────────────────────────────────────────────────────────────
@app.get("/api/search")
async def search_history(
    q: str,
    limit: int = 10,
):
    """
    Search diagnosis history by crop name, disease, or farmer notes.
    Uses FTS5 full-text search (falls back to LIKE search).
    """
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")

    db = get_database()
    entries = await db.search(query=q.strip(), limit=min(limit, 50))
    return {
        "query": q,
        "results": len(entries),
        "entries": [_entry_to_dict(e) for e in entries],
    }


# ── Stats ─────────────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def get_stats():
    """Return aggregate statistics: total diagnoses, top crops, top diseases."""
    db = get_database()
    return await db.get_stats()


# ── Image Serving ──────────────────────────────────────────────────────────────
@app.get("/api/image/{entry_id}")
async def get_image(entry_id: int):
    """Serve the crop image for a given diagnosis ID."""
    db = get_database()
    entry = await db.get_by_id(entry_id)
    if not entry or not entry.image_path:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = Path(entry.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(str(image_path), media_type="image/jpeg")


# ── Frontend ──────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """Serve the KrishiDrishti web UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            content={"error": "Frontend not found", "api_docs": "/docs"},
            status_code=404,
        )
    return FileResponse(str(index_file))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _entry_to_dict(entry) -> dict:
    """Convert DiagnosisHistoryEntry to a JSON-serializable dict."""
    d = entry.diagnosis.model_dump()
    d["id"] = entry.id
    d["timestamp"] = entry.timestamp.isoformat()
    d["farmer_note"] = entry.farmer_note
    d["analysis_mode"] = entry.analysis_mode
    d["has_image"] = entry.image_path is not None
    return d
