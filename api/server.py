"""
NHAI FaceGuard SDK — FastAPI REST Server
Provides REST endpoints for Datalake 3.0 integration.
All processing is 100% local — no internet calls at runtime.

Endpoints:
    GET  /health           — SDK health & enrolled count
    POST /register         — Enroll a new face
    POST /identify         — 1:N identify from database
    POST /verify           — 1:1 verify against a specific person
    GET  /faces            — List all enrolled persons
    DELETE /face/{id}      — Remove a person from DB
    GET  /attendance/unsynced — View queued offline logs
    POST /attendance/sync    — Sync logs to AWS & purge local data
"""

import sys
import os
import logging
import base64
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.detector   import FaceDetector
from core.liveness   import LivenessDetector, LivenessResult
from core.recognizer import FaceRecognizer
from core.database   import FaceDatabase
from utils.image_utils import base64_to_image, align_face
from utils.config import RECOGNITION_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Init ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="NHAI FaceGuard SDK",
    description="Offline facial recognition & liveness detection API for Datalake 3.0",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pipeline Components (lazy init on first request) ──────────────────────
_detector   = None
_liveness   = None
_recognizer = None
_db         = None

def get_pipeline():
    global _detector, _liveness, _recognizer, _db
    if _detector is None:
        logger.info("Initializing pipeline components...")
        _detector   = FaceDetector()
        _liveness   = LivenessDetector()
        _recognizer = FaceRecognizer()
        _db         = FaceDatabase()
        logger.info("Pipeline ready.")
    return _detector, _liveness, _recognizer, _db


# ── Pydantic Models ────────────────────────────────────────────────────────

class ImagePayload(BaseModel):
    """Base64-encoded image payload."""
    image_b64: str          # base64 image (JPEG/PNG, data-URI ok)

class RegisterRequest(ImagePayload):
    person_id: str          # Unique ID (employee number, etc.)
    person_name: str        # Display name
    overwrite: bool = False # Overwrite if person_id already exists
    metadata: str = "{}"   # Optional JSON metadata

class VerifyRequest(ImagePayload):
    person_id: str          # Who to verify against

class IdentifyRequest(ImagePayload):
    top_k: int = 1                         # Number of results to return
    threshold: float = RECOGNITION_THRESHOLD  # Cosine distance threshold
    check_liveness: bool = True            # Run liveness check first

class FaceMatch(BaseModel):
    person_id: str
    person_name: str
    distance: float
    confidence: float  # 1 - distance, normalized

class IdentifyResponse(BaseModel):
    success: bool
    liveness: Optional[str] = None
    liveness_score: Optional[float] = None
    matches: List[FaceMatch] = []
    message: str = ""

class VerifyResponse(BaseModel):
    success: bool
    is_match: bool
    liveness: Optional[str] = None
    liveness_score: Optional[float] = None
    distance: float = -1.0
    message: str = ""

class RegisterResponse(BaseModel):
    success: bool
    person_id: str
    message: str = ""

class HealthResponse(BaseModel):
    status: str
    enrolled_count: int
    liveness_ready: bool
    sdk_version: str = "1.0.0"


# ── Helpers ────────────────────────────────────────────────────────────────

def _decode_and_detect(image_b64: str):
    """Decode base64 image and detect the largest face."""
    try:
        img = base64_to_image(image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    detector, liveness, recognizer, db = get_pipeline()
    face = detector.detect_largest(img)
    if face is None:
        raise HTTPException(status_code=422, detail="No face detected in the image.")
    return img, face, detector, liveness, recognizer, db


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check SDK health and readiness."""
    _, liveness, _, db = get_pipeline()
    return HealthResponse(
        status="ok",
        enrolled_count=db.count(),
        liveness_ready=liveness.is_ready,
    )


@app.post("/register", response_model=RegisterResponse, tags=["Enrollment"])
def register_face(req: RegisterRequest):
    """
    Enroll a new face in the database.
    Provide a clear, front-facing photo for best accuracy.
    """
    img, face, _, _, recognizer, db = _decode_and_detect(req.image_b64)
    embedding = recognizer.get_embedding_from_full_image(
        img, face.bbox, face.landmarks
    )
    if embedding is None:
        raise HTTPException(status_code=422, detail="Could not extract face embedding.")

    success = db.register(
        req.person_id, req.person_name, embedding,
        metadata=req.metadata, overwrite=req.overwrite
    )
    if not success:
        raise HTTPException(
            status_code=409,
            detail=f"Person ID '{req.person_id}' already exists. Use overwrite=true to update."
        )
    return RegisterResponse(
        success=True,
        person_id=req.person_id,
        message=f"Successfully enrolled '{req.person_name}'.",
    )


@app.post("/identify", response_model=IdentifyResponse, tags=["Recognition"])
def identify_face(req: IdentifyRequest):
    """
    1:N identification — find who this person is from the enrolled database.
    Optionally runs liveness check first to reject spoof attacks.
    Automatically logs successful verifications to the local attendance ledger.
    """
    img, face, _, liveness_det, recognizer, db = _decode_and_detect(req.image_b64)

    liveness_label = None
    liveness_score = None

    # Step 1: Liveness check (optional but recommended)
    if req.check_liveness and liveness_det.is_ready:
        live_result, live_score = liveness_det.check(img, face.bbox)
        liveness_label = live_result.value
        liveness_score = round(live_score, 4)
        if live_result == LivenessResult.SPOOF:
            return IdentifyResponse(
                success=False,
                liveness=liveness_label,
                liveness_score=liveness_score,
                message="Liveness check failed — spoof attack detected.",
            )

    # Step 2: Get embedding
    embedding = recognizer.get_embedding_from_full_image(img, face.bbox, face.landmarks)
    if embedding is None:
        raise HTTPException(status_code=422, detail="Could not extract face embedding.")

    # Step 3: Search database
    matches = db.identify(embedding, top_k=req.top_k, threshold=req.threshold)
    if not matches:
        return IdentifyResponse(
            success=True,
            liveness=liveness_label,
            liveness_score=liveness_score,
            matches=[],
            message="No matching face found in database.",
        )

    face_matches = [
        FaceMatch(
            person_id=pid,
            person_name=name,
            distance=round(dist, 4),
            confidence=round(max(0.0, 1.0 - dist), 4),
        )
        for pid, name, dist in matches
    ]

    # ── LOG ATTENDANCE LOCALLY ON SUCCESS (For Sync & Purge) ──
    if liveness_label == "REAL":
        db.log_attendance(
            person_id=face_matches[0].person_id,
            liveness_score=liveness_score or 1.0,
            verification_mode="passive-liveness"
        )

    return IdentifyResponse(
        success=True,
        liveness=liveness_label,
        liveness_score=liveness_score,
        matches=face_matches,
        message=f"Identified as: {face_matches[0].person_name}",
    )


@app.post("/verify", response_model=VerifyResponse, tags=["Recognition"])
def verify_face(req: VerifyRequest):
    """
    1:1 verification — confirm if the face matches a specific enrolled person.
    """
    img, face, _, liveness_det, recognizer, db = _decode_and_detect(req.image_b64)

    liveness_label = None
    liveness_score = None

    # Liveness check
    if liveness_det.is_ready:
        live_result, live_score = liveness_det.check(img, face.bbox)
        liveness_label = live_result.value
        liveness_score = round(live_score, 4)
        if live_result == LivenessResult.SPOOF:
            return VerifyResponse(
                success=False,
                is_match=False,
                liveness=liveness_label,
                liveness_score=liveness_score,
                message="Liveness check failed — spoof attack detected.",
            )

    # Get embedding
    embedding = recognizer.get_embedding_from_full_image(img, face.bbox, face.landmarks)
    if embedding is None:
        raise HTTPException(status_code=422, detail="Could not extract face embedding.")

    # Verify against specific person
    is_match, distance = db.verify(req.person_id, embedding)

    # ── LOG ATTENDANCE LOCALLY ON SUCCESS ──
    if is_match and liveness_label == "REAL":
        db.log_attendance(
            person_id=req.person_id,
            liveness_score=liveness_score or 1.0,
            verification_mode="passive-liveness-1:1"
        )

    return VerifyResponse(
        success=True,
        is_match=is_match,
        liveness=liveness_label,
        liveness_score=liveness_score,
        distance=round(distance, 4),
        message="Identity verified." if is_match else "Identity not matched.",
    )


@app.get("/faces", tags=["Enrollment"])
def list_faces():
    """List all enrolled persons (without embeddings)."""
    _, _, _, db = get_pipeline()
    return {"enrolled": db.list_all(), "count": db.count()}


@app.delete("/face/{person_id}", tags=["Enrollment"])
def delete_face(person_id: str):
    """Remove a person from the database."""
    _, _, _, db = get_pipeline()
    success = db.delete(person_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Person ID '{person_id}' not found.")
    return {"success": True, "message": f"Removed '{person_id}' from database."}


# ── Sync & Purge Endpoints ────────────────────────────────────────────────

@app.get("/attendance/unsynced", tags=["Sync & Purge"])
def get_unsynced_attendance():
    """Retrieve all offline biometric verify sessions waiting to be synced."""
    _, _, _, db = get_pipeline()
    logs = db.get_unsynced_logs()
    return {"unsynced_logs": logs, "count": len(logs)}


@app.post("/attendance/sync", tags=["Sync & Purge"])
def sync_attendance():
    """
    Simulate uploading unsynced local logs to AWS.
    Upon successful API Gateway response, logs are immediately PURGED
    from local SQLite storage to free device memory.
    """
    _, _, _, db = get_pipeline()
    logs = db.get_unsynced_logs()
    if not logs:
        return {"success": True, "message": "No unsynced logs to upload.", "purged_count": 0}

    # Extract IDs to sync
    log_ids = [log["id"] for log in logs]

    # Mock AWS upload process
    logger.info("Uploading %d logs to AWS API Gateway endpoint...", len(log_ids))

    # Mark logs as synced
    db.mark_logs_as_synced(log_ids)

    # Purge them locally to free up device memory
    purged_count = db.purge_synced_logs()

    return {
        "success": True,
        "message": f"Successfully synced {len(log_ids)} logs to AWS.",
        "synced_count": len(log_ids),
        "purged_count": purged_count
    }


# ── UI Route ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def read_root():
    """Serve the web-based interactive demo."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    if os.path.exists(html_path):
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        except Exception as e:
            return HTMLResponse(content=f"<h1>Error loading UI</h1><p>{str(e)}</p>", status_code=500)
    return HTMLResponse(
        content="""
        <html>
            <head><title>NHAI FaceGuard</title></head>
            <body style="font-family: sans-serif; text-align: center; padding-top: 10%; background: #0b0f19; color: #fff;">
                <h1>NHAI FaceGuard API is Running</h1>
                <p>Interactive web demo not found. Please place <code>index.html</code> in <code>api/templates/</code>.</p>
                <p>View Swagger API documentation at <a href="/docs" style="color: #00f2fe;">/docs</a>.</p>
            </body>
        </html>
        """
    )


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from utils.config import API_HOST, API_PORT

    host = os.environ.get("HOST", API_HOST)
    port = int(os.environ.get("PORT", API_PORT))

    logger.info("Starting NHAI FaceGuard API on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
