"""
NHAI FaceGuard SDK — Configuration
All thresholds, paths, and constants in one place.
"""

import os

# ── Base Paths ─────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DB_PATH    = os.path.join(ROOT_DIR, "facedb.sqlite")

# ── Model File Names ───────────────────────────────────────────────────────
LIVENESS_MODEL_V1 = os.path.join(MODELS_DIR, "MiniFASNetV1.onnx")
LIVENESS_MODEL_V2 = os.path.join(MODELS_DIR, "MiniFASNetV2.onnx")

# ── Detection Settings ─────────────────────────────────────────────────────
DETECTION_SIZE    = (640, 640)   # Input resolution for SCRFD detector
DET_THRESHOLD     = 0.5          # Face detection confidence threshold
DET_NMS_THRESHOLD = 0.4          # Non-maximum suppression threshold

# ── Liveness Settings ──────────────────────────────────────────────────────
LIVENESS_THRESHOLD   = 0.6       # Score > threshold → REAL; else → SPOOF
LIVENESS_INPUT_SIZE  = (80, 80)  # MiniFASNet input resolution
# Scale factors for multi-scale liveness check (improves robustness)
LIVENESS_SCALES      = [2.7, 4.0]

# ── Recognition Settings ───────────────────────────────────────────────────
RECOGNITION_SIZE      = (112, 112)  # ArcFace input resolution
RECOGNITION_THRESHOLD = 0.35        # Cosine distance threshold for match
                                    # Lower = stricter. Tune per use-case.
EMBEDDING_DIM         = 512         # ArcFace embedding dimension

# ── InsightFace Settings ────────────────────────────────────────────────────
# Model pack: 'buffalo_sc' = lightweight MobileFaceNet + SCRFD_500M
# Downloaded ONCE to ~/.insightface/models/buffalo_sc/ then used offline
INSIGHTFACE_MODEL_PACK = "buffalo_sc"
INSIGHTFACE_PROVIDERS  = ["CPUExecutionProvider"]

# ── API Settings ───────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
