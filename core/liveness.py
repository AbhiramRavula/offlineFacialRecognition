"""
NHAI FaceGuard SDK — Passive Liveness Detection

Two-tier system:
  Tier 1 (Primary): MiniFASNet ONNX — if model files present in ./models/
  Tier 2 (Fallback): Texture + Frequency Analysis via OpenCV — always available

The fallback uses Laplacian variance + HSV saturation analysis + frequency
domain energy to distinguish real faces from flat printed photos/screens.
This works offline with zero model download.
"""

import os
import logging
import numpy as np
import cv2
import onnxruntime as ort
from enum import Enum
from typing import Tuple, Optional

from utils.config import (
    LIVENESS_MODEL_V1,
    LIVENESS_MODEL_V2,
    LIVENESS_THRESHOLD,
    LIVENESS_INPUT_SIZE,
    LIVENESS_SCALES,
    INSIGHTFACE_PROVIDERS,
)

logger = logging.getLogger(__name__)


class LivenessResult(str, Enum):
    REAL    = "REAL"
    SPOOF   = "SPOOF"
    UNKNOWN = "UNKNOWN"


# ── Texture-Based Fallback (OpenCV only, no ML model needed) ───────────────

# Tunable thresholds — calibrated for typical webcam conditions
_TEXTURE_LAP_THRESHOLD   = 80.0   # Laplacian variance: real > this value
_TEXTURE_FREQ_THRESHOLD  = 0.08   # High-freq energy ratio: real > this
_TEXTURE_SAT_THRESHOLD   = 20.0   # HSV saturation std: real > this


def _laplacian_variance(gray: np.ndarray) -> float:
    """Measure texture sharpness — real faces have higher variance."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _high_freq_energy(gray: np.ndarray) -> float:
    """
    FFT-based high-frequency energy ratio.
    Real faces contain more high-frequency micro-texture detail.
    Printed photos are band-limited by the printing/display process.
    """
    f   = np.fft.fft2(gray.astype(np.float32))
    mag = np.abs(np.fft.fftshift(f))
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 6   # Low-freq radius
    Y, X = np.ogrid[:h, :w]
    low_mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    total     = mag.sum() + 1e-8
    high_freq = mag[~low_mask].sum()
    return float(high_freq / total)


def _saturation_variation(face_bgr: np.ndarray) -> float:
    """
    HSV saturation std — real skin has natural color variation;
    photos often have flatter, more uniform saturation.
    """
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    return float(sat.std())


def _texture_liveness(
    img: np.ndarray,
    bbox: np.ndarray,
) -> Tuple[LivenessResult, float]:
    """
    Texture-based liveness using three OpenCV cues.
    Returns (LivenessResult, confidence 0-1).
    """
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox[:4]]
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return LivenessResult.UNKNOWN, 0.0

    face_resized = cv2.resize(face, (128, 128))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

    lap   = _laplacian_variance(gray)
    freq  = _high_freq_energy(gray)
    sat   = _saturation_variation(face_resized)

    logger.debug("Texture cues — Laplacian: %.1f, FreqEnergy: %.3f, Saturation: %.1f",
                 lap, freq, sat)

    # Score each cue (0=spoof, 1=real)
    lap_score  = min(1.0, lap  / (_TEXTURE_LAP_THRESHOLD  * 2))
    freq_score = min(1.0, freq / (_TEXTURE_FREQ_THRESHOLD * 2))
    sat_score  = min(1.0, sat  / (_TEXTURE_SAT_THRESHOLD  * 2))

    # Weighted combination (texture sharpness is most discriminative)
    combined = 0.5 * lap_score + 0.3 * freq_score + 0.2 * sat_score

    # Majority vote on individual cues
    votes_real = sum([
        lap  > _TEXTURE_LAP_THRESHOLD,
        freq > _TEXTURE_FREQ_THRESHOLD,
        sat  > _TEXTURE_SAT_THRESHOLD,
    ])

    if votes_real >= 2:
        result = LivenessResult.REAL
    else:
        result = LivenessResult.SPOOF
        combined = 1.0 - combined

    return result, round(combined, 4)


# ── MiniFASNet Preprocessing ───────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _preprocess_for_fas(
    img: np.ndarray,
    bbox: np.ndarray,
    scale: float,
    input_size: Tuple[int, int],
) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox[:4].astype(int)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * scale / 2.0
    bh = (y2 - y1) * scale / 2.0

    nx1 = max(0, int(cx - bw))
    ny1 = max(0, int(cy - bh))
    nx2 = min(w, int(cx + bw))
    ny2 = min(h, int(cy + bh))

    crop = img[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        crop = img[y1:y2, x1:x2]

    crop = cv2.resize(crop, input_size, interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = crop.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop = (crop - mean) / std
    crop = crop.transpose(2, 0, 1)
    crop = np.expand_dims(crop, axis=0)
    return crop.astype(np.float32)


# ── Main Liveness Detector ─────────────────────────────────────────────────

class LivenessDetector:
    """
    Passive face anti-spoofing with automatic fallback.

    Priority:
      1. MiniFASNet ONNX ensemble (if models/ files present) — highest accuracy
      2. Texture/Frequency/Saturation analysis (OpenCV) — always available

    Usage:
        detector = LivenessDetector()
        result, score = detector.check(frame, bbox)
    """

    def __init__(self) -> None:
        self._sessions: list = []
        self._onnx_loaded = False
        self._load_models()
        mode = "MiniFASNet ONNX" if self._onnx_loaded else "Texture Fallback (OpenCV)"
        logger.info("LivenessDetector ready — mode: %s", mode)

    def _load_models(self) -> None:
        model_paths = [LIVENESS_MODEL_V1, LIVENESS_MODEL_V2]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        for path in model_paths:
            if not os.path.isfile(path):
                continue
            try:
                session = ort.InferenceSession(
                    path, sess_options=opts,
                    providers=INSIGHTFACE_PROVIDERS,
                )
                self._sessions.append(session)
                logger.info("Loaded MiniFASNet model: %s", os.path.basename(path))
            except Exception as e:
                logger.error("Failed to load liveness model %s: %s", path, e)

        self._onnx_loaded = len(self._sessions) > 0
        if not self._onnx_loaded:
            logger.info(
                "MiniFASNet ONNX not found — using texture-based fallback. "
                "Run download_models.py to enable higher accuracy."
            )

    @property
    def is_ready(self) -> bool:
        """Always True — texture fallback is always available."""
        return True

    @property
    def mode(self) -> str:
        return "MiniFASNet" if self._onnx_loaded else "TextureFallback"

    def check(
        self,
        img: np.ndarray,
        bbox: np.ndarray,
    ) -> Tuple[LivenessResult, float]:
        """
        Run liveness check. Uses MiniFASNet if available, texture analysis otherwise.

        Args:
            img: Full BGR image frame.
            bbox: [x1, y1, x2, y2] bounding box from detector.

        Returns:
            (LivenessResult.REAL | LivenessResult.SPOOF, confidence 0-1)
        """
        if self._onnx_loaded:
            return self._check_minifasnet(img, bbox)
        else:
            return _texture_liveness(img, bbox)

    def _check_minifasnet(
        self,
        img: np.ndarray,
        bbox: np.ndarray,
    ) -> Tuple[LivenessResult, float]:
        scale_scores = []
        for scale in LIVENESS_SCALES:
            patch = _preprocess_for_fas(img, bbox, scale, LIVENESS_INPUT_SIZE)
            session_scores = []
            for session in self._sessions:
                input_name = session.get_inputs()[0].name
                try:
                    output = session.run(None, {input_name: patch})[0]
                    probs = _softmax(output[0])
                    real_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    session_scores.append(real_prob)
                except Exception as e:
                    logger.error("MiniFASNet inference error: %s", e)
            if session_scores:
                scale_scores.append(float(np.mean(session_scores)))

        if not scale_scores:
            return _texture_liveness(img, bbox)   # Fallback on error

        score  = float(np.mean(scale_scores))
        result = LivenessResult.REAL if score >= LIVENESS_THRESHOLD else LivenessResult.SPOOF
        return result, score
