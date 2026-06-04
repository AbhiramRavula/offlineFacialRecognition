"""
NHAI FaceGuard SDK — Face Detector
Uses SCRFD (Sample and Computation Redistribution Face Detection) via InsightFace.
Runs entirely offline using locally cached ONNX models.
"""

import logging
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import insightface
from insightface.app import FaceAnalysis

from utils.config import (
    DETECTION_SIZE,
    DET_THRESHOLD,
    INSIGHTFACE_MODEL_PACK,
    INSIGHTFACE_PROVIDERS,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """Represents a single detected face."""
    bbox: np.ndarray          # [x1, y1, x2, y2, confidence]
    landmarks: np.ndarray     # 5-point landmarks [[x,y], ...]
    confidence: float

    @property
    def x1(self) -> int: return int(self.bbox[0])
    @property
    def y1(self) -> int: return int(self.bbox[1])
    @property
    def x2(self) -> int: return int(self.bbox[2])
    @property
    def y2(self) -> int: return int(self.bbox[3])

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class FaceDetector:
    """
    SCRFD-based face detector wrapping InsightFace.

    On first use, InsightFace downloads the 'buffalo_sc' model pack
    (~5 MB) to ~/.insightface/models/buffalo_sc/. All subsequent runs
    are fully offline.

    Usage:
        detector = FaceDetector()
        faces = detector.detect(frame)
        largest = detector.detect_largest(frame)
    """

    def __init__(self) -> None:
        logger.info("Initializing FaceDetector (model: %s)...", INSIGHTFACE_MODEL_PACK)
        self._app = FaceAnalysis(
            name=INSIGHTFACE_MODEL_PACK,
            providers=INSIGHTFACE_PROVIDERS,
            allowed_modules=["detection"],  # Only load detector, not recognizer
        )
        self._app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
        logger.info("FaceDetector ready. Detection size: %s", DETECTION_SIZE)

    def detect(self, img: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in an image.

        Args:
            img: BGR image as numpy array (HxWx3).

        Returns:
            List of DetectedFace objects sorted by confidence (desc).
        """
        if img is None or img.size == 0:
            return []

        raw_faces = self._app.get(img)
        results: List[DetectedFace] = []

        for face in raw_faces:
            det_score = float(face.det_score)
            if det_score < DET_THRESHOLD:
                continue
            results.append(DetectedFace(
                bbox=face.bbox,
                landmarks=face.kps,   # 5-point: LE, RE, Nose, LM, RM
                confidence=det_score,
            ))

        results.sort(key=lambda f: f.confidence, reverse=True)
        return results

    def detect_largest(self, img: np.ndarray) -> Optional[DetectedFace]:
        """
        Detect and return the largest (by area) face in the image.
        Useful for single-person verification scenarios.

        Returns:
            Single DetectedFace or None if no face found.
        """
        faces = self.detect(img)
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)

    def draw_detections(
        self, img: np.ndarray, faces: List[DetectedFace]
    ) -> np.ndarray:
        """Draw bounding boxes and landmarks on a copy of the image."""
        out = img.copy()
        for face in faces:
            cv2.rectangle(
                out,
                (face.x1, face.y1), (face.x2, face.y2),
                (0, 200, 0), 2,
            )
            cv2.putText(
                out, f"{face.confidence:.2f}",
                (face.x1, face.y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1,
            )
            for pt in face.landmarks:
                cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (0, 100, 255), -1)
        return out
