"""
NHAI FaceGuard SDK — Face Recognizer
Uses ArcFace (MobileFaceNet) via InsightFace ONNX Runtime.
Generates 512-dim L2-normalized embeddings for face comparison.
"""

import os
import glob
import logging
import numpy as np
import cv2
from typing import Optional, Tuple

import insightface
from insightface.app import FaceAnalysis

from utils.config import (
    RECOGNITION_THRESHOLD,
    INSIGHTFACE_MODEL_PACK,
    INSIGHTFACE_PROVIDERS,
    RECOGNITION_SIZE,
)
from utils.image_utils import align_face, normalize_face

logger = logging.getLogger(__name__)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance between two L2-normalized embedding vectors.
    Range: 0.0 (identical) to 2.0 (opposite).
    Threshold ~0.35 works well for most ArcFace models.
    """
    return float(1.0 - np.dot(a, b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity: 1.0 = identical, -1.0 = opposite."""
    return float(np.dot(a, b))


class FaceRecognizer:
    """
    ArcFace MobileFaceNet face recognizer.

    Extracts 512-dimensional face embeddings that can be stored and
    compared for 1:1 verification or 1:N identification.

    On first use, downloads buffalo_sc model pack (~5 MB total with
    detector) to ~/.insightface/models/. Fully offline afterwards.

    Usage:
        rec = FaceRecognizer()
        emb = rec.get_embedding(face_crop)
        dist = rec.compare(emb1, emb2)
        is_match = rec.is_same_person(emb1, emb2)
    """

    def __init__(self) -> None:
        logger.info("Initializing FaceRecognizer (model: %s)...", INSIGHTFACE_MODEL_PACK)
        # Load the recognition model directly via model_zoo to avoid
        # FaceAnalysis's internal assertion that 'detection' must be present.
        self._rec_model = self._load_rec_model()
        logger.info("FaceRecognizer ready.")

    def _load_rec_model(self):
        """Load MobileFaceNet recognition model from local InsightFace cache."""
        models_dir = os.path.expanduser(
            f"~/.insightface/models/{INSIGHTFACE_MODEL_PACK}/"
        )
        # Search for recognition ONNX (buffalo_sc uses w600k_mbf.onnx)
        for pattern in ["w600k_mbf.onnx", "*mbf*.onnx", "*recognition*.onnx"]:
            matches = glob.glob(os.path.join(models_dir, pattern))
            if matches:
                model_path = matches[0]
                logger.info(
                    "Loading recognition model: %s", os.path.basename(model_path)
                )
                model = insightface.model_zoo.get_model(
                    model_path, providers=INSIGHTFACE_PROVIDERS
                )
                model.prepare(ctx_id=0)
                return model

        # Model not cached — trigger download via FaceAnalysis (needs internet once)
        logger.warning(
            "Recognition model not found at %s. Triggering InsightFace download...",
            models_dir,
        )
        app = FaceAnalysis(
            name=INSIGHTFACE_MODEL_PACK,
            providers=INSIGHTFACE_PROVIDERS,
        )
        app.prepare(ctx_id=0)
        return app.models.get("recognition")

    def get_embedding(
        self,
        face_crop: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract a 512-dim L2-normalized face embedding.

        Args:
            face_crop: BGR face image (any size, will be resized).
            landmarks: Optional 5-point landmarks for better alignment.
                       If None, uses center-crop alignment.

        Returns:
            512-dim numpy float32 array (L2 normalized), or None on error.
        """
        try:
            if landmarks is not None:
                aligned = align_face(face_crop, landmarks, output_size=RECOGNITION_SIZE)
            else:
                aligned = cv2.resize(face_crop, RECOGNITION_SIZE)

            # InsightFace recognition model expects a face object
            # We create a minimal face object with normed embedding
            embedding = self._rec_model.get_feat(aligned)
            if embedding is None:
                return None

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return None
            return (embedding / norm).astype(np.float32)

        except Exception as e:
            logger.error("Embedding extraction failed: %s", e)
            return None

    def get_embedding_from_full_image(
        self,
        full_img: np.ndarray,
        bbox: np.ndarray,
        landmarks: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Convenience method: align & embed from full image + detection result.

        Args:
            full_img: Full BGR frame.
            bbox: [x1, y1, x2, y2] bounding box.
            landmarks: 5-point landmarks.

        Returns:
            512-dim embedding or None.
        """
        aligned = align_face(full_img, landmarks, output_size=RECOGNITION_SIZE)
        return self.get_embedding(aligned)

    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compare two embeddings. Returns cosine distance (lower = more similar).

        Returns:
            float: 0.0 (identical) to 2.0 (completely different).
        """
        return cosine_distance(emb1, emb2)

    def is_same_person(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        threshold: float = RECOGNITION_THRESHOLD,
    ) -> Tuple[bool, float]:
        """
        Determine if two embeddings belong to the same person.

        Args:
            emb1, emb2: L2-normalized 512-dim embeddings.
            threshold: Cosine distance threshold (default from config).

        Returns:
            (is_match: bool, distance: float)
        """
        dist = self.compare(emb1, emb2)
        return dist <= threshold, dist

    def find_best_match(
        self,
        query_emb: np.ndarray,
        database: dict,  # {person_id: embedding_array}
        threshold: float = RECOGNITION_THRESHOLD,
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching person in an embedding dictionary.

        Args:
            query_emb: Query embedding (512-dim, L2 normalized).
            database: Dict mapping person_id → embedding.
            threshold: Maximum distance to accept as a match.

        Returns:
            (best_person_id or None, best_distance)
        """
        if not database:
            return None, float("inf")

        best_id = None
        best_dist = float("inf")

        for person_id, stored_emb in database.items():
            dist = self.compare(query_emb, stored_emb)
            if dist < best_dist:
                best_dist = dist
                best_id = person_id

        if best_dist > threshold:
            return None, best_dist  # No match within threshold

        return best_id, best_dist
