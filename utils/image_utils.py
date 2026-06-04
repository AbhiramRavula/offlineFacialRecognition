"""
NHAI FaceGuard SDK — Image Utilities
Preprocessing, alignment, and conversion helpers.
"""

import cv2
import numpy as np
from PIL import Image
import base64
import io
from typing import Tuple, Optional


# Standard 5-point facial landmarks for 112×112 ArcFace alignment
ARCFACE_REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV default) to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to (width, height)."""
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def normalize_face(img: np.ndarray) -> np.ndarray:
    """
    Normalize a face crop for ArcFace input.
    Returns float32 array in range [-1, 1], shape (1, 3, 112, 112).
    """
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0  # ArcFace standard normalization
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def align_face(
    img: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Align a face to canonical ArcFace orientation using a similarity transform.

    Args:
        img: Full image (BGR or RGB, HxWxC).
        landmarks: 5-point landmarks [[x,y], ...] from face detector.
        output_size: Output face image size (width, height).

    Returns:
        Aligned face crop as uint8 numpy array.
    """
    dst = ARCFACE_REFERENCE_LANDMARKS.copy()
    if output_size != (112, 112):
        scale_x = output_size[0] / 112.0
        scale_y = output_size[1] / 112.0
        dst[:, 0] *= scale_x
        dst[:, 1] *= scale_y

    src = landmarks.astype(np.float32)

    # Estimate similarity transform (rotation + scale + translation)
    tform = _estimate_similarity_transform(src, dst)
    aligned = cv2.warpAffine(
        img, tform, output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned


def _estimate_similarity_transform(
    src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """
    Compute 2×3 affine transform matrix from src→dst using least-squares.
    Uses umeyama algorithm approximation via cv2.estimateAffinePartial2D.
    """
    tform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if tform is None:
        # Fallback: identity-ish transform
        tform = np.eye(2, 3, dtype=np.float32)
    return tform


def crop_face(
    img: np.ndarray,
    bbox: np.ndarray,
    scale: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Crop a face region from the image given a bounding box.

    Args:
        img: Full image (HxWxC).
        bbox: [x1, y1, x2, y2] bounding box.
        scale: Scale factor to expand crop area (>1 adds margin).

    Returns:
        Cropped face region or None if crop is invalid.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox[:4].astype(int)

    if scale != 1.0:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
        x1 = int(cx - bw)
        y1 = int(cy - bh)
        x2 = int(cx + bw)
        y2 = int(cy + bh)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def base64_to_image(b64_string: str) -> np.ndarray:
    """
    Decode a base64-encoded image string to a numpy BGR array.
    Accepts standard base64 or data-URI format.
    """
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def image_to_base64(img: np.ndarray, fmt: str = "JPEG") -> str:
    """Encode a BGR numpy array to a base64 JPEG/PNG string."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def draw_results(
    img: np.ndarray,
    bbox: Optional[np.ndarray],
    liveness: Optional[str],
    identity: Optional[str],
    confidence: Optional[float],
) -> np.ndarray:
    """
    Draw detection, liveness, and recognition results on frame.

    Args:
        img: BGR image to annotate.
        bbox: [x1, y1, x2, y2] face bounding box.
        liveness: 'REAL' | 'SPOOF' | None
        identity: Identified person name or None
        confidence: Match confidence (0-1) or None

    Returns:
        Annotated BGR image.
    """
    out = img.copy()
    if bbox is None:
        cv2.putText(out, "No Face Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
        return out

    x1, y1, x2, y2 = bbox[:4].astype(int)

    # Box color: green=real, red=spoof, yellow=unknown
    if liveness == "REAL":
        box_color = (0, 220, 0)
    elif liveness == "SPOOF":
        box_color = (0, 0, 220)
    else:
        box_color = (0, 200, 220)

    cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

    # Liveness label
    live_label = f"[{liveness}]" if liveness else "[UNKNOWN]"
    cv2.putText(out, live_label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)

    # Identity label
    if identity:
        id_text = f"{identity} ({confidence:.1%})" if confidence else identity
        cv2.putText(out, id_text, (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out
