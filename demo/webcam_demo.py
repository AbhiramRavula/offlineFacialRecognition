"""
NHAI FaceGuard SDK — Live Webcam Demo
Real-time facial recognition + liveness detection demo.

Controls:
    R - Register current detected face (enter ID + name in terminal)
    Q - Quit
    S - Save current frame as screenshot
"""

import sys
import os
import cv2
import time
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector   import FaceDetector
from core.liveness   import LivenessDetector, LivenessResult
from core.recognizer import FaceRecognizer
from core.database   import FaceDatabase
from utils.image_utils import draw_results

logging.basicConfig(level=logging.WARNING)  # Suppress info spam in demo


WINDOW_TITLE = "NHAI FaceGuard — Live Demo  |  R=Register  Q=Quit"
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")


def draw_hud(frame: np.ndarray, fps: float, enrolled_count: int, live_mode: str = "") -> np.ndarray:
    """Draw heads-up display with FPS and enrollment count."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
    cv2.putText(frame, f"Enrolled: {enrolled_count}", (150, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
    mode_label = f"Liveness: {live_mode}" if live_mode else "NHAI FaceGuard SDK v1.0"
    cv2.putText(frame, mode_label, (w - 310, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 150), 1)
    return frame


def draw_liveness_badge(frame: np.ndarray, result: str, score: float, bbox) -> np.ndarray:
    """Draw a colored liveness badge over the bounding box."""
    if result == "REAL":
        color  = (0, 210, 0)
        label  = f"✓ REAL  {score:.0%}"
    elif result == "SPOOF":
        color  = (0, 50, 220)
        label  = f"✗ SPOOF {score:.0%}"
    else:
        color  = (0, 200, 220)
        label  = "? CHECKING"

    x1, y1 = int(bbox[0]), int(bbox[1])
    x2, y2 = int(bbox[2]), int(bbox[3])

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Badge background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def draw_identity_badge(frame: np.ndarray, name: str, distance: float, bbox) -> np.ndarray:
    """Draw identity label below bounding box."""
    x1, y2 = int(bbox[0]), int(bbox[3])
    confidence = max(0.0, 1.0 - distance)

    if distance < 0.3:
        color = (0, 220, 0)
    elif distance < 0.45:
        color = (0, 180, 220)
    else:
        color = (80, 80, 220)

    label = f"{name}  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y2), (x1 + tw + 8, y2 + th + 12), color, -1)
    cv2.putText(frame, label, (x1 + 4, y2 + th + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def register_face_interactive(
    recognizer: FaceRecognizer,
    db: FaceDatabase,
    img: np.ndarray,
    face,
) -> None:
    """Prompt terminal for ID + name, then register the current face."""
    print("\n" + "=" * 50)
    print("  REGISTER NEW FACE")
    print("=" * 50)
    person_id   = input("  Enter Person ID (e.g. EMP001): ").strip()
    person_name = input("  Enter Person Name: ").strip()

    if not person_id or not person_name:
        print("  [!] Cancelled — ID and name cannot be empty.")
        return

    embedding = recognizer.get_embedding_from_full_image(img, face.bbox, face.landmarks)
    if embedding is None:
        print("  [!] Failed to extract embedding. Try again with a clearer photo.")
        return

    success = db.register(person_id, person_name, embedding)
    if success:
        print(f"  [✓] Registered '{person_name}' (ID: {person_id}) successfully!")
    else:
        overwrite = input(f"  [!] ID '{person_id}' already exists. Overwrite? (y/N): ")
        if overwrite.lower() == "y":
            db.register(person_id, person_name, embedding, overwrite=True)
            print(f"  [✓] Updated '{person_name}'.")
    print("=" * 50 + "\n")


def run_demo(camera_index: int = 0) -> None:
    print("=" * 60)
    print("  NHAI FaceGuard SDK — Live Demo")
    print("  Initializing pipeline (first run downloads models)...")
    print("=" * 60)

    detector   = FaceDetector()
    liveness   = LivenessDetector()
    recognizer = FaceRecognizer()
    db         = FaceDatabase()

    if not liveness.is_ready:
        print("\n  [WARNING] Liveness models not found!")
        print("  Run:  python download_models.py\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    fps_counter = 0
    fps_timer   = time.time()
    fps         = 0.0

    # Process liveness every N frames (liveness is slower)
    LIVENESS_EVERY_N = 5
    frame_count = 0
    cached_liveness = None
    cached_live_score = 0.0

    print("\n  [✓] Camera open. Starting demo...\n")
    print("  Controls:  R = Register face | S = Screenshot | Q = Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        frame_count += 1
        display = frame.copy()

        # Detect largest face
        face = detector.detect_largest(frame)

        if face is not None:
            # Liveness (every N frames for performance)
            if frame_count % LIVENESS_EVERY_N == 0 or cached_liveness is None:
                if liveness.is_ready:
                    result, score = liveness.check(frame, face.bbox)
                    cached_liveness   = result.value
                    cached_live_score = score
                else:
                    cached_liveness   = "UNKNOWN"
                    cached_live_score = 0.0

            draw_liveness_badge(display, cached_liveness, cached_live_score, face.bbox)

            # Recognition (only if REAL)
            if cached_liveness == "REAL":
                emb = recognizer.get_embedding_from_full_image(frame, face.bbox, face.landmarks)
                if emb is not None:
                    matches = db.identify(emb, top_k=1)
                    if matches:
                        pid, name, dist = matches[0]
                        draw_identity_badge(display, name, dist, face.bbox)
                    else:
                        cv2.putText(display, "Unknown Person",
                                    (int(face.bbox[0]), int(face.bbox[3]) + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        else:
            cv2.putText(display, "No face detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)

        # FPS
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        draw_hud(display, fps, db.count(), liveness.mode)
        cv2.imshow(WINDOW_TITLE, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            if face is not None:
                register_face_interactive(recognizer, db, frame, face)
            else:
                print("[!] No face detected. Position yourself in front of the camera.")
        elif key == ord("s"):
            path = os.path.join(SCREENSHOT_DIR, f"screenshot_{int(time.time())}.jpg")
            cv2.imwrite(path, display)
            print(f"[✓] Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[✓] Demo closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NHAI FaceGuard Live Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()
    run_demo(camera_index=args.camera)
