"""
NHAI FaceGuard SDK — Live Webcam Demo with Active Challenge-Response
Real-time facial recognition + active liveness + passive anti-spoofing.

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

logging.basicConfig(level=logging.WARNING)

WINDOW_TITLE = "NHAI FaceGuard — Active Liveness Demo | R=Register Q=Quit"
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")


# ── Active Liveness: Landmark-Based Metrics ─────────────────────────────────

def get_active_metrics(landmarks: np.ndarray):
    """
    Compute smile_ratio and turn_ratio from 5-point face landmarks.

    Landmark indices (InsightFace kps):
        0: Left Eye (LE)
        1: Right Eye (RE)
        2: Nose Tip
        3: Left Mouth Corner (LM)
        4: Right Mouth Corner (RM)

    Returns:
        smile_ratio: mouth_width / eye_distance (higher = smiling)
        turn_ratio:  horizontal nose position between eyes (0.5 = centered)
    """
    le = landmarks[0]
    re = landmarks[1]
    nose = landmarks[2]
    lm = landmarks[3]
    rm = landmarks[4]

    # Smile: ratio of mouth width to inter-eye distance
    eye_dist = np.linalg.norm(re - le)
    mouth_width = np.linalg.norm(rm - lm)
    smile_ratio = mouth_width / (eye_dist + 1e-6)

    # Head turn: horizontal position of nose tip relative to eyes
    # 0.5 = centered, <0.35 = turned left, >0.65 = turned right
    dx_eyes = re[0] - le[0] + 1e-6
    turn_ratio = (nose[0] - le[0]) / dx_eyes

    return smile_ratio, turn_ratio


# ── HUD Drawing ─────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, fps: float, enrolled_count: int, state: str) -> np.ndarray:
    """Draw heads-up display with FPS, enrollment count, and current state."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
    cv2.putText(frame, f"Enrolled: {enrolled_count}", (150, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
    cv2.putText(frame, f"STATE: {state}", (w - 260, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return frame


def draw_challenge_prompt(frame: np.ndarray, challenge: str, time_left: float) -> np.ndarray:
    """Draw the active challenge instruction card on the frame."""
    cv2.rectangle(frame, (30, 30), (480, 145), (25, 25, 25), -1)
    cv2.rectangle(frame, (30, 30), (480, 145), (0, 180, 240), 2)
    cv2.putText(frame, f"ACTIVE CHALLENGE: {challenge}", (50, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Time Left: {time_left:.1f}s", (50, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def draw_result_badge(frame: np.ndarray, passed: bool, text: str) -> np.ndarray:
    """Draw a PASSED or FAILED result badge on the frame."""
    color = (0, 200, 0) if passed else (0, 0, 220)
    label = "LIVENESS PASSED (REAL)" if passed else "LIVENESS REJECTED"
    cv2.rectangle(frame, (30, 30), (550, 145), (25, 25, 25), -1)
    cv2.rectangle(frame, (30, 30), (550, 145), color, 2)
    cv2.putText(frame, label, (50, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, text, (50, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


# ── Registration ────────────────────────────────────────────────────────────

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


# ── Main Demo Loop ──────────────────────────────────────────────────────────

def run_demo(camera_index: int = 0) -> None:
    print("=" * 60)
    print("  NHAI FaceGuard — Active Challenge Live Demo")
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

    # ── Active Liveness State Machine ──────────────────────────────────────
    STATE_IDLE      = "IDLE"
    STATE_CHALLENGE = "CHALLENGE"
    STATE_PASSED    = "PASSED"
    STATE_FAILED    = "FAILED"

    current_state        = STATE_IDLE
    current_challenge    = None
    challenge_timeout    = 6.0   # seconds allowed to perform the action
    challenge_start_time = 0.0
    verified_name        = ""
    state_display_timer  = 0.0

    CHALLENGES = ["SMILE", "TURN LEFT", "TURN RIGHT"]

    print("\n  [✓] Camera open. Starting Active Challenge demo...\n")
    print("  Controls:  R = Register face | S = Screenshot | Q = Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        display = frame.copy()
        face = detector.detect_largest(frame)

        if face is not None:
            # Draw landmarks and bounding box
            for pt in face.landmarks:
                cv2.circle(display, (int(pt[0]), int(pt[1])), 4, (0, 140, 255), -1)
            cv2.rectangle(display, (face.x1, face.y1), (face.x2, face.y2),
                          (255, 255, 255), 1)

            # Compute active metrics from landmarks
            smile_ratio, turn_ratio = get_active_metrics(face.landmarks)

            # ── State Machine ──────────────────────────────────────────────

            if current_state == STATE_IDLE:
                # Pick a random challenge when a face appears
                current_challenge = np.random.choice(CHALLENGES)
                current_state = STATE_CHALLENGE
                challenge_start_time = time.time()
                print(f"[ACTIVE LIVENESS] Challenge issued: {current_challenge}")

            elif current_state == STATE_CHALLENGE:
                elapsed   = time.time() - challenge_start_time
                time_left = max(0.0, challenge_timeout - elapsed)

                draw_challenge_prompt(display, current_challenge, time_left)

                # Check if the user performed the requested action
                passed = False
                if current_challenge == "SMILE" and smile_ratio > 0.83:
                    passed = True
                elif current_challenge == "TURN LEFT" and turn_ratio < 0.33:
                    passed = True
                elif current_challenge == "TURN RIGHT" and turn_ratio > 0.67:
                    passed = True

                if passed:
                    # Active check passed — now run passive anti-spoofing
                    print("[ACTIVE] Challenge passed. Running passive spoof check...")
                    if liveness.is_ready:
                        p_res, p_score = liveness.check(frame, face.bbox)
                        if p_res == LivenessResult.SPOOF:
                            current_state = STATE_FAILED
                            state_display_timer = time.time()
                            print("[PASSIVE] SPOOF detected after active pass!")
                        else:
                            current_state = STATE_PASSED
                            state_display_timer = time.time()

                            # Perform recognition and log attendance
                            emb = recognizer.get_embedding_from_full_image(
                                frame, face.bbox, face.landmarks
                            )
                            if emb is not None:
                                matches = db.identify(emb, top_k=1)
                                if matches:
                                    pid, name, dist = matches[0]
                                    conf = max(0.0, 1.0 - dist)
                                    verified_name = f"{name} ({conf:.0%})"
                                    db.log_attendance(
                                        pid, p_score, "active-passive-liveness"
                                    )
                                    print(f"[VERIFIED] {name} (distance: {dist:.4f})")
                                else:
                                    verified_name = "Unknown — Not in Database"
                            else:
                                verified_name = "Embedding extraction failed"
                    else:
                        # No passive model — pass anyway
                        current_state = STATE_PASSED
                        state_display_timer = time.time()
                        verified_name = "Real Face (passive model unavailable)"

                elif elapsed >= challenge_timeout:
                    current_state = STATE_FAILED
                    state_display_timer = time.time()
                    print("[ACTIVE] Challenge timed out!")

            elif current_state == STATE_PASSED:
                draw_result_badge(display, True, f"Verified: {verified_name}")
                if time.time() - state_display_timer >= 3.0:
                    current_state = STATE_IDLE

            elif current_state == STATE_FAILED:
                draw_result_badge(display, False, "Spoof Attack or Timeout Detected")
                if time.time() - state_display_timer >= 3.0:
                    current_state = STATE_IDLE

        else:
            # No face detected — reset state machine
            current_state = STATE_IDLE
            cv2.putText(display, "No face detected — position yourself in front of the camera",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 140, 255), 2)

        # FPS counter
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        draw_hud(display, fps, db.count(), current_state)
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
    parser = argparse.ArgumentParser(description="NHAI FaceGuard Active Liveness Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()
    run_demo(camera_index=args.camera)
