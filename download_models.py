"""
NHAI FaceGuard SDK — Model Downloader
Run this ONCE (with internet) to download all required ONNX models.
After this, the entire SDK runs 100% offline.

Usage:
    python download_models.py
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# ── Model Definitions ───────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"

# MiniFASNet Anti-Spoofing Models (ONNX format)
# Source: HuggingFace — garciafido/minifasnet-v2-anti-spoofing-onnx
# and    yakhyo/face-anti-spoofing
LIVENESS_MODELS = {
    "MiniFASNetV1.onnx": (
        "https://huggingface.co/yakhyo/face-anti-spoofing/resolve/main"
        "/weights/MiniFASNetV1SE.onnx"
    ),
    "MiniFASNetV2.onnx": (
        "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx"
        "/resolve/main/MiniFASNetV2.onnx"
    ),
}

# InsightFace handles its own models (buffalo_sc pack)
# They are auto-downloaded to ~/.insightface/models/ on first FaceAnalysis call.
# This script only needs to download the liveness models.

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         NHAI FaceGuard SDK — Model Downloader            ║
║  This downloads liveness ONNX models (one-time setup)    ║
║  InsightFace models are auto-managed separately.         ║
╚══════════════════════════════════════════════════════════╝
"""


def download_file(url: str, dest: Path) -> bool:
    """Download a file with a progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            desc=dest.name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to download {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def verify_insightface_models() -> None:
    """
    Trigger InsightFace to download its buffalo_sc model pack.
    This ensures the detection + recognition models are cached locally.
    """
    print("\n[2/2] Verifying InsightFace models (buffalo_sc)...")
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_sc",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("  [✓] InsightFace models ready (cached in ~/.insightface/models/)")
    except Exception as e:
        print(f"  [ERROR] InsightFace model setup failed: {e}")
        print("  Try: pip install insightface")


def main():
    print(BANNER)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download liveness models
    print("[1/2] Downloading liveness models (MiniFASNet)...\n")
    all_ok = True
    for filename, url in LIVENESS_MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"  [✓] {filename} already exists, skipping.")
            continue
        print(f"  Downloading {filename}...")
        ok = download_file(url, dest)
        if not ok:
            all_ok = False
            print(f"\n  ⚠ Could not download {filename} automatically.")
            print(f"  Please download manually from:")
            print(f"    {url}")
            print(f"  And place it at: {dest}\n")

    # Step 2: InsightFace models
    verify_insightface_models()

    print("\n" + "=" * 60)
    if all_ok:
        print("  [✓] All models ready! SDK is now fully offline.")
    else:
        print("  [!] Some models need manual download (see above).")
        print("  The SDK will still run — liveness detection will")
        print("  return UNKNOWN until models are placed in ./models/")
    print("=" * 60)
    print("\nNext steps:")
    print("  Start API:   python api/server.py")
    print("  Live demo:   python demo/webcam_demo.py")
    print()


if __name__ == "__main__":
    main()
