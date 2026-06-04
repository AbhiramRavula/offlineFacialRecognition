# NHAI FaceGuard SDK — README

## Offline Facial Recognition & Liveness Detection
**NHAI Innovation Hackathon 7.0 Submission**

> 100% offline · CPU-only · Zero network calls at runtime

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Models (One-Time, Needs Internet)
```bash
python download_models.py
```
After this, **no internet required ever again**.

### 3. Start the API Server
```bash
python api/server.py
```
API runs at: `http://127.0.0.1:8000`
Interactive docs: `http://127.0.0.1:8000/docs`

### 4. Run Live Webcam Demo
```bash
python demo/webcam_demo.py
```
**Controls:** `R` = Register face | `S` = Screenshot | `Q` = Quit

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | SDK status & enrolled count |
| `POST` | `/register` | Enroll a new face |
| `POST` | `/identify` | 1:N identify from database |
| `POST` | `/verify` | 1:1 verify identity |
| `GET` | `/faces` | List enrolled persons |
| `DELETE` | `/face/{id}` | Remove a person |

Full API docs available at `/docs` when server is running.

---

## Example API Usage

### Register a Face
```bash
curl -X POST http://127.0.0.1:8000/register \
  -H "Content-Type: application/json" \
  -d '{
    "person_id": "EMP001",
    "person_name": "Arjun Sharma",
    "image_b64": "<BASE64_ENCODED_IMAGE>"
  }'
```

### Identify a Face
```bash
curl -X POST http://127.0.0.1:8000/identify \
  -H "Content-Type: application/json" \
  -d '{
    "image_b64": "<BASE64_ENCODED_IMAGE>",
    "check_liveness": true
  }'
```

### Sample Response
```json
{
  "success": true,
  "liveness": "REAL",
  "liveness_score": 0.923,
  "matches": [
    {
      "person_id": "EMP001",
      "person_name": "Arjun Sharma",
      "distance": 0.18,
      "confidence": 0.82
    }
  ],
  "message": "Identified as: Arjun Sharma"
}
```

---

## Architecture

```
Camera Frame
    │
    ▼
┌──────────────┐
│ Face Detect  │  SCRFD 500M (via InsightFace ONNX)
│ (detector.py)│  → Bounding box + 5-point landmarks
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Liveness    │  MiniFASNet V1 + V2 ensemble (passive)
│ (liveness.py)│  → REAL / SPOOF classification
└──────┬───────┘
       │ (only if REAL)
       ▼
┌──────────────┐
│  Recognize   │  ArcFace MobileFaceNet (via InsightFace ONNX)
│(recognizer.py│  → 512-dim L2 embedding
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Database    │  SQLite + in-memory cosine search
│(database.py) │  → Person ID + name + confidence
└──────────────┘
```

---

## Models Used

| Model | Task | Size | Accuracy |
|-------|------|------|----------|
| SCRFD 500M | Face Detection | ~1 MB | mAP 0.85 |
| MiniFASNet V1+V2 | Liveness Detection | ~2 MB | >98% anti-spoof |
| ArcFace MobileFaceNet | Face Recognition | ~4 MB | 99.5% LFW |

**Total model footprint: ~7 MB**

---

## Project Structure

```
tool/
├── models/              ← ONNX model files (after download)
├── core/
│   ├── detector.py      ← SCRFD face detection
│   ├── liveness.py      ← MiniFASNet anti-spoofing
│   ├── recognizer.py    ← ArcFace face recognition
│   └── database.py      ← SQLite face database
├── api/
│   └── server.py        ← FastAPI REST server
├── utils/
│   ├── config.py        ← Thresholds & paths
│   └── image_utils.py   ← Preprocessing helpers
├── demo/
│   └── webcam_demo.py   ← Live webcam demo
├── download_models.py   ← One-time model setup
└── requirements.txt
```

---

## Offline Guarantee

All inference uses **ONNX Runtime** on CPU. After `download_models.py` completes:
- ✅ Zero network calls
- ✅ Works in airplane mode
- ✅ Works in RF-shielded zones
- ✅ No cloud API keys needed

---

## Hackathon Submission
**NHAI Innovation Hackathon 7.0**
Challenge: Offline Facial Recognition & Liveness Detection for Datalake 3.0
Submission Deadline: 05.06.2026
