<p align="center">
  <strong>🛡 NHAI FaceGuard SDK</strong>
</p>

<h3 align="center">Offline Facial Recognition & Liveness Detection for Datalake 3.0</h3>

<p align="center">
  <em>NHAI Innovation Hackathon 7.0 Submission</em>
</p>

<p align="center">
  <code>100% Offline</code> · <code>CPU-Only</code> · <code>~7 MB Models</code> · <code>Zero Cloud Dependency</code>
</p>

---

## ✨ What Is This?

A production-ready, edge-optimized facial biometric SDK designed for NHAI's remote highway zones, toll plazas, and construction sites — locations where internet connectivity is unreliable or nonexistent.

The system handles the complete biometric pipeline on-device:

```
Camera Frame → Face Detection → Liveness Check → Face Recognition → Local Database Match
```

**Key highlights:**
- 🔒 **100% Offline** — Zero network calls after one-time model download
- 🎭 **Anti-Spoofing** — Passive (MiniFASNet ensemble) + Active (challenge-response) liveness detection
- 📱 **Mobile-Ready** — Built-in web UI accessible from any phone browser on the same network
- 🔄 **Sync & Purge** — Local attendance ledger with cloud sync capability
- ⚡ **< 200ms latency** — Full pipeline on CPU in under 200 milliseconds

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Models (One-Time, Needs Internet)
```bash
python download_models.py
```
This downloads **MiniFASNet V1+V2** liveness models to `./models/` and caches **InsightFace buffalo_sc** models to `~/.insightface/models/`. After this step, **no internet is ever needed again**.

### 3. Start the API Server
```bash
python api/server.py
```

| Resource | URL |
|----------|-----|
| Interactive Web UI | `http://localhost:8000` |
| Swagger API Docs | `http://localhost:8000/docs` |

### 4. Run the Live Webcam Demo (Desktop)
```bash
python demo/webcam_demo.py
```
**Controls:** `R` = Register face · `S` = Screenshot · `Q` = Quit

---

## 📱 Mobile Phone Access

The SDK includes a **mobile-responsive web UI** served directly by the API server. Any phone on the same Wi-Fi network can access it — no app installation needed.

### Setup Steps

1. **Start the server** (it binds to `0.0.0.0` by default):
   ```bash
   python api/server.py
   ```

2. **Find your machine's LAN IP:**
   ```bash
   # Windows
   ipconfig
   # Look for Wi-Fi adapter → IPv4 Address (e.g., 192.168.1.5)
   ```

3. **Open on your phone's browser:**
   ```
   http://<your-laptop-ip>:8000
   ```

### ⚠️ Important Notes

**Windows Firewall** — You must allow inbound connections on port 8000:
```powershell
netsh advfirewall firewall add rule name="NHAI FaceGuard" dir=in action=allow protocol=TCP localport=8000
```

**HTTPS for Camera** — Browsers block camera access over plain HTTP on non-localhost origins. Use one of these workarounds:

| Method | How |
|--------|-----|
| **Chrome Flag** (easiest) | On phone Chrome: `chrome://flags` → search "Insecure origins treated as secure" → add `http://<your-ip>:8000` → Relaunch |
| **ngrok tunnel** | Run `ngrok http 8000` to get an HTTPS URL |
| **Self-signed cert** | Run uvicorn with `--ssl-keyfile` / `--ssl-certfile` flags |

### Mobile UI Features

The web UI has three modes accessible via bottom navigation tabs:

| Tab | Action |
|-----|--------|
| **Identify** | Capture a face → 1:N search against enrolled database → shows name, confidence, liveness |
| **Register** | Enter name + ID → capture face → enroll in database |
| **Verify** | Enter a Person ID → capture face → 1:1 identity confirmation |

---

## 🔌 API Endpoints

### Core Biometrics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/register` | Enroll a new face (base64 image + person ID + name) |
| `POST` | `/identify` | 1:N identification with optional liveness check |
| `POST` | `/verify` | 1:1 verification against a specific person ID |
| `GET` | `/faces` | List all enrolled persons |
| `DELETE` | `/face/{id}` | Remove a person from the database |
| `GET` | `/health` | SDK status, enrolled count, liveness readiness |

### Sync & Purge (Offline Attendance Ledger)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/attendance/unsynced` | View queued offline attendance logs |
| `POST` | `/attendance/sync` | Upload logs to AWS & purge local records |

Successful identifications/verifications with `liveness = REAL` are automatically logged to a local SQLite attendance table. When connectivity is restored, these logs can be batch-synced to cloud infrastructure and purged from the device to free storage.

---

## 📡 API Usage Examples

### Register a Face
```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{
    "person_id": "EMP001",
    "person_name": "Arjun Sharma",
    "image_b64": "<BASE64_ENCODED_IMAGE>"
  }'
```

### Identify a Face
```bash
curl -X POST http://localhost:8000/identify \
  -H "Content-Type: application/json" \
  -d '{
    "image_b64": "<BASE64_ENCODED_IMAGE>",
    "check_liveness": true,
    "top_k": 3
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

## 🏗 Architecture

```
Camera Frame
    │
    ▼
┌──────────────────┐
│  Face Detection   │  SCRFD 500M (via InsightFace ONNX)
│  (detector.py)    │  → Bounding box + 5-point landmarks
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Liveness Check   │  Tier 1: MiniFASNet V1+V2 ensemble (passive)
│  (liveness.py)    │  Tier 2: Texture + FFT + HSV fallback (OpenCV)
└────────┬─────────┘
         │ (only if REAL)
         ▼
┌──────────────────┐
│  Face Recognition │  ArcFace MobileFaceNet (via InsightFace ONNX)
│  (recognizer.py)  │  → 512-dim L2-normalized embedding
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Local Database   │  SQLite + in-memory cosine search
│  (database.py)    │  → Person ID + name + confidence
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Attendance Log   │  Local offline ledger
│  (database.py)    │  → Sync to cloud when network available
└──────────────────┘
```

### Liveness Detection — Two-Tier System

| Tier | Method | When Used |
|------|--------|-----------|
| **Primary** | MiniFASNet V1+V2 ONNX ensemble | When model files present in `./models/` |
| **Fallback** | Laplacian variance + FFT frequency analysis + HSV saturation | Always available (pure OpenCV, zero download) |

The **webcam demo** also includes an **Active Challenge-Response** system that prompts users to smile or turn their head, verified via landmark geometry, before running the passive spoof check.

---

## 🧠 Models Used

| Model | Task | Size | Accuracy |
|-------|------|------|----------|
| SCRFD 500M | Face Detection | ~1 MB | mAP 0.85 |
| MiniFASNet V1+V2 | Passive Liveness | ~2 MB | >98% anti-spoof |
| ArcFace MobileFaceNet | Face Recognition | ~4 MB | 99.5% LFW |

**Total model footprint: ~7 MB** — runs on any CPU, no GPU required.

---

## 📂 Project Structure

```
tool/
├── api/
│   ├── server.py              ← FastAPI REST server (8 endpoints + Sync & Purge)
│   └── templates/
│       └── index.html         ← Mobile-responsive web UI (camera + identify/register/verify)
├── core/
│   ├── __init__.py            ← Package exports
│   ├── detector.py            ← SCRFD face detection (InsightFace)
│   ├── liveness.py            ← MiniFASNet passive anti-spoofing + OpenCV fallback
│   ├── recognizer.py          ← ArcFace 512-dim embedding extraction
│   └── database.py            ← SQLite face store + attendance logging + sync/purge
├── demo/
│   ├── webcam_demo.py         ← Desktop webcam demo with active challenge-response
│   └── screenshots/           ← Saved screenshots from demo
├── utils/
│   ├── config.py              ← All thresholds, paths, and constants
│   └── image_utils.py         ← Alignment, base64 decoding, drawing helpers
├── models/                    ← ONNX model files (after download_models.py)
├── download_models.py         ← One-time model setup script
├── requirements.txt           ← Python dependencies
├── facedb.sqlite              ← Local face database (auto-created)
└── nhai_faceguard_proposal.md ← Hackathon proposal document
```

---

## 🔐 Offline Guarantee

All inference runs via **ONNX Runtime** on CPU. After `download_models.py` completes:

- ✅ Zero network calls at runtime
- ✅ Works in airplane mode
- ✅ Works in RF-shielded zones (tunnels, underpasses)
- ✅ No cloud API keys needed
- ✅ No images leave the device — only 512-dim math vectors are stored

---

## ⚙️ Configuration

All tunable parameters are centralized in [`utils/config.py`](utils/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DET_THRESHOLD` | `0.5` | Face detection confidence cutoff |
| `LIVENESS_THRESHOLD` | `0.6` | Score > threshold → REAL |
| `RECOGNITION_THRESHOLD` | `0.35` | Cosine distance cutoff for match (lower = stricter) |
| `API_HOST` | `0.0.0.0` | Server bind address (LAN-accessible) |
| `API_PORT` | `8000` | Server port |

---

## 📈 Performance vs. Cloud Solutions

| Metric | NHAI FaceGuard (Offline) | Cloud Biometrics (AWS/Azure) |
|--------|--------------------------|------------------------------|
| **Network** | None required | Active low-latency internet |
| **Latency** | ~120–180ms | ~800–2500ms |
| **Data Privacy** | Maximum (nothing leaves device) | Images sent to remote servers |
| **Cost per Transaction** | ₹0 | Recurring per-request fees |
| **Spoof Detection** | Dual-model passive + active challenge | Often requires premium tier |

---

## 📋 Dependencies

| Package | Purpose |
|---------|---------|
| `onnxruntime` | Offline CPU inference engine |
| `opencv-python` | Frame capture & image processing |
| `insightface` | SCRFD detector + ArcFace recognizer |
| `numpy`, `scipy`, `Pillow` | Array ops, transforms, image I/O |
| `fastapi`, `uvicorn` | REST API server |
| `requests`, `tqdm` | One-time model download only |

---

## 🏆 Hackathon

**NHAI Innovation Hackathon 7.0**
Challenge: *Develop a mobile-based secure offline facial recognition and liveness detection system for remote locations*
