<div align="center">

# 🎯 FaceTrack

### Real-Time Face Recognition, Emotion Analysis & Servo-Based Camera Tracking

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Arduino](https://img.shields.io/badge/Arduino-Uno%20R3-00979D?style=for-the-badge&logo=arduino&logoColor=white)](https://arduino.cc)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> Real-time face recognition + servo tracking using **ArcFace**, **Flask** & **Arduino**.  
> P-controller, emotion analysis, attendance system & GPT-4.1-nano AI assistant.  
> **WCE Sangli — B.Tech Project-II — Electronics Engineering — 2025–26**

<br/>

![FaceTrack Banner](https://img.shields.io/badge/FEED%20STATUS-LIVE-00ff87?style=for-the-badge)
![Faces](https://img.shields.io/badge/FACES%20DETECTED-UP%20TO%204-00e5ff?style=for-the-badge)
![Confidence](https://img.shields.io/badge/RECOGNITION-61%%20–%2083%25-ff6b35?style=for-the-badge)
![Latency](https://img.shields.io/badge/SERIAL%20LATENCY-%3C5ms-9b59b6?style=for-the-badge)

</div>

---

## 📌 Overview

**FaceTrack** is a real-time face detection, recognition, and servo-based camera tracking system built on a standard Windows laptop paired with an **Arduino Uno** microcontroller.

The system uses **InsightFace (ArcFace)** for 512-dimensional face recognition, **DeepFace** for 7-class emotion analysis, a **Proportional (P) Controller** for precise servo actuation that actively *centres* the detected face rather than just following it, and a **Flask web dashboard** for live monitoring, attendance management, and AI-powered analytics via **GPT-4.1-nano**.

This project replaces the earlier Raspberry Pi + HTTP architecture with a direct USB serial link to Arduino, reducing servo latency from **100–200 ms → <5 ms** and hardware cost from **₹6,600 → ₹3,200**.

> 🏫 **Academic Project — Group 14**  
> Department of Electronics Engineering, Walchand College of Engineering, Sangli  
> Academic Year 2025–2026

---

## ✨ Features

| Category | Feature |
|---|---|
| 👁️ **Vision** | Real-time face detection via InsightFace at ~12 fps |
| 🧑‍💼 **Recognition** | ArcFace 512-d cosine similarity matching |
| 😊 **Emotion** | 7-class emotion analysis via DeepFace (happy, sad, angry, fear, disgust, surprise, neutral) |
| 🎯 **Tracking** | P-controller servo tracking — face **actively centred**, not just followed |
| 🤖 **Hardware** | Arduino Uno pan-tilt control via USB serial at 115,200 baud |
| 📊 **Dashboard** | Flask web UI — live MJPEG stream, attendance, alerts, user profiles |
| 🗃️ **Database** | SQLite attendance with check-in / check-out and emotion logging |
| 🔔 **Audio** | Non-blocking TTS via gTTS + pygame with priority queue |
| 🧠 **AI** | Daily briefing, anomaly detection, per-person summary via GPT-4.1-nano |
| 📷 **Cameras** | Multi-camera support with live one-click switching |
| 🌐 **Network** | LAN-accessible dashboard from any device |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   USB Webcam                            │
│              640 × 480  ·  30 fps                       │
└───────────────────────┬─────────────────────────────────┘
                        │  USB
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Python / Flask  (Laptop)                   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │            Camera Pipeline (3 Threads)          │   │
│  │                                                 │   │
│  │  Thread 1: Camera Worker    → 30 fps capture   │   │
│  │  Thread 2: Detection Worker → ~12 fps           │   │
│  │                               InsightFace IoU  │   │
│  │  Thread 3: Recognition      → ArcFace cosine   │   │
│  │                               identity cache   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  P-Controller│  │   SQLite DB  │  │ GPT-4.1-nano│  │
│  │  Kp·err·sign │  │  Attendance  │  │ AI Assistant│  │
│  └──────┬───────┘  └──────────────┘  └─────────────┘  │
│         │                                               │
│  ┌──────▼───────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Arduino      │  │  gTTS+pygame │  │    Flask    │  │
│  │ Serial Bridge│  │  Speaker TTS │  │  MJPEG+REST │  │
│  └──────┬───────┘  └──────────────┘  └─────────────┘  │
└─────────│───────────────────────────────────────────────┘
          │  USB Serial  115,200 baud
          │  "pan,tilt\n" / "centre\n" / "quench\n"
          ▼
┌─────────────────────────────────────┐
│           Arduino Uno R3            │
│                                     │
│  Pin 9  (PWM Timer1) → Pan  SG90   │
│  Pin 10 (PWM Timer1) → Tilt SG90   │
│                                     │
│  External 5V ──── Servo VCC (×2)   │
│  Common  GND ──── Servo GND (×2)   │
└─────────────────────────────────────┘
```

---

## 🔬 The Core Innovation — P-Controller

The critical architectural fix over Project-I is replacing **direct pixel-to-angle mapping** with a **proportional controller on the error signal**.

### ❌ Wrong approach (Project-I)

```python
# Maps face POSITION → servo ANGLE
# Servo settles at fixed offset — face is NEVER centred
servo_angle = f(face_pixel_x)
```

If face is at `pixel 400` (80 px right of centre `320`), the servo moves to `~78°` and **stays there forever**.  
The face remains 80 px off-centre. Tracked, but never centred.

### ✅ Correct approach (FaceTrack)

```python
# Maps face ERROR → servo CORRECTION
error     = face_px - frame_centre_px       # how far off-centre
smoothed  = α * smoothed + (1 - α) * error  # EMA filter
new_angle = current_angle + Kp * smoothed   # P-step
```

When `error → 0`, `Δangle → 0` — servo holds still only when the face **IS** centred.  
**Self-correcting by construction.**

### Controller Parameters

| Parameter | Value | Description |
|---|---|---|
| `Kp_PAN` | `0.12 °/px` | Pan proportional gain |
| `Kp_TILT` | `0.10 °/px` | Tilt proportional gain |
| `DEADZONE_PX` | `25 px` | No-move zone at frame centre |
| `Initial α` | `0.20` | Fast EMA at face acquisition |
| `Stable α` | `0.65` | Damped EMA when face is centred |
| `Ramp` | `45 frames (~1.5s)` | α transition duration |
| `Min Δangle` | `1.0°` | Sub-degree suppression threshold |

---

## 🛠️ Hardware Requirements

| Component | Specification | Qty | Cost (₹) |
|---|---|---|---|
| Laptop / PC | Windows 10/11, i5/Ryzen 5, 8 GB RAM | 1 | existing |
| USB Webcam | 640×480, 30 fps, plug-and-play | 1 | 1,200 |
| Arduino Uno R3 | ATmega328P, USB-B | 1 | 550 |
| SG90 Servo Motor | 5V, 0–180°, 1.8 kg·cm | 2 | 400 |
| Pan-Tilt Bracket | Acrylic / 3D-printed, SG90-compatible | 1 | 350 |
| USB-A to USB-B Cable | 1.5 m | 1 | 120 |
| External 5V Supply | ≥ 2A output (USB power bank or 4×AA) | 1 | 200 |
| Jumper Wires | Male-to-male, 20 cm | 1 pack | 80 |
| Speaker | 3.5 mm / USB active speaker | 1 | 300 |
| **Total (new parts)** | | | **~₹ 3,200** |

### ⚡ Wiring Diagram

```
Arduino Uno
───────────────────────────────────────────────────────
Pin 9  (Hardware PWM) ──────── Pan  Servo  Signal (orange)
Pin 10 (Hardware PWM) ──────── Tilt Servo  Signal (orange)
GND                   ──────── Both Servo  GND    (brown)
GND                   ──────── External 5V GND
───────────────────────────────────────────────────────
External 5V Supply    ──────── Both Servo  VCC    (red)
───────────────────────────────────────────────────────
```

> ⚠️ **Critical:** Never power servos from Arduino's `5V` pin.  
> Peak draw ≈ 1.4 A combined will reset or damage the Arduino.  
> Always use a **dedicated external 5V supply** with a **shared GND**.

---

## 💻 Software Requirements

```
Python          3.10+
Arduino IDE     1.8+  (or Arduino CLI)
OS              Windows 10/11  or  Ubuntu 20.04+
RAM             8 GB minimum
Storage         4 GB (for Python env + InsightFace model weights)
GPU             Not required (CPU-only inference)
```

### Python Dependencies

```txt
flask>=2.0
opencv-python>=4.8
insightface>=0.7
deepface>=0.0.79
pyserial>=3.5
sqlalchemy>=2.0
gtts>=2.3
pygame>=2.5
python-dotenv>=1.0
openai>=1.0
numpy>=1.24
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/FaceTrack-RealTime-Recognition-ServoTracking.git
cd FaceTrack-RealTime-Recognition-ServoTracking
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> 📦 InsightFace will auto-download ArcFace model weights (~500 MB) on first run.  
> Ensure you have an internet connection for the first launch.

### 4. Configure Environment Variables

```bash
cp _env .env
```

Open `.env` and fill in your values:

```env
# ── Arduino ──────────────────────────────────────────────────
# Windows: COM3 / COM7 etc.   Linux: /dev/ttyACM0
ARDUINO_PORT=COM7

# ── AI Assistant ─────────────────────────────────────────────
# Free token at: https://aipipe.org
AIPIPE_TOKEN=your_aipipe_token_here

# Optional: direct OpenAI key (if not using AIPipe)
OPENAI_API_KEY=

# ── Camera ───────────────────────────────────────────────────
CAMERA_INDEX=0
FRAME_WIDTH=640
FRAME_HEIGHT=480

# ── Recognition ──────────────────────────────────────────────
# Cosine similarity threshold (0.0 – 1.0)
RECOGNITION_THRESHOLD=0.5

# Minimum seconds between attendance log entries
ATTENDANCE_COOLDOWN=60

# ── Servo ────────────────────────────────────────────────────
SERVO_UPDATE_HZ=25
```

### 5. Upload Arduino Firmware

**Using Arduino IDE:**
1. Open `arduino_servo/arduino_servo.ino`
2. Select **Board:** `Arduino Uno`
3. Select **Port:** `COM7` (or your port)
4. Click **Upload** ✓

**Using Arduino CLI:**
```bash
arduino-cli compile --fqbn arduino:avr:uno arduino_servo/
arduino-cli upload  --fqbn arduino:avr:uno --port COM7 arduino_servo/
```

### 6. Initialise Database

```bash
python -c "from database import init_db; init_db(); print('✓ Database ready')"
```

### 7. Run FaceTrack

```bash
python app.py
```

Open browser at:

```
Local:   http://localhost:5001
Network: http://YOUR_LAPTOP_IP:5001
```

---

## 📋 Usage Guide

### 👤 Enrolling a Person

1. Navigate to **`/users`** → click **`+ Enrol New Person`**
2. Enter Name, Employee ID, Role, Department
3. Stand in front of the camera
4. Click **Start Capture** — system collects 10 aligned face crops
5. Click **Save** — ArcFace embeddings stored in SQLite

### 📹 Starting the Camera

1. Navigate to **`/camera`**
2. Click **`▶ Start Camera`**
3. Wait for **LIVE** indicator → green ✓
4. Detection and recognition begin automatically

### 🎯 Servo Tracking

1. Go to **Settings** → set **Arduino Port** → enable **Servo**
2. Camera physically pans/tilts to follow the primary face
3. To lock onto a specific person:  
   → their profile page → **`◎ Track with Servo`**
4. To stop tracking: **`■ Stop Tracking`**
5. To release servos: servo will receive `quench` command

### 📊 Attendance

```
/attendance   → full log with date and department filter
/user/<id>    → per-person profile with visit history
/alerts       → active alerts with resolve button
```

### 🤖 AI Assistant

Navigate to **`/`** → AI Assistant tab and ask:

```
"Who was present today?"
"How many unknown persons were detected this week?"
"Generate today's briefing"
"Check for anomalies in the last 48 hours"
"Summarise Akshat's attendance pattern"
```

---

## 📁 Project Structure

```
FaceTrack-RealTime-Recognition-ServoTracking/
│
├── 📄 app.py                   Main Flask app + camera worker threads
├── 📄 database.py              SQLAlchemy ORM models
├── 📄 face_registry.py         InsightFace enrolment + recognition
├── 📄 arduino_serial.py        P-controller + ArduinoBridge serial bridge
├── 📄 servo_controller.py      ServoController wrapper (app.py interface)
├── 📄 ai_assistant.py          GPT-4.1-nano FaceTrackAI class
├── 📄 Speaker.py               gTTS + pygame non-blocking TTS
├── 📄 migrate_embeddings.py    dlib → ArcFace migration utility
│
├── 📁 arduino_servo/
│   └── 📄 arduino_servo.ino    Arduino Uno firmware (Servo.h)
│
├── 📁 templates/
│   ├── 📄 base.html            Base layout — dark surveillance theme
│   ├── 📄 index.html           Surveillance dashboard
│   ├── 📄 live.html            Live tracking feed
│   ├── 📄 camera.html          Camera feed + controls
│   ├── 📄 attendance.html      Attendance log table
│   ├── 📄 alerts.html          Alert management
│   ├── 📄 users.html           User list + enrolment form
│   └── 📄 user_detail.html     Per-person profile page
│
├── 📁 static/
│   └── 📁 snapshots/           Auto-captured face snapshots
│
├── 📄 database.db              SQLite database (auto-created on first run)
├── 📄 requirements.txt         Python dependencies
├── 📄 _env                     Environment variable template
└── 📄 README.md
```

---

## 🌐 API Reference

### Camera

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/video_feed` | MJPEG live camera stream |
| `POST` | `/camera/start` | Start camera worker thread |
| `POST` | `/camera/stop` | Stop camera worker thread |
| `GET` | `/api/camera/status` | Camera running state + frame info |
| `GET` | `/api/camera/list` | List all detected camera devices |
| `POST` | `/api/camera/switch` | Switch active camera by index |

### Servo

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/servo/configure` | Set port, enable/disable servo |
| `POST` | `/api/track/<uid>` | Lock servo to specific person |
| `POST` | `/api/track/stop` | Stop servo tracking (AUTO mode) |
| `GET` | `/api/track/status` | Current tracking target |

### Attendance & Alerts

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/attendance` | Paginated attendance log (JSON) |
| `GET` | `/api/alerts` | Active alerts (JSON) |
| `POST` | `/alert/<id>/resolve` | Resolve an alert |

### Persons

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/persons` | All enrolled persons |
| `POST` | `/api/persons/enroll` | Enrol new person with face crops |
| `GET` | `/user/<id>` | Per-person profile page |
| `POST` | `/user/<id>/flag` | Flag a person |
| `POST` | `/user/<id>/clear` | Clear a person's flag |

### AI Assistant

| Method | Endpoint | Description |
|---|---|---|
| `GET/POST` | `/api/ai/chat` | Free-form question with DB context |
| `GET` | `/api/ai/briefing` | GPT-generated daily summary |
| `GET` | `/api/ai/anomalies` | AI anomaly detection (last 48h) |
| `GET` | `/api/ai/person/<id>/summary` | Per-person behavioural summary |

---

## 📊 Performance Results

| Metric | Result | Condition |
|---|---|---|
| Recognition confidence range | **61% – 83%** | Indoor, frontal lighting |
| Peak confidence | **83%** (Gayatri) | Good frontal lighting, ~1m |
| Max simultaneous faces detected | **4** | Crowd scene (outdoor) |
| False positive identities | **0** | All test sessions |
| Max continuous session | **1,093 s (~18 min)** | No dropout or flip error |
| Serial latency (PC → Arduino) | **< 5 ms** | 115,200 baud USB |
| Total system latency | **120–160 ms** | Dominated by SG90 mechanical |
| Camera capture rate | **30 fps** | 640×480 MJPEG |
| InsightFace detection rate | **~12 fps** | CPU-only, Thread 2 |
| Hardware cost (new parts only) | **₹ 3,200** | Excl. laptop |

---

## 🐛 Troubleshooting

<details>
<summary><b>Arduino not detected / serial port error</b></summary>

```bash
# Windows — check Device Manager → Ports (COM & LPT)
# Update .env: ARDUINO_PORT=COM3  (or whichever COM port appears)

# Linux — find the port
ls /dev/ttyACM* /dev/ttyUSB*

# Linux — add user to dialout group (then log out & back in)
sudo usermod -aG dialout $USER
```
</details>

<details>
<summary><b>InsightFace model download fails</b></summary>

```bash
# Models auto-download on first run (~500 MB)
# Ensure internet connection for first launch
# Cached at: C:\Users\<you>\.insightface\models\  (Windows)
#            ~/.insightface/models/               (Linux)

# Manual download if needed:
python -c "import insightface; app = insightface.app.FaceAnalysis(); app.prepare(ctx_id=0)"
```
</details>

<details>
<summary><b>Camera not opening (black screen)</b></summary>

```bash
# Try a different camera index in .env
CAMERA_INDEX=1   # or 2

# Verify camera in Python
python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"
```
</details>

<details>
<summary><b>Servo moving in wrong direction</b></summary>

```python
# In arduino_serial.py — flip the sign constant:
PAN_SIGN  = -1   # change +1 → -1  if pan direction is reversed
TILT_SIGN = +1   # change -1 → +1  if tilt direction is reversed
```
</details>

<details>
<summary><b>No audio / gTTS errors</b></summary>

```bash
pip install gTTS pygame

# Ensure speaker is connected
# Ensure system volume is not muted
# Check internet connection (gTTS requires Google API access)
```
</details>

<details>
<summary><b>Recognition confidence too low</b></summary>

- Enrol more face samples per person (different angles, lighting)
- Ensure face is well-lit and camera is at eye level
- Lower threshold in `.env`: `RECOGNITION_THRESHOLD=0.4`
- Avoid strong backlighting during enrolment
</details>

<details>
<summary><b>AI assistant returns "unavailable"</b></summary>

```bash
# Check your .env file has a valid token:
AIPIPE_TOKEN=your_actual_token_here

# Test the token:
python -c "from ai_assistant import ai_assistant; print(ai_assistant._client)"
```
</details>

---

## 🔮 Future Scope

- [ ] **PID controller** — add I and D terms for faster, overshoot-free tracking
- [ ] **GPU acceleration** — NVIDIA Jetson Nano for 30+ fps InsightFace detection
- [ ] **Liveness detection** — blink/depth anti-spoofing to prevent photo attacks
- [ ] **OpenAI Whisper** — on-device voice commands without internet dependency
- [ ] **Brushless gimbal motor** — eliminate 80–120 ms mechanical response bottleneck
- [ ] **Cloud sync** — Firebase / AWS DynamoDB for multi-site attendance aggregation
- [ ] **YOLOv8 integration** — multi-class object tracking (people, vehicles, packages)
- [ ] **Raspberry Pi 5 edge deployment** — fully self-contained node, no laptop needed
- [ ] **RL adaptive controller** — self-tuning Kp and EMA via reinforcement learning
- [ ] **3-axis IMU stabilisation** — MPU-6050 for mobile/vehicle deployment

---

## 👥 Team

<div align="center">

| | Name | Roll No. | Contribution |
|---|---|---|---|
| 👩‍💻 | **Gayatri Dabhade** | 22410052 | Flask dashboard, database design, AI assistant |
| 👩‍💻 | **Achala Paradeshi** | 22410053 | Face recognition pipeline, enrolment system |
| 👨‍💻 | **Akshat B Gupta** | 22410063 | Arduino firmware, P-controller, serial bridge |

**Project Guide:** Prof. Dr. S. D. Ruikar  
*Head of Department, Electronics Engineering*  
Walchand College of Engineering, Sangli *(Government-Aided Autonomous Institute)*

</div>

---

## 📄 License

```
MIT License

Copyright (c) 2026 Gayatri Dabhade, Achala Paradeshi, Akshat B Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 📚 Citation

```bibtex
@misc{facetrack2026,
  title   = {FaceTrack: Real-Time Face Recognition, Emotion Analysis
             and Proportional Servo Tracking via Arduino USB Serial},
  author  = {Dabhade, Gayatri and Paradeshi, Achala and
             Gupta, Akshat B and Ruikar, S. D.},
  year    = {2026},
  school  = {Walchand College of Engineering, Sangli},
  note    = {B.Tech Project-II, Department of Electronics Engineering,
             Academic Year 2025--2026}
}
```

---

## 🙏 Acknowledgements

- [**InsightFace**](https://insightface.ai) — ArcFace production-ready recognition models
- [**DeepFace**](https://github.com/serengil/deepface) — Emotion and face analysis library
- [**Flask**](https://flask.palletsprojects.com) — Lightweight Python web framework
- [**OpenCV**](https://opencv.org) — Computer vision and image processing
- [**Arduino**](https://arduino.cc) — Open-source microcontroller platform
- [**gTTS**](https://gtts.readthedocs.io) — Google Text-to-Speech Python library
- [**AIPipe**](https://aipipe.org) — GPT-4.1-nano API proxy

---

<div align="center">

**⭐ Star this repository if you found it useful!**

*Built with ❤️ at Walchand College of Engineering, Sangli — 2025–2026*

![Footer](https://img.shields.io/badge/WCE%20Sangli-Electronics%20Engineering-1F3864?style=for-the-badge)
![Footer](https://img.shields.io/badge/B.Tech%20Project%20II-Group%2014-2E5796?style=for-the-badge)
![Footer](https://img.shields.io/badge/AY-2025--2026-00e5ff?style=for-the-badge)

</div>
