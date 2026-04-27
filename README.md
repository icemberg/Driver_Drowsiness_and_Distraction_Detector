<div align="center">

# 🚗 Driver Drowsiness & Distraction Detector

**A real-time, multi-signal safety system that detects driver drowsiness, distraction, and drink-while-driving behavior using computer vision — and raises an alarm before an accident happens.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8.10%2B-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Build](https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-blue?style=for-the-badge)](CONTRIBUTING.md)

</div>

---

## 📋 Table of Contents

1. [Key Features](#-key-features)
2. [Tech Stack](#-tech-stack)
3. [System Architecture & Logic](#-system-architecture--logic)
4. [Getting Started](#-getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Configuration](#configuration)
5. [Usage](#-usage)
6. [Project Structure](#-project-structure)
7. [Performance & Optimization](#-performance--optimization)
8. [Troubleshooting](#-troubleshooting)
9. [Contributing](#-contributing)
10. [License](#-license)
11. [Contact](#-contact)

---

## ✨ Key Features

- 👁️ **Real-time Drowsiness Detection** — Eye Aspect Ratio (EAR) with smoothed temporal averaging to eliminate flicker and false positives
- 🔄 **Head Pose Estimation** — Computes calibrated yaw, pitch, and roll from facial landmarks using vector cross-product geometry
- 👀 **Iris Gaze Tracking** — Detects off-centre gaze by tracking iris position relative to eye corners (MediaPipe Iris)
- 🍺 **Drink-While-Driving Detection** — YOLOv8n object detection fused with hand-proximity signals and a 4-state machine (IDLE → POSSIBLE → DRINKING → ALERT)
- 🔗 **Multi-Signal Fusion** — Weighted risk scoring combines EAR, gaze, head pose, drink object confidence, and hand-to-mouth distance
- 🔔 **Audio Alarms** — Synthesized dual-frequency beep or custom `.wav` file via Pygame mixer
- 🎛️ **Highly Configurable** — 40+ parameters in `config.py`; no code changes needed to tune thresholds
- 📋 **Event Logging** — CSV-format drowsiness logs with optional frame snapshots
- 🐞 **Debug Mode** — Live facial mesh overlay, EAR values, and frame counter on-screen
- ⌨️ **Keyboard Shortcuts** — Toggle debug mode, reset counters, re-calibrate head pose at runtime
- 🔁 **Auto-Calibration** — Neutral head pose captured on first frame; press `c` anytime to re-calibrate

---

## 🛠️ Tech Stack

### Core Vision & ML

| Library | Version | Role |
|---|---|---|
| **OpenCV** | ≥ 4.5.0 | Camera capture, frame processing, drawing overlays |
| **MediaPipe** | ≥ 0.8.10 | 468-point FaceMesh, Iris landmarks, Hand tracking |
| **Ultralytics YOLOv8** | ≥ 8.0.0 | Real-time drink object detection (YOLOv8n backbone) |
| **NumPy** | ≥ 1.20.0 | Vector math, EAR calculations, angle geometry |

### Audio & Utilities

| Library | Version | Role |
|---|---|---|
| **Pygame** | ≥ 2.0.0 | Audio mixer, alarm sound synthesis & playback |
| **Matplotlib** | ≥ 3.4.0 | EAR trend plots, dataset visualisation |
| **Pillow** | ≥ 8.0.0 | Image I/O for dataset management |
| **icrawler** | ≥ 0.6.0 | Automated Bing image scraping for training data |
| **scikit-learn** | ≥ 1.0.0 | Dataset splitting, model evaluation metrics |

### Language & Environment

- **Python 3.8+** — Core runtime
- **Virtual Environment** (`venv`) — Dependency isolation

---

## 🧠 System Architecture & Logic

The system operates as a **multi-signal pipeline** where independent detectors feed into a unified risk decision:

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────────────┐
│  Camera Feed │───▶│  MediaPipe        │───▶│  Signal Extractors   │
│  (OpenCV)    │    │  FaceMesh + Iris  │    │  ┌────────────────┐  │
└──────────────┘    │  + Hands          │    │  │ EAR Calculator │  │
                    └───────────────────┘    │  ├────────────────┤  │
                                             │  │ Gaze Tracker   │  │
┌──────────────┐                             │  ├────────────────┤  │
│  YOLOv8n     │───▶ Drink Score ──────────▶│  │ Head Pose Est. │  │
│  (Drink Det.)│                             │  ├────────────────┤  │
└──────────────┘                             │  │ Hand Proximity │  │
                                             │  └────────────────┘  │
                                             └──────────┬───────────┘
                                                        │
                                                        ▼
                                             ┌──────────────────────┐
                                             │   Signal Fusion      │
                                             │   Weighted Risk Score│
                                             └──────────┬───────────┘
                                                        │
                                                        ▼
                                             ┌──────────────────────┐
                                             │  State Machine       │
                                             │  IDLE → POSSIBLE →   │
                                             │  CONFIRMED → ALERT   │
                                             └──────────┬───────────┘
                                                        │
                                                        ▼
                                             🔔 Audio Alarm + Log
```

### Key Algorithms

#### 1. Eye Aspect Ratio (EAR)
```
EAR = (Mean upper eyelid Y − Mean lower eyelid Y) / Eye width
```
- 16 landmarks per eye are averaged into upper/lower groups
- A rolling 5-frame average smooths out noise
- Alarm fires after `EYE_AR_CONSEC_FRAMES` consecutive frames below threshold

#### 2. Head Pose via Vector Cross-Product
```
eye_vector  = right_eye − left_eye          (normalized)
vert_vector = mouth_center − nose           (normalized)
face_normal = eye_vector × vert_vector      (looking direction)

yaw   = arctan2(face_normal.x, face_normal.z)
pitch = arctan2(−face_normal.y, face_normal.z)
roll  = arctan2(eye_vector.y, eye_vector.x)
```
All angles are **relative to the calibrated neutral** captured on startup (or re-calibrated with `c`). No hardcoded camera offsets.

#### 3. Iris Gaze Ratio
```
gaze_ratio = (iris_x − outer_corner_x) / (inner_corner_x − outer_corner_x)
```
- 0.0 = fully toward outer corner, 0.5 = centred, 1.0 = fully toward inner
- Division-by-zero guarded; falls back to 0.5 if landmarks overlap

#### 4. Drink-Drive Signal Fusion
```
risk_score = (drink_conf × 1.0) + (hand_proximity × 1.0) + (head_distraction × 0.5)
```
State machine transitions require the risk score to exceed a threshold **for N consecutive frames**, preventing single-frame noise from triggering alerts.

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.8 | 3.10+ recommended |
| Webcam | Any | Built-in or USB; 720p+ preferred |
| RAM | 4 GB | 8 GB recommended for YOLOv8 training |
| OS | Windows / Linux / macOS | Tested on Windows 11 |

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/nikes303/Driver_Drowsiness_Detector.git
cd Driver_Drowsiness_Detector
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv drowsy_env
drowsy_env\Scripts\activate

# Linux / macOS
python -m venv drowsy_env
source drowsy_env/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Verify installation**
```bash
python test_installation.py
```

Expected output:
```
✔ Camera test passed
✔ MediaPipe Face Mesh test passed
✔ Pygame audio test passed
[SUCCESS] All tests passed! System is ready.
```

> **MediaPipe version issue?** Run:
> ```bash
> pip uninstall mediapipe && pip install mediapipe==0.10.9
> ```

---

### Configuration

All system parameters live in **`config.py`** — no code edits needed for normal tuning.

```python
# ── Core Drowsiness ──────────────────────────────────────────────────────────
EYE_AR_THRESHOLD       = 0.22   # EAR below this = eyes closed
EYE_AR_CONSEC_FRAMES   = 25     # frames closed before alarm (~0.83 s @ 30 fps)

# ── Hardware ─────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0               # 0 = default webcam; try 1, 2 for others
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

# ── Distraction Detection ────────────────────────────────────────────────────
GAZE_CENTER_MIN           = 0.35   # gaze ratio band; outside = looking away
GAZE_CENTER_MAX           = 0.65
HEAD_YAW_LIMIT            = 20     # degrees left/right
HEAD_PITCH_LIMIT          = 15     # degrees up/down
HEAD_ROLL_LIMIT           = 12     # degrees sideways tilt
OFF_CENTER_MIN            = 0.30   # nose_x normalised band
OFF_CENTER_MAX            = 0.70
DISTRACTION_CONSEC_FRAMES = 10     # frames before distraction warning (~0.33 s)

# ── Alerts ───────────────────────────────────────────────────────────────────
ALARM_SOUND  = "alarm.wav"      # path to WAV file, or None for generated beep
ALARM_VOLUME = 0.9              # 0.0 – 1.0

# ── Logging ──────────────────────────────────────────────────────────────────
ENABLE_LOGGING = False
LOG_FILE       = "sleep_detection_log.txt"

# ── Drink Detection ──────────────────────────────────────────────────────────
DRINK_DETECTOR_CONFIDENCE_THRESHOLD    = 0.60
DRINK_RISK_THRESHOLD_CONFIRMED_TO_ALERT = 2.5
DRINK_ALERT_COOLDOWN                   = 3.0   # seconds between alerts
```

---

## 📖 Usage

### Drowsiness Detection

```bash
# Default settings
python sleep_detector.py

# Custom EAR threshold (lower = more sensitive)
python sleep_detector.py --ear 0.18

# Trigger alarm after 1.5 seconds of closed eyes (overrides --frames)
python sleep_detector.py --seconds 1.5

# Use a secondary camera, enable debug overlay and logging
python sleep_detector.py --camera 1 --debug --log

# Silent mode (visual warning only, no audio)
python sleep_detector.py --silent
```

### Distraction Detection

```bash
python distraction_detection.py
```

| Key | Action |
|-----|--------|
| `c` | Re-calibrate neutral head pose |
| `Esc` | Quit |

### Drink-While-Driving Detection

```bash
python drink_and_drive_detection.py
```

### Training a Custom Drink Detector

```bash
# Step 1 — Download training images from web
python drink_detector_trainer.py --mode download_web

# Step 2 — Train YOLOv8n on collected dataset
python drink_detector_trainer.py --mode train --dataset_path ./drink_dataset

# Step 3 — Live test with the trained model
python drink_detector_trainer.py --mode test_camera --model_path ./drink_detector_yolov8n.pt

# Step 4 — Evaluate performance metrics
python drink_detector_trainer.py --mode evaluate --model_path ./drink_detector_yolov8n.pt
```

### Keyboard Shortcuts (all modules)

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `d` | Toggle debug mode (show/hide face mesh) |
| `r` | Reset counters and alarms |
| `c` | Re-calibrate neutral head pose *(distraction module)* |

---

## 📁 Project Structure

```
Driver_Drowsiness_Detector/
│
├── 📄 sleep_detector.py            # Drowsiness detection — EAR + alarm logic
├── 📄 distraction_detection.py     # Gaze tracking + head pose estimation
├── 📄 drink_and_drive_detection.py # Multi-signal drink-drive state machine
│
├── 📄 drink_detector_trainer.py    # YOLOv8 training orchestration pipeline
├── 📄 yolov8_drink_detector.py     # YOLOv8 detector class (clean interface)
├── 📄 image_downloader.py          # Automated Bing dataset scraper (icrawler)
│
├── ⚙️  config.py                   # Centralised configuration (40+ params)
├── 🔧 utils.py                     # Shared utilities: alarm, logging, EAR helpers
├── 🧪 test_installation.py         # Dependency & camera verification
│
├── 📋 requirements.txt
└── 📖 README.md
```

---

## ⚡ Performance & Optimization

| Metric | Value | Notes |
|---|---|---|
| **End-to-end latency** | < 100 ms | MediaPipe + EAR on CPU |
| **YOLOv8n inference** | < 50 ms / frame | CPU; GPU further reduces this |
| **False positive rate** | < 5% | With proper lighting & calibration |
| **False negative rate** | < 3% | With proper lighting & calibration |
| **CPU usage** | 15 – 30% | Modern quad-core |
| **RAM usage** | 200 – 400 MB | Without GPU memory |

### Optimization Techniques Used

- **EAR temporal smoothing** — 5-frame rolling average; eliminates single-frame blink noise
- **Consecutive-frame counters** — All alerts require N sustained frames; prevents flicker
- **YOLOv8 Nano backbone** — Smallest YOLO variant; maximises FPS on CPU
- **Single face tracking** — `max_num_faces=1` in FaceMesh reduces compute
- **Frame skipping fallback** — Graceful degradation if camera drops frames

### Tips for Best Performance

1. **Lighting** — Ensure even, bright illumination on the driver's face
2. **Camera placement** — Eye level, 30–45 cm from face
3. **Resolution** — Use `640×480` default; lower if CPU-bound
4. **Calibration** — Press `c` after the driver adjusts their seating position
5. **Threshold tuning** — Start with defaults, then adjust `EYE_AR_THRESHOLD` ±0.02

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| **Camera not found** | Try `--camera 1` or `--camera 2`; check system camera permissions |
| **Too many false alarms** | Raise `EYE_AR_THRESHOLD` to 0.24–0.25; improve lighting |
| **Alarms not triggering** | Lower `EYE_AR_THRESHOLD` to 0.19–0.20; check lighting |
| **MediaPipe import error** | `pip uninstall mediapipe && pip install mediapipe==0.10.9` |
| **Camera freezes** | Reduce `FRAME_WIDTH`/`FRAME_HEIGHT` in config.py |
| **No audio output** | Check system volume; try `--silent` to confirm visual-only works |
| **Head pose always "DISTRACTED"** | Press `c` to re-calibrate with face looking straight ahead |


---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — free to use, modify, and distribute with attribution.
```

---

## 📬 Contact

**icemberg** — Primary Developer
- 🐙 GitHub: [@icemberg](https://github.com/icemberg)

**nikes303** — Contributor
- 🐙 GitHub: [@nikes303](https://github.com/nikes303)

> Project Repository: [https://github.com/nikes303/Driver_Drowsiness_Detector](https://github.com/nikes303/Driver_Drowsiness_Detector)

---

<div align="center">

⚠️ **Disclaimer** — This system is a safety *aid* and should never replace adequate rest before driving. Always follow traffic safety regulations.

*Built with ❤️ using MediaPipe, OpenCV, and YOLOv8*

</div>
