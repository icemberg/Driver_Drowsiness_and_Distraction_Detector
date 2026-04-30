<div align="center">

# 🚗 Driver Drowsiness & Distraction Detector

**A real-time, multi-signal safety system that detects driver drowsiness, distraction, yawning, drink-while-driving behavior, and phone usage using computer vision — and raises an alarm before an accident happens.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8.10%2B-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
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

- 🖥️ **Unified Dashboard** — Streamlit-based live dashboard runs all five detectors on a single camera feed simultaneously
- 👁️ **Real-time Drowsiness Detection** — Eye Aspect Ratio (EAR) with smoothed temporal averaging to eliminate flicker and false positives
- 🔄 **Head Pose Estimation** — Computes calibrated yaw, pitch, and roll from facial landmarks using vector cross-product geometry
- 👀 **Iris Gaze Tracking** — Detects off-centre gaze by tracking iris position relative to eye corners (MediaPipe Iris)
- 😮 **Yawning Detection** — Mouth Aspect Ratio (MAR) + eye closure overlap + hand-near-mouth scoring with cooldown logic
- 🍺 **Drink-While-Driving Detection** — YOLOv8n object detection fused with hand-proximity signals and a 4-state machine (IDLE → POSSIBLE → DRINKING → ALERT) with grace-frame tolerance for noisy hand tracking
- 📱 **Phone Use Detection** — YOLOv8 phone detector with a 4-state machine (IDLE → POSSIBLE_PHONE_USE → CONFIRMED_PHONE_USE → ALERT); alarm fires on confirmed use
- 🔗 **Multi-Signal Fusion** — Weighted risk scoring combines EAR, gaze, head pose, drink object confidence, and hand-to-mouth distance
- 🧩 **DRY Modular Architecture** — Shared MediaPipe setup (`setup/setup.py`), per-feature `process_frame()` functions, pure orchestrator entry point
- 🔔 **Audio Alarms** — Synthesized dual-frequency beep or custom `.wav` file via Pygame mixer with global cooldown to prevent overlapping sounds
- 🎛️ **Highly Configurable** — 50+ parameters in `config/config.py`; no code changes needed to tune thresholds
- 📋 **Event Logging** — CSV-format drowsiness and drink-event logs with optional frame snapshots
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
| **Ultralytics YOLOv8** | ≥ 8.0.0 | Real-time drink object & phone detection (YOLOv8n backbone) |
| **NumPy** | ≥ 1.20.0 | Vector math, EAR calculations, angle geometry |
| **PyTorch** | ≥ 2.0.0 | YOLOv8 model backend |

### Dashboard & Utilities

| Library | Version | Role |
|---|---|---|
| **Streamlit** | ≥ 1.28.0 | Real-time web dashboard with live camera feed |
| **Pygame** | ≥ 2.0.0 | Audio mixer, alarm sound synthesis & playback |
| **Matplotlib** | ≥ 3.4.0 | EAR trend plots, dataset visualisation |
| **Pillow** | ≥ 8.0.0 | Image I/O for dataset management |
| **icrawler** | ≥ 0.6.0 | Automated Bing image scraping for training data |
| **scikit-learn** | ≥ 1.0.0 | Dataset splitting, model evaluation metrics |

### Language & Environment

- **Python 3.8+** — Core runtime (3.10+ recommended)
- **Virtual Environment** (`venv`) — Dependency isolation

---

## 🧠 System Architecture & Logic

The system follows a **DRY modular architecture** where a single shared MediaPipe setup feeds all five feature detectors through one orchestrator.

#### Module & Import Structure

```mermaid
flowchart TD
    S["setup/setup.py\nShared FaceMesh + Hands + mp_drawing"]

    S -->|imported by| M["main.py\nDriverSafetyPipeline\nCLI Orchestrator"]
    S -->|imported by| A["app.py\nStreamlit Dashboard"]
    S -->|imported by| F1["features/sleep_detector.py"]
    S -->|imported by| F2["features/distraction_detection.py"]
    S -->|imported by| F3["features/yawning_detection.py"]
    S -->|imported by| F4["features/drink_and_drive/\ndrink_and_drive_detection.py"]

    A -->|imports DriverSafetyPipeline| M
    M -->|imports process_frame| F1
    M -->|imports process_frame| F2
    M -->|imports process_frame| F3
    M -->|imports process_frame| F4
    M -->|imports process_frame| F5["features/phone_tracking/\nphone_tracker.py"]

    style S fill:#1e3a5f,color:#90caf9,stroke:#42a5f5
    style M fill:#1b3a2d,color:#a5d6a7,stroke:#66bb6a
    style A fill:#3b1f2b,color:#f48fb1,stroke:#f06292
    style F1 fill:#2a2a1e,color:#fff59d,stroke:#ffee58
    style F2 fill:#2a2a1e,color:#fff59d,stroke:#ffee58
    style F3 fill:#2a2a1e,color:#fff59d,stroke:#ffee58
    style F4 fill:#2a2a1e,color:#fff59d,stroke:#ffee58
    style F5 fill:#2a2a1e,color:#fff59d,stroke:#ffee58
```

#### Per-Frame Detection Pipeline

```mermaid
flowchart LR
    CAM["Camera\nframe"]
    MP["MediaPipe\nOne shared pass\nFaceMesh + Hands"]
    LMS["face_landmarks\nhand_results"]

    SLEEP["sleep_detector\nprocess_frame\nEAR - Drowsiness"]
    DIST["distraction_detection\nprocess_frame\nGaze + Head Pose"]
    YAWN["yawning_detection\nprocess_frame\nMAR + Score"]
    DRINK["drink_and_drive\nprocess_frame\nYOLO + State Machine"]
    PHONE["phone_tracker\nprocess_frame\nYOLO Phone Detection"]

    ALARM["Alarm\n+ Overlay\n+ Log"]

    CAM --> MP
    MP --> LMS
    LMS --> SLEEP
    LMS --> DIST
    LMS --> YAWN
    LMS --> DRINK
    CAM --> PHONE

    SLEEP -->|drowsy| ALARM
    DIST -->|distracted| ALARM
    YAWN -->|yawning| ALARM
    DRINK -->|ALERT state| ALARM
    PHONE -->|confirmed use| ALARM

    style CAM fill:#263238,color:#eceff1,stroke:#546e7a
    style MP fill:#1e3a5f,color:#90caf9,stroke:#42a5f5
    style LMS fill:#1e3a5f,color:#90caf9,stroke:#42a5f5
    style SLEEP fill:#1b3a2d,color:#a5d6a7,stroke:#66bb6a
    style DIST fill:#1b3a2d,color:#a5d6a7,stroke:#66bb6a
    style YAWN fill:#1b3a2d,color:#a5d6a7,stroke:#66bb6a
    style DRINK fill:#1b3a2d,color:#a5d6a7,stroke:#66bb6a
    style PHONE fill:#1b3a2d,color:#a5d6a7,stroke:#66bb6a
    style ALARM fill:#4a1010,color:#ef9a9a,stroke:#e53935
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

#### 4. Yawning Detection (Score-Based)
```
score = (1.0 if mouth_open ≥ MIN_YAWN_DURATION)
      + (1.0 if eye_closed_overlap ≥ MIN_EYE_CLOSED_OVERLAP)
      + (0.25 if hand_near_mouth)
```
Yawn is confirmed when `score ≥ 2.0`. Cooldown timer prevents duplicate counting.

#### 5. Drink-Drive Signal Fusion
```
risk = 0.0
if head_distracted:            risk += 0.7    # Preparatory signal
if hand_near_mouth:            risk += 0.6    # Primary drinking indicator
  if hand_very_close (<0.3):   risk += 0.2    # Bonus for very close
if drink_object_detected:      risk += 0.5    # Independent confirmatory signal
  if head_distracted:          risk += 0.3    # Bonus: head + object
```
State machine transitions require risk score to exceed a threshold **for N consecutive frames** with **3-frame grace tolerance** for noisy hand tracking, preventing single-frame noise from triggering or resetting alerts. Alarm fires when risk exceeds 1.5 or when DRINKING state is confirmed.

#### 6. Phone Use Detection
```
State: IDLE → POSSIBLE_PHONE_USE (5 frames) → CONFIRMED_PHONE_USE → ALERT (10 frames)
```
- YOLOv8 model (`phone_brain.pt`) detects phone objects at >50% confidence
- Alarm fires at `CONFIRMED_PHONE_USE` and re-triggers at `ALERT` escalation
- Falls back to IDLE if phone disappears at any stage

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
git clone https://github.com/icemberg/Driver_Drowsiness_and_Distraction_Detector.git
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

All system parameters live in **`config/config.py`** — no code edits needed for normal tuning.

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
GAZE_CENTER_MAX           = 0.60
HEAD_YAW_LIMIT            = 20     # degrees left/right
HEAD_PITCH_LIMIT          = 15     # degrees up/down
HEAD_ROLL_LIMIT           = 12     # degrees sideways tilt
DISTRACTION_CONSEC_FRAMES = 10     # frames before distraction warning (~0.33 s)

# ── Yawning Detection ───────────────────────────────────────────────────────
MAR_THRESHOLD             = 0.5    # mouth aspect ratio threshold
EAR_THRESHOLD             = 0.23   # eye closure threshold for yawn scoring
MIN_YAWN_DURATION         = 1.2    # seconds mouth must stay open
YAWN_COOLDOWN             = 2.0    # seconds between counted yawns

# ── Alerts ───────────────────────────────────────────────────────────────────
ALARM_SOUND  = "alarm.wav"      # path to WAV file, or None for generated beep
ALARM_VOLUME = 0.9              # 0.0 – 1.0

# ── Drink Detection ──────────────────────────────────────────────────────────
DRINK_RISK_THRESHOLD_IDLE_TO_POSSIBLE     = 0.4
DRINK_RISK_THRESHOLD_POSSIBLE_TO_CONFIRMED = 0.9
DRINK_RISK_THRESHOLD_CONFIRMED_TO_ALERT    = 1.3
DRINK_RISK_FALLBACK_THRESHOLD              = 0.3
DRINK_FRAMES_POSSIBLE_TO_CONFIRMED         = 5    # ~0.17 s @ 30 fps
DRINK_FRAMES_CONFIRMED_TO_ALERT            = 8    # ~0.27 s @ 30 fps

# ── Phone Detection ──────────────────────────────────────────────────────────
ENABLE_PHONE_DETECTION     = True
PHONE_FRAMES_IDLE_TO_POSSIBLE  = 5   # ~0.17 seconds to trigger warning
PHONE_FRAMES_POSSIBLE_TO_ALERT = 10  # ~0.33 seconds to trigger alarm
PHONE_LOCKOUT_DURATION         = 5.0 # seconds to stay in ALERT mode
```

---

## 📖 Usage

### 🖥️ Unified Dashboard (Recommended)

Run **all five detectors simultaneously** on a single camera feed:

```bash
# CLI mode (OpenCV window)
python main.py

# Streamlit web dashboard (with live metrics, toggles, and alert history)
streamlit run app.py
```

**Dashboard features:**
- ▶/⏹ Start/Stop buttons
- Per-detector toggle checkboxes (Drowsiness, Distraction, Yawning, Drink & Drive)
- Real-time metric cards with severity coloring (green/orange/red)
- Alert banner + scrollable alert history log
- Silent mode toggle
- Head pose recalibration button

### Individual Feature Modules (Standalone)

Each feature module can still be run independently:

```bash
# Drowsiness Detection
python features/sleep_detector.py
python features/sleep_detector.py --ear 0.18 --seconds 1.5 --debug

# Distraction Detection
python features/distraction_detection.py

# Yawning Detection
python features/yawning_detection.py

# Drink-While-Driving Detection
python features/drink_and_drive/drink_and_drive_detection.py
```

### Training a Custom Drink Detector

```bash
# Step 1 — Download training images from web
python features/drink_and_drive/drink_detector_trainer.py --mode download_web

# Step 2 — Train YOLOv8n on collected dataset
python features/drink_and_drive/drink_detector_trainer.py --mode train --dataset_path ./drink_dataset

# Step 3 — Live test with the trained model
python features/drink_and_drive/drink_detector_trainer.py --mode test_camera --model_path ./drink_detector_yolov8n.pt

# Step 4 — Evaluate performance metrics
python features/drink_and_drive/drink_detector_trainer.py --mode evaluate --model_path ./drink_detector_yolov8n.pt
```

### Keyboard Shortcuts

#### CLI mode (`python main.py`)

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit the application |
| `r` | Reset all counters and alarms |
| `c` | Re-calibrate neutral head pose |

#### Standalone modules

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
├── 📄 main.py                              # Pure orchestrator — imports process_frame() from each feature
├── 📄 app.py                               # Streamlit dashboard — imports DriverSafetyPipeline from main.py
│
├── 📁 setup/                               # Shared MediaPipe resources (DRY)
│   ├── __init__.py
│   └── setup.py                            # Single FaceMesh + Hands instance for all modules
│
├── 📁 config/
│   └── config.py                           # Centralised configuration (50+ params)
│
├── 📁 utils/
│   └── utils.py                            # Shared utilities: alarm, EAR/MAR, logging, risk scoring
│
├── 📁 features/                            # Feature modules — each has main() + process_frame()
│   ├── sleep_detector.py                   # Drowsiness detection (EAR + smoothing + alarm)
│   ├── distraction_detection.py            # Gaze tracking + head pose + off-centre detection
│   ├── yawning_detection.py                # MAR + eye overlap + hand proximity yawn scoring
│   ├── 📁 drink_and_drive/
│   │   ├── drink_and_drive_detection.py    # 4-state machine drink-drive detector
│   │   ├── drink_detector_trainer.py       # YOLOv8 training pipeline
│   │   ├── yolov8_drink_detector.py        # YOLOv8 detector class (clean interface)
│   │   └── image_downloader.py            # Automated Bing dataset scraper (icrawler)
│   └── 📁 phone_tracking/
│       ├── phone_tracker.py               # 4-state machine phone use detector (YOLOv8)
│       └── guide.md                       # Phone detection setup guide
│
├── 📁 models/                              # Trained model weights (git-ignored)
│   ├── yolov8n_drink.pt                   # Drink object detector
│   └── phone_brain.pt                     # Phone use detector (in features/phone_tracking/)
│
├── 🧪 test_installation.py                # Dependency & camera verification
├── 📋 requirements.txt
└── 📖 README.md
```

### Architecture Principles

- **DRY (Don't Repeat Yourself)** — MediaPipe setup exists once in `setup/setup.py`; all modules import from it
- **Separation of Concerns** — Feature modules contain logic, utils contain helpers, entry points contain orchestration
- **Scalability** — Adding a new detector: create `features/new_detector.py` with `process_frame()`, import it in `main.py`
- **Backward Compatibility** — Each feature's `main()` still works standalone with `python features/<module>.py`

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

- **Shared MediaPipe instances** — One `FaceMesh` + one `Hands` instance processes each frame; results shared across all 5 detectors
- **EAR temporal smoothing** — 5-frame rolling average; eliminates single-frame blink noise
- **Consecutive-frame counters** — All alerts require N sustained frames; prevents flicker
- **Grace-frame tolerance** — Drink detection tolerates up to 3 brief tracking dropouts without resetting state
- **Global alarm cooldown** — Prevents overlapping beeps when multiple detectors trigger simultaneously
- **YOLOv8 Nano backbone** — Smallest YOLO variant; maximises FPS on CPU
- **Single face tracking** — `max_num_faces=1` in FaceMesh reduces compute
- **Frame skipping fallback** — Graceful degradation if camera drops frames
- **Flicker-free Streamlit feed** — `while`-loop with `st.empty()` placeholder updates instead of `st.rerun()`

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
| **Camera not found** | Try `--camera 1` or `--camera 2`; change `CAMERA_INDEX` in config |
| **Too many false alarms** | Raise `EYE_AR_THRESHOLD` to 0.24–0.25; improve lighting |
| **Alarms not triggering** | Lower `EYE_AR_THRESHOLD` to 0.19–0.20; check lighting |
| **MediaPipe import error** | `pip uninstall mediapipe && pip install mediapipe==0.10.9` |
| **Camera freezes** | Reduce `FRAME_WIDTH`/`FRAME_HEIGHT` in `config/config.py` |
| **No audio output** | Check system volume; use Silent Mode to confirm visual-only works |
| **Head pose always "DISTRACTED"** | Press `c` / click Recalibrate with face looking straight ahead |
| **Streamlit video blinking** | Ensure you're using Streamlit ≥ 1.28; the app uses `while`-loop (not `st.rerun()`) |
| **Drink detection not working** | Check risk score in dashboard; ensure hand is near mouth with head tilt |
| **Phone detection not loading** | Verify `features/phone_tracking/phone_brain.pt` model file exists |
---

## 🔍 How It Works

### 1. Face Detection

The system uses MediaPipe's FaceMesh to detect 468 facial landmarks in real-time with high accuracy. A single shared MediaPipe pass produces both face and hand landmarks, which are distributed to all five detector modules.

### 2. Eye Landmark Extraction

Specific landmarks are extracted for:

- **Left Eye**: 16 key points defining eye contour
- **Right Eye**: 16 key points defining eye contour

### 3. Eye Aspect Ratio (EAR) Calculation

```
EAR = (Distance between upper and lower eyelid) / (Width of eye)
```

- High EAR (>0.22) = Eyes Open
- Low EAR (<0.22) = Eyes Closed
- A 5-frame rolling average smooths out single-frame blink noise

### 4. Drowsiness Detection Logic

- If EAR < threshold for `EYE_AR_CONSEC_FRAMES` consecutive frames → Drowsiness detected
- Hysteresis band (`threshold + 0.02`) prevents rapid toggling
- Audio alarm triggers once per drowsiness episode; resets when eyes reopen

### 5. Distraction Detection

- Monitors head rotation (yaw, pitch, roll angles) relative to calibrated neutral
- Tracks iris position within eye corners using MediaPipe Iris landmarks
- Detects off-centre face position using nose landmark
- Any ONE signal (gaze, head turn, off-centre) independently triggers distraction
- Requires `DISTRACTION_CONSEC_FRAMES` sustained frames before raising alarm

### 6. Yawning Detection

- Mouth Aspect Ratio (MAR) detects mouth opening
- Score combines mouth duration, eye closure overlap, and hand-near-mouth proximity
- Yawn confirmed when score ≥ 2.0 with cooldown to prevent duplicate counting

### 7. Drink-While-Driving Detection

- **Signal fusion**: Hand-to-mouth distance + YOLOv8 object detection + head pose
- **4-state machine**: IDLE → POSSIBLE_DRINKING → DRINKING → ALERT
- **Grace-frame tolerance**: Tolerates up to 3 noisy hand-tracking dropouts without resetting
- **Risk-based alarm**: Plays alarm when risk score exceeds 1.5

### 8. Phone Use Detection

- YOLOv8 model (`phone_brain.pt`) detects phone objects at >50% confidence
- **4-state machine**: IDLE → POSSIBLE_PHONE_USE → CONFIRMED_PHONE_USE → ALERT
- Alarm fires on confirmed phone use (5 consecutive frames) and re-triggers at ALERT (10 frames)
- Falls back to IDLE with lockout timer (`PHONE_LOCKOUT_DURATION = 5s`)

---

## 📦 Dependencies Details

### MediaPipe

- Provides 468 high-accuracy facial landmarks + iris tracking
- Real-time face detection on CPU with refine_landmarks enabled
- Robust to various head poses and lighting conditions

### OpenCV

- Video capture and frame processing
- Image transformations and visualization (BGR/RGB conversion)
- Real-time video rendering and text overlay

### NumPy

- Numerical computations for EAR/MAR calculations
- Vector cross-product geometry for head pose estimation
- Distance and angle computations

### Pygame

- Audio playback for alarm sounds
- Synthesized dual-frequency tone generation (880 Hz + 1320 Hz)
- Global cooldown prevents overlapping alarm sounds

### Ultralytics YOLOv8

- YOLOv8 Nano backbone for real-time object detection
- Custom-trained models for drink objects and phone detection
- Built-in training pipeline with data augmentation

### Streamlit

- Real-time web dashboard with live camera feed
- Interactive sidebar with detector toggles and settings
- Metric cards with severity-based color coding
- Alert history log with timestamps

---

## 🔮 Future Enhancements

- Support for multiple driver detection
- Machine learning model for fatigue prediction
- Cloud-based alert system and remote monitoring
- Mobile application integration
- Driver profile customization with personalized thresholds
- Advanced gaze tracking with 3D eye models
- Integration with vehicle CAN bus for speed-aware alerts
- Voice-based driver engagement detection

---

## 💬 Support

For issues, questions, or feedback:

1. Check the [Troubleshooting](#-troubleshooting) section above
2. Review configuration parameters in `config/config.py`
3. Run `python test_installation.py` to verify setup
4. Check system requirements (Python 3.8+, webcam, 4GB+ RAM)
5. Open an issue on [GitHub](https://github.com/icemberg/Driver_Drowsiness_and_Distraction_Detector/issues)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs and issues
- Suggest improvements
- Submit pull requests with enhancements
- Improve documentation

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

> Project Repository: [https://github.com/icemberg/Driver_Drowsiness_and_Distraction_Detector](https://github.com/icemberg/Driver_Drowsiness_and_Distraction_Detector)

---

<div align="center">

⚠️ **Disclaimer** — This system is a safety *aid* and should never replace adequate rest before driving. Always follow traffic safety regulations.

*Built with ❤️ using MediaPipe, OpenCV, YOLOv8 & Streamlit*

</div>
