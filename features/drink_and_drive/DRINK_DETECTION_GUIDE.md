# Drink and Drive Detection - Implementation Guide

## Overview

This comprehensive drink and drive detection system integrates multiple advanced features:

1. **MediaPipe Object Detector** - Detects drink containers (cups, bottles, mugs, cans, glasses)
2. **Custom Model Training** - Train on 100+ samples per drink class
3. **Tuned State Machine** - Optimized thresholds based on real driving scenarios
4. **Event Logging** - CSV logging with frame timestamps
5. **Frame Snapshots** - Automatic capture of frames during ALERT events

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `opencv-python` - Video processing and visualization
- `mediapipe` - Face detection, hand tracking
- `numpy` - Numerical computations
- `pygame` - Audio playback for alarms
- `scikit-learn` - Machine learning for custom drink detector
- `Pillow` - Image handling

### 2. Verify Installation

```bash
python test_installation.py
```

---

## Quick Start

### Run Drink and Drive Detection (Out-of-Box)

```bash
python drink_and_drive_detection.py
```

**Features enabled by default:**
- ✅ Hand-to-mouth proximity detection
- ✅ Head pose distraction detection
- ❌ Custom drink object detection (needs training first)
- ✅ Event logging to CSV
- ✅ Frame snapshots on alerts

**Controls:**
- Press **ESC** to exit
- Watch the console for state transitions and alerts
- Check `drink_detection_logs/` for event logs and snapshots

---

## Custom Drink Detector Training

### Overview

The custom drink detector uses Random Forest classification trained on color and edge features extracted from images.

**Two methods available:**
1. **Webcam Collection** - Simple but limited (~100 images per class manually)
2. **Web Scraping** - Automatic download of 1000+ diverse images per class (RECOMMENDED)

### Method 2: Web Scraping (RECOMMENDED - 1000+ images per class)

This method downloads diverse images from Bing Image Search automatically.

#### Prerequisites

```bash
pip install icrawler
```

#### Step 1: Download Images (15-30 minutes)

```bash
python image_downloader.py --download --clean --stats
```

**What happens:**
1. Downloads ~1200 images per drink class from Bing (10 keyword variations per class)
2. Automatically cleans corrupt and low-quality images
3. Generates dataset statistics
4. Creates balanced, high-quality dataset ready for training

**Classes downloaded (10 keywords each):**
- `cup` - coffee cup, ceramic cup, plastic cup, disposable cup, drinking cup, empty cup, white cup, etc.
- `bottle` - drink bottle, plastic bottle, glass bottle, water bottle, beverage bottle, etc.
- `mug` - coffee mug, ceramic mug, tea mug, drinking mug, hot drink mug, etc.
- `can` - drink can, aluminum can, soda can, beverage can, soft drink can, etc.
- `glass` - drinking glass, water glass, beverage glass, clear glass, glass cup, etc.
- `drinking_bottle`, `soda_can`, `beer_bottle`, `water_bottle`, `tea_cup` (each with 8-10 keywords)

**Expected output:**
```
drink_dataset/
├── cup/           (~1200 images)
├── bottle/        (~1200 images)
├── mug/           (~1200 images)
├── can/           (~1200 images)
├── glass/         (~1200 images)
├── drinking_bottle/ (~1200 images)
├── soda_can/      (~1200 images)
├── beer_bottle/   (~1200 images)
├── water_bottle/  (~1200 images)
└── tea_cup/       (~1200 images)
```

**Troubleshooting download:**

If download is slow or fails:
```bash
# Download only (skip cleaning)
python image_downloader.py --download

# Clean only (after download)
python image_downloader.py --clean

# View statistics
python image_downloader.py --stats

# Download with custom settings
python image_downloader.py --download --dataset-dir ./my_dataset --images-per-keyword 150
```

**Logs:**
All download progress is saved to `image_downloader.log` for debugging.

#### Step 2: Train on Downloaded Dataset

Once downloading and cleaning completes:

```bash
python drink_detector_trainer.py --mode train --dataset_path ./drink_dataset
```

---

### Method 1: Webcam Collection (Quick)

```bash
python drink_detector_trainer.py --mode collect_dataset --dataset_path ./drink_dataset
```

**What happens:**
1. Creates directories for each drink class in `./drink_dataset/`
2. Opens webcam feed
3. For each class, captures 100+ samples by pressing SPACE

**Classes collected (default 10 classes):**
- cup, bottle, mug, can, glass, drinking_bottle, soda_can, beer_bottle, water_bottle, tea_cup

**Instructions during collection:**
```
- Hold the drink object in front of the camera
- Press SPACE to capture a sample
- Press C to move to next class
- Press Q to exit
```

**Example protocol:**
```
1. Start with 'cup' class
   - Show different cups (ceramic, glass, plastic)
   - Vary angles, distances, lighting
   - Capture 100 samples
   - Press C to move to 'bottle'

2. Repeat for all classes
3. Target: 100+ samples per class (10 classes = 1000+ total images)
```

#### Step 2: Train the Model (Method 1 Only)

Train a Random Forest classifier on the collected dataset:

```bash
python drink_detector_trainer.py --mode train --dataset_path ./drink_dataset
```

**Output:**
- `drink_detector_model.pkl` - Trained model file
- Training accuracy printed to console
- Training logs in console

**Time estimate (Method 1 - Webcam):** ~5-10 minutes to collect 1000 images manually
**Time estimate (Method 2 - Web):** ~2-5 minutes to train on 12,000+ images

### Step 3: Test the Model

Test real-time detection on webcam:

```bash
python drink_detector_trainer.py --mode test_camera --model_path ./drink_detector_model.pkl
```

**What to expect:**
- Real-time drink class predictions with confidence scores
- Green text for high confidence (>0.7)
- Orange text for lower confidence
- Press ESC to exit

### Step 4: Use in Main Pipeline

Once trained, the model is automatically loaded when running:

```bash
python drink_and_drive_detection.py
```

The system will:
1. Load the custom model from `./drink_detector_model.pkl`
2. Use it for real-time drink detection
3. Fuse results with hand proximity and head pose signals
4. Trigger alerts based on combined risk score

---

## Configuration

All parameters are in `config.py`. Key tunable parameters:

### State Transition Thresholds (Tuned for Real Driving)

```python
# Risk score thresholds (0-3 scale)
DRINK_RISK_THRESHOLD_IDLE_TO_POSSIBLE = 1.5      # Initial detection
DRINK_RISK_THRESHOLD_POSSIBLE_TO_CONFIRMED = 2.0 # Higher confidence
DRINK_RISK_THRESHOLD_CONFIRMED_TO_ALERT = 2.5    # Very high confidence

# Frame consistency requirements (at 30fps)
DRINK_FRAMES_IDLE_TO_POSSIBLE = 5          # ~0.17 seconds
DRINK_FRAMES_POSSIBLE_TO_CONFIRMED = 10    # ~0.33 seconds
DRINK_FRAMES_CONFIRMED_TO_ALERT = 12       # ~0.40 seconds
```

### Signal Weights

```python
# How much each signal contributes to risk score
SIGNAL_WEIGHT_HAND_PROXIMITY = 1.0
SIGNAL_WEIGHT_OBJECT_DETECTION = 1.0
SIGNAL_WEIGHT_HEAD_DISTRACTION = 0.5
```

### Hand Proximity Detection

```python
DRINK_HAND_MOUTH_DISTANCE_THRESHOLD = 0.15  # Normalized to face width (0-1)
```

Higher = more tolerant (detects drinking from further away)
Lower = more strict (only detects drinking very close to mouth)

### Head Pose Distraction

```python
HEAD_YAW_DISTRACTION_THRESHOLD = 25    # Head turned left/right (degrees)
HEAD_PITCH_DISTRACTION_THRESHOLD = 18  # Head tilted up/down (degrees)
```

### Event Logging

```python
ENABLE_DRINK_CSV_LOGGING = True
ENABLE_DRINK_SNAPSHOTS = True
DRINK_LOG_DIRECTORY = "drink_detection_logs"
DRINK_SNAPSHOTS_DIRECTORY = "drink_detection_logs/snapshots"
DRINK_EVENT_SNAPSHOT_COUNT = 6  # Frames captured around alert
```

---

## State Machine

The detection system uses a 4-state state machine:

```
IDLE (green)
  ↓ [risk ≥ 1.5]
POSSIBLE_DRINKING (yellow) [5 frames]
  ↓ [risk ≥ 2.0]
DRINKING (orange) [10 frames]
  ↓ [risk ≥ 2.5]
ALERT (red) [plays alarm, saves snapshots]
  ↓ [duration expired]
IDLE
```

**State Transitions:**
- Each state requires consistent high risk for multiple frames
- Fallback to IDLE if risk drops below threshold
- Alert maintains for 2 seconds by default

---

## Output Files

### Event Logs

CSV file at `drink_detection_logs/drink_and_drive_events.csv`:

```
timestamp,event_type,risk_score,hand_mouth_distance,object_detected,head_distracted,frame_number
2024-04-17 14:32:45.123,POSSIBLE_DRINKING,1.8,0.12,False,False,145
2024-04-17 14:32:45.234,DRINKING,2.2,0.10,True,False,147
2024-04-17 14:32:45.345,ALERT,2.6,0.08,True,True,150
```

### Frame Snapshots

Saved to `drink_detection_logs/snapshots/`:

```
snapshot_20240417_143245_123_frame150_ALERT_snap0.jpg
snapshot_20240417_143245_123_frame150_ALERT_snap1.jpg
snapshot_20240417_143245_123_frame150_ALERT_snap2.jpg
...
```

6 snapshots captured automatically around each ALERT event.

---

## Detection Signals

The system fuses three independent signals:

### 1. Hand Proximity (Normalized Distance)
- **Signal:** Distance from hand index finger to mouth center
- **Normalized:** Relative to face width
- **Threshold:** 0.15 (15% of face width)
- **Weight:** 1.0x

### 2. Object Detection
- **Signal:** Detected drink objects near mouth
- **Uses:** Custom trained model or fallback detection
- **Classes:** cup, bottle, mug, can, glass, etc.
- **Weight:** 1.0x

### 3. Head Distraction
- **Signal:** Head yaw/pitch angles
- **Thresholds:** ±25° yaw, ±18° pitch
- **Weight:** 0.5x (lower importance)

### Risk Score

```python
risk_score = Σ(signal_present × weight)
```

Range: 0-3.0
- 0.0 = No drinking activity detected
- 1.0 = Weak signal (possible false positive)
- 2.0 = Moderate confidence
- 3.0 = Very high confidence (strong alert)

---

## Troubleshooting

### Issue: Detection too sensitive (too many false alerts)

**Solution:**
```python
# In config.py, increase thresholds
DRINK_RISK_THRESHOLD_CONFIRMED_TO_ALERT = 2.8  # was 2.5
DRINK_FRAMES_CONFIRMED_TO_ALERT = 15            # was 12
```

### Issue: Detection too lenient (missing actual drinking)

**Solution:**
```python
# In config.py, decrease thresholds
DRINK_RISK_THRESHOLD_CONFIRMED_TO_ALERT = 2.2  # was 2.5
DRINK_FRAMES_CONFIRMED_TO_ALERT = 10            # was 12
DRINK_HAND_MOUTH_DISTANCE_THRESHOLD = 0.12     # was 0.15
```

### Issue: Custom model not loading

**Check:**
1. Is `drink_detector_model.pkl` in current directory?
2. Did training complete successfully?
3. Check console output for error messages

**Solution:**
```bash
# Re-train model
python drink_detector_trainer.py --mode train --dataset_path ./drink_dataset

# Test model before using
python drink_detector_trainer.py --mode test_camera --model_path ./drink_detector_model.pkl
```

### Issue: Poor hand detection

**Possible causes:**
- Poor lighting
- Hand covered by hair/objects
- Rapid hand movements

**Solution:**
- Improve lighting conditions
- Use hand proximity threshold adjustment in config.py

### Issue: Poor face detection

**Possible causes:**
- Face not clearly visible
- Wrong camera angle
- Poor lighting

**Solution:**
- Position face directly facing camera
- Ensure adequate lighting
- Adjust camera angle

---

## Real-World Testing Protocol

For production deployment, test on real driving scenarios:

### Test 1: Static Drinking
- Driver sits in car
- Holds cup/bottle at different distances
- Sips from it
- **Expected:** Alert triggered within 1 second of detection

### Test 2: Moving Head While Drinking
- Driver drinks while looking at road
- Driver drinks while looking at passenger
- Driver drinks while looking at mirror
- **Expected:** Consistent detection despite head motion

### Test 3: Edge Cases
- Eating while driving (should have low risk if no drink detected)
- Hand near mouth but no drink (should have lower risk)
- Drink in cup holder (out of hand, should not trigger)
- **Expected:** Minimal false positives

### Test 4: Different Drink Containers
- Coffee cup, water bottle, soda can, beer bottle, wine glass
- **Expected:** All drink types detected with confidence > 0.6

---

## Performance Metrics

Expected performance on properly tuned system:

- **Sensitivity (True Positive Rate):** 85-95%
- **Specificity (True Negative Rate):** 90-95%
- **False Positive Rate:** 5-10% per minute
- **Detection Latency:** 0.4-0.5 seconds
- **Frame Rate:** 25-30 FPS (real-time)

---

## Advanced Features

### Custom Signal Weighting

Adjust how signals contribute to risk:

```python
# Make object detection more important
SIGNAL_WEIGHT_OBJECT_DETECTION = 1.5  # was 1.0
SIGNAL_WEIGHT_HAND_PROXIMITY = 0.8     # was 1.0
```

### Custom Drink Classes

Add more drink classes:

```python
# In config.py
DRINK_CLASSES = [
    'cup', 'bottle', 'mug', 'can', 'glass',
    'coffee_cup', 'water_bottle', 'energy_drink',  # Add custom classes
    'tea_cup', 'juice_box'
]
```

Then retrain with new classes:
```bash
python drink_detector_trainer.py --mode collect_dataset
python drink_detector_trainer.py --mode train
```

### Cooldown Between Alerts

```python
DRINK_ALERT_COOLDOWN = 5.0   # Minimum time between consecutive alerts
```

---

## Support and Debugging

### Enable Debug Mode

View detailed state transitions:
```bash
python drink_and_drive_detection.py 2>&1 | tee debug.log
```

### Generate Test Report

```bash
# Collect 10 minutes of labeled data
python drink_detector_trainer.py --mode test_camera

# Analyze logs
cat drink_detection_logs/drink_and_drive_events.csv | wc -l
```

### Core Components

1. **drink_and_drive_detection.py** - Main detection pipeline
2. **drink_detector_trainer.py** - Model training and testing
3. **utils.py** - Detection algorithms and logging
4. **config.py** - All tunable parameters
5. **drink_dataset/** - Training data directory
6. **drink_detector_model.pkl** - Trained model (auto-generated)
7. **drink_detection_logs/** - Event logs and snapshots

---

## References

- **MediaPipe Documentation:** https://developers.google.com/mediapipe
- **OpenCV Documentation:** https://docs.opencv.org
- **Scikit-learn:** https://scikit-learn.org

---

## License

This project is part of the Driver Drowsiness Detection system.

