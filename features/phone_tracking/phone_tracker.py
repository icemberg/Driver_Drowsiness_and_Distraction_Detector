import cv2
import time
from enum import Enum
from ultralytics import YOLO

# Import configuration and shared utilities
import config.config as config
from utils.utils import play_alarm

# ---------------------------------------------------------
# STATE MACHINE SETUP
# ---------------------------------------------------------
class PhoneState(Enum):
    IDLE = 0
    POSSIBLE_PHONE_USE = 1
    CONFIRMED_PHONE_USE = 2
    ALERT = 3

# Module-level Global Variables (The Orchestrator will manage these)
_phone_state = PhoneState.IDLE
_phone_frame_ctr = 0
_phone_events = 0
_phone_alert_until = 0.0
_phone_alarm_on = False

# Lazy-load the YOLO model (so it only loads once)
_phone_model = None

def reset_phone_state():
    """Reset module-level phone state."""
    global _phone_state, _phone_frame_ctr, _phone_events
    global _phone_alert_until, _phone_alarm_on
    
    _phone_state = PhoneState.IDLE
    _phone_frame_ctr = 0
    _phone_events = 0
    _phone_alert_until = 0.0
    _phone_alarm_on = False

def process_frame(frame, w, h, now, silent=False):
    """
    Process one frame for phone detection using a State Machine.
    """
    global _phone_state, _phone_frame_ctr, _phone_events
    global _phone_alert_until, _phone_alarm_on, _phone_model

    # 1. Load the model on the very first frame
    if _phone_model is None:
        try:
            print("[INFO] Loading YOLOv8 Phone Detector...")
            _phone_model = YOLO("features/phone_tracking/phone_brain.pt")
        except Exception as e:
            print(f"[ERROR] Could not load phone model: {e}")
            return {"state": "ERROR", "phone_detected": False, "risk": 0.0}

    # 2. Run the YOLO Inference
    phone_detected = False
    risk_score = 0.0
    
    # We use verbose=False so it doesn't spam the console every frame
    results = _phone_model.predict(frame, verbose=False)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf > 0.50:  # 50% confidence threshold
                phone_detected = True
                risk_score = 3.0 # Max risk if phone is detected

    # 3. The State Machine Logic
    _phone_frame_ctr += 1

    if _phone_state == PhoneState.IDLE:
        if phone_detected:
            _phone_frame_ctr = 0
            _phone_state = PhoneState.POSSIBLE_PHONE_USE

    elif _phone_state == PhoneState.POSSIBLE_PHONE_USE:
        # Require 5 consecutive frames to confirm
        if phone_detected:
            if _phone_frame_ctr >= 5: 
                _phone_frame_ctr = 0
                _phone_state = PhoneState.CONFIRMED_PHONE_USE
                # ── Play alarm on confirmed phone use ──
                if not _phone_alarm_on:
                    _phone_alarm_on = True
                    if not silent:
                        play_alarm(config.ALARM_SOUND, config.ALARM_VOLUME)
        else:
            # Fallback to IDLE if phone goes away
            _phone_frame_ctr = 0
            _phone_state = PhoneState.IDLE

    elif _phone_state == PhoneState.CONFIRMED_PHONE_USE:
        # Require 10 consecutive frames to escalate to ALERT
        if phone_detected:
            if _phone_frame_ctr >= 10:
                _phone_frame_ctr = 0
                _phone_state = PhoneState.ALERT
                _phone_alert_until = now + config.PHONE_LOCKOUT_DURATION
                _phone_events += 1
                # Re-trigger alarm on escalation to ALERT
                if not silent:
                    play_alarm(config.ALARM_SOUND, config.ALARM_VOLUME)
        else:
            # Phone disappeared — fall back to IDLE
            _phone_frame_ctr = 0
            _phone_state = PhoneState.IDLE
            _phone_alarm_on = False

    elif _phone_state == PhoneState.ALERT:
        # Stay in Alert mode until the lockout timer expires
        if now >= _phone_alert_until:
            _phone_frame_ctr = 0
            _phone_state = PhoneState.IDLE
            _phone_alarm_on = False

    # 4. Return the standardized dictionary to the Orchestrator
    return {
        "state": _phone_state.name,
        "risk": risk_score,
        "events": _phone_events,
        "phone_detected": phone_detected
    }