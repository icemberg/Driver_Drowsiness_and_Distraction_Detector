import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from enum import Enum
from collections import deque
from config.config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    DRINK_HAND_MOUTH_DISTANCE_THRESHOLD,
    DRINK_SUSTAINED_FRAMES,
    DRINK_POSSIBLE_FRAMES,
    DRINK_CONFIRMED_FRAMES,
    DRINK_ALERT_FRAMES,
    HEAD_YAW_DISTRACTION_THRESHOLD,
    HEAD_PITCH_DISTRACTION_THRESHOLD,
    DRINK_DETECTION_CONFIDENCE_THRESHOLD,
    DRINK_BBOX_FACE_PROXIMITY_THRESHOLD,
    DRINK_ALERT_COOLDOWN,
    DRINK_ALERT_DURATION,
    ALARM_SOUND,
    ALARM_VOLUME,
    ENABLE_LOGGING,
    LOG_FILE,
    # New config parameters
    ENABLE_DRINK_DETECTION,
    ENABLE_DRINK_SNAPSHOTS,
    ENABLE_DRINK_CSV_LOGGING,
    DRINK_LOG_DIRECTORY,
    DRINK_DETECTOR_MODEL_PATH,
    DRINK_SNAPSHOTS_DIRECTORY,
    DRINK_EVENT_SNAPSHOT_COUNT,
    DRINK_RISK_THRESHOLD_IDLE_TO_POSSIBLE,
    DRINK_RISK_THRESHOLD_POSSIBLE_TO_CONFIRMED,
    DRINK_RISK_THRESHOLD_CONFIRMED_TO_ALERT,
    DRINK_FRAMES_IDLE_TO_POSSIBLE,
    DRINK_FRAMES_POSSIBLE_TO_CONFIRMED,
    DRINK_FRAMES_CONFIRMED_TO_ALERT,
    DRINK_RISK_FALLBACK_THRESHOLD,
)
from utils.utils import (
    play_alarm,
    initialize_logger,
    mouth_center,
    hand_near_mouth,
    calculate_mouth_aspect_ratio,
    calculate_eye_aspect_ratio,
    draw_face_mesh,
    estimate_face_width,
    normalize_hand_mouth_distance,
    estimate_head_pose,
    check_drink_bbox_near_mouth,
    get_hand_index_tip,
    
    # New utility functions
    initialize_drink_detection_logger,
    log_drink_event,
    save_frame_snapshot,
    is_drink_object_near_mouth,
    calculate_weighted_risk_score,
)

# ============================================================
# MediaPipe setup — imported from shared setup
# ============================================================
from setup.setup import (
    face_mesh as _shared_face_mesh,
    hands as _shared_hands,
    mp_face_mesh, mp_hands, mp_drawing,
)

# ============================================================
# State Machine
# ============================================================
class DrinkState(Enum):
    IDLE = 0
    POSSIBLE_DRINKING = 1
    DRINKING = 2
    ALERT = 3


# ============================================================
# Helpers
# ============================================================
def fuse_three_signals(
    hand_mouth_dist_normalized,
    drink_near_mouth,
    head_distracted,
    hand_threshold=DRINK_HAND_MOUTH_DISTANCE_THRESHOLD,
):
    """
    Fuse three detection signals with weighted scoring:
    1. Hand-to-mouth distance (normalized by face width)
    2. Drink object near mouth
    3. Head pose indicates distraction
    
    Returns: risk_score (0-3, higher = more risk)
    """
    return calculate_weighted_risk_score(
        hand_mouth_dist_normalized,
        drink_near_mouth,
        head_distracted,
        hand_threshold
    )


def load_custom_drink_detector():
    """
    Load YOLOv8 drink detector model.
    Falls back if model not found (will use hand proximity only).
    """
    try:
        from features.drink_and_drive.yolov8_drink_detector import YOLOv8DrinkDetector
        model_path = DRINK_DETECTOR_MODEL_PATH
        
        if __import__('os').path.exists(model_path):
            detector = YOLOv8DrinkDetector.load(model_path)
            if detector and detector.model:
                print(f"[INFO] YOLOv8 drink detector loaded: {model_path}")
                return detector
        else:
            print(f"[INFO] Trained model not found: {model_path}")
            print("[INFO] Using hand-proximity detection only")
    except Exception as e:
        print(f"[INFO] Could not load YOLOv8 detector: {e}")
    
    return None


# ============================================================
# Module-level state for process_frame() (used by orchestrator)
# ============================================================
_drink_state       = DrinkState.IDLE
_drink_frame_ctr   = 0
_drink_grace_ctr   = 0          # counts consecutive low-risk frames before fallback
_DRINK_GRACE_FRAMES = 3         # tolerate up to 3 noisy low-risk frames
_drink_events      = 0
_drink_alert_until = 0.0
_drink_alarm_on    = False
_drink_model       = None
_drink_csv_path    = None
_drink_initialized = False


def _ensure_drink_init():
    """Lazy-init drink model and CSV logger on first process_frame() call."""
    global _drink_model, _drink_csv_path, _drink_initialized
    if _drink_initialized:
        return
    _drink_initialized = True
    if ENABLE_DRINK_DETECTION:
        _drink_model = load_custom_drink_detector()
    if ENABLE_DRINK_CSV_LOGGING:
        try:
            _drink_csv_path = initialize_drink_detection_logger()
        except Exception as e:
            print(f"[WARN] Drink CSV logger failed: {e}")


def process_frame(face_landmarks, hand_results, frame, w, h, now, frame_number=0, silent=False):
    """
    Process one frame for drink-and-drive detection.

    Args:
        face_landmarks: mediapipe face_landmarks object (or None).
        hand_results:   mediapipe hands results object.
        frame:          BGR frame (for YOLOv8 object detection).
        w, h:           frame dimensions.
        now:            current time.time().
        frame_number:   current frame count.
        silent:         if True, suppress alarm.

    Returns:
        dict with keys: face, state, risk, events, hand_dist, drink_obj, head_dist
    """
    global _drink_state, _drink_frame_ctr, _drink_grace_ctr, _drink_events
    global _drink_alert_until, _drink_alarm_on

    _ensure_drink_init()

    base = {"face": False, "state": _drink_state.name,
            "risk": 0.0, "events": _drink_events,
            "hand_dist": None, "drink_obj": False, "head_dist": False}

    if face_landmarks is None:
        return base

    lms  = face_landmarks.landmark
    m_pt = mouth_center(lms, w, h)
    fw   = estimate_face_width(lms, w)

    if m_pt is None or fw is None or fw < 10 or fw > w:
        return {**base, "face": True}

    # Head distraction
    yaw, pitch = estimate_head_pose(lms)
    head_dist  = (abs(yaw) > HEAD_YAW_DISTRACTION_THRESHOLD or
                  abs(pitch) > HEAD_PITCH_DISTRACTION_THRESHOLD)

    # Hand proximity
    hand_norm = None
    if hand_results.multi_hand_landmarks:
        for hl in hand_results.multi_hand_landmarks:
            d = normalize_hand_mouth_distance(hl.landmark[8], m_pt, fw, w, h)
            if d is not None:
                hand_norm = min(hand_norm, d) if hand_norm is not None else d

    # YOLOv8 drink object detection
    drink_obj = False
    if _drink_model is not None:
        try:
            dets = _drink_model.predict(frame)
            drink_obj = any(
                det.get("confidence", 0) >= DRINK_DETECTION_CONFIDENCE_THRESHOLD
                for det in dets
            )
        except Exception:
            pass

    # Risk score
    risk = fuse_three_signals(hand_norm, drink_obj, head_dist,
                              DRINK_HAND_MOUTH_DISTANCE_THRESHOLD)

    # ── Play alarm when risk is high enough ──
    if risk > 1.5 and not _drink_alarm_on and not silent:
        _drink_alarm_on = True
        play_alarm(ALARM_SOUND, ALARM_VOLUME)

    # State machine (with grace-frame tolerance for noisy hand tracking)
    _drink_frame_ctr += 1

    if _drink_state == DrinkState.IDLE:
        if risk >= DRINK_RISK_THRESHOLD_IDLE_TO_POSSIBLE:
            _drink_frame_ctr = 0
            _drink_grace_ctr = 0
            _drink_state = DrinkState.POSSIBLE_DRINKING

    elif _drink_state == DrinkState.POSSIBLE_DRINKING:
        if risk >= DRINK_RISK_THRESHOLD_POSSIBLE_TO_CONFIRMED:
            _drink_grace_ctr = 0                     # good frame — reset grace
            if _drink_frame_ctr >= DRINK_FRAMES_POSSIBLE_TO_CONFIRMED:
                _drink_frame_ctr = 0
                _drink_state = DrinkState.DRINKING
                # ── Play alarm on confirmed drinking ──
                if not _drink_alarm_on:
                    _drink_alarm_on = True
                    if not silent:
                        play_alarm(ALARM_SOUND, ALARM_VOLUME)
        elif risk < DRINK_RISK_FALLBACK_THRESHOLD:
            _drink_grace_ctr += 1
            if _drink_grace_ctr >= _DRINK_GRACE_FRAMES:
                _drink_frame_ctr = 0
                _drink_grace_ctr = 0
                _drink_state = DrinkState.IDLE
        else:
            # Risk between fallback and confirmed — keep waiting, don't reset
            _drink_grace_ctr = 0

    elif _drink_state == DrinkState.DRINKING:
        if risk >= DRINK_RISK_THRESHOLD_CONFIRMED_TO_ALERT:
            _drink_grace_ctr = 0
            if _drink_frame_ctr >= DRINK_FRAMES_CONFIRMED_TO_ALERT:
                _drink_frame_ctr   = 0
                _drink_state       = DrinkState.ALERT
                _drink_alert_until = now + DRINK_ALERT_DURATION
                _drink_events     += 1
                if _drink_csv_path:
                    log_drink_event(_drink_csv_path, "ALERT", risk,
                                    hand_norm, drink_obj, head_dist, frame_number)
                if not _drink_alarm_on:
                    _drink_alarm_on = True
                    if not silent:
                        play_alarm(ALARM_SOUND, ALARM_VOLUME)
        elif risk < DRINK_RISK_FALLBACK_THRESHOLD:
            _drink_grace_ctr += 1
            if _drink_grace_ctr >= _DRINK_GRACE_FRAMES:
                _drink_frame_ctr = 0
                _drink_grace_ctr = 0
                _drink_state = DrinkState.IDLE
        else:
            # Risk between fallback and alert threshold — keep waiting
            _drink_grace_ctr = 0

    elif _drink_state == DrinkState.ALERT:
        if now >= _drink_alert_until:
            _drink_frame_ctr = 0
            _drink_grace_ctr = 0
            _drink_state = DrinkState.IDLE
            _drink_alarm_on = False

    return {"face": True, "state": _drink_state.name,
            "risk": round(risk, 2), "events": _drink_events,
            "hand_dist": hand_norm, "drink_obj": drink_obj, "head_dist": head_dist}


def reset_drink_state():
    """Reset module-level drink state."""
    global _drink_state, _drink_frame_ctr, _drink_grace_ctr, _drink_events
    global _drink_alert_until, _drink_alarm_on
    _drink_state = DrinkState.IDLE; _drink_frame_ctr = 0; _drink_grace_ctr = 0
    _drink_events = 0; _drink_alert_until = 0.0; _drink_alarm_on = False


# ============================================================
# Standalone Main Detection Pipeline
# ============================================================
def main():
    if ENABLE_LOGGING:
        initialize_logger(LOG_FILE)
    
    # Initialize drink detection logger
    csv_logger_path = None
    if ENABLE_DRINK_CSV_LOGGING:
        csv_logger_path = initialize_drink_detection_logger()
    
    # State machine
    state = DrinkState.IDLE
    frame_count_in_state = 0
    last_alert_time = 0.0
    
    # Temporal tracking
    sustained_frame_counter = 0
    drink_frame_counter = deque(maxlen=10)
    alert_snapshot_buffer = deque(maxlen=DRINK_EVENT_SNAPSHOT_COUNT)
    
    # Metrics
    drink_and_drive_count = 0
    alert_until = 0.0
    frame_number = 0
    
    last_frame_time = time.time()
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Load custom drink detector
    custom_drink_detector = load_custom_drink_detector() if ENABLE_DRINK_DETECTION else None
    
    print("="*60)
    print("DRINK AND DRIVE DETECTION PIPELINE")
    print("="*60)
    print(f"[CONFIG] Using tuned thresholds:")
    print(f"  - Risk thresholds: {DRINK_RISK_THRESHOLD_IDLE_TO_POSSIBLE} → {DRINK_RISK_THRESHOLD_POSSIBLE_TO_CONFIRMED} → {DRINK_RISK_THRESHOLD_CONFIRMED_TO_ALERT}")
    print(f"  - Frame requirements: {DRINK_FRAMES_IDLE_TO_POSSIBLE} → {DRINK_FRAMES_POSSIBLE_TO_CONFIRMED} → {DRINK_FRAMES_CONFIRMED_TO_ALERT}")
    print(f"  - Drink detection: {'ENABLED' if ENABLE_DRINK_DETECTION else 'DISABLED'}")
    print(f"  - Event logging: {'ENABLED' if ENABLE_DRINK_CSV_LOGGING else 'DISABLED'}")
    print(f"  - Frame snapshots: {'ENABLED' if ENABLE_DRINK_SNAPSHOTS else 'DISABLED'}")
    print("="*60)
    print("Press ESC to exit\n")

    try:
        face_mesh_ctx = _shared_face_mesh
        hands_ctx     = _shared_hands

        while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                now = time.time()
                dt = now - last_frame_time
                last_frame_time = now

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                face_results = face_mesh_ctx.process(rgb)
                hand_results = hands_ctx.process(rgb)
                rgb.flags.writeable = True

                # Initialize detection signals
                hand_mouth_dist_normalized = None
                drink_near_mouth = False
                head_distracted = False
                risk_score = 0.0
                mouth_point = None
                face_width = None
                yaw = 0.0
                pitch = 0.0
                drink_objects = []
                face_detected = False

                # ============================================================
                # FACE DETECTION: Optional but preferred
                # ============================================================
                if face_results.multi_face_landmarks:
                    face_detected = True
                    face_landmarks = face_results.multi_face_landmarks[0].landmark
                    
                    # Get mouth center and face width
                    mouth_point = mouth_center(face_landmarks, w, h)
                    face_width = estimate_face_width(face_landmarks, w)
                    
                    # Validate face measurements
                    if mouth_point is not None and face_width is not None and face_width >= 10 and face_width <= w:
                        # Estimate head pose
                        yaw, pitch = estimate_head_pose(face_landmarks)
                        head_distracted = (
                            abs(yaw) > HEAD_YAW_DISTRACTION_THRESHOLD
                            or abs(pitch) > HEAD_PITCH_DISTRACTION_THRESHOLD
                        )
                    else:
                        # Face detected but measurements invalid
                        face_detected = False
                        mouth_point = None
                        face_width = None
                    
                    # Draw face mesh if valid
                    if face_detected:
                        draw_face_mesh(frame, face_results.multi_face_landmarks[0])

                # ============================================================
                # HAND DETECTION: Works independently of face
                # ============================================================
                if hand_results.multi_hand_landmarks and mouth_point is not None and face_width is not None:
                    # Hand detected AND we have valid face measurements
                    for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                        # Get hand index finger tip
                        index_tip = get_hand_index_tip(hand_landmarks, w, h)
                        
                        # Compute normalized hand-to-mouth distance
                        norm_dist = normalize_hand_mouth_distance(
                            hand_landmarks.landmark[8],
                            mouth_point,
                            face_width,
                            w,
                            h,
                        )
                        
                        if norm_dist is not None:
                            if hand_mouth_dist_normalized is None:
                                hand_mouth_dist_normalized = norm_dist
                            else:
                                hand_mouth_dist_normalized = min(hand_mouth_dist_normalized, norm_dist)
                        
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )

                # ============================================================
                # DRINK OBJECT DETECTION: Use as signal
                # ============================================================
                if ENABLE_DRINK_DETECTION and custom_drink_detector is not None:
                    try:
                        detections = custom_drink_detector.predict(frame)

                        valid_detections = []

                        for det in detections:
                            obj_class = det.get('class', 'unknown').lower()
                            
                            # SKIP glass objects (high false positive rate)
                            if 'glass' in obj_class or 'cup' in obj_class:
                                continue
                            
                            x1, y1, x2, y2 = det["bbox"]
                            bw = x2 - x1
                            bh = y2 - y1

                            # ❌ REJECT FULL FRAME / HUGE BOXES (noise)
                            if bw > 0.8 * w and bh > 0.8 * h:
                                continue

                            # ❌ REJECT TOO SMALL NOISE (<20px)
                            if bw < 20 or bh < 20:
                                continue
                            
                            # ✅ VALID CUP/BOTTLE/CAN/MUG object
                            valid_detections.append(det)

                        # If we have valid drink objects, mark as detected
                        # The risk scorer will weight based on hand proximity + head position
                        if valid_detections:
                            drink_near_mouth = True  # Valid object detected
                            drink_objects = valid_detections
                            if frame_number % 30 == 0:
                                print(f"[DRINK-OBJ] {len(valid_detections)} valid object(s) detected: {[d['class'] for d in valid_detections]}")
                        else:
                            drink_near_mouth = False
                            drink_objects = []

                        # Draw all valid detections on frame (not just valid ones)
                        if valid_detections:
                            frame = custom_drink_detector.draw_detections(
                                frame, valid_detections, color=(0, 255, 0), thickness=2
                            )
                            
                    except Exception as e:
                        print(f"[ERROR] YOLOv8 detection error: {e}")
                        drink_near_mouth = False
                        drink_objects = []

                # ============================================================
                # RISK SCORING: Behavior-based approach
                # ============================================================
                if face_detected and mouth_point is not None and face_width is not None:
                    # Primary signal: hand near mouth
                    # This is most reliable even without YOLO
                    risk_score = fuse_three_signals(
                        hand_mouth_dist_normalized,
                        drink_near_mouth,
                        head_distracted,
                    )
                    
                    # Debug: Show signal values
                    if frame_number % 20 == 0:
                        hand_str = f"{hand_mouth_dist_normalized:.2f}" if hand_mouth_dist_normalized else "N/A"
                        print(f"[HEAD-POSE] yaw={yaw:.1f}° pitch={pitch:.1f}° distracted={head_distracted}")
                        print(f"[SIGNALS] hand={hand_str}, drink={drink_near_mouth}, distracted={head_distracted} → risk={risk_score:.2f}")
                else:
                    # Face not detected - skip state machine updates but keep displaying
                    if frame_number % 30 == 0:
                        print(f"[WARN] Face detection failed on frame {frame_number}")
                    risk_score = 0.0
                    
                print(f"[DEBUG] hand={hand_mouth_dist_normalized} | drink={drink_near_mouth} | head={head_distracted} | risk={risk_score}")

                # ============================================================
                # State Machine Update with Tuned Thresholds
                # (Only run if face is detected for reliable risk scoring)
                # ============================================================
                if face_detected and risk_score is not None:
                    frame_count_in_state += 1
                    drink_frame_counter.append(risk_score)
                    
                    # Store frame for potential snapshot
                    if ENABLE_DRINK_SNAPSHOTS:
                        alert_snapshot_buffer.append(frame.copy())
                    
                    if state == DrinkState.IDLE:
                        if risk_score >= DRINK_RISK_THRESHOLD_IDLE_TO_POSSIBLE:
                            frame_count_in_state = 0
                            state = DrinkState.POSSIBLE_DRINKING
                            print(f"[STATE] IDLE → POSSIBLE_DRINKING (risk={risk_score:.1f})")
                            if csv_logger_path:
                                log_drink_event(csv_logger_path, "POSSIBLE_DRINKING", risk_score, 
                                              hand_mouth_dist_normalized, drink_near_mouth, head_distracted, frame_number)

                    elif state == DrinkState.POSSIBLE_DRINKING:
                        if risk_score >= DRINK_RISK_THRESHOLD_POSSIBLE_TO_CONFIRMED:
                            if frame_count_in_state >= DRINK_FRAMES_POSSIBLE_TO_CONFIRMED:
                                print(f"[STATE] POSSIBLE_DRINKING → DRINKING (consistent for {frame_count_in_state} frames)")
                                frame_count_in_state = 0
                                state = DrinkState.DRINKING
                                if csv_logger_path:
                                    log_drink_event(csv_logger_path, "DRINKING", risk_score, 
                                                  hand_mouth_dist_normalized, drink_near_mouth, head_distracted, frame_number)
                        else:
                            # Fallback to IDLE if risk drops
                            if risk_score < DRINK_RISK_FALLBACK_THRESHOLD:
                                frame_count_in_state = 0
                                state = DrinkState.IDLE

                    elif state == DrinkState.DRINKING:
                        if risk_score >= DRINK_RISK_THRESHOLD_CONFIRMED_TO_ALERT:
                            if frame_count_in_state >= DRINK_FRAMES_CONFIRMED_TO_ALERT:
                                frame_count_in_state = 0
                                state = DrinkState.ALERT
                                alert_until = now + DRINK_ALERT_DURATION
                                last_alert_time = now
                                drink_and_drive_count += 1
                                
                                # Play alarm
                                play_alarm(ALARM_SOUND, ALARM_VOLUME)
                                print(f"[ALERT] 🚨 DRINK AND DRIVE DETECTED! Event #{drink_and_drive_count}")
                                
                                # Log alert event
                                if csv_logger_path:
                                    log_drink_event(csv_logger_path, "ALERT", risk_score, 
                                                  hand_mouth_dist_normalized, drink_near_mouth, head_distracted, frame_number)
                                
                                # Save snapshots
                                if ENABLE_DRINK_SNAPSHOTS and len(alert_snapshot_buffer) > 0:
                                    for snap_idx, snap_frame in enumerate(alert_snapshot_buffer):
                                        snap_path = save_frame_snapshot(snap_frame, "ALERT", frame_number, snap_idx)
                                        if snap_path:
                                            print(f"  [SNAPSHOT] Saved: {snap_path}")
                        else:
                            # Fallback to IDLE if risk drops
                            if risk_score < DRINK_RISK_FALLBACK_THRESHOLD:
                                print(f"[STATE] DRINKING → IDLE (risk dropped to {risk_score:.1f})")
                                frame_count_in_state = 0
                                state = DrinkState.IDLE

                    elif state == DrinkState.ALERT:
                        if now >= alert_until:
                            frame_count_in_state = 0
                            state = DrinkState.IDLE
                            print(f"[STATE] ALERT → IDLE (alert duration expired)")
                else:
                    # Face not detected - don't update state machine but keep last state
                    if ENABLE_DRINK_SNAPSHOTS:
                        alert_snapshot_buffer.append(frame.copy())

                # ============================================================
                # Overlay and Visualization
                # ============================================================
                state_color = (0, 255, 0)  # Default: green (IDLE)
                state_text = state.name
                
                if state == DrinkState.POSSIBLE_DRINKING:
                    state_color = (0, 255, 255)  # Yellow
                elif state == DrinkState.DRINKING:
                    state_color = (0, 165, 255)  # Orange
                elif state == DrinkState.ALERT:
                    state_color = (0, 0, 255)  # Red

                # Draw state information
                cv2.putText(
                    frame, f'State: {state_text}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2
                )
                cv2.putText(
                    frame, f'Risk Score: {risk_score:.1f}/3.0', (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
                cv2.putText(
                    frame, f'Events: {drink_and_drive_count}', (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
                cv2.putText(
                    frame, f'Frame: {frame_number}', (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1
                )

                # Draw signal details
                y_offset = 180
                
                # Face detection status
                face_status_color = (100, 255, 100) if face_detected else (255, 100, 100)
                cv2.putText(
                    frame, f'Face Detected: {"Yes" if face_detected else "No"}', (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_status_color, 2
                )
                y_offset += 30
                
                if hand_mouth_dist_normalized is not None:
                    hand_color = (100, 255, 100) if hand_mouth_dist_normalized < DRINK_HAND_MOUTH_DISTANCE_THRESHOLD else (255, 100, 100)
                    cv2.putText(
                        frame, f'Hand-Mouth: {hand_mouth_dist_normalized:.2f} (threshold: {DRINK_HAND_MOUTH_DISTANCE_THRESHOLD:.2f})', (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2
                    )
                else:
                    cv2.putText(
                        frame, f'Hand-Mouth: No hand detected', (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2
                    )
                y_offset += 30

                drinks_color = (100, 255, 100) if drink_near_mouth else (255, 100, 100)
                cv2.putText(
                    frame, f'Drink Near Mouth: {"Yes" if drink_near_mouth else "No"}', (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, drinks_color, 2
                )
                y_offset += 30

                # Head distraction with pose angles
                head_color = (100, 255, 100) if not head_distracted else (255, 100, 100)
                head_status = f"Yes (yaw={yaw:.0f}°, pitch={pitch:.0f}°)" if head_distracted else f"No (yaw={yaw:.0f}°, pitch={pitch:.0f}°)"
                cv2.putText(
                    frame, f'Head Distracted: {head_status}', (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, head_color, 2
                )

                # Draw alert message
                if now < alert_until:
                    cv2.putText(
                        frame, '🚨 DRINK AND DRIVE ALERT! 🚨', (10, h-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
                    )

                cv2.imshow('Drink and Drive Detection (Enhanced)', frame)

                # Key handling
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Total frames processed: {frame_number}")
        print(f"Total drink & drive events detected: {drink_and_drive_count}")
        if csv_logger_path:
            print(f"Event log saved to: {csv_logger_path}")
        if ENABLE_DRINK_SNAPSHOTS:
            print(f"Snapshots saved to: {DRINK_SNAPSHOTS_DIRECTORY}")
        print("="*60)


if __name__ == '__main__':
    main()
