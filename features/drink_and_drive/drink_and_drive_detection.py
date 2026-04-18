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
# MediaPipe setup
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
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
# Main Detection Pipeline
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
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh_ctx, mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands_ctx:

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
                drink_objects = []

                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0].landmark
                    
                    # Get mouth center and face width
                    mouth_point = mouth_center(face_landmarks, w, h)
                    face_width = estimate_face_width(face_landmarks, w)
                    
                    # Estimate head pose
                    yaw, pitch = estimate_head_pose(face_landmarks)
                    head_distracted = (
                        abs(yaw) > HEAD_YAW_DISTRACTION_THRESHOLD
                        or abs(pitch) > HEAD_PITCH_DISTRACTION_THRESHOLD
                    )
                    
                    # Draw face mesh
                    draw_face_mesh(frame, face_results.multi_face_landmarks[0])
                    
                    # Process hands
                    if hand_results.multi_hand_landmarks and mouth_point is not None:
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
                            
                            if hand_mouth_dist_normalized is None or norm_dist < hand_mouth_dist_normalized:
                                hand_mouth_dist_normalized = norm_dist
                            
                            # Draw hand landmarks
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                            )
                    
                    # ============================================================
                    # ENHANCED: Drink object detection using YOLOv8 model
                    # ============================================================
                    if ENABLE_DRINK_DETECTION and custom_drink_detector is not None:
                        try:
                            # YOLOv8 prediction
                            detections = custom_drink_detector.predict(frame)
                            
                            if detections:
                                drink_near_mouth = True
                                drink_objects = detections
                                
                                # Draw detections on frame
                                frame = custom_drink_detector.draw_detections(
                                    frame, detections, color=(0, 165, 255), thickness=2
                                )
                                
                                # Log detection info
                                for det in detections:
                                    print(f"[DETECT] {det['class']}: {det['confidence']:.2f}")
                        except Exception as e:
                            print(f"[ERROR] YOLOv8 detection error: {e}")
                    else:
                        # No detector or detector disabled
                        drink_near_mouth = False
                    
                    # Compute risk score from three signals
                    risk_score = fuse_three_signals(
                        hand_mouth_dist_normalized,
                        drink_near_mouth,
                        head_distracted,
                    )

                # ============================================================
                # State Machine Update with Tuned Thresholds
                # ============================================================
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
                            frame_count_in_state = 0
                            state = DrinkState.DRINKING
                            print(f"[STATE] POSSIBLE_DRINKING → DRINKING (consistent for {frame_count_in_state} frames)")
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
                            frame_count_in_state = 0
                            state = DrinkState.IDLE

                elif state == DrinkState.ALERT:
                    if now >= alert_until:
                        frame_count_in_state = 0
                        state = DrinkState.IDLE
                        print(f"[STATE] ALERT → IDLE (alert duration expired)")

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
                if hand_mouth_dist_normalized is not None:
                    hand_color = (100, 255, 100) if hand_mouth_dist_normalized < DRINK_HAND_MOUTH_DISTANCE_THRESHOLD else (255, 100, 100)
                    cv2.putText(
                        frame, f'Hand-Mouth: {hand_mouth_dist_normalized:.2f}', (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2
                    )
                    y_offset += 30

                drinks_color = (100, 255, 100) if drink_near_mouth else (255, 100, 100)
                cv2.putText(
                    frame, f'Drink Object: {"Yes" if drink_near_mouth else "No"}', (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, drinks_color, 2
                )
                y_offset += 30

                head_color = (100, 255, 100) if not head_distracted else (255, 100, 100)
                cv2.putText(
                    frame, f'Head Distracted: {"Yes" if head_distracted else "No"}', (20, y_offset),
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
