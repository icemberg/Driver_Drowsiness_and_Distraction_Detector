import numpy as np
import cv2
import pygame
import time
import logging
from datetime import datetime
import os
import mediapipe as mp
import config.config as config

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
_alarm_last_played = 0.0   # timestamp of last alarm play


def initialize_logger(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("driver_drowsiness session started")
    
    



def play_alarm(sound_file=None, volume=1.0):
    global _alarm_last_played

    # ── Global cooldown: prevent rapid-fire overlapping beeps ──
    now = time.time()
    cooldown = getattr(config, 'ALARM_DURATION', 1.0) + 0.5
    if now - _alarm_last_played < cooldown:
        return  # another alarm is still playing / just played

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(
                config.ALARM_SAMPLE_RATE,
                config.ALARM_AUDIO_FORMAT,
                config.ALARM_CHANNELS,
                config.ALARM_BUFFER_SIZE,
            )
        pygame.mixer.music.set_volume(volume)
        if sound_file is None or not os.path.exists(sound_file):
            samples = int(config.ALARM_DURATION * config.ALARM_SAMPLE_RATE)
            t = np.linspace(0, config.ALARM_DURATION, samples, False)
            wave1 = np.sin(2 * np.pi * config.ALARM_FREQUENCY * t) * config.ALARM_AMPLITUDE * 0.7
            wave2 = np.sin(2 * np.pi * (config.ALARM_FREQUENCY * 1.5) * t) * config.ALARM_AMPLITUDE * 0.3
            tone = wave1 + wave2
            buffer = np.zeros((samples, 2), dtype=np.int16)
            buffer[:, 0] = tone  
            buffer[:, 1] = tone  
            sound = pygame.sndarray.make_sound(buffer)
            sound.play()
            print("[INFO] Alarm sound playing (generated beep)")
        else:
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            print(f"[INFO] Alarm sound playing from file: {sound_file}")

        _alarm_last_played = time.time()
    except Exception as e:
        print(f"[ERROR] Failed to play alarm: {e}")
        print("\a" * 3)  
        
        
def calculate_eye_aspect_ratio(face_landmarks, eye_indices, w, h):
    """
    EAR = (p2-p6 + p3-p5) / (2 * p1-p4)
    Lower values mean the eye is more closed.
    """
    p1 = get_point(face_landmarks, eye_indices[0], w, h)
    p2 = get_point(face_landmarks, eye_indices[1], w, h)
    p3 = get_point(face_landmarks, eye_indices[2], w, h)
    p4 = get_point(face_landmarks, eye_indices[3], w, h)
    p5 = get_point(face_landmarks, eye_indices[4], w, h)
    p6 = get_point(face_landmarks, eye_indices[5], w, h)

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal < 1e-6:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))




def is_looking_away(
    face_landmarks,
    frame_width,
    frame_height,
    horizontal_min_ratio=config.GAZE_HORIZONTAL_MIN_RATIO,
    horizontal_max_ratio=config.GAZE_HORIZONTAL_MAX_RATIO,
    vertical_min_ratio=config.GAZE_VERTICAL_MIN_RATIO,
    vertical_max_ratio=config.GAZE_VERTICAL_MAX_RATIO,
):
    """
    Check if driver is looking away based on face landmarks.
    MediaPipe FaceMesh indices for eyes:
    Left eye: indices 33, 133 (left and right corners)
    Right eye: indices 362, 263 (left and right corners)
    """
    if face_landmarks is None:
        return False
    
    try:
        # Get eye landmarks (normalized 0-1)
        # Left eye outer corner and inner corner
        left_eye_x = face_landmarks.landmark[33].x
        right_eye_x = face_landmarks.landmark[362].x
        
        # Check horizontal gaze
        if left_eye_x < horizontal_min_ratio or left_eye_x > horizontal_max_ratio:
            return True
        if right_eye_x < horizontal_min_ratio or right_eye_x > horizontal_max_ratio:
            return True
        
        # Check vertical gaze (left and right eye vertical positions)
        left_eye_y = face_landmarks.landmark[33].y
        right_eye_y = face_landmarks.landmark[362].y
        
        if left_eye_y < vertical_min_ratio or left_eye_y > vertical_max_ratio:
            return True
        if right_eye_y < vertical_min_ratio or right_eye_y > vertical_max_ratio:
            return True
        
        return False
    except (AttributeError, IndexError):
        return False



def detect_eye_closure(
    eye_roi_gray,
    bins=config.EYE_CLOSURE_HIST_BINS,
    weight_start=config.EYE_CLOSURE_WEIGHT_START,
    weight_end=config.EYE_CLOSURE_WEIGHT_END,
):
    eye_roi_eq = cv2.equalizeHist(eye_roi_gray)
    hist = cv2.calcHist([eye_roi_eq], [0], None, [bins], [0, 256])
    hist = hist / hist.sum()
    weights = np.linspace(weight_start, weight_end, bins)
    closure_score = np.sum(hist.flatten() * weights)
    return closure_score


def log_drowsiness_event(log_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_file, "a") as f:
            f.write(f"{timestamp} - Drowsiness detected\n")
        logging.info("Drowsiness event logged")
    except Exception as e:
        print(f"[ERROR] Failed to log drowsiness event: {e}")
        
        
def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def get_point(landmarks, idx, w, h):
    return np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float32)


def mouth_center(landmarks, w, h):
    points = np.array([
        get_point(landmarks, config.UPPER_LIP_LANDMARK, w, h),
        get_point(landmarks, config.LOWER_LIP_LANDMARK, w, h),
        get_point(landmarks, config.LEFT_MOUTH_LANDMARK, w, h),
        get_point(landmarks, config.RIGHT_MOUTH_LANDMARK, w, h),
    ], dtype=np.float32)
    return np.mean(points, axis=0)


def hand_near_mouth(hand_landmarks, mouth_point, w, h, threshold_px=config.HAND_MOUTH_DISTANCE_PX):
    min_dist = float("inf")
    for lm in hand_landmarks.landmark:
        point = np.array([lm.x * w, lm.y * h], dtype=np.float32)
        min_dist = min(min_dist, np.linalg.norm(point - mouth_point))
    return min_dist < threshold_px, min_dist


def draw_face_mesh(frame, face_landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing.draw_landmarks(
        frame,
        face_landmarks,
        mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=1, circle_radius=1,
        ),
    )

def dist(a, b):
    return np.linalg.norm(a - b)


def calculate_mouth_aspect_ratio(landmarks, w, h):
    top = np.array([landmarks[config.UPPER_LIP_LANDMARK].x * w, landmarks[config.UPPER_LIP_LANDMARK].y * h])
    bottom = np.array([landmarks[config.LOWER_LIP_LANDMARK].x * w, landmarks[config.LOWER_LIP_LANDMARK].y * h])

    left = np.array([landmarks[config.LEFT_MOUTH_LANDMARK].x * w, landmarks[config.LEFT_MOUTH_LANDMARK].y * h])
    right = np.array([landmarks[config.RIGHT_MOUTH_LANDMARK].x * w, landmarks[config.RIGHT_MOUTH_LANDMARK].y * h])

    vertical_dist = np.linalg.norm(top - bottom)
    horizontal_dist = np.linalg.norm(left - right)

    return vertical_dist / (horizontal_dist + 1e-6)  # avoid division by zero


def estimate_face_width(landmarks, w):
    """
    Estimate face width using inter-ocular distance (distance between eyes).
    Returns width in pixels.
    """
    left_eye_x = landmarks[33].x * w  # LEFT_EYE outer corner
    right_eye_x = landmarks[263].x * w  # RIGHT_EYE outer corner
    face_width = abs(right_eye_x - left_eye_x)
    return max(face_width, 1.0)


def normalize_hand_mouth_distance(hand_landmark, mouth_point, face_width, w, h):
    """
    Compute normalized hand-to-mouth distance (0-1 scale relative to face width).
    Lower values = hand closer to mouth.
    """
    hand_point = np.array([hand_landmark.x * w, hand_landmark.y * h], dtype=np.float32)
    distance = np.linalg.norm(hand_point - mouth_point)
    normalized = distance / max(face_width, 1.0)
    return normalized


def estimate_head_pose(landmarks):
    """
    Estimate head yaw and pitch from facial landmarks using proper geometry.
    Returns: (yaw_degrees, pitch_degrees)
    
    Yaw: positive = face turned right, negative = face turned left
    Pitch: positive = face tilted up, negative = face turned down
    """
    try:
        # Key landmarks (normalized 0-1)
        nose_tip = np.array([landmarks[1].x, landmarks[1].y])
        nose_bridge = np.array([landmarks[168].x, landmarks[168].y])
        
        # Eyes
        left_eye = np.array([landmarks[33].x, landmarks[33].y])
        right_eye = np.array([landmarks[263].x, landmarks[263].y])
        
        # Mouth
        mouth_left = np.array([landmarks[61].x, landmarks[61].y])
        mouth_right = np.array([landmarks[291].x, landmarks[291].y])
        
        # Chin (bottom of face)
        chin = np.array([landmarks[152].x, landmarks[152].y])
        
        # Face reference measurements
        eye_center = (left_eye + right_eye) / 2.0
        face_width = np.linalg.norm(right_eye - left_eye)
        face_height = np.linalg.norm(chin - nose_bridge)
        
        # Avoid division by zero
        if face_width < 1e-6 or face_height < 1e-6:
            return 0.0, 0.0
        
        # YAW: Horizontal offset of nose from eye center
        # Normalize by face width for scale invariance
        nose_horizontal_offset = (nose_tip[0] - eye_center[0]) / face_width
        # Convert to degrees: ±1.0 offset ≈ ±45 degrees
        yaw = np.clip(nose_horizontal_offset * 45, -90, 90)
        
        # PITCH: Vertical position of nose relative to face center
        # Normalize by face height
        face_center_y = (nose_bridge[1] + chin[1]) / 2.0
        nose_vertical_offset = (nose_tip[1] - face_center_y) / face_height
        # Convert to degrees: ±1.0 offset ≈ ±40 degrees
        pitch = np.clip(nose_vertical_offset * 40, -90, 90)
        
        return float(yaw), float(pitch)
        
    except (AttributeError, IndexError, TypeError) as e:
        print(f"[ERROR] Head pose estimation failed: {e}")
        return 0.0, 0.0


def check_drink_bbox_near_mouth(drink_bbox, mouth_point, face_width, frame_width, frame_height):
    """
    Check if a drink object bounding box is near the mouth region.
    drink_bbox format: [x_min, y_min, x_max, y_max] (in pixel coordinates)
    Returns: True if bbox is close to mouth (horizontally and vertically).
    """
    if drink_bbox is None or len(drink_bbox) < 4 or mouth_point is None or face_width is None:
        return False
    
    try:
        x_min, y_min, x_max, y_max = [float(v) for v in drink_bbox]
        bbox_center_x = (x_min + x_max) / 2.0
        bbox_center_y = (y_min + y_max) / 2.0
        
        mouth_x = float(mouth_point[0])
        mouth_y = float(mouth_point[1])
        face_w = float(face_width)
        
        # VERY RELAXED thresholds: detect drinks anywhere in upper half of frame
        vertical_upper = mouth_y - face_w * 1.5  # 150% above
        vertical_lower = mouth_y + face_w * 1.5  # 150% below
        
        horizontal_threshold = face_w * 1.5  # 150% of face width
        
        vertical_dist = abs(bbox_center_y - mouth_y)
        horizontal_dist = abs(bbox_center_x - mouth_x)
        
        # Debug every detection
        print(f"[PROX-CHECK] bbox=({x_min:.0f},{y_min:.0f},{x_max:.0f},{y_max:.0f}) center=({bbox_center_x:.0f},{bbox_center_y:.0f})")
        print(f"  mouth=({mouth_x:.0f},{mouth_y:.0f}) face_w={face_w:.0f}")
        print(f"  V: {vertical_dist:.0f} < {face_w*1.5:.0f}? {vertical_dist < face_w*1.5}")
        print(f"  H: {horizontal_dist:.0f} < {face_w*1.5:.0f}? {horizontal_dist < face_w*1.5}")
        
        result = (vertical_dist < face_w * 1.5) and (horizontal_dist < face_w * 1.5)
        print(f"  -> {result}")
        return result
        
    except Exception as e:
        print(f"[ERROR] Proximity check crashed: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_hand_index_tip(hand_landmarks, w, h):
    """
    Extract index finger tip from hand landmarks.
    Returns: 2D point (x, y) in pixel coordinates.
    """
    # Index finger tip is landmark index 8 in MediaPipe hand landmarks
    index_tip = hand_landmarks.landmark[8]
    return np.array([index_tip.x * w, index_tip.y * h], dtype=np.float32)


# ============================================================
# DRINK DETECTION LOGGING AND SNAPSHOT FUNCTIONS
# ============================================================

def initialize_drink_detection_logger():
    """Initialize directories and CSV logger for drink detection events."""
    try:
        # Create log directories using proper path construction
        log_dir = os.path.normpath(config.DRINK_LOG_DIRECTORY)
        os.makedirs(log_dir, exist_ok=True)
        
        if config.ENABLE_DRINK_SNAPSHOTS:
            snapshots_dir = os.path.normpath(config.DRINK_SNAPSHOTS_DIRECTORY)
            os.makedirs(snapshots_dir, exist_ok=True)
        
        # Initialize CSV file with headers (use normpath for consistency)
        csv_path = os.path.normpath(os.path.join(log_dir, config.DRINK_LOG_FILENAME))
        
        # Create or append to CSV
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', buffering=1) as f:  # Line buffering for immediate writes
                f.write("timestamp,event_type,risk_score,hand_mouth_distance,object_detected,head_distracted,frame_number\n")
                f.flush()
        
        print(f"[INFO] Drink detection logger initialized. Logs will be saved to {csv_path}")
        return csv_path
    except Exception as e:
        print(f"[ERROR] Failed to initialize drink detection logger: {e}")
        import traceback
        traceback.print_exc()
        return None


def log_drink_event(csv_path, event_type, risk_score, hand_mouth_distance, object_detected, head_distracted, frame_number):
    """
    Log a drink detection event to CSV file.
    
    Args:
        csv_path: Path to the CSV log file
        event_type: Type of event (IDLE, POSSIBLE_DRINKING, DRINKING, ALERT)
        risk_score: Current risk score (0-3)
        hand_mouth_distance: Normalized hand-to-mouth distance
        object_detected: Whether a drink object was detected
        head_distracted: Whether head pose indicates distraction
        frame_number: Current frame number
    """
    try:
        if csv_path is None:
            return
        
        if not config.ENABLE_DRINK_CSV_LOGGING:
            return
        
        # Normalize path
        csv_path = os.path.normpath(csv_path)
        
        # Ensure directory exists
        csv_dir = os.path.dirname(csv_path)
        os.makedirs(csv_dir, exist_ok=True)
        
        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Format hand_mouth_distance (handle None)
        hand_dist_str = f"{hand_mouth_distance:.3f}" if hand_mouth_distance is not None else "N/A"
        
        # Build CSV row
        row = f"{timestamp},{event_type},{risk_score:.2f},{hand_dist_str},{object_detected},{head_distracted},{frame_number}\n"
        
        # Write with line buffering (immediate flush)
        with open(csv_path, 'a', buffering=1) as f:
            f.write(row)
            f.flush()  # Explicit flush to ensure write
        
        # Debug logging
        print(f"[LOG-WRITE] {event_type}: risk={risk_score:.2f}, drink={object_detected}, hand={hand_dist_str}, frame={frame_number}")
        
    except Exception as e:
        print(f"[ERROR] Failed to log drink event to {csv_path}: {e}")
        import traceback
        traceback.print_exc()


def save_frame_snapshot(frame, event_type, frame_number, snapshot_index):
    """
    Save a frame snapshot when drink and drive event is detected.
    
    Args:
        frame: The frame to save (OpenCV image)
        event_type: Type of event (IDLE, POSSIBLE_DRINKING, DRINKING, ALERT)
        frame_number: Current frame number
        snapshot_index: Index of snapshot (0-5 for before/during/after ALERT)
    
    Returns:
        Path to saved snapshot
    """
    try:
        if not config.ENABLE_DRINK_SNAPSHOTS:
            return None
        
        # Ensure snapshot directory exists
        snapshots_dir = config.DRINK_SNAPSHOTS_DIRECTORY
        os.makedirs(snapshots_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}_frame{frame_number}_{event_type}_snap{snapshot_index}.jpg"
        filepath = os.path.join(snapshots_dir, filename)
        
        # Save frame
        cv2.imwrite(filepath, frame)
        return filepath
    except Exception as e:
        print(f"[ERROR] Failed to save frame snapshot: {e}")
        return None


def detect_drink_objects_with_mediapipe(frame, detector):
    """
    Detect drink objects using MediaPipe Object Detector.
    
    Args:
        frame: Input frame (OpenCV image, BGR format)
        detector: MediaPipe ObjectDetector instance
    
    Returns:
        List of detected drink objects with bounding boxes and confidence scores
    """
    try:
        # Convert frame to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect objects
        detections = detector.detect(rgb)
        
        drink_objects = []
        if hasattr(detections, 'detections'):
            for detection in detections.detections:
                # Extract category and confidence
                label = detection.categories[0].category_name if detection.categories else None
                confidence = detection.categories[0].score if detection.categories else 0.0
                
                # Check if it's a drink class and confidence is high enough
                if label in config.DRINK_CLASSES and confidence >= config.DRINK_DETECTOR_CONFIDENCE_THRESHOLD:
                    # Extract bounding box
                    bbox = detection.bounding_box
                    h, w = frame.shape[:2]
                    
                    # Convert normalized coords to pixel coords
                    x_min = int(bbox.origin_x * w)
                    y_min = int(bbox.origin_y * h)
                    x_max = int((bbox.origin_x + bbox.width) * w)
                    y_max = int((bbox.origin_y + bbox.height) * h)
                    
                    # Calculate bounding box area
                    bbox_area = (x_max - x_min) * (y_max - y_min)
                    
                    # Filter by minimum area
                    if bbox_area >= config.DRINK_DETECTOR_BOX_AREA_MIN_PIXELS:
                        drink_objects.append({
                            'class': label,
                            'confidence': float(confidence),
                            'bbox': [x_min, y_min, x_max, y_max],
                            'area': bbox_area
                        })
        
        return drink_objects
    except Exception as e:
        print(f"[ERROR] Failed to detect drink objects: {e}")
        return []


def draw_drink_detections(frame, drink_objects, color=(0, 255, 0)):
    """
    Draw drink object detections on frame.
    
    Args:
        frame: Input frame (OpenCV image)
        drink_objects: List of detected drink objects
        color: BGR color for bounding boxes
    
    Returns:
        Frame with drawn detections
    """
    try:
        for obj in drink_objects:
            x_min, y_min, x_max, y_max = obj['bbox']
            
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw label and confidence
            label = f"{obj['class']} ({obj['confidence']:.2f})"
            cv2.putText(
                frame,
                label,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return frame
    except Exception as e:
        print(f"[ERROR] Failed to draw drink detections: {e}")
        return frame


def is_drink_object_near_mouth(drink_objects, mouth_point, face_width, frame_width=None, frame_height=None):
    """
    Check if any detected drink object is near the mouth region.
    
    Args:
        drink_objects: List of detected drink objects
        mouth_point: 2D point of mouth center [x, y]
        face_width: Width of face in pixels
        frame_width: Frame width (optional)
        frame_height: Frame height (optional)
    
    Returns:
        True if any drink object is near the mouth
    """
    try:
        if mouth_point is None or face_width is None or not drink_objects:
            return False
        
        for idx, obj in enumerate(drink_objects):
            bbox = obj['bbox']
            fwidth = frame_width or 640
            fheight = frame_height or 480
            result = check_drink_bbox_near_mouth(bbox, mouth_point, face_width, fwidth, fheight)
            
            if result:
                print(f"[PROX-OK] Object {idx} IS near mouth!")
                return True
        
        return False
    except Exception as e:
        print(f"[ERROR] Error checking drink object proximity to mouth: {e}")
        return False


def calculate_weighted_risk_score(hand_proximity, object_detected, head_distracted, hand_threshold=0.70):
    """
    Calculate weighted risk score from multiple signals.
    
    BEHAVIOR-BASED DRINKING DETECTION:
    Progressively increasing score as signals align:
    1. Head tilt = 0.7 (preparatory)
    2. Head + hand near = 1.3+ (drinking action)  
    3. Head + hand + object = 1.8+ (confirmed)
    
    Args:
        hand_proximity: Normalized hand-to-mouth distance (0-1)
        object_detected: Boolean, whether drink object was detected
        head_distracted: Boolean, whether head is distracted/tilting
        hand_threshold: Threshold for hand proximity (default 0.70 = RELAXED)
    
    Returns:
        Weighted risk score (0-3.0)
    """
    risk = 0.0

    # SIGNAL 1: Head distracted/tilting (always if true)
    # Drinking involves tilting head back (positive pitch) - first indicator
    if head_distracted:
        risk += 0.7  # Preparatory signal

    # SIGNAL 2: Hand near mouth (PRIMARY drinking indicator)  
    # Most reliable - hand moving to mouth for drinking
    if hand_proximity is not None and hand_proximity < hand_threshold:
        risk += 0.6  # Hand near mouth signal
        
        # BONUS: If very close (<0.3), boost confidence
        if hand_proximity < 0.30:
            risk += 0.2  # Extra confidence when very close
    elif hand_proximity is not None and hand_proximity < hand_threshold * 1.5:
        # Hand somewhat far but detectable - weak signal
        risk += 0.2

    # SIGNAL 3: Object detection (weak but confirmatory)
    # Only trust if head is tilting (drinking behavior confirmed)
    if object_detected and head_distracted:
        risk += 0.5  # Confirms what hand is carrying

    return min(risk, 3.0)
