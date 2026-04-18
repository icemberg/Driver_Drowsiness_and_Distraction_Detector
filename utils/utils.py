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


def initialize_logger(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("driver_drowsiness session started")
    
    
def play_alarm(sound_file=None, volume=1.0):
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
    Estimate head yaw and pitch from facial landmarks.
    Returns: (yaw_degrees, pitch_degrees)
    
    Yaw: positive = face turned right, negative = face turned left
    Pitch: positive = face tilted up, negative = face tilted down
    """
    # Nose tip (landmark 1) and nose bridge (landmark 168)
    nose_tip_x = landmarks[1].x
    nose_base_x = landmarks[168].x
    
    # Eye centers
    left_eye_center = (landmarks[33].x + landmarks[133].x) / 2
    right_eye_center = (landmarks[362].x + landmarks[263].x) / 2
    
    # Horizontal offset indicates yaw
    eye_center_x = (left_eye_center + right_eye_center) / 2
    yaw = (nose_tip_x - eye_center_x) * 90  # Rough angle estimation
    
    # Vertical: use mouth and nose landmarks
    mouth_center_y = (landmarks[13].y + landmarks[14].y) / 2
    nose_y = landmarks[1].y
    nose_bridge_y = landmarks[168].y
    
    pitch = (mouth_center_y - nose_y) * 80  # Rough angle estimation
    
    return yaw, pitch


def check_drink_bbox_near_mouth(drink_bbox, mouth_point, face_width, frame_width, frame_height):
    """
    Check if a drink object bounding box is near the mouth region.
    drink_bbox format: [x_min, y_min, x_max, y_max] (in pixel coordinates)
    Returns: True if bbox is in lower face region and close to mouth horizontally.
    """
    if drink_bbox is None or len(drink_bbox) < 4:
        return False
    
    x_min, y_min, x_max, y_max = drink_bbox
    bbox_center_x = (x_min + x_max) / 2
    bbox_center_y = (y_min + y_max) / 2
    
    # Check if bbox is in lower half of face (below mouth level)
    mouth_y = mouth_point[1]
    lower_face_start = mouth_y + face_width * 0.05
    
    if bbox_center_y < lower_face_start:
        return False
    
    # Check horizontal proximity to mouth
    mouth_x = mouth_point[0]
    horizontal_dist = abs(bbox_center_x - mouth_x)
    proximity_threshold = face_width * 0.4  # Within 40% of face width
    
    return horizontal_dist < proximity_threshold


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
        # Create log directories
        if not os.path.exists(config.DRINK_LOG_DIRECTORY):
            os.makedirs(config.DRINK_LOG_DIRECTORY)
        
        if config.ENABLE_DRINK_SNAPSHOTS and not os.path.exists(config.DRINK_SNAPSHOTS_DIRECTORY):
            os.makedirs(config.DRINK_SNAPSHOTS_DIRECTORY)
        
        # Initialize CSV file with headers
        csv_path = os.path.join(config.DRINK_LOG_DIRECTORY, config.DRINK_LOG_FILENAME)
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write("timestamp,event_type,risk_score,hand_mouth_distance,object_detected,head_distracted,frame_number\n")
        
        print(f"[INFO] Drink detection logger initialized. Logs will be saved to {config.DRINK_LOG_DIRECTORY}")
        return csv_path
    except Exception as e:
        print(f"[ERROR] Failed to initialize drink detection logger: {e}")
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
        if csv_path is None or not config.ENABLE_DRINK_CSV_LOGGING:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(csv_path, 'a') as f:
            f.write(f"{timestamp},{event_type},{risk_score:.2f},{hand_mouth_distance if hand_mouth_distance else 'N/A'},{object_detected},{head_distracted},{frame_number}\n")
    except Exception as e:
        print(f"[ERROR] Failed to log drink event: {e}")


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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}_frame{frame_number}_{event_type}_snap{snapshot_index}.jpg"
        filepath = os.path.join(config.DRINK_SNAPSHOTS_DIRECTORY, filename)
        
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


def is_drink_object_near_mouth(drink_objects, mouth_point, face_width):
    """
    Check if any detected drink object is near the mouth region.
    
    Args:
        drink_objects: List of detected drink objects
        mouth_point: 2D point of mouth center [x, y]
        face_width: Width of face in pixels
    
    Returns:
        True if any drink object is near the mouth
    """
    try:
        for obj in drink_objects:
            bbox = obj['bbox']
            if check_drink_bbox_near_mouth(bbox, mouth_point, face_width, 640, 480):
                return True
        return False
    except Exception as e:
        print(f"[ERROR] Error checking drink object proximity to mouth: {e}")
        return False


def calculate_weighted_risk_score(hand_proximity, object_detected, head_distracted, hand_threshold=0.15):
    """
    Calculate weighted risk score from multiple signals.
    
    Args:
        hand_proximity: Normalized hand-to-mouth distance (0-1)
        object_detected: Boolean, whether drink object was detected
        head_distracted: Boolean, whether head is distracted
        hand_threshold: Threshold for hand proximity (default 0.15)
    
    Returns:
        Weighted risk score (0-3.0)
    """
    score = 0.0
    
    # Hand proximity signal
    if hand_proximity is not None and hand_proximity < hand_threshold:
        hand_signal = 1.0 * config.SIGNAL_WEIGHT_HAND_PROXIMITY
        score += hand_signal
    
    # Object detection signal
    if object_detected:
        object_signal = 1.0 * config.SIGNAL_WEIGHT_OBJECT_DETECTION
        score += object_signal
    
    # Head distraction signal
    if head_distracted:
        distraction_signal = 1.0 * config.SIGNAL_WEIGHT_HEAD_DISTRACTION
        score += distraction_signal
    
    return min(score, 3.0)  # Cap at 3.0
