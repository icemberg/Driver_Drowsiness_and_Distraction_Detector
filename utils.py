import numpy as np
import cv2
import pygame
import time
import logging
from datetime import datetime
import os
import mediapipe as mp
import config

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
    eyes,
    frame_height,
    frame_width,
    horizontal_min_ratio=config.GAZE_HORIZONTAL_MIN_RATIO,
    horizontal_max_ratio=config.GAZE_HORIZONTAL_MAX_RATIO,
    vertical_min_ratio=config.GAZE_VERTICAL_MIN_RATIO,
    vertical_max_ratio=config.GAZE_VERTICAL_MAX_RATIO,
):
    if len(eyes) < 2:
        return False
    eye_centers = []
    for x, y, w, h in eyes[:2]:
        center_x = x + w / 2
        center_y = y + h / 2
        eye_centers.append((center_x, center_y))
    for cx, cy in eye_centers:
        if cx < frame_width * horizontal_min_ratio or cx > frame_width * horizontal_max_ratio:
            return True
        if cy < frame_height * vertical_min_ratio or cy > frame_height * vertical_max_ratio:
            return True
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
