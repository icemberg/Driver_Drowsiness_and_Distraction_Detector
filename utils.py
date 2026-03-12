import numpy as np
from scipy.spatial import distance as dist
import cv2
import pygame
import time
import logging
from datetime import datetime
import os
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
            pygame.mixer.init(44100, -16, 2, 1024)
        pygame.mixer.music.set_volume(volume)
        if sound_file is None or not os.path.exists(sound_file):
            duration = 1.0  
            frequency = 880  
            sample_rate = 44100
            samples = int(duration * sample_rate)
            t = np.linspace(0, duration, samples, False)
            wave1 = np.sin(2 * np.pi * frequency * t) * 32767 * 0.7  
            wave2 = np.sin(2 * np.pi * (frequency*1.5) * t) * 32767 * 0.3  
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
def calculate_eye_aspect_ratio(eye1, eye2):
    x1, y1, w1, h1 = eye1
    x2, y2, w2, h2 = eye2
    ratio1 = h1 / max(w1, 1)  
    ratio2 = h2 / max(w2, 1)
    area1 = w1 * h1
    area2 = w2 * h2
    avg_ratio = (ratio1 + ratio2) / 2.0
    ear = 0.27 * avg_ratio
    size_factor = min(0.03, (area1 + area2) / 20000)
    ear += size_factor
    return min(max(ear, 0.15), 0.35)
def is_looking_away(eyes, frame_height, frame_width):
    if len(eyes) < 2:
        return False
    eye_centers = []
    for x, y, w, h in eyes[:2]:  
        center_x = x + w/2
        center_y = y + h/2
        eye_centers.append((center_x, center_y))
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    for cx, cy in eye_centers:
        if cx < frame_width * 0.2 or cx > frame_width * 0.8:
            return True
        if cy < frame_height * 0.15 or cy > frame_height * 0.6:
            return True
    return False
def detect_eye_closure(eye_roi_gray):
    eye_roi_eq = cv2.equalizeHist(eye_roi_gray)
    hist = cv2.calcHist([eye_roi_eq], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    weights = np.linspace(1.0, 0.1, 256)  
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
