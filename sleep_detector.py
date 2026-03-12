import cv2
import time
import argparse
import numpy as np
import pygame
import mediapipe as mp
from datetime import datetime
import config
from utils import play_alarm, log_drowsiness_event, resize_frame
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
LEFT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]
RIGHT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Advanced eye-based drowsiness detection system')
    parser.add_argument('--ear', type=float, default=0.17,
                        help='Eye aspect ratio threshold')
    parser.add_argument('--frames', type=int, default=60,
                        help='Number of consecutive frames to trigger alarm (default 60)')
    parser.add_argument('--seconds', type=float, default=None,
                        help='Number of seconds of eye-closure to trigger alarm (overrides --frames using camera fps)')
    parser.add_argument('--camera', type=int, default=config.CAMERA_INDEX,
                        help='Camera index (default from config)')
    parser.add_argument('--log', action='store_true', default=config.ENABLE_LOGGING,
                        help='Enable logging of drowsiness events')
    parser.add_argument('--debug', action='store_true', default=config.SHOW_EYE_PROCESSING,
                        help='Show debug information for eye processing')
    parser.add_argument('--silent', action='store_true', default=False,
                        help='Run in silent mode (no alarm sound)')
    return parser.parse_args()

def calculate_ear(eye_landmarks):
    if len(eye_landmarks) != 16:
        return 0.0
    upper_landmarks = eye_landmarks[:8]
    lower_landmarks = eye_landmarks[8:]
    upper_y = np.mean([lm.y for lm in upper_landmarks])
    lower_y = np.mean([lm.y for lm in lower_landmarks])
    eye_width = abs(max([lm.x for lm in eye_landmarks]) - min([lm.x for lm in eye_landmarks]))
    eye_height = abs(upper_y - lower_y)
    if eye_width > 0:
        return eye_height / eye_width
    return 0.0
def extract_eye_landmarks(face_landmarks, eye_indices):
    return [face_landmarks.landmark[idx] for idx in eye_indices]
def main():
    args = parse_arguments()
    if args.log:
        from utils import initialize_logger
        initialize_logger(config.LOG_FILE)
    pygame.mixer.init()
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    COUNTER = 0
    ALARM_ON = False
    ear_history = []
    print("[INFO] Starting video stream...")
    vs = cv2.VideoCapture(args.camera)
    fps = vs.get(cv2.CAP_PROP_FPS)
    try:
        fps = float(fps)
    except Exception:
        fps = 0.0
    if fps <= 0.0 or np.isnan(fps):
        fps = 30.0
        print("[WARN] Camera FPS returned 0 or invalid. Falling back to 30 FPS for timing calculations.")
    if args.seconds is not None:
        computed_frames = int(max(1, round(args.seconds * fps)))
        print(f"[INFO] --seconds provided: {args.seconds}s -> threshold set to {computed_frames} frames (@{fps:.1f} FPS).")
        args.frames = computed_frames
    else:
        effective_seconds = args.frames / fps
        print(f"[INFO] Using frames threshold: {args.frames} frames (~{effective_seconds:.2f} seconds at {fps:.1f} FPS).")
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    time.sleep(1.0)
    print("[INFO] Press 'q' to quit the application")
    print("[INFO] Press 'd' to toggle debug mode")
    print("[INFO] Press 'r' to reset counters")
    debug_mode = args.debug
    while True:
        ret, frame = vs.read()
        if not ret:
            print("[ERROR] Failed to grab frame - check your camera connection")
            break
        frame = resize_frame(frame, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        status_text = "Status: Awake"
        if debug_mode:
            debug_frame = frame.copy()
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            if debug_mode:
                mp_drawing.draw_landmarks(
                    image=debug_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=debug_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                mp_drawing.draw_landmarks(
                    image=debug_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            left_eye_landmarks = extract_eye_landmarks(face_landmarks, LEFT_EYE_INDICES)
            right_eye_landmarks = extract_eye_landmarks(face_landmarks, RIGHT_EYE_INDICES)
            left_ear = calculate_ear(left_eye_landmarks)
            right_ear = calculate_ear(right_eye_landmarks)
            ear = (left_ear + right_ear) / 2.0
            ear_history.append(ear)
            if len(ear_history) > 5:
                ear_history.pop(0)
            smoothed_ear = sum(ear_history) / len(ear_history)
            cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR, 2)
            if smoothed_ear < args.ear:
                COUNTER += 1
                if debug_mode:
                    seconds_equiv = COUNTER / fps
                    cv2.putText(frame, f"Closed frames: {COUNTER}/{args.frames} ({seconds_equiv:.2f}s)",
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if COUNTER >= args.frames:
                    if not ALARM_ON:
                        ALARM_ON = True
                        if not args.silent:
                            play_alarm(config.ALARM_SOUND, config.ALARM_VOLUME)
                        if args.log:
                            log_drowsiness_event(config.LOG_FILE)
                    status_text = "Status: DROWSY!"
                    cv2.putText(frame, "WAKE UP!", (10, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR, 2)
            else:
                if smoothed_ear > (args.ear + 0.02):
                    COUNTER = 0
                    ALARM_ON = False
        else:
            if debug_mode:
                cv2.putText(frame, "No face detected", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            COUNTER = max(0, COUNTER - 1)
        cv2.putText(frame, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR, 2)
        timestamp = datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imshow("Sleep Detector (MediaPipe)", frame)
        if debug_mode:
            cv2.imshow("Debug View", debug_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            debug_mode = not debug_mode
            if not debug_mode and 'debug_frame' in locals():
                cv2.destroyWindow("Debug View")
        elif key == ord("r"):
            COUNTER = 0
            ear_history = []
            print("[INFO] Counters reset")
    vs.release()
    cv2.destroyAllWindows()
    print("[INFO] Sleep detection system stopped")
if __name__ == "__main__":
    main()