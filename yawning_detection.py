import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from utils import play_alarm, initialize_logger, mouth_aspect_ratio
from config import MAR_THRESHOLD, MIN_YAWN_DURATION, ALARM_SOUND, ALARM_VOLUME

# -------------------------------
# Initialize MediaPipe Face Mesh
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# For smoothing
mar_history = deque(maxlen=5)

# State variables
yawn_start_time = None
yawn_in_progress = False
yawn_detected = False
alarm_on = False

# -------------------------------
# Start Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

print("Press ESC to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    mar = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Compute MAR
            mar = mouth_aspect_ratio(landmarks, w, h)

            # Smooth MAR
            mar_history.append(mar)
            mar = np.mean(mar_history)

            # Draw face mesh (optional)
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                )
            )

            # -------------------------------
            # Yawning Detection Logic
            # -------------------------------
            if mar > MAR_THRESHOLD:
                if not yawn_in_progress:
                    yawn_start_time = time.time()
                    yawn_in_progress = True

                duration = time.time() - yawn_start_time
                if duration > MIN_YAWN_DURATION:
                    if not yawn_detected:
                        yawn_detected = True
                        if not alarm_on:
                            alarm_on = True
                            play_alarm(ALARM_SOUND, ALARM_VOLUME)
            else:
                yawn_start_time = None
                yawn_in_progress = False
                yawn_detected = False
                alarm_on = False

            # -------------------------------
            # Display Info
            # -------------------------------
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if yawn_in_progress:
                duration = time.time() - yawn_start_time if yawn_start_time else 0
                cv2.putText(frame, f"Open Time: {duration:.2f}s", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if yawn_detected:
                cv2.putText(frame, "YAWNING DETECTED!", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show frame
    cv2.imshow("Yawning Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()