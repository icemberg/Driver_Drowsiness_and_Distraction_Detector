import cv2
import mediapipe as mp
import numpy as np
import math
import config
from utils import play_alarm

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -------------------------------
# Head Rotation Metrics Function
# -------------------------------
neutral_yaw = None
neutral_pitch = None
neutral_roll = None

def reset_calibration():
    """Reset neutral head-pose calibration so next frame becomes the new neutral."""
    global neutral_yaw, neutral_pitch, neutral_roll
    neutral_yaw = None
    neutral_pitch = None
    neutral_roll = None

def vector_head_pose(landmarks, w, h):

    global neutral_yaw, neutral_pitch, neutral_roll
    
    def p(i):
        return np.array([
            landmarks.landmark[i].x * w,
            landmarks.landmark[i].y * h,
            landmarks.landmark[i].z * w
        ])

    # ---- Stable landmarks ----
    left_eye = p(33)
    right_eye = p(263)
    nose = p(1)
    left_mouth = p(61)
    right_mouth = p(291)

    # ---- Face horizontal axis ----
    eye_vector = right_eye - left_eye
    eye_vector = eye_vector / np.linalg.norm(eye_vector)

    # ---- Face vertical axis ----
    mouth_center = (left_mouth + right_mouth) / 2
    vertical_vector = mouth_center - nose
    vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)

    # ---- Face normal (looking direction) ----
    face_normal = np.cross(eye_vector, vertical_vector)
    face_normal = face_normal / np.linalg.norm(face_normal)

    # ---- Compute angles ----
    yaw = np.degrees(np.arctan2(face_normal[0], face_normal[2]))
    pitch = np.degrees(np.arctan2(-face_normal[1], face_normal[2]))
    roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))

    # Calibrate against the first (or re-calibrated) frame
    if neutral_yaw is None:
        neutral_yaw = yaw
        neutral_pitch = pitch
        neutral_roll = roll
        
    yaw -= neutral_yaw
    pitch -= neutral_pitch
    roll -= neutral_roll
    
    forward = (
        abs(yaw) < config.HEAD_YAW_LIMIT and
        abs(pitch) < config.HEAD_PITCH_LIMIT and
        abs(roll) < config.HEAD_ROLL_LIMIT
    )

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "forward": forward
    }    

# -------------------------------
# Status Function
# -------------------------------
def get_status(face_landmarks, w, h):

    # ----- Left Eye -----
    l_iris = face_landmarks.landmark[468]
    l_inner = face_landmarks.landmark[133]
    l_outer = face_landmarks.landmark[33]

    denom_l = l_inner.x - l_outer.x
    left_ratio = (l_iris.x - l_outer.x) / denom_l if abs(denom_l) > 1e-6 else 0.5

    # ----- Right Eye -----
    r_iris = face_landmarks.landmark[473]
    r_inner = face_landmarks.landmark[362]
    r_outer = face_landmarks.landmark[263]

    denom_r = r_outer.x - r_inner.x
    right_ratio = (r_iris.x - r_inner.x) / denom_r if abs(denom_r) > 1e-6 else 0.5

    # ----- Average Gaze -----
    gaze_ratio = (left_ratio + right_ratio) / 2

    # 0.5 means center — values outside the config band mean looking away
    is_gazing_away = not (config.GAZE_CENTER_MIN < gaze_ratio < config.GAZE_CENTER_MAX)

    # --- Head Pose ---
    metrics = vector_head_pose(face_landmarks, w, h)

    # --- Center Check ---
    nose_x = face_landmarks.landmark[1].x
    is_off_center = not (config.OFF_CENTER_MIN < nose_x < config.OFF_CENTER_MAX)

    return is_gazing_away, metrics, is_off_center


# -------------------------------
# Main Loop
# -------------------------------
def main():

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    # Consecutive-frame counter for temporal smoothing
    distraction_counter = 0
    alarm_on = False
    no_face_counter = 0

    print("[INFO] Distraction detection started.")
    print("[INFO] Press 'Esc' to quit, 'c' to re-calibrate neutral head pose.")

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            no_face_counter = 0  # reset when face is found

            for landmarks in results.multi_face_landmarks:

                # Draw Mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

                # Run Detection
                gaze, metrics, center = get_status(landmarks, w, h)

                # Print angles
                if "yaw" in metrics:
                    print(f"Yaw={metrics['yaw']:.2f} Pitch={metrics['pitch']:.2f} Roll={metrics['roll']:.2f}")

                # --- Distraction decision ---
                # Any ONE signal independently means distraction
                frame_distracted = gaze or (not metrics["forward"]) or center

                if frame_distracted:
                    distraction_counter += 1
                else:
                    distraction_counter = 0
                    alarm_on = False

                # Only show warning after sustained distraction
                if distraction_counter >= config.DISTRACTION_CONSEC_FRAMES:
                    cv2.putText(frame, "DISTRACTED", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Trigger alarm once per distraction episode
                    if not alarm_on:
                        alarm_on = True
                        play_alarm(config.ALARM_SOUND, config.ALARM_VOLUME)
                else:
                    cv2.putText(frame, "ATTENTIVE", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                text = f"Gaze:{gaze} HeadTurn:{not metrics['forward']} OffCenter:{center}"

                cv2.putText(frame, text, (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
            # No face detected
            no_face_counter += 1
            distraction_counter = 0
            alarm_on = False

            cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2)

            # If face has been absent for a long time, escalate warning
            if no_face_counter >= config.DISTRACTION_CONSEC_FRAMES * 3:
                cv2.putText(frame, "DRIVER ABSENT!", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Face Mesh & Attention Detection", frame)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # Esc
            break
        elif key == ord('c'):
            reset_calibration()
            print("[INFO] Neutral head-pose re-calibrated.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Distraction detection stopped.")


if __name__ == "__main__":
    main()