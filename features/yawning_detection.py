import cv2
import numpy as np
import time
from collections import deque
from config.config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT
)
from utils.utils import play_alarm, initialize_logger, mouth_center, hand_near_mouth, calculate_mouth_aspect_ratio, calculate_eye_aspect_ratio, draw_face_mesh
from config.config import (
    MAR_THRESHOLD,
    EAR_THRESHOLD,
    MIN_YAWN_DURATION,
    MIN_EYE_CLOSED_OVERLAP,
    HAND_MOUTH_DISTANCE_PX,
    SMOOTH_WINDOW,
    YAWN_COOLDOWN,
    YAWN_ALERT_DURATION,
    ALARM_SOUND,
    ALARM_VOLUME,
    ENABLE_LOGGING,
    LOG_FILE,
)
from setup.setup import (
    face_mesh as _shared_face_mesh,
    hands as _shared_hands,
    mp_face_mesh, mp_hands, mp_drawing,
)

# ============================================================
# Landmark indices
# ============================================================
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308


RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


# ── Module-level state for process_frame() (used by orchestrator) ─────────────
_mar_hist    = deque(maxlen=SMOOTH_WINDOW)
_ear_y_hist  = deque(maxlen=SMOOTH_WINDOW)
_yawn_count  = 0
_yawn_alert_until = 0.0
_yawn_cooldown    = 0.0
_mouth_open_t     = None
_eye_closed_overlap     = 0.0
_hand_seen_during_mouth = False
_yawn_counted_this_event = False
_yawn_alarm_on = False


def process_frame(face_landmarks, hand_results, w, h, now, dt, silent=False):
    """
    Process one frame for yawning detection.

    Args:
        face_landmarks: mediapipe face_landmarks object (or None).
        hand_results:   mediapipe hands results object.
        w, h:           frame dimensions.
        now:            current time.time().
        dt:             delta time since last frame.
        silent:         if True, suppress alarm.

    Returns:
        dict with keys: face, mar, ear, yawning, yawn_count, hand_near
    """
    global _mar_hist, _ear_y_hist, _yawn_count, _yawn_alert_until
    global _yawn_cooldown, _mouth_open_t, _eye_closed_overlap
    global _hand_seen_during_mouth, _yawn_counted_this_event, _yawn_alarm_on

    if face_landmarks is None:
        return {"face": False, "mar": 0.0, "ear": 0.0,
                "yawning": False, "yawn_count": _yawn_count, "hand_near": False}

    lms = face_landmarks.landmark
    mar = calculate_mouth_aspect_ratio(lms, w, h)
    le  = calculate_eye_aspect_ratio(lms, LEFT_EYE, w, h)
    re  = calculate_eye_aspect_ratio(lms, RIGHT_EYE, w, h)
    ear = (le + re) / 2.0

    _mar_hist.append(mar)
    _ear_y_hist.append(ear)
    sm_mar = float(np.mean(_mar_hist))
    sm_ear = float(np.mean(_ear_y_hist))

    mouth_open  = sm_mar > MAR_THRESHOLD
    eyes_closed = sm_ear < EAR_THRESHOLD

    m_pt = mouth_center(lms, w, h)
    hand_close = False
    if hand_results.multi_hand_landmarks and m_pt is not None:
        for hl in hand_results.multi_hand_landmarks:
            near, _ = hand_near_mouth(hl, m_pt, w, h, HAND_MOUTH_DISTANCE_PX)
            hand_close = hand_close or near

    # Score-based yawn detection (same logic as standalone main)
    if mouth_open:
        if _mouth_open_t is None:
            _mouth_open_t            = now
            _eye_closed_overlap      = 0.0
            _hand_seen_during_mouth  = hand_close
            _yawn_counted_this_event = False
        else:
            _hand_seen_during_mouth = _hand_seen_during_mouth or hand_close

        if eyes_closed:
            _eye_closed_overlap += dt

        dur   = now - _mouth_open_t
        score = (1.0 if dur >= MIN_YAWN_DURATION else 0.0) + \
                (1.0 if _eye_closed_overlap >= MIN_EYE_CLOSED_OVERLAP else 0.0) + \
                (0.25 if _hand_seen_during_mouth else 0.0)

        if (not _yawn_counted_this_event
                and now >= _yawn_cooldown
                and score >= 2.0):
            _yawn_count += 1
            _yawn_alert_until    = now + YAWN_ALERT_DURATION
            _yawn_cooldown       = now + YAWN_COOLDOWN
            _yawn_counted_this_event = True
            if not _yawn_alarm_on:
                _yawn_alarm_on = True
                if not silent:
                    play_alarm(ALARM_SOUND, ALARM_VOLUME)
    else:
        _mouth_open_t            = None
        _eye_closed_overlap      = 0.0
        _hand_seen_during_mouth  = False
        _yawn_counted_this_event = False
        _yawn_alarm_on           = False

    is_yawning = now < _yawn_alert_until
    return {"face": True, "mar": round(sm_mar, 3), "ear": round(sm_ear, 3),
            "yawning": is_yawning, "yawn_count": _yawn_count, "hand_near": hand_close}


def reset_yawning_state():
    """Reset module-level yawning state."""
    global _yawn_count, _yawn_alert_until, _yawn_cooldown
    global _mouth_open_t, _eye_closed_overlap, _hand_seen_during_mouth
    global _yawn_counted_this_event, _yawn_alarm_on
    _mar_hist.clear(); _ear_y_hist.clear()
    _yawn_count = 0; _yawn_alert_until = 0.0; _yawn_cooldown = 0.0
    _mouth_open_t = None; _eye_closed_overlap = 0.0
    _hand_seen_during_mouth = False; _yawn_counted_this_event = False
    _yawn_alarm_on = False


# ── Standalone main ───────────────────────────────────────────────────────────
def main():
    if ENABLE_LOGGING:
        initialize_logger(LOG_FILE)

    mar_history = deque(maxlen=SMOOTH_WINDOW)
    ear_history = deque(maxlen=SMOOTH_WINDOW)

    # Per-mouth-open event state
    mouth_open_start = None
    eye_closed_overlap = 0.0
    peak_mar = 0.0
    hand_seen_during_mouth = False
    yawn_counted_this_event = False
    
    # Hand-on-mouth event tracking
    hand_mouth_start = None
    hand_mouth_eye_closed = 0.0
    hand_mouth_counted = False

    # Global state
    yawn_count = 0
    yawn_alert_until = 0.0
    cooldown_until = 0.0

    last_frame_time = time.time()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Press ESC to exit")

    face_mesh = _shared_face_mesh  # use shared instance
    hands = _shared_hands           # use shared instance

    try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                now = time.time()
                dt = now - last_frame_time
                last_frame_time = now

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                face_results = face_mesh.process(rgb)
                hand_results = hands.process(rgb)
                rgb.flags.writeable = True

                mar_smooth = 0.0
                ear_smooth = 0.0
                mouth_open = False
                eyes_closed = False
                hand_close = False
                hand_dist = None
                mouth_point = None

                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0].landmark

                    mar = calculate_mouth_aspect_ratio(face_landmarks, w, h)
                    
                    left_ear = calculate_eye_aspect_ratio(face_landmarks, LEFT_EYE, w, h)
                    right_ear = calculate_eye_aspect_ratio(face_landmarks, RIGHT_EYE, w, h)
                    ear = (left_ear + right_ear) / 2.0

                    mar_history.append(mar)
                    ear_history.append(ear)

                    mar_smooth = float(np.mean(mar_history))
                    ear_smooth = float(np.mean(ear_history))

                    mouth_open = mar_smooth > MAR_THRESHOLD
                    eyes_closed = ear_smooth < EAR_THRESHOLD
                    mouth_point = mouth_center(face_landmarks, w, h)

                    if hand_results.multi_hand_landmarks and mouth_point is not None:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            near, d = hand_near_mouth(
                                hand_landmarks,
                                mouth_point,
                                w,
                                h,
                                HAND_MOUTH_DISTANCE_PX,
                            )
                            hand_close = hand_close or near
                            hand_dist = d if hand_dist is None else min(hand_dist, d)
                            if mouth_open_start is not None or mouth_open:
                                print(f"[DEBUG] Hand: dist={d:.1f}px, threshold={HAND_MOUTH_DISTANCE_PX}px, near={near}")

                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                            )

                    draw_face_mesh(frame, face_results.multi_face_landmarks[0])

                    if mouth_open:
                        if mouth_open_start is None:
                            mouth_open_start = now
                            eye_closed_overlap = 0.0
                            peak_mar = mar_smooth
                            hand_seen_during_mouth = hand_close
                            yawn_counted_this_event = False
                        else:
                            peak_mar = max(peak_mar, mar_smooth)
                            hand_seen_during_mouth = hand_seen_during_mouth or hand_close

                        if eyes_closed:
                            eye_closed_overlap += dt

                        open_duration = now - mouth_open_start

                        # Score-based yawn detection:
                        # - Strong yawn requires mouth open + some eye closure
                        # - Occluded yawns (with hand near mouth) are allowed
                        yawn_score = 0.0

                        if open_duration >= MIN_YAWN_DURATION:
                            yawn_score += 1.0
                        if eye_closed_overlap >= MIN_EYE_CLOSED_OVERLAP:
                            yawn_score += 1.0
                        if hand_seen_during_mouth:
                            # Hand proximity is acceptable during yawn (e.g., covering mouth)
                            yawn_score += 0.25

                        print(f"[DEBUG] Yawn event - duration={open_duration:.2f}s, eye_overlap={eye_closed_overlap:.2f}s, hand={hand_seen_during_mouth}, score={yawn_score:.2f}")

                        # Detect yawn when score is strong enough and cooldown expired
                        if (
                            not yawn_counted_this_event
                            and now >= cooldown_until
                            and yawn_score >= 2.0
                        ):
                            yawn_count += 1
                            yawn_alert_until = now + YAWN_ALERT_DURATION
                            cooldown_until = now + YAWN_COOLDOWN
                            yawn_counted_this_event = True
                            play_alarm(ALARM_SOUND, ALARM_VOLUME)

                    else:
                        # Reset the event state when mouth closes
                        mouth_open_start = None
                        eye_closed_overlap = 0.0
                        peak_mar = 0.0
                        hand_seen_during_mouth = False
                        yawn_counted_this_event = False
                    # Alternative yawn detection: hand-on-mouth with eye closure
                    # Detects yawns where hand covers the mouth
                    if hand_close and eyes_closed:
                        if hand_mouth_start is None:
                            hand_mouth_start = now
                            hand_mouth_eye_closed = 0.0
                            hand_mouth_counted = False
                        else:
                            hand_mouth_eye_closed += dt

                        hand_mouth_duration = now - hand_mouth_start
                        
                        # Yawn triggered if hand near mouth + eyes closed long enough
                        if (
                            not hand_mouth_counted
                            and now >= cooldown_until
                            and hand_mouth_duration >= MIN_YAWN_DURATION
                            and hand_mouth_eye_closed >= MIN_EYE_CLOSED_OVERLAP
                        ):
                            yawn_count += 1
                            yawn_alert_until = now + YAWN_ALERT_DURATION
                            cooldown_until = now + YAWN_COOLDOWN
                            hand_mouth_counted = True
                            play_alarm(ALARM_SOUND, ALARM_VOLUME)
                            print(f'[DEBUG] Hand-on-mouth yawn detected: duration={hand_mouth_duration:.2f}s, eye_closed={hand_mouth_eye_closed:.2f}s')
                    else:
                        # Reset hand-mouth tracking when hand moves away or eyes open
                        hand_mouth_start = None
                        hand_mouth_eye_closed = 0.0
                        hand_mouth_counted = False
                    # Overlay text
                    cv2.putText(
                        frame, f'MAR: {mar_smooth:.2f}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
                    cv2.putText(
                        frame, f"EAR: {ear_smooth:.2f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
                    cv2.putText(
                        frame, f"Yawns: {yawn_count}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )

                    if hand_dist is not None:
                        hand_color = (100, 255, 100) if hand_close else (255, 100, 100)
                        cv2.putText(
                            frame, f'Hand: {hand_dist:.0f}px / {HAND_MOUTH_DISTANCE_PX}px', (20, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2
                        )
                    if hand_mouth_start is not None and hand_close:
                        hand_mouth_duration = now - hand_mouth_start
                        cv2.putText(
                            frame, f'Hand-mouth event: {hand_mouth_duration:.2f}s, eyes closed: {hand_mouth_eye_closed:.2f}s', (20, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 255), 2
                        )
                    if mouth_open_start is not None:
                        open_duration = now - mouth_open_start
                        cv2.putText(
                            frame, f"Mouth open: {open_duration:.2f}s", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                        )
                        cv2.putText(
                            frame, f"Eye closed overlap: {eye_closed_overlap:.2f}s", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                        )

                        if hand_seen_during_mouth:
                            cv2.putText(
                                frame, "Hand near mouth", (20, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2
                            )

                        if now < yawn_alert_until:
                            cv2.putText(
                                frame, "YAWNING DETECTED", (20, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
                            )
                        elif hand_seen_during_mouth and not eyes_closed and open_duration < MIN_YAWN_DURATION:
                            cv2.putText(
                                frame, "Likely drinking / talking", (20, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
                            )

                cv2.imshow("Advanced Yawning Detection", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()