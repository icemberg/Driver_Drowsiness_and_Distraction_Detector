#!/usr/bin/env python3
"""
Driver Safety Monitor — CLI Orchestrator (main.py)

Pure orchestrator: imports process_frame() from each feature module,
runs one camera + one shared MediaPipe pass, and calls all four
detectors on every frame.  Zero feature logic lives here.

Controls:  q / ESC = quit   r = reset   c = recalibrate head pose
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

# ── Shared MediaPipe modules (constants, drawing utils) ──────────────────────
from setup.setup import mp_face_mesh, mp_hands

# ── Config ────────────────────────────────────────────────────────────────────
import config.config as config

# ── Feature process_frame() imports ──────────────────────────────────────────
from features.sleep_detector import (
    process_frame as sleep_process,
    reset_sleep_state,
)
from features.distraction_detection import (
    process_frame as distraction_process,
    reset_calibration,
    reset_distraction_state,
)
from features.yawning_detection import (
    process_frame as yawning_process,
    reset_yawning_state,
)
from features.drink_and_drive.drink_and_drive_detection import (
    process_frame as drink_process,
    reset_drink_state,
)
from features.drink_and_drive.drink_and_drive_detection import (
    process_frame as drink_process,
    reset_drink_state,
)

# ADD YOUR FEATURE HERE:
from features.phone_tracking.phone_tracker import (
    process_frame as phone_process,
    reset_phone_state,
)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline wrapper (thin orchestrator)
# ─────────────────────────────────────────────────────────────────────────────
class DriverSafetyPipeline:
    """
    Runs one shared MediaPipe pass per frame, then delegates to each
    feature module's process_frame().  No detection logic lives here.
    """

    def __init__(self, silent: bool = False) -> None:
        self.silent = silent
        self.enable_sleep       = True
        self.enable_distraction = True
        self.enable_yawning     = True
        self.enable_drink       = True
        self.enable_phone       = True
        self._last_t = time.time()
        self._frame_number = 0

        # Pipeline-local MediaPipe instances (NOT the shared module-level ones)
        # so that close() doesn't destroy them for other users.
        self._fm = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    # ── Per-frame entry point ─────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> dict:
        """Run all enabled detectors. Returns {sleep, distraction, yawning, drink}."""
        now = time.time()
        dt  = now - self._last_t
        self._last_t = now
        self._frame_number += 1

        h, w = frame.shape[:2]

        # Single MediaPipe pass (pipeline-local instances)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        face_res = self._fm.process(rgb)
        hand_res = self._hands.process(rgb)
        rgb.flags.writeable = True

        # Extract landmarks once
        face_lms = (face_res.multi_face_landmarks[0]
                    if face_res.multi_face_landmarks else None)

        # Delegate to each feature module's process_frame()
        results = {}

        results["sleep"] = (
            sleep_process(face_lms, w, h, silent=self.silent)
            if self.enable_sleep else None
        )
        results["distraction"] = (
            distraction_process(face_lms, w, h, silent=self.silent)
            if self.enable_distraction else None
        )
        results["yawning"] = (
            yawning_process(face_lms, hand_res, w, h, now, dt, silent=self.silent)
            if self.enable_yawning else None
        )
        results["drink"] = (
            drink_process(face_lms, hand_res, frame, w, h, now,
                          self._frame_number, silent=self.silent)
            if self.enable_drink else None
        )
        results["phone"] = (
            phone_process(frame, w, h, now, silent=self.silent)
            if self.enable_phone else None
        )

        return results

    # ── Annotation (display overlay) ──────────────────────────────────────────
    def annotate(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Draw status overlays for all enabled detectors."""
        out    = frame.copy()
        alerts = []
        y      = 30

        def put(text, color=(200, 200, 200), scale=0.65, thick=2):
            nonlocal y
            cv2.putText(out, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
            y += 26

        s = results.get("sleep")
        if s:
            col = (0, 0, 255) if s["drowsy"] else (0, 255, 0)
            put(f"[Sleep]  {'DROWSY!' if s['drowsy'] else 'Awake'}  EAR:{s['ear']:.3f}", col)
            if s["drowsy"]:
                alerts.append("DROWSY")

        d = results.get("distraction")
        if d:
            col = (0, 0, 255) if d["distracted"] else (0, 255, 0)
            put(f"[Distract] {'DISTRACTED' if d['distracted'] else 'Attentive'}"
                f"  Y:{d['yaw']:.0f} P:{d['pitch']:.0f}", col)
            if d["distracted"]:
                alerts.append("DISTRACTED")

        yn = results.get("yawning")
        if yn:
            col = (0, 140, 255) if yn["yawning"] else (0, 255, 0)
            put(f"[Yawn]  {'YAWNING' if yn['yawning'] else 'Normal'}"
                f"  MAR:{yn['mar']:.3f}  #{yn['yawn_count']}", col)
            if yn["yawning"]:
                alerts.append("YAWNING")

        dk = results.get("drink")
        if dk:
            state_col = {"IDLE": (0,255,0), "POSSIBLE_DRINKING": (0,255,255),
                         "DRINKING": (0,165,255), "ALERT": (0,0,255)}
            col = state_col.get(dk["state"], (200, 200, 200))
            put(f"[Drink]  {dk['state']}  Risk:{dk['risk']:.1f}/3.0", col)
            if dk["state"] == "ALERT":
                alerts.append("DRINK & DRIVE")

        ph = results.get("phone")
        if ph:
            state_col = {"IDLE": (0, 255, 0), "POSSIBLE_PHONE_USE": (0, 255, 255),
                         "CONFIRMED_PHONE_USE": (0, 165, 255), "ALERT": (0, 0, 255)}
            col = state_col.get(ph["state"], (200, 200, 200))
            put(f"[Phone]  {ph['state']}  Risk:{ph['risk']:.1f}/3.0", col)
            if ph["state"] == "ALERT":
                alerts.append("PHONE USE")

        if alerts:
            h_  = out.shape[0]
            ov  = out.copy()
            cv2.rectangle(ov, (0, h_-55), (out.shape[1], h_), (0, 0, 160), -1)
            cv2.addWeighted(ov, 0.6, out, 0.4, 0, out)
            cv2.putText(out, "  WARNING: " + "  |  ".join(alerts),
                        (10, h_-18), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

        cv2.putText(out, datetime.now().strftime("%H:%M:%S"),
                    (out.shape[1]-85, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1)
        return out

    # ── Controls ──────────────────────────────────────────────────────────────
    def recalibrate(self):
        """Resets distraction module's neutral head pose."""
        reset_calibration()
        print("[INFO] Head-pose recalibrated.")

    def reset(self):
        """Resets all feature-module counters."""
        reset_sleep_state()
        reset_distraction_state()
        reset_yawning_state()
        reset_drink_state()
        reset_phone_state()
        self._frame_number = 0
        print("[INFO] All counters reset.")

    def close(self):
        """Release pipeline-local MediaPipe resources."""
        self._fm.close()
        self._hands.close()


# ─── CLI main ─────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("  Driver Safety Monitor")
    print("  q/ESC=quit   r=reset   c=recalibrate head pose")
    print("=" * 55)

    pipeline = DriverSafetyPipeline()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera read failed."); break

            frame     = cv2.flip(frame, 1)
            results   = pipeline.process(frame)
            annotated = pipeline.annotate(frame, results)
            cv2.imshow("Driver Safety Monitor", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  break
            elif key == ord("r"):      pipeline.reset()
            elif key == ord("c"):      pipeline.recalibrate()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
