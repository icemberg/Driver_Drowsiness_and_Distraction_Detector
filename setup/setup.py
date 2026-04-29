"""
Shared MediaPipe Setup — Single Source of Truth

All feature modules and the orchestrator import from here
instead of creating their own MediaPipe instances.

Resources created:
    face_mesh         — FaceMesh instance (1 face, refined landmarks)
    hands             — Hands instance (2 hands)
    mp_face_mesh      — mp.solutions.face_mesh module
    mp_hands          — mp.solutions.hands module
    mp_drawing        — mp.solutions.drawing_utils
    mp_drawing_styles — mp.solutions.drawing_styles
"""

import mediapipe as mp

# ── Module references (used for constants like FACEMESH_TESSELATION, HAND_CONNECTIONS) ──
mp_face_mesh    = mp.solutions.face_mesh
mp_hands        = mp.solutions.hands
mp_drawing      = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Shared instances (identical config across all 4 feature modules) ──────────
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
