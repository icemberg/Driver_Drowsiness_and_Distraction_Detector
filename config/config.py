"""
driver_drowsiness Configuration Module

This module contains all configurable parameters for the driver_drowsiness drowsiness detection system.
Adjust these settings to fine-tune the system for different users, environments, and hardware.

Note: After changing parameters, restart the application for changes to take effect.
"""

###################
# CORE PARAMETERS #
###################

# Eye Aspect Ratio (EAR) threshold for determining closed eyes
# Range: 0.15-0.25 (typical values)
# - Lower values (e.g., 0.18) = More sensitive detection (detects slightly closed eyes)
# - Higher values (e.g., 0.25) = Less sensitive (only detects fully closed eyes)
# Adjust based on individual facial features and lighting conditions
EYE_AR_THRESHOLD = 0.22

# Number of consecutive frames the eye must be below threshold to trigger alarm
# Range: 15-40 frames (at 30fps, this equals 0.5-1.3 seconds)
# - Lower values = Faster alerts but more false positives
# - Higher values = Fewer false positives but delayed alerts
# Adjust based on use case (lower for safety-critical applications)
EYE_AR_CONSEC_FRAMES = 25

#####################
# HARDWARE SETTINGS #
#####################

# Camera settings
CAMERA_INDEX = 0      # Camera device index (0=first camera, 1=second camera, etc.)
FRAME_WIDTH = 640     # Frame width in pixels (higher = more detail but slower processing)
FRAME_HEIGHT = 480    # Frame height in pixels (higher = more detail but slower processing)

####################
# MEDIAPIPE PARAMS #
####################

# MediaPipe Face Mesh detection parameters
# Note: These are set in the sleep_detector.py file:
# - max_num_faces=1: Focus on single user (driver)
# - refine_landmarks=True: Better accuracy for eye landmarks
# - min_detection_confidence=0.5: Balance between detection rate and false positives
# - min_tracking_confidence=0.5: Balance between tracking stability and adaptability

####################
# ALERT SETTINGS   #
####################

# Audio alert settings
ALARM_SOUND = "alarm.wav"    # Path to sound file (default=built-in beeping)
ALARM_VOLUME = 0.9           # Volume level (0.0=silent, 1.0=maximum)

#########################
# AUDIO ALERT SETTINGS #
#########################
# Internal fallback tone generator used by utils.play_alarm.
ALARM_SAMPLE_RATE = 44100
ALARM_AUDIO_FORMAT = -16
ALARM_CHANNELS = 2
ALARM_BUFFER_SIZE = 1024
ALARM_DURATION = 1.0
ALARM_FREQUENCY = 880
ALARM_AMPLITUDE = 32767

####################
# UI SETTINGS      #
####################

# Visual elements configuration
SHOW_EAR_VALUE = True        # Display the EAR value on screen
SHOW_LANDMARKS = True        # Show facial landmarks visualization in debug mode

# Colors (in BGR format - OpenCV uses BGR instead of RGB)
TEXT_COLOR = (255, 0, 0)     # Alert text color (Blue=0, Green=0, Red=255) = Red
FRAME_COLOR = (0, 255, 0)    # Face frame color (Blue=0, Green=255, Red=0) = Green

##############################
# DISTRACTION DETECTION      #
##############################

# Gaze ratio range — iris position relative to eye corners
# 0.0 = fully toward outer corner, 1.0 = fully toward inner corner, 0.5 = centered
# Values outside this band trigger "gazing away"
GAZE_CENTER_MIN = 0.35
GAZE_CENTER_MAX = 0.60

# Head-pose angle limits (degrees from calibrated neutral)
# Exceeding any limit means the head is "not forward"
HEAD_YAW_LIMIT = 20       # left/right turn
HEAD_PITCH_LIMIT = 15     # up/down tilt
HEAD_ROLL_LIMIT = 12      # sideways head tilt

# Face off-centre band (nose_x normalised 0–1)
# Outside this band ⇒ face is off-centre in the frame
OFF_CENTER_MIN = 0.30
OFF_CENTER_MAX = 0.70

# Consecutive distracted frames required before showing warning
# Prevents single-frame flicker. At 30 fps, 10 frames ≈ 0.33 s
DISTRACTION_CONSEC_FRAMES = 10

####################
# DEBUG SETTINGS   #
####################

# Debugging options
SHOW_EYE_PROCESSING = True   # Enable detailed eye processing visualization

####################
# LOGGING SETTINGS #
####################

# Data logging for analysis
ENABLE_LOGGING = False                  # Enable/disable event logging
LOG_FILE = "sleep_detection_log.txt"    # Path to log file

########################
# UTILITY CONSTANTS    #
########################
# Internal helper constants used by utility functions across modules.
EAR_RATIO_SCALE = 0.27
EAR_SIZE_FACTOR_MAX = 0.03
EAR_AREA_DIVISOR = 20000
EAR_MIN = 0.15
EAR_MAX = 0.35
GAZE_HORIZONTAL_MIN_RATIO = 0.2
GAZE_HORIZONTAL_MAX_RATIO = 0.8
GAZE_VERTICAL_MIN_RATIO = 0.15
GAZE_VERTICAL_MAX_RATIO = 0.6
EYE_CLOSURE_HIST_BINS = 256
EYE_CLOSURE_WEIGHT_START = 1.0
EYE_CLOSURE_WEIGHT_END = 0.1
UPPER_LIP_LANDMARK = 13
LOWER_LIP_LANDMARK = 14
LEFT_MOUTH_LANDMARK = 78
RIGHT_MOUTH_LANDMARK = 308

#####################
# YAWNING DETECTION #
#####################

MAR_THRESHOLD = 0.5
EAR_THRESHOLD = 0.23
MIN_EYE_CLOSED_OVERLAP = 0.3
HAND_MOUTH_DISTANCE_PX = 120  # Increased from 90 for better hand detection tolerance
SMOOTH_WINDOW = 7
YAWN_COOLDOWN = 2.0
YAWN_ALERT_DURATION = 1.5
YAWN_FRAMES = 25   # ~1 sec at 25 FPS
MIN_YAWN_DURATION = 1.2   # seconds to confirm a yawn

##########################
# DRINK AND DRIVE DETECTION#
##########################

# Hand-to-mouth detection thresholds (TUNED FOR ACTUAL DRINKING DETECTION)
DRINK_HAND_MOUTH_DISTANCE_THRESHOLD = 0.70  # Normalized distance (0-1): RELAXED to 0.5 for realistic detection
DRINK_SUSTAINED_FRAMES = 12  # Consecutive frames (at 30 fps, ~0.4 seconds)

# State machine thresholds
DRINK_POSSIBLE_FRAMES = 3  # Frames to transition to POSSIBLE_DRINKING
DRINK_CONFIRMED_FRAMES = 8  # Frames to transition to DRINKING
DRINK_ALERT_FRAMES = 15  # Frames to trigger ALERT state

# Head pose distraction indicators (degrees) - REDUCED for pitch-up detection
HEAD_YAW_DISTRACTION_THRESHOLD = 15  # Head turned left/right (yaw) - kept loose
HEAD_PITCH_DISTRACTION_THRESHOLD = 6   # Head tilted up/down (pitch) - REDUCED to catch upward tilt

# Drink object detector settings
DRINK_DETECTION_CONFIDENCE_THRESHOLD = 0.5
DRINK_OBJECT_CLASSES = ['cup', 'bottle', 'mug', 'can', 'glass']

# Drink-object position relative to face (normalized)
DRINK_BBOX_FACE_PROXIMITY_THRESHOLD = 0.3  # How close bbox should be to lower face region

# Cooldown between alerts (seconds)
DRINK_ALERT_COOLDOWN = 3.0
DRINK_ALERT_DURATION = 2.0

##############################################
# DRINK OBJECT DETECTION (YOLOv8)           #
##############################################

# YOLOv8 Object Detector parameters
DRINK_OBJECT_DETECTOR_MODEL = "yolov8n"  # YOLOv8 Nano (lightweight, real-time)
DRINK_DETECTOR_CONFIDENCE_THRESHOLD = 0.6  # 60% confidence for detection
DRINK_DETECTOR_BOX_AREA_MIN_PIXELS = 500  # Minimum bounding box area
DRINK_DETECTOR_MODEL_PATH = "./models/yolov8n_drink.pt"  # Path to trained model

# Custom drink classes to detect (must match training labels)
DRINK_CLASSES = ['cup', 'bottle', 'mug', 'can', 'glass', 'drinking_bottle', 'soda_can', 'beer_bottle', 'water_bottle', 'tea_cup']
DRINK_OBJECT_DETECTION_ENABLED = True

# Mouth region definition (relative to face bounding box)
MOUTH_REGION_TOP_RATIO = 0.65  # 65% down from top of face
MOUTH_REGION_BOTTOM_RATIO = 0.95  # 95% down from top of face
MOUTH_REGION_LEFT_RATIO = 0.25  # 25% from left edge of face
MOUTH_REGION_RIGHT_RATIO = 0.75  # 75% from right edge of face

############################################
# TUNED STATE TRANSITION THRESHOLDS       #
############################################
# These are optimized for real driving scenarios

# State transition thresholds (based on risk score 0-3) - TUNED FOR BEHAVIOR DETECTION
DRINK_RISK_THRESHOLD_IDLE_TO_POSSIBLE = 0.4  # Low threshold: head tilt alone triggers investigation
DRINK_RISK_THRESHOLD_POSSIBLE_TO_CONFIRMED = 0.9  # Hand + head signals together (primary drinking indicator)
DRINK_RISK_THRESHOLD_CONFIRMED_TO_ALERT = 1.3  # Hand + head + object detected (high confidence)

# Frame consistency requirements (at 30fps)
DRINK_FRAMES_IDLE_TO_POSSIBLE = 5  # ~0.17 seconds
DRINK_FRAMES_POSSIBLE_TO_CONFIRMED = 10  # ~0.33 seconds
DRINK_FRAMES_CONFIRMED_TO_ALERT = 12  # ~0.40 seconds

# Fallback thresholds (drop to idle if risk falls below)
DRINK_RISK_FALLBACK_THRESHOLD = 1.0

# Signal weights for fused risk score
SIGNAL_WEIGHT_HAND_PROXIMITY = 1.0
SIGNAL_WEIGHT_OBJECT_DETECTION = 1.0
SIGNAL_WEIGHT_HEAD_DISTRACTION = 0.5

############################################
# EVENT LOGGING AND SNAPSHOTS             #
############################################

# Logging directory and settings
DRINK_LOG_DIRECTORY = "features/drink_and_drive/drink_detection_logs"
DRINK_LOG_FILENAME = "drink_and_drive_events.csv"
DRINK_SNAPSHOTS_DIRECTORY = "features/drink_and_drive/drink_detection_logs/snapshots"
DRINK_EVENT_SNAPSHOT_COUNT = 6  # Number of frames to capture around alert

# Enable/disable features
ENABLE_DRINK_DETECTION = True
ENABLE_DRINK_SNAPSHOTS = True
ENABLE_DRINK_CSV_LOGGING = True
############################################
# PHONE DETECTION PARAMETERS               #
############################################

# Enable/disable the phone detection feature
ENABLE_PHONE_DETECTION = True

# Frame consistency requirements (at 30fps)
PHONE_FRAMES_IDLE_TO_POSSIBLE = 5    # ~0.17 seconds to trigger warning
PHONE_FRAMES_POSSIBLE_TO_ALERT = 10  # ~0.33 seconds to trigger alarm

# Lockout and Cooldown
PHONE_LOCKOUT_DURATION = 5.0         # Seconds to stay in ALERT mode

