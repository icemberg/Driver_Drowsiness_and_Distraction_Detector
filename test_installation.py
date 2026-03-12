#!/usr/bin/env python3
"""
Test installation script for driver_drowsiness.
This script verifies that all required dependencies are installed correctly.
"""

import sys
import importlib.util
import subprocess

def check_module(module_name):
    """Check if a Python module is installed."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def check_installation():
    """Check if all required modules are installed."""
    required_modules = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "pygame": "pygame",
        "mediapipe": "mediapipe",
        "matplotlib": "matplotlib"
    }

    missing_modules = []

    for module, package in required_modules.items():
        if not check_module(module):
            missing_modules.append(package)

    if missing_modules:
        print(f"[ERROR] The following required packages are missing: {', '.join(missing_modules)}")
        print("[INFO] Install them using: pip install " + " ".join(missing_modules))
        return False

    return True

def test_camera():
    """Test if camera can be accessed."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            print("[SUCCESS] Camera test passed.")
            return True
        else:
            print("[ERROR] Camera test failed. Could not read frame.")
            return False
    except Exception as e:
        print(f"[ERROR] Camera test failed: {e}")
        return False

def test_mediapipe():
    """Test if MediaPipe Face Mesh works."""
    try:
        import cv2
        import mediapipe as mp
        import numpy as np

        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Try to process the image
        results = face_mesh.process(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))

        print("[SUCCESS] MediaPipe Face Mesh test passed.")
        return True
    except Exception as e:
        print(f"[ERROR] MediaPipe Face Mesh test failed: {e}")
        return False

def test_pygame():
    """Test if pygame audio works."""
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.quit()
        print("[SUCCESS] Pygame audio test passed.")
        return True
    except Exception as e:
        print(f"[ERROR] Pygame audio test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("===== driver_drowsiness Installation Test =====")

    # Check all required modules
    print("\nChecking required modules...")
    if not check_installation():
        return False
    print("[SUCCESS] All required modules are installed.")

    # Test camera
    print("\nTesting camera access...")
    camera_test = test_camera()

    # Test MediaPipe Face Mesh
    print("\nTesting MediaPipe Face Mesh...")
    mediapipe_test = test_mediapipe()

    # Test pygame audio
    print("\nTesting pygame audio...")
    pygame_test = test_pygame()

    # Overall result
    print("\n===== Test Results =====")
    all_passed = camera_test and mediapipe_test and pygame_test

    if all_passed:
        print("\n[SUCCESS] All tests passed! driver_drowsiness is ready to use.")
        print("\nRun the system with: python sleep_detector.py")
        return True
    else:
        print("\n[WARNING] Some tests failed. driver_drowsiness might not work correctly.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
