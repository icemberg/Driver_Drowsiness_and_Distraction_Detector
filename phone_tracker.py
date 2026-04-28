import cv2
import pyttsx3
import threading
import winsound
import time
import torch  # Added to check for GPU
from ultralytics import YOLO

def speak_warning():
    """
    By initializing pyttsx3 INSIDE the thread, we prevent the Windows 
    audio engine from silently crashing after the first use.
    """
    try:
        engine = pyttsx3.init()
        engine.say("Please do not use phone and focus on driving.")
        engine.runAndWait()
    except Exception as e:
        print(f"Audio Error: {e}")

def play_beep():
    """Plays a pure electronic beep. 
    1500 is the frequency (pitch), 150 is the duration in milliseconds."""
    winsound.Beep(1500, 150)

def main():
    # Model Setup (Phase 3)
    model = YOLO("phone_brain.pt") 
    cap = cv2.VideoCapture(0)

    # ==========================================
    # PHASE 4: THE FRAME COUNTER & HARDWARE SETUP
    # ==========================================
    distraction_counter = 0
    
    # Check if the computer has an active NVIDIA GPU
    if torch.cuda.is_available():
        DISTRACTION_THRESHOLD = 90  # ~3 seconds at 30 FPS for GPU
        print("Hardware Detected: GPU. Setting threshold to 90.")
    else:
        DISTRACTION_THRESHOLD = 30  # ~3 seconds at 10 FPS for CPU
        print("Hardware Detected: CPU. Setting threshold to 30.")
    
    # Beep Throttle Variable
    last_beep_time = 0
    BEEP_INTERVAL = 0.5  # Will only beep once every 0.5 seconds

    # ==========================================
    # PHASE 5: LOCKOUT SETUP
    # ==========================================
    lockout_active_until = 0  
    LOCKOUT_DURATION = 5.0    

    print("Starting Phone Tracker Demo. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        
        # Inference Step
        results = model.predict(frame, verbose=False)
        phone_detected_this_frame = False

        # Data Extraction
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf > 0.50:  
                    phone_detected_this_frame = True

        # ==========================================
        # PHASE 5: THE LOCKOUT LOGIC
        # ==========================================
        is_locked_out = current_time < lockout_active_until

        # ==========================================
        # PHASE 4: LOGIC & ALARMS
        # ==========================================
        if phone_detected_this_frame and not is_locked_out:
            distraction_counter += 1
            
            # The Uniform Beep: Only trigger if 0.5 seconds have passed since the last beep
            if current_time - last_beep_time > BEEP_INTERVAL:
                # Trigger the pure tone beep in the background so video doesn't lag
                threading.Thread(target=play_beep, daemon=True).start()
                last_beep_time = current_time
                
        # NOTICE: The "else" block that used to decrease the counter has been DELETED. 
        # Now, if the phone is lowered, the counter simply pauses and remembers its place.

        # ==========================================
        # PHASE 4 & 5: VOICE COMMAND & RESET
        # ==========================================
        if distraction_counter > DISTRACTION_THRESHOLD:
            # Fire the voice warning in the background
            threading.Thread(target=speak_warning, daemon=True).start()
            
            # Reset the counter
            distraction_counter = 0
            
            # Activate the Lockout Timer
            lockout_active_until = current_time + LOCKOUT_DURATION
            print("System Locked Out for 5 seconds. Put the phone down!")

        # --- VISUAL FEEDBACK ---
        annotated_frame = results[0].plot()

        if is_locked_out:
            cv2.putText(annotated_frame, "WARNING TRIGGERED - PLEASE FOCUS", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            cv2.putText(annotated_frame, f"Distraction Score: {distraction_counter}/{DISTRACTION_THRESHOLD}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Phone Tracker Demo", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()