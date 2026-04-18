"""
Drink Detector Trainer

This module provides tools to train and manage custom drink object detection models.
It supports dataset collection, model training, and evaluation with 100+ samples per class.

Usage:
    python drink_detector_trainer.py --mode collect_dataset
    python drink_detector_trainer.py --mode train --dataset_path ./drink_dataset
    python drink_detector_trainer.py --mode evaluate --model_path ./drink_detector_model.pkl
    python drink_detector_trainer.py --mode test_camera --model_path ./drink_detector_model.pkl
    
    # Download images from web (see image_downloader.py)
    python image_downloader.py --download --clean --stats
"""

import cv2
import os
import json
import numpy as np
from datetime import datetime
import argparse
from pathlib import Path
import pickle
import mediapipe as mp
from collections import defaultdict

import config.config as config
from utils.utils import initialize_logger, play_alarm


class DrinkDatasetCollector:
    """
    Interactive tool for collecting drink dataset samples.
    Collects 100+ samples per drink class using webcam.
    """
    
    def __init__(self, dataset_path="./drink_dataset"):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_objectron = mp.solutions.objectron
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.drink_classes = config.DRINK_CLASSES
        self.samples_per_class = int(input(f"Enter number of samples per class (default 100): ") or "100")
        
    def create_class_directories(self):
        """Create subdirectories for each drink class."""
        for drink_class in self.drink_classes:
            class_dir = self.dataset_path / drink_class / "images"
            class_dir.mkdir(parents=True, exist_ok=True)
            
            annot_dir = self.dataset_path / drink_class / "annotations"
            annot_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Created directory: {class_dir}")
    
    def collect_samples(self):
        """Collect samples for each drink class from webcam."""
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        print("\n" + "="*60)
        print("DRINK DATASET COLLECTION")
        print("="*60)
        print(f"Target: {self.samples_per_class} samples per class")
        print(f"Total samples needed: {len(self.drink_classes) * self.samples_per_class}")
        print("\nInstructions:")
        print("  - Hold the drink object in front of the camera")
        print("  - Press SPACE to capture a sample")
        print("  - Press C to move to next class")
        print("  - Press Q to exit collection")
        print("="*60 + "\n")
        
        class_idx = 0
        
        while class_idx < len(self.drink_classes):
            current_class = self.drink_classes[class_idx]
            sample_count = 0
            
            class_dir = self.dataset_path / current_class / "images"
            
            print(f"\nCollecting samples for: {current_class}")
            print(f"Progress: {class_idx + 1}/{len(self.drink_classes)}")
            print(f"Samples collected: {sample_count}/{self.samples_per_class}")
            
            while cap.isOpened() and sample_count < self.samples_per_class:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Display instructions
                cv2.putText(frame, f"Class: {current_class}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Samples: {sample_count}/{self.samples_per_class}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "SPACE=Capture | C=Next Class | Q=Quit", (10, h-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Drink Dataset Collection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # SPACE: capture sample
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    img_filename = f"{current_class}_{sample_count:03d}_{timestamp}.jpg"
                    img_path = class_dir / img_filename
                    
                    cv2.imwrite(str(img_path), frame)
                    
                    # Save annotation (just store frame info for now)
                    annot_dir = self.dataset_path / current_class / "annotations"
                    annot_filename = img_filename.replace('.jpg', '.json')
                    annot_path = annot_dir / annot_filename
                    
                    annotation = {
                        'class': current_class,
                        'timestamp': timestamp,
                        'frame_width': w,
                        'frame_height': h,
                        'index': sample_count
                    }
                    
                    with open(str(annot_path), 'w') as f:
                        json.dump(annotation, f)
                    
                    sample_count += 1
                    print(f"  Captured: {img_filename}")
                    play_alarm(volume=0.3)  # Soft beep feedback
                    
                elif key == ord('c') or key == ord('C'):  # C: next class
                    if sample_count < self.samples_per_class:
                        print(f"  ⚠ Warning: Only collected {sample_count}/{self.samples_per_class} samples")
                    class_idx += 1
                    break
                    
                elif key == ord('q') or key == ord('Q'):  # Q: quit
                    print("\nDataset collection cancelled.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            # Check if we completed this class
            if sample_count >= self.samples_per_class:
                print(f"✓ Completed: {current_class} ({sample_count} samples)")
                class_idx += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("✓ Dataset collection completed!")
        print(f"Total samples collected: {len(self.drink_classes) * self.samples_per_class}")
        print(f"Dataset saved to: {self.dataset_path}")
        print("="*60)


# ============================================================
# Note: Old DrinkDetectorModel class removed - now using YOLOv8DrinkDetector
# imported from yolov8_drink_detector.py
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Drink Detector Training Tool")
    parser.add_argument('--mode', type=str, default='collect_dataset',
                       choices=['collect_dataset', 'download_web', 'train', 'evaluate', 'test_camera'],
                       help='Operation mode')
    parser.add_argument('--dataset_path', type=str, default='./drink_dataset',
                       help='Path to training dataset')
    parser.add_argument('--model_path', type=str, default='./drink_detector_model.pkl',
                       help='Path to save/load model')
    parser.add_argument('--download-threads', type=int, default=4,
                       help='Number of download threads for web mode')
    parser.add_argument('--images-per-keyword', type=int, default=120,
                       help='Images per keyword to download')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("DRINK DETECTOR TRAINER")
    print("="*60)
    
    if args.mode == 'collect_dataset':
        print("\nMode: Collect Dataset")
        collector = DrinkDatasetCollector(dataset_path=args.dataset_path)
        collector.create_class_directories()
        collector.collect_samples()
    
    elif args.mode == 'download_web':
        print("\nMode: Download Images from Web")
        try:
            from features.drink_and_drive.image_downloader import DrinkImageDownloader
            print("✓ Image downloader module loaded")
            
            downloader = DrinkImageDownloader(base_dir=args.dataset_path)
            
            print("\nStarting download pipeline...")
            print("1. Downloading images (this may take 15-30 minutes)...")
            downloader.download_all()
            
            print("\n2. Cleaning corrupt images...")
            downloader.clean_corrupt_images()
            
            print("\n3. Generating statistics...")
            stats = downloader.get_dataset_statistics()
            
            print("\n" + "="*60)
            print("✓ Download pipeline complete!")
            print("="*60)
            print(f"Dataset ready at: {args.dataset_path}")
            print("Next step: Train the model with:")
            print(f"  python drink_detector_trainer.py --mode train --dataset_path {args.dataset_path}")
            
        except ImportError:
            print("❌ image_downloader module not found!")
            print("Make sure icrawler is installed:")
            print("  pip install icrawler")
    
    elif args.mode == 'train':
        print("\nMode: Train YOLOv8 Model")
        try:
            from features.drink_and_drive.yolov8_drink_detector import YOLOv8DrinkDetector, create_dataset_yaml
            
            # Create YOLO formatted dataset.yaml
            print("Preparing YOLO dataset structure...")
            dataset_yaml = create_dataset_yaml(args.dataset_path, output_path="dataset.yaml")
            
            if dataset_yaml is None:
                print("❌ Failed to create dataset.yaml")
                return
            
            print(f"✓ Dataset configuration created: {dataset_yaml}")
            
            # Initialize and train YOLOv8 detector
            print("\nInitializing YOLOv8n detector...")
            detector = YOLOv8DrinkDetector()
            
            print(f"\nTraining YOLOv8n on drink dataset...")
            print(f"Dataset: {dataset_yaml}")
            print(f"Model save path: {args.model_path}")
            
            detector.train(data_yaml=dataset_yaml, epochs=50, imgsz=640)
            
            # Save trained model
            detector.save(args.model_path)
            print(f"\n✓ Model trained and saved to: {args.model_path}")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Make sure ultralytics is installed: pip install ultralytics")
        except Exception as e:
            print(f"❌ Training failed: {e}")
    
    elif args.mode == 'evaluate':
        print("\nMode: Evaluate YOLOv8 Model")
        try:
            from features.drink_and_drive.yolov8_drink_detector import YOLOv8DrinkDetector
            
            model = YOLOv8DrinkDetector.load(args.model_path)
            if model is not None:
                print("✓ YOLOv8 model loaded successfully!")
                print(f"Model path: {args.model_path}")
                print("Model is ready for inference")
            else:
                print("❌ Failed to load model")
        except ImportError as e:
            print(f"❌ Import error: {e}")
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
    
    elif args.mode == 'test_camera':
        print("\nMode: Test Camera (Real-time Detection with YOLOv8)")
        try:
            from features.drink_and_drive.yolov8_drink_detector import YOLOv8DrinkDetector
            
            model = YOLOv8DrinkDetector.load(args.model_path)
            if model is None:
                print("❌ Cannot load model. Train model first with --mode train")
                return
            
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            
            print("Testing YOLOv8 drink detector on camera feed...")
            print("Press ESC to exit\n")
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                frame_count += 1
                
                # Run YOLOv8 detection
                detections = model.predict(frame)
                
                if detections:
                    # Draw detections
                    frame = model.draw_detections(frame, detections, color=(0, 255, 0), thickness=2)
                    
                    # Print detection info
                    if frame_count % 10 == 0:  # Print every 10 frames
                        for det in detections:
                            print(f"[{frame_count}] {det['class']}: {det['confidence']:.2f} (area: {det['area']})")
                
                # Show frame
                cv2.putText(frame, f"FPS-ready (detections: {len(detections)})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("YOLOv8 Drink Detector Test", frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print("\n✓ Camera test completed")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
        except Exception as e:
            print(f"❌ Camera test failed: {e}")


if __name__ == '__main__':
    main()
