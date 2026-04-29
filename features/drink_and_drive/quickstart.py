"""
Quick Start Script - Complete Training Pipeline

This script automates the entire workflow:
  1. Download 1000+ images per class from Bing
  2. Clean corrupt images
  3. Train Random Forest model
  4. Test on webcam
  5. Use in main detection pipeline

Usage:
    python quickstart.py

or with options:
    python quickstart.py --skip-download    # Use existing dataset
    python quickstart.py --skip-train       # Use existing model
    python quickstart.py --webcam-only      # Manual webcam collection
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def banner(text):
    """Print formatted banner."""
    width = 70
    print("\n" + "="*width)
    print(f"  {text.center(width-4)}")
    print("="*width)

def run_command(cmd, description):
    """Run command and handle errors."""
    print(f"\n▶ {description}")
    print(f"  Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True)
        else:
            result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"❌ Failed: {description}")
            return False
        else:
            print(f"✅ Complete: {description}")
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    banner("DRINK DETECTOR - COMPLETE TRAINING PIPELINE")
    
    parser = argparse.ArgumentParser(description="Quick start training pipeline")
    parser.add_argument('--skip-download', action='store_true', help='Skip image download')
    parser.add_argument('--skip-clean', action='store_true', help='Skip image cleaning')
    parser.add_argument('--skip-train', action='store_true', help='Skip model training')
    parser.add_argument('--skip-test', action='store_true', help='Skip testing')
    parser.add_argument('--webcam-only', action='store_true', help='Use webcam collection instead')
    parser.add_argument('--dataset-dir', type=str, default='drink_dataset', help='Dataset directory')
    parser.add_argument('--model-path', type=str, default='drink_detector_model.pkl', help='Model path')
    
    args = parser.parse_args()
    
    # Step 1: Download Dataset
    if args.webcam_only:
        print("\n[INFO] Using webcam collection mode")
        banner("STEP 1: COLLECT DATASET FROM WEBCAM")
        run_command(
            ["python", "drink_detector_trainer.py", "--mode", "collect_dataset", 
             "--dataset_path", args.dataset_dir],
            "Collect 100+ images per class using webcam"
        )
    else:
        if not args.skip_download:
            banner("STEP 1: DOWNLOAD 1000+ IMAGES PER CLASS FROM WEB")
            print("\n[INFO] This will take 15-30 minutes. Internet connection required.")
            print("[INFO] You can monitor progress in image_downloader.log\n")
            
            run_command(
                ["python", "image_downloader.py", "--download", "--clean", "--stats",
                 "--dataset-dir", args.dataset_dir],
                "Download and clean dataset from Bing Image Search"
            )
        else:
            print("\n⏭️ Skipping download (using existing dataset)")
    
    # Verify dataset exists
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists() or len(list(dataset_path.glob("*/images/*.jpg"))) == 0:
        print("\n❌ Dataset not found! Run without --skip-download")
        sys.exit(1)
    
    # Step 2: Train Model
    if not args.skip_train:
        banner("STEP 2: TRAIN RANDOM FOREST MODEL")
        print(f"\n[INFO] Training on dataset: {args.dataset_dir}")
        print("[INFO] This will take 2-5 minutes\n")
        
        success = run_command(
            ["python", "drink_detector_trainer.py", "--mode", "train",
             "--dataset_path", args.dataset_dir, "--model_path", args.model_path],
            "Train Random Forest classifier"
        )
        
        if not success:
            print("\n❌ Training failed!")
            sys.exit(1)
    else:
        print("\n⏭️ Skipping training (using existing model)")
    
    # Verify model exists
    if not Path(args.model_path).exists():
        print(f"\n❌ Model not found at {args.model_path}!")
        print("Run without --skip-train to train a model")
        sys.exit(1)
    
    # Step 3: Test Model
    if not args.skip_test:
        banner("STEP 3: TEST MODEL ON WEBCAM")
        print("\n[INFO] Testing real-time detection")
        print("[INFO] Instructions:")
        print("  - Show different drink objects to the camera")
        print("  - Press ESC to exit\n")
        
        input("  Press ENTER to start...")
        
        run_command(
            ["python", "drink_detector_trainer.py", "--mode", "test_camera",
             "--model_path", args.model_path],
            "Test model on webcam feed"
        )
    else:
        print("\n⏭️ Skipping webcam test")
    
    # Step 4: Configuration Info
    banner("STEP 4: CONFIGURATION & USAGE")
    
    print("\n✅ Setup complete! Your model is ready to use.")
    print(f"\nModel location: {args.model_path}")
    print(f"Dataset location: {args.dataset_dir}")
    
    print("\n" + "="*70)
    print("  NEXT STEPS")
    print("="*70)
    
    print("\n1. Run main detection pipeline:")
    print("   python drink_and_drive_detection.py")
    
    print("\n2. View/modify configuration:")
    print("   - config.py - All tunable parameters")
    print("   - DRINK_DETECTION_GUIDE.md - Configuration guide")
    
    print("\n3. Advanced:")
    print("   - Retrain model: python drink_detector_trainer.py --mode train")
    print("   - Download more images: python image_downloader.py --download")
    print("   - Collect from webcam: python drink_detector_trainer.py --mode collect_dataset")
    
    print("\n4. View logs:")
    print("   - Event logs: drink_detection_logs/drink_and_drive_events.csv")
    print("   - Snapshots: drink_detection_logs/snapshots/")
    print("   - Download log: image_downloader.log")
    
    print("\n" + "="*70)
    print("  DOCUMENTATION")
    print("="*70)
    print("- DRINK_DETECTION_GUIDE.md")
    print("- IMAGE_DOWNLOADER_GUIDE.md")
    print("- README.md")
    
    banner("🎉 SETUP COMPLETE!")
    print("\nYou're ready to use the Drink and Drive Detection system!")
    print("\nRun: python drink_and_drive_detection.py")
    print("\nPress CTRL+C to exit this script, or...")
    
    # Offer to start detection
    try:
        response = input("\nStart detection pipeline now? (y/n): ").lower().strip()
        if response == 'y':
            print("\nStarting drink_and_drive_detection.py...")
            run_command(["python", "drink_and_drive_detection.py"], "Start detection")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
