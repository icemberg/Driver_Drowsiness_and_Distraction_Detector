"""
YOLOv8 Drink Object Detector

This module provides YOLOv8n-based drink object detection for the drink and drive system.
Supports training from custom dataset and real-time inference.

Features:
    - YOLOv8 Nano model (lightweight, real-time)
    - Support for 10 drink classes
    - Custom training from image dataset
    - Real-time inference at 30+ FPS
    - Automatic model download

Usage:
    # Train model
    detector = YOLOv8DrinkDetector()
    detector.train(data_yaml_path="dataset.yaml", epochs=50)
    
    # Inference
    results = detector.predict(frame)
    
    # Save/Load
    detector.save("drink_detector.pt")
    detector = YOLOv8DrinkDetector.load("drink_detector.pt")
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging

import config.config as config

logger = logging.getLogger(__name__)


class YOLOv8DrinkDetector:
    """
    YOLOv8n-based drink object detector.
    Lightweight model optimized for real-time detection on CPU/GPU.
    """
    
    def __init__(self, model_name: str = "yolov8n", model_path: str = None):
        """
        Initialize YOLOv8 drink detector.
        
        Args:
            model_name: Base model name (yolov8n, yolov8s, etc.)
            model_path: Path to pretrained model. If None, uses base model
        """
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.trained = False
        
        # Load model
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
            self.trained = True
        else:
            self._load_model(f"{model_name}.pt")
    
    def _load_model(self, model_path: str):
        """Load YOLOv8 model."""
        try:
            self.model = YOLO(model_path)
            logger.info(f"[OK] YOLOv8 model loaded: {model_path}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
            raise
    
    def train(self, data_yaml: str, epochs: int = 50, imgsz: int = 640, 
              device: str = "cpu", patience: int = 20, batch: int = 32):
        """
        Train YOLOv8 model on drink dataset.
        
        Args:
            data_yaml: Path to dataset.yaml (YOLO format)
            epochs: Number of training epochs
            imgsz: Image size for training
            device: GPU device ID (0 for first GPU, cpu for CPU)
            patience: Early stopping patience
            batch: Batch size (smaller=less memory, default 8 for CPU)
        
        Returns:
            Training results
        """
        logger.info("\n" + "="*60)
        logger.info("TRAINING YOLOv8 DRINK DETECTOR")
        logger.info("="*60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Dataset: {data_yaml}")
        logger.info(f"Epochs: {epochs}, Image Size: {imgsz}")
        logger.info(f"Device: {device}")
        
        try:
            # Get project root directory (go up 3 levels from this file)
            project_root = Path(__file__).resolve().parent.parent.parent
            runs_dir = project_root / "features" / "drink_and_drive" / "runs"
            
            # Train model
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                patience=patience,
                batch=batch,
                cache=True,  # Disable caching to reduce disk I/O
                save=True,
                verbose=True,
                project=str(runs_dir),
                name="drink_detector"
            )
            
            self.trained = True
            logger.info("[OK] Training complete!")
            logger.info(f"Best model: {self.model.trainer.best}")
            
            return results
        
        except Exception as e:
            logger.error(f"[ERROR] Training failed: {e}")
            raise
    
    def predict(self, frame: np.ndarray, confidence: float = None) -> list:
        """
        Detect drink objects in frame.
        
        Args:
            frame: Input image (BGR, OpenCV format)
            confidence: Confidence threshold (uses config default if None)
        
        Returns:
            List of detections with format:
            [
                {
                    'class': str,           # Class name (cup, bottle, etc.)
                    'class_id': int,        # Class ID
                    'confidence': float,    # Confidence score (0-1)
                    'bbox': [x1, y1, x2, y2],  # Bounding box (pixels)
                    'area': int             # Bounding box area
                },
                ...
            ]
        """
        if not self.model:
            logger.warning("[WARN] Model not loaded")
            return []
        
        # Use config threshold if not specified
        if confidence is None:
            confidence = config.DRINK_DETECTOR_CONFIDENCE_THRESHOLD
        
        try:
            # Run inference
            results = self.model.predict(
                source=frame,
                conf=confidence,
                verbose=False,
                device="cpu"
            )
            
            detections = []
            
            # Parse results
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # Extract box coordinates (pixel space)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Get class info
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]
                        confidence_score = float(box.conf[0].cpu().numpy())
                        
                        # Calculate area
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        # Filter by class (only drink classes)
                        if class_name.lower() in config.DRINK_CLASSES:
                            # Filter by minimum area
                            if area >= config.DRINK_DETECTOR_BOX_AREA_MIN_PIXELS:
                                detections.append({
                                    'class': class_name,
                                    'class_id': class_id,
                                    'confidence': confidence_score,
                                    'bbox': [x1, y1, x2, y2],
                                    'area': area
                                })
            
            return detections
        
        except Exception as e:
            logger.error(f"[ERROR] Prediction failed: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: list, 
                       color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.
        
        Args:
            frame: Input image
            detections: List of detections from predict()
            color: BGR color for boxes
            thickness: Box line thickness
        
        Returns:
            Frame with drawn detections
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with confidence
            label = f"{det['class']} ({det['confidence']:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw background rectangle
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1 - 5),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 5),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness
            )
        
        return frame
    
    def save(self, model_path: str):
        """
        Save trained model.
        
        Args:
            model_path: Path to save model
        """
        try:
            if self.model:
                # Create parent directories if they don't exist
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                self.model.save(model_path)
                logger.info(f"[OK] Model saved: {model_path}")
            else:
                logger.warning("[WARN] No model to save")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save model: {e}")
    
    @staticmethod
    def load(model_path: str) -> 'YOLOv8DrinkDetector':
        """
        Load trained model from file.
        
        Args:
            model_path: Path to trained model
        
        Returns:
            YOLOv8DrinkDetector instance
        """
        try:
            detector = YOLOv8DrinkDetector(model_path=model_path)
            logger.info(f"[OK] Model loaded: {model_path}")
            return detector
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
            return None


def create_dataset_yaml(dataset_dir: str, output_path: str = "dataset.yaml"):
    """
    Create YOLO format dataset structure from nested class-organized images.
    Reorganizes images into train/val/test splits and creates label files.
    
    Expected input structure:
        drink_dataset/
            ├── beer_bottle/
            │   └── images/
            │       └── image1.jpg, image2.jpg, ...
            ├── cup/
            │   └── images/
            │       └── image1.jpg, image2.jpg, ...
            ...
    
    Output structure:
        drink_dataset/
            ├── images/
            │   ├── train/
            │   ├── val/
            │   └── test/
            ├── labels/
            │   ├── train/
            │   ├── val/
            │   └── test/
    
    Args:
        dataset_dir: Root directory of drink_dataset
        output_path: Output path for dataset.yaml
    """
    import shutil
    import random
    from pathlib import Path
    
    dataset_dir = Path(dataset_dir)
    
    # Get all drink classes from subdirectories (exclude images/labels)
    classes = sorted([d.name for d in dataset_dir.iterdir() 
                     if d.is_dir() and d.name not in ['images', 'labels']])
    
    if not classes:
        logger.warning("[WARN] No class directories found")
        return None
    
    # Create YOLO directory structure
    yolo_images_train = dataset_dir / "images" / "train"
    yolo_images_val = dataset_dir / "images" / "val"
    yolo_images_test = dataset_dir / "images" / "test"
    yolo_labels_train = dataset_dir / "labels" / "train"
    yolo_labels_val = dataset_dir / "labels" / "val"
    yolo_labels_test = dataset_dir / "labels" / "test"
    
    for dir_path in [yolo_images_train, yolo_images_val, yolo_images_test,
                     yolo_labels_train, yolo_labels_val, yolo_labels_test]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Reorganize images into train/val/test splits
    logger.info("[INFO] Reorganizing dataset into YOLO format...")
    logger.info(f"[INFO] Found {len(classes)} classes: {', '.join(classes)}")
    
    total_images = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for class_idx, class_name in enumerate(classes):
        # Look for images in class_name/images/ subdirectory
        class_images_dir = dataset_dir / class_name / "images"
        
        if not class_images_dir.exists():
            logger.warning(f"[WARN] Images directory not found: {class_images_dir}")
            continue
        
        # Get ALL image files with any common extension (case-insensitive)
        images = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', 
                    '*.gif', '*.GIF', '*.bmp', '*.BMP', '*.webp', '*.WEBP']:
            images.extend(class_images_dir.glob(ext))
        
        # Remove duplicates (in case of overlapping patterns)
        images = list(set(images))
        
        if not images:
            logger.warning(f"[WARN] No images found in {class_name}/images/")
            continue
        
        logger.info(f"  Processing class '{class_name}': {len(images)} images")
        
        # Split: 70% train, 15% val, 15% test
        random.seed(42)
        random.shuffle(images)
        
        train_split = int(0.7 * len(images))
        val_split = int(0.85 * len(images))
        
        train_imgs = images[:train_split]
        val_imgs = images[train_split:val_split]
        test_imgs = images[val_split:]
        
        logger.info(f"    -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
        
        # Copy train images
        for img in train_imgs:
            try:
                # Include class name in filename to avoid collisions (e.g., beer_bottle_000001.jpg)
                unique_name = f"{class_name}_{img.name}"
                dest = yolo_images_train / unique_name
                shutil.copy2(img, dest)
                
                # Create YOLO label file (full image bbox: class_id 0 0.5 0.5 1.0 1.0)
                label_file = yolo_labels_train / (unique_name.split('.')[0] + ".txt")
                label_file.write_text(f"{class_idx} 0.5 0.5 1.0 1.0\n")
                split_counts['train'] += 1
            except Exception as e:
                logger.warning(f"[WARN] Failed to copy {img.name}: {e}")
        
        # Copy val images
        for img in val_imgs:
            try:
                # Include class name in filename to avoid collisions
                unique_name = f"{class_name}_{img.name}"
                dest = yolo_images_val / unique_name
                shutil.copy2(img, dest)
                
                label_file = yolo_labels_val / (unique_name.split('.')[0] + ".txt")
                label_file.write_text(f"{class_idx} 0.5 0.5 1.0 1.0\n")
                split_counts['val'] += 1
            except Exception as e:
                logger.warning(f"[WARN] Failed to copy {img.name}: {e}")
        
        # Copy test images
        for img in test_imgs:
            try:
                # Include class name in filename to avoid collisions
                unique_name = f"{class_name}_{img.name}"
                dest = yolo_images_test / unique_name
                shutil.copy2(img, dest)
                
                label_file = yolo_labels_test / (unique_name.split('.')[0] + ".txt")
                label_file.write_text(f"{class_idx} 0.5 0.5 1.0 1.0\n")
                split_counts['test'] += 1
            except Exception as e:
                logger.warning(f"[WARN] Failed to copy {img.name}: {e}")
        
        total_images += len(images)
    
    if total_images == 0:
        logger.error("[ERROR] No images were processed. Check your dataset structure.")
        return None
    
    logger.info(f"\n[OK] Dataset reorganized: {total_images} total images")
    logger.info(f"    Train: {split_counts['train']} ({split_counts['train']/total_images*100:.1f}%)")
    logger.info(f"    Val:   {split_counts['val']} ({split_counts['val']/total_images*100:.1f}%)")
    logger.info(f"    Test:  {split_counts['test']} ({split_counts['test']/total_images*100:.1f}%)")
    
    # Create class mapping (0-indexed)
    class_map = {i: cls for i, cls in enumerate(classes)}
    
    # Create YAML content
    yaml_content = f"""path: {dataset_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(classes)}
names: {class_map}
"""
    
    # Write YAML
    output_file = Path(output_path)
    output_file.write_text(yaml_content)
    
    logger.info(f"\n[OK] Dataset YAML created: {output_path}")
    logger.info(f"    Classes: {len(classes)}")
    logger.info(f"    Class names: {', '.join(classes)}")
    
    return output_path


if __name__ == "__main__":
    # Example usage
    detector = YOLOv8DrinkDetector()
    print("[OK] YOLOv8 Drink Detector ready")
