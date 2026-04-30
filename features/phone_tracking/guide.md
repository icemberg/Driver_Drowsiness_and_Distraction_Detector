# Phone Using Detection - Model Training Guide

This document contains the dataset details, setup instructions, and the code required to train our YOLOv8 model using our downloaded dataset.

⚠️ **Note for CPU Users:** Training computer vision models on a standard processor (CPU) instead of a graphics card (GPU) is possible, but it will be **significantly slower**. We have optimized the code below specifically for CPU training to prevent RAM crashes and ensure stability.

---

## 📊 Dataset Information

- **Project Name:** Phone using detection - v6
- **Export Date:** 2024-09-04 8:15am
- **Image Count:** 2026 images
- **Annotation Format:** YOLOv8
- **Preprocessing:** Resize to 640x640 (Stretch)
- **Augmentations:** None applied
- **Source:** [Roboflow Universe - yurals-pro/phone-using-detection](https://universe.roboflow.com/yurals-pro/phone-using-detection-h0hzo)
- **License:** CC BY 4.0

---

## ⚙️ Setup & Installation

Before running the code, ensure you have the required computer vision library installed. Run this in your terminal or command prompt:

```bash
pip install ultralytics
```

---

## 📥 Step 1: Prepare the Dataset

We downloaded this dataset directly from the Roboflow website. To set it up for training:

1. Ensure you have downloaded the dataset in **YOLOv8 format**.
2. Extract the downloaded `.zip` file into a folder.
3. Move that extracted folder into the same directory as your Python script.
4. Inside that folder, you will see a file named `data.yaml`. Keep track of where this file is, as YOLO needs it to find the images!

---

## 🧠 Step 2: Training the YOLOv8 Model (CPU Optimized)

Create a Python file (e.g., `train.py`) and paste the code below.

Since you are running this on a CPU, we have made the following crucial adjustments to the code:

1. **Nano Model:** We are strictly using `yolov8n.pt` (Nano). It is the smallest, fastest model and the only realistic option for CPU training.
2. **Device Enforcement:** `device='cpu'` forces the system to use the processor, preventing errors if you have mismatched GPU drivers.
3. **Lower Batch Size:** `batch=8` prevents the CPU's memory (RAM) from overloading during training.
4. **Lower Workers:** `workers=2` limits how many CPU threads are used to load images, preventing your computer from completely freezing.

```python
from ultralytics import YOLO

# Load the fastest/smallest pre-trained YOLOv8 model for CPU
model = YOLO('yolov8n.pt')

print("Starting CPU training. Note: This may take several hours to complete.")

# Train the model with CPU-safe parameters
results = model.train(
    # IMPORTANT: Change this path to point to the data.yaml file inside your extracted dataset folder!
    data="path/to/your/extracted_folder/data.yaml",

    epochs=20,       # Start with 20 to test how long it takes. Increase to 50 later if needed.
    imgsz=640,       # Matches our dataset preprocessing
    device='cpu',    # Explicitly tells YOLO to use the CPU
    batch=8,         # Lowers memory usage
    workers=2,       # Prevents CPU thread bottlenecking
    plots=True       # Generates training graphs and confusion matrices
)

print("Training complete! Weights saved to runs/detect/train/weights/best.pt")
```

> **💡 Pro-Tip:** If training on your local CPU takes too many days to complete, consider uploading your dataset to Google Drive and running this exact same Python code in a free **Google Colab** notebook. Google provides free cloud GPUs that can train this model in minutes instead of hours!

---

## 🔍 Step 3: Running Inference (Testing)

Once training is finally complete, you can test your new model on a fresh image to see if it detects phones properly.

```python
from ultralytics import YOLO

# Load the 'best' weights from your recent training run
best_model = YOLO('runs/detect/train/weights/best.pt')

# Run inference on a test image (replace with a real path to a photo)
results = best_model('path/to/your/test_image.jpg')

# Display the results in a popup window
results[0].show()

# Alternatively, save the resulting image with bounding boxes drawn
results[0].save(filename='result_image.jpg')
```

```

```
