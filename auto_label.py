from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np

def generate_labels_from_model(model_path, images_dir, labels_dir, conf_threshold=0.05):
    """
    Generate YOLO format labels using existing trained model predictions
    """
    model = YOLO(model_path)
    
    # Create labels directory if it doesn't exist
    os.makedirs(labels_dir, exist_ok=True)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(images_dir).glob(ext))
    
    print(f"Found {len(image_files)} images to process")
    
    generated_count = 0
    detection_count = 0
    
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        
        # Load image to get dimensions
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Warning: Could not load {img_path}")
            continue
            
        height, width = image.shape[:2]
        
        # Run detection
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        
        # Create label file path
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"
        
        # Write labels
        with open(label_path, 'w') as f:
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    # Get bounding box coordinates (normalized)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Convert to YOLO format (center_x, center_y, width, height)
                    center_x = (x1 + x2) / 2 / width
                    center_y = (y1 + y2) / 2 / height
                    bbox_width = (x2 - x1) / width
                    bbox_height = (y2 - y1) / height
                    
                    # Class ID (0 for fish)
                    class_id = 0
                    confidence = box.conf[0].item()
                    
                    # Write YOLO format: class_id center_x center_y width height
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                    detection_count += 1
                    
                print(f"  Generated {len(results[0].boxes) if results[0].boxes is not None else 0} labels")
            else:
                print(f"  No detections found")
                
        generated_count += 1
    
    print(f"\nSummary:")
    print(f"- Processed {generated_count} images")
    print(f"- Generated {detection_count} total detections")
    print(f"- Labels saved to: {labels_dir}")

def create_empty_labels_for_missing():
    """
    Create empty label files for images that don't have labels
    """
    train_images = Path("data/images/train")
    val_images = Path("data/images/val")
    train_labels = Path("data/labels/train")
    val_labels = Path("data/labels/val")
    
    for images_dir, labels_dir in [(train_images, train_labels), (val_images, val_labels)]:
        if not images_dir.exists():
            continue
            
        os.makedirs(labels_dir, exist_ok=True)
        
        for img_file in images_dir.glob("*.jpg"):
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                # Create empty label file
                label_file.touch()
                print(f"Created empty label: {label_file}")

if __name__ == "__main__":
    # First, create empty labels for all missing files
    print("Creating empty label files for missing labels...")
    create_empty_labels_for_missing()
    
    # Check if trained model exists
    model_path = "runs/detect/train10/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Available models:")
        for run_dir in Path("runs/detect").glob("train*"):
            weights_path = run_dir / "weights" / "best.pt"
            if weights_path.exists():
                print(f"  - {weights_path}")
        exit(1)
    
    print(f"\nUsing model: {model_path}")
    
    # Generate labels for training images
    print("\nGenerating labels for training images...")
    generate_labels_from_model(
        model_path=model_path,
        images_dir="data/images/train",
        labels_dir="data/labels/train",
        conf_threshold=0.05  # Very low threshold to catch any possible detections
    )
    
    # Generate labels for validation images  
    print("\nGenerating labels for validation images...")
    generate_labels_from_model(
        model_path=model_path,
        images_dir="data/images/val", 
        labels_dir="data/labels/val",
        conf_threshold=0.05
    )
    
    print("\nDone! Please review the generated labels and manually correct any errors.")
    print("You can use the label_viewer.py script to review the labels visually.")
