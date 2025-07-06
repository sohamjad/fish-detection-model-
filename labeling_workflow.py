#!/usr/bin/env python3
"""
Complete labeling workflow for fish detection dataset
"""

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import cv2
        import ultralytics
        print("\u2713 All required packages are installed")
        return True
    except ImportError as e:
        print(f"\u2717 Missing package: {e}")
        print("Please install missing packages with: pip install opencv-python ultralytics")
        return False

def count_labels():
    """Count labeled vs unlabeled images"""
    train_images = len(list(Path("data/images/train").glob("*.jpg")))
    val_images = len(list(Path("data/images/val").glob("*.jpg")))
    train_labels_with_fish = 0
    val_labels_with_fish = 0
    for label_file in Path("data/labels/train").glob("*.txt"):
        if label_file.stat().st_size > 0:
            train_labels_with_fish += 1
    for label_file in Path("data/labels/val").glob("*.txt"):
        if label_file.stat().st_size > 0:
            val_labels_with_fish += 1
    total_images = train_images + val_images
    total_with_fish = train_labels_with_fish + val_labels_with_fish
    return total_images, total_with_fish

def main():
    print("\U0001F41F Fish Detection Labeling Workflow")
    print("=" * 40)
    
    if not check_requirements():
        return
    
    # Check current status
    total_images, total_with_fish = count_labels()
    
    print(f"\nWorkflow Options:")
    print("1. \U0001F440 View/Review current labels (recommended first step)")
    print("2. \u270F\uFE0F  Manual labeling (add/edit labels by hand)")
    print("3. \U0001F504 Re-run auto-labeling (if you improved the model)")
    print("4. \U0001F680 Train new model with current labels")
    print("5. \U0001F4CA Show detailed statistics")
    print("6. \u274C Exit")
    
    while True:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            print("\nLaunching label viewer...")
            print("This will show you all images with their current labels.")
            print("Use this to:")
            print("- Review auto-generated labels for accuracy")
            print("- Delete incorrect labels (press 'd')")
            print("- Navigate through your dataset")
            try:
                subprocess.run([sys.executable, "label_viewer.py"], check=True)
            except subprocess.CalledProcessError:
                print("Error running label viewer")
                
        elif choice == "2":
            print("\nLaunching manual labeler...")
            print("This lets you draw bounding boxes around fish.")
            print("Controls:")
            print("- Click and drag to draw boxes around fish")
            print("- Press 'n' for next image")
            print("- Press 's' to save")
            try:
                subprocess.run([sys.executable, "manual_labeler.py"], check=True)
            except subprocess.CalledProcessError:
                print("Error running manual labeler")
                
        elif choice == "3":
            print("\nRe-running auto-labeling...")
            confirm = input("This will overwrite existing auto-generated labels. Continue? (y/n): ")
            if confirm.lower() == 'y':
                try:
                    subprocess.run([sys.executable, "auto_label.py"], check=True)
                except subprocess.CalledProcessError:
                    print("Error running auto-labeling")
            
        elif choice == "4":
            print("\nStarting training with current labels...")
            total_images, total_with_fish = count_labels()
            
            if total_with_fish < 10:
                print(f"\u26A0\uFE0F  Warning: Only {total_with_fish} images have fish labels.")
                print("For good results, you should have at least 20-50 labeled images.")
                confirm = input("Continue anyway? (y/n): ")
                if confirm.lower() != 'y':
                    continue
            
            print("This will train a new model. This may take 10-30 minutes...")
            try:
                subprocess.run([sys.executable, "train.py"], check=True)
                print("\n\u2705 Training completed!")
                print("You can now test the new model with detect.py")
            except subprocess.CalledProcessError:
                print("Error during training")
                
        elif choice == "5":
            count_labels()
            
            # Show detailed breakdown
            print("\nDetailed Analysis:")
            
            # Check if any images might have fish but no labels
            unlabeled_count = 0
            for img_path in Path("data/images/train").glob("*.jpg"):
                label_path = Path("data/labels/train") / f"{img_path.stem}.txt"
                if not label_path.exists() or label_path.stat().st_size == 0:
                    unlabeled_count += 1
                    
            for img_path in Path("data/images/val").glob("*.jpg"):
                label_path = Path("data/labels/val") / f"{img_path.stem}.txt"
                if not label_path.exists() or label_path.stat().st_size == 0:
                    unlabeled_count += 1
            
            print(f"Images that might need manual review: {unlabeled_count}")
            
            if total_with_fish < 20:
                print("\n\ud83d\udd0d Recommendations:")
                print("1. Use the manual labeler to add more fish annotations")
                print("2. Review the current labels with the label viewer")
                print("3. Consider collecting more fish images")
                
        elif choice == "6":
            print("\nGoodbye! \U0001F41F")
            break
            
        else:
            print("Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()
