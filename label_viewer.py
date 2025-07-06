import cv2
import os
from pathlib import Path
import numpy as np

class LabelViewer:
    def __init__(self, images_dir, labels_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.current_idx = 0
        
        # Get list of images
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        if not self.image_files:
            print(f"No images found in {images_dir}")
            return
            
        self.image_files.sort()
        print(f"Found {len(self.image_files)} images")
        print("\nControls:")
        print("- 'n' or right arrow: Next image")
        print("- 'p' or left arrow: Previous image")
        print("- 'd': Delete current label file (mark as no fish)")
        print("- 'r': Refresh current image")
        print("- 'q' or ESC: Quit")
        print("- 's': Show statistics")
        
    def load_labels(self, img_path):
        """Load YOLO format labels for an image"""
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        labels = []
        
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                center_x = float(parts[1])
                                center_y = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                labels.append([class_id, center_x, center_y, width, height])
            except Exception as e:
                print(f"Error reading label file {label_path}: {e}")
        
        return labels
    
    def draw_boxes(self, image, labels, img_width, img_height):
        """Draw bounding boxes on image"""
        if not labels:
            return image
            
        image_copy = image.copy()
        
        for label in labels:
            class_id, center_x, center_y, width, height = label
            
            # Convert from YOLO format to pixel coordinates
            x1 = int((center_x - width/2) * img_width)
            y1 = int((center_y - height/2) * img_height)
            x2 = int((center_x + width/2) * img_width)
            y2 = int((center_y + height/2) * img_height)
            
            # Draw bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"fish"
            cv2.putText(image_copy, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return image_copy
    
    def delete_label_file(self, img_path):
        """Delete the label file for current image"""
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            os.remove(label_path)
            print(f"Deleted label file: {label_path}")
        else:
            print(f"No label file to delete for: {img_path.name}")
    
    def show_statistics(self):
        """Show labeling statistics"""
        total_images = len(self.image_files)
        labeled_count = 0
        total_detections = 0
        
        for img_path in self.image_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists() and label_path.stat().st_size > 0:
                labeled_count += 1
                labels = self.load_labels(img_path)
                total_detections += len(labels)
        
        print(f"\nStatistics:")
        print(f"- Total images: {total_images}")
        print(f"- Images with labels: {labeled_count}")
        print(f"- Images without labels: {total_images - labeled_count}")
        print(f"- Total detections: {total_detections}")
        print(f"- Average detections per labeled image: {total_detections/max(labeled_count,1):.2f}")
    
    def run(self):
        """Main viewer loop"""
        if not self.image_files:
            return
            
        while self.current_idx < len(self.image_files):
            img_path = self.image_files[self.current_idx]
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Could not load image: {img_path}")
                self.current_idx += 1
                continue
                
            img_height, img_width = image.shape[:2]
            
            # Load labels
            labels = self.load_labels(img_path)
            
            # Draw boxes
            display_image = self.draw_boxes(image, labels, img_width, img_height)
            
            # Resize image if too large
            max_height = 800
            if display_image.shape[0] > max_height:
                scale = max_height / display_image.shape[0]
                new_width = int(display_image.shape[1] * scale)
                display_image = cv2.resize(display_image, (new_width, max_height))
            
            # Add info text
            info_text = f"Image {self.current_idx + 1}/{len(self.image_files)}: {img_path.name}"
            label_text = f"Labels: {len(labels)} detections"
            
            cv2.putText(display_image, info_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, label_text, (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show image
            cv2.imshow("Label Viewer", display_image)
            
            # Handle key presses
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('n') or key == 83:  # 'n' or right arrow
                self.current_idx = min(self.current_idx + 1, len(self.image_files) - 1)
            elif key == ord('p') or key == 81:  # 'p' or left arrow
                self.current_idx = max(self.current_idx - 1, 0)
            elif key == ord('d'):  # Delete label
                self.delete_label_file(img_path)
            elif key == ord('r'):  # Refresh
                pass  # Just redraw
            elif key == ord('s'):  # Statistics
                self.show_statistics()
        
        cv2.destroyAllWindows()

def main():
    print("Label Viewer")
    print("Choose dataset to review:")
    print("1. Training set")
    print("2. Validation set")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        images_dir = "data/images/train"
        labels_dir = "data/labels/train"
    elif choice == "2":
        images_dir = "data/images/val"
        labels_dir = "data/labels/val"
    else:
        print("Invalid choice")
        return
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
        
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return
    
    viewer = LabelViewer(images_dir, labels_dir)
    viewer.run()

if __name__ == "__main__":
    main()
