import cv2
import os
from pathlib import Path

class ManualLabeler:
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
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_boxes = []
        self.temp_box = None
        
        os.makedirs(self.labels_dir, exist_ok=True)
        
        print(f"Found {len(self.image_files)} images")
        print("\nControls:")
        print("- Mouse: Click and drag to draw bounding box")
        print("- 'n': Next image")
        print("- 'p': Previous image")
        print("- 'u': Undo last box")
        print("- 'c': Clear all boxes")
        print("- 's': Save current labels")
        print("- 'q': Quit")
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_box = (self.start_point, (x, y))
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                
                # Add box if it's large enough
                if abs(end_point[0] - self.start_point[0]) > 10 and abs(end_point[1] - self.start_point[1]) > 10:
                    self.current_boxes.append((self.start_point, end_point))
                    print(f"Added box: {self.start_point} to {end_point}")
                
                self.temp_box = None
    
    def load_existing_labels(self, img_path):
        """Load existing labels for current image"""
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        self.current_boxes = []
        
        if label_path.exists():
            try:
                # Load image to get dimensions
                image = cv2.imread(str(img_path))
                img_height, img_width = image.shape[:2]
                
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                center_x = float(parts[1])
                                center_y = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # Convert from YOLO format to pixel coordinates
                                x1 = int((center_x - width/2) * img_width)
                                y1 = int((center_y - height/2) * img_height)
                                x2 = int((center_x + width/2) * img_width)
                                y2 = int((center_y + height/2) * img_height)
                                
                                self.current_boxes.append(((x1, y1), (x2, y2)))
            except Exception as e:
                print(f"Error loading labels: {e}")
    
    def save_labels(self, img_path):
        """Save current boxes to label file"""
        if not self.current_boxes:
            # Create empty file
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            label_path.touch()
            print("Saved empty label file")
            return
            
        # Load image to get dimensions
        image = cv2.imread(str(img_path))
        img_height, img_width = image.shape[:2]
        
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        
        with open(label_path, 'w') as f:
            for start_point, end_point in self.current_boxes:
                # Convert to YOLO format
                x1, y1 = start_point
                x2, y2 = end_point
                
                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                center_x = (x1 + x2) / 2 / img_width
                center_y = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Class ID 0 for fish
                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Saved {len(self.current_boxes)} labels to {label_path}")
    
    def draw_boxes(self, image):
        """Draw all bounding boxes on image"""
        image_copy = image.copy()
        
        # Draw saved boxes in green
        for start_point, end_point in self.current_boxes:
            cv2.rectangle(image_copy, start_point, end_point, (0, 255, 0), 2)
        
        # Draw temporary box in red
        if self.temp_box:
            cv2.rectangle(image_copy, self.temp_box[0], self.temp_box[1], (0, 0, 255), 2)
        
        return image_copy
    
    def run(self):
        """Main labeling loop"""
        if not self.image_files:
            return
            
        cv2.namedWindow("Manual Labeler")
        cv2.setMouseCallback("Manual Labeler", self.mouse_callback)
        
        while self.current_idx < len(self.image_files):
            img_path = self.image_files[self.current_idx]
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Could not load image: {img_path}")
                self.current_idx += 1
                continue
            
            # Load existing labels
            self.load_existing_labels(img_path)
            
            while True:
                # Draw image with boxes
                display_image = self.draw_boxes(image)
                
                # Add info text
                info_text = f"Image {self.current_idx + 1}/{len(self.image_files)}: {img_path.name}"
                box_text = f"Boxes: {len(self.current_boxes)}"
                
                cv2.putText(display_image, info_text, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_image, box_text, (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Manual Labeler", display_image)
                
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q'):
                    return
                elif key == ord('n'):
                    self.save_labels(img_path)
                    self.current_idx = min(self.current_idx + 1, len(self.image_files) - 1)
                    break
                elif key == ord('p'):
                    self.save_labels(img_path)
                    self.current_idx = max(self.current_idx - 1, 0)
                    break
                elif key == ord('u'):  # Undo
                    if self.current_boxes:
                        self.current_boxes.pop()
                        print("Removed last box")
                elif key == ord('c'):  # Clear
                    self.current_boxes = []
                    print("Cleared all boxes")
                elif key == ord('s'):  # Save
                    self.save_labels(img_path)
        
        cv2.destroyAllWindows()

def main():
    print("Manual Labeler")
    print("Choose dataset to label:")
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
    
    labeler = ManualLabeler(images_dir, labels_dir)
    labeler.run()

if __name__ == "__main__":
    main()
