from ultralytics import YOLO

def train():
    model = YOLO('yolov8s.pt')  # Pretrained YOLOv8 nano model
    
    # Train with better parameters for small dataset
    model.train(
        data='data/dataset.yaml', 
        epochs=200,  # More epochs for small dataset
        imgsz=640, 
        batch=8,     # Smaller batch for small dataset
        lr0=0.001,   # Lower learning rate
        patience=20, # Early stopping patience
        save_period=10,  # Save checkpoints
        augment=True,    # Enable data augmentation
        mosaic=0.5,      # Mosaic augmentation
        mixup=0.1,       # Mixup augmentation
        copy_paste=0.1   # Copy-paste augmentation
    )

if __name__ == '__main__':
    train()
