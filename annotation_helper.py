import os

def convert_bbox_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def save_yolo_label(label_path, class_id, bbox):
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

if __name__ == '__main__':
    # Example usage:
    # Image size (width, height)
    size = (1280, 720)
    # Bounding box in pixel coordinates (xmin, xmax, ymin, ymax)
    box = (100, 400, 200, 600)
    yolo_bbox = convert_bbox_to_yolo(size, box)
    print("YOLO format bbox:", yolo_bbox)
    # Save label file
    import os

# Ensure the directory exists before saving labels
def save_yolo_label(label_path, class_id, bbox):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

# Example usage
bbox = (50, 100, 150, 200)  # xmin, ymin, xmax, ymax
img_width, img_height = 256, 256

def convert_bbox_to_yolo(bbox, img_width, img_height):
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return (x_center, y_center, width, height)

yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
print(f"YOLO format bbox: {yolo_bbox}")

save_yolo_label('data/labels/train/frame_00000.txt', 0, yolo_bbox)