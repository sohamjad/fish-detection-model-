import argparse
import os
import cv2
from ultralytics import YOLO

def detect(video_paths, conf_threshold=0.1, weights_path='runs/detect/train17/weights/best.pt', output_dir='outputs'):
    model = YOLO(weights_path).to('cpu')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🔍 Testing model with confidence threshold: {conf_threshold}")

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video file {video_path}")
            continue

        print(f"\n📹 Processing {video_path}...")

        ret, test_frame = cap.read()
        if ret:
            test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            test_results = model(test_frame_rgb, conf=conf_threshold)
            print(f"🧪 Test frame detections: {len(test_results[0].boxes) if test_results[0].boxes is not None else 0}")
            if test_results[0].boxes is not None:
                for i, box in enumerate(test_results[0].boxes):
                    conf = box.conf[0].item()
                    print(f"  ➤ Detection {i+1}: confidence = {conf:.4f}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = os.path.join(output_dir, os.path.basename(video_path).replace('.mp4', '_output.mp4'))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 640))

        frame_count = 0
        detections_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            resized_frame = cv2.resize(frame, (640, 640))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            results = model(rgb_frame, conf=conf_threshold)
            if results[0].boxes is not None:
                detections_count += len(results[0].boxes)

            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            if frame_count % 30 == 0:
                print(f"  ✅ Processed {frame_count}/{total_frames} frames — {detections_count} detections")

        cap.release()
        out.release()
        print(f"✅ Done: {video_path}")
        print(f"📊 Total frames: {frame_count}, Total detections: {detections_count}")
        print(f"💾 Saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fish detection using YOLOv8')
    parser.add_argument('--weights', type=str, default='runs/detect/train17/weights/best.pt', help='Path to model weights')
    parser.add_argument('--input', type=str, nargs='*', help='List of video files to process (optional)')
    parser.add_argument('--input_dir', type=str, help='Folder containing .mp4 videos (optional)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save output videos')
    parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold for detection')
    args = parser.parse_args()

    video_files = []

    if args.input_dir:
        for file in os.listdir(args.input_dir):
            if file.endswith('.mp4'):
                video_files.append(os.path.join(args.input_dir, file))
    elif args.input:
        video_files = args.input
    else:
        print("❗ Please provide either --input or --input_dir")
        exit(1)

    detect(video_files, conf_threshold=args.conf, weights_path=args.weights, output_dir=args.output_dir)
