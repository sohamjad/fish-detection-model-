import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate) if frame_rate > 0 else 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        count += 1
    cap.release()

if __name__ == '__main__':
    videos = ['task1vid1.mp4', 'task1vid2.mp4']
    for i, video in enumerate(videos):
        output_dir = f'data/images/train' if i == 0 else f'data/images/val'
        extract_frames(video, output_dir, frame_rate=1)  # Extract 1 frame per second