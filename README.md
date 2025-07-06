
FISH DETECTION IN VIDEOS USING YOLOv8
=====================================

This project uses Ultralytics YOLOv8 to detect fish in underwater or lab-recorded videos.
It supports batch processing of multiple .mp4 files and outputs annotated videos showing fish detections.

------------------------------------------------------------

PROJECT STRUCTURE:

fish-detection/
├── detect.py            - Main detection script
├── runs/                - YOLO training outputs (optional)
├── outputs/             - Annotated output videos
├── videos/              - Input folder for .mp4 videos
└── README.txt           - This file

------------------------------------------------------------

REQUIREMENTS:

- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)

INSTALL DEPENDENCIES:
    pip install ultralytics opencv-python

------------------------------------------------------------

USAGE:

1. Detect Fish in a Single Video:
    python detect.py --input task1vid1.mp4

2. Detect in Multiple Videos:
    python detect.py --input task1vid1.mp4 task2vid2.mp4

3. Process All .mp4 Files in a Folder:
    python detect.py --input_dir videos/

4. Adjust Confidence Threshold:
    python detect.py --input task1vid1.mp4 --conf 0.5

------------------------------------------------------------

ARGUMENTS:

--weights     Path to model weights (default: runs/detect/train17/weights/best.pt)
--input       List of video files to process
--input_dir   Folder containing .mp4 videos
--output_dir  Folder to save output annotated videos (default: outputs/)
--conf        Confidence threshold for detection (default: 0.1)

------------------------------------------------------------

OUTPUT:

- Annotated videos are saved in the outputs/ folder.
- Output file example: task1vid1_output.mp4

------------------------------------------------------------

TROUBLESHOOTING:

- Too many detections? → Increase --conf to 0.4 or 0.5
- No detections? → Lower --conf or retrain with better data
- Model not found? → Check --weights path

------------------------------------------------------------

AUTHOR:

Built by Soham Jadhav
GitHub: https://github.com/
