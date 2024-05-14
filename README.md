# Real-Time Object Detection and Tracking

This project implements real-time object detection and tracking using the YOLOv5 model integrated with the SORT (Simple Online and Realtime Tracking) algorithm. The software uses PyTorch, OpenCV, and ultralytics YOLO implementations for robust and efficient object tracking in video streams from webcams.

## Features

- **Real-Time Detection**: Leverages YOLOv5 for high-accuracy object detection.
- **Object Tracking**: Integrates SORT algorithm for stable and reliable object tracking.
- **CUDA Support**: Optimized for CUDA-enabled devices for accelerated computing performance.
- **Flexible Input Options**: Configurable for different webcam sources and settings.

## Prerequisites

- Python 3.8 or newer
- PyTorch 1.7 or newer
- OpenCV 4.x
- NumPy
- ultralytics YOLOv5 (automatically downloaded via PyTorch Hub)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/fei123ilike/realtime-object-detection-and-tracking.git
   cd realtime-object-detection-and-tracking
   ```

2. **Set Up Python Environment::**
    ```bash
        python3 -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Set Up Python Environment::**
    ```bash
        pip3 install -r requirement.txt
    ```
## Usage

To start the object detection and tracking, simply run the script with the appropriate arguments:
        
        python3 detection.py --source 0 --model_weight yolov5s --iou_threshold 0.5
        python3 tracking.py --source 0 --max_age 30 --iou_threshold 0.5

arguments:

--source: 0 (webcam) or xxx.mmp4 local video

--max_age: Maximum number of frames to keep alive a track without associated detections

--iou_threshold: bounding boxes overlapping ratio in two consecutive frames

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Thanks to the ultralytics team for providing an accessible YOLOv5 model implementation.

The SORT algorithm is developed by Alex Bewley[https://arxiv.org/abs/1602.00763].