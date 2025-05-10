<h1 align="center">SpeedVision</h1>
The code implements vehicle detection, tracking, and speed estimation using YOLOv8 for object detection, ByteTrack for tracking and OpenCV for video processing, annotating a video with bounding boxes, labels and speed calculations.

## Execution Guide:
1. Clone the repoistory:
   ```
   git clone https://github.com/kr1shnasomani/TraffiSense.git
   cd TraffiSense/SpeedVision
   ```

2. Download the dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Optional: Download the YOLOv8 model from the link - **https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt**
   Note: The code will automatically download the YOLOv8 model

4. On running the code it you will get the result video by the name of `vehicles-result.mp4`

## Model Prediction:

   Input Video:

   [input](https://github.com/user-attachments/assets/87f0202f-2582-47ea-a417-f2f4ef56afeb)

   Output Video:

   [output](https://github.com/user-attachments/assets/ae44c488-8690-442c-92d2-98cbf0461ceb)
