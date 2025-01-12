<h1 align="center">SpeedVision</h1>
The code implements vehicle detection, tracking, and speed estimation using YOLOv8 for object detection, ByteTrack for tracking and OpenCV for video processing, annotating a video with bounding boxes, labels and speed calculations.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install opencv-python numpy supervision tqdm ultralytics
   ```

2. Download the YOLOv8 model from the link - **https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt**

3. Copy paste the path of the model in the code
  
4. Download the `vehicles.mp4` file

5. Copy the path of the file `vehicles.mp4` and paste it in the code

6. Once that is done run the code and you will get the result video by the name of `vehicles-result.mp4`

## Model Prediction:

   Input Video:

   ![input](https://github.com/user-attachments/assets/87f0202f-2582-47ea-a417-f2f4ef56afeb)

   Output Video:

   ![output](https://github.com/user-attachments/assets/ae44c488-8690-442c-92d2-98cbf0461ceb)

## Overview:
This project uses object detection and tracking techniques to detect vehicles and estimate their speed in a video. The key components include:

1. **Video Processing**: The code reads and processes video frames from a source video file (`vehicles.mp4`), and a target video (`vehicles-result.mp4`) is generated with annotations.

2. **Object Detection**: The YOLOv8 model is employed to detect objects (vehicles) in each video frame. Only objects with a confidence score above a threshold (0.3) and excluding class 0 (person) are considered.

3. **Polygon Zone**: A source polygon is defined on the video to filter detected objects that fall within this region, ensuring only relevant detections are tracked.

4. **Tracking**: The ByteTrack tracker is used to track vehicles across frames. The tracker updates with each new detection and stores coordinates of the tracked objects.

5. **Speed Estimation**: For each tracked object, the code calculates its speed based on the change in position over time (using the coordinates stored by the tracker). The speed is then converted to kilometers per hour (km/h).

6. **Annotation**: The frame is annotated with bounding boxes around detected vehicles, their tracker IDs, and calculated speeds. Annotations are applied using custom annotators that handle bounding boxes, labels, and traces.

7. **Result Video**: Finally, the annotated frames are compiled into a target video that shows the detected vehicles with their speeds, providing real-time tracking and speed estimation.

In summary, the code integrates object detection, tracking, and speed estimation to analyze vehicle movement in a video, providing an annotated output with the vehicle IDs and their speeds.
