<h1 align="center">TraffiSense</h1>
<p align="center" style="margin-top:30px;">
  <img src="https://github.com/user-attachments/assets/7c28ee29-4ee0-44c0-8ee9-43076d6c53b0" height="250cm"/>
</p>

## Project Index:
| Name | Descritpion |
|------|-------------|
| [PlateScan](https://github.com/kr1shnasomani/TraffiSense/tree/main/PlateScan) | The project detects and extracts text from vehicle number plates using OpenCV for image processing, EasyOCR for text recognition, and imutils for contour detection. It identifies the number plate region, crops it, extracts the text, and saves the result without displaying the images. |
| [SpeedVision](https://github.com/kr1shnasomani/TraffiSense/tree/main/SpeedVision) | The code implements vehicle detection, tracking, and speed estimation using YOLOv8 for object detection, ByteTrack for tracking and OpenCV for video processing, annotating a video with bounding boxes, labels and speed calculations. |
| [StreetScanner](https://github.com/kr1shnasomani/TraffiSense/tree/main/StreetScanner) | The system leverages DeepLabv3 ResNet-50 to detect and segment pedestrians and vehicles, with color-coded masks for easy visualization. This solution is designed for applications in traffic monitoring, pedestrian safety and smart city solutions. Built using PyTorch, Semantic Segmentation and Computer Vision techniques. |


## Respository Structure:
```
TraffiSense/
├── PlateScan/
│   ├── code/
│   │   └── main.py
│   ├── dataset/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── result/
│   │   ├── number-plate-1.jpg
│   │   └── number-plate-2.jpg
│   └── README.md  
├── SpeedVision/
│   ├── code/
│   │   └── main.py
│   └── README.md
├── StreetScanner/
│   ├── code/
|   │   ├── both.py
│   │   ├── pedestrian.py
│   │   └── vehicle.py
│   ├── dataset/
│   │   └── image.jpeg
│   ├── result/
|   │   ├── both-resultant-image.png
│   │   ├── pedestrian-resultant-image.png
│   │   └── vehicle-resultant-image.png
│   └── README.md
└── README.md
```
