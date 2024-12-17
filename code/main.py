# Import the required libraries
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, deque

# Define video paths
SOURCE_VIDEO_PATH = r"C:\Users\krish\OneDrive\Desktop\Projects\Vehicle Speed Detection\dataset\vehicles.mp4"
TARGET_VIDEO_PATH = r"C:\Users\krish\OneDrive\Desktop\Projects\Vehicle Speed Detection\result\vehicles-result.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_NAME = r"C:\Users\krish\OneDrive\Desktop\Projects\Vehicle Speed Detection\code\yolov8x.pt"
MODEL_RESOLUTION = 1280

# Define source polygon for tracking
SOURCE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])

# Define target polygon for transformation
TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

# Define color map for unique tracker IDs
def get_color(tracker_id: int):
    np.random.seed(tracker_id)  
    return tuple(np.random.randint(0, 255, size=3).tolist())

# View transformer class for perspective transformation
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# Initialize components
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
model = YOLO(MODEL_NAME)
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

byte_track = sv.ByteTrack(frame_rate=video_info.fps)

thickness = 3  
text_scale = 1.0 

box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.BOTTOM_CENTER
)
trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=video_info.fps * 2,
    position=sv.Position.BOTTOM_CENTER
)

polygon_zone = sv.PolygonZone(polygon=SOURCE)
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

# Open target video file
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        # Run YOLO model
        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Filter detections by confidence and polygon
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections[polygon_zone.trigger(detections)]

        # Apply non-max suppression and tracking
        detections = detections.with_nms(IOU_THRESHOLD)
        detections = byte_track.update_with_detections(detections=detections)

        # Transform points for speed calculation
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER        )
        points = view_transformer.transform_points(points=points).astype(int)

        # Update coordinates for each tracked object
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        # Calculate speed and create labels
        labels = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # Calculate speed
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6  # Convert to km/h
                labels.append(f"#{tracker_id} {int(speed)} km/h")

        # Generate colors for each tracker_id
        colors = [get_color(tracker_id) for tracker_id in detections.tracker_id]

        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Write annotated frame to the output video
        sink.write_frame(annotated_frame)