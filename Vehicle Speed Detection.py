import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, deque

# Define video paths
SOURCE_VIDEO_PATH = "C:\\Users\\krish\\OneDrive\\Desktop\\Projects\\vehicles.mp4"
TARGET_VIDEO_PATH = "vehicles-result.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_NAME = "yolov8x.pt"
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

# Initialize the frame generator and view transformer
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
frame_iterator = iter(frame_generator)
frame = next(frame_iterator)

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
    
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# Initialize YOLO model
model = YOLO(MODEL_NAME)

# Get video info and frame generator
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# Initialize the tracker
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps
)

# Manually configure annotators (set thickness and text scale)
thickness = 2  # Default thickness
text_scale = 0.5  # Default text scale

bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
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

# Define a polygon zone and initialize coordinates
polygon_zone = sv.PolygonZone(
    polygon=SOURCE,
    frame_resolution_wh=video_info.resolution_wh
)
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

# Open the target video file to write annotated frames
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:

    # Loop over each frame in the source video
    for frame in tqdm(frame_generator, total=video_info.total_frames):

        # Run the YOLO model on the frame
        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Filter detections by confidence and class
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections[detections.class_id != 0]

        # Filter detections within the defined polygon zone
        detections = detections[polygon_zone.trigger(detections)]

        # Apply non-max suppression
        detections = detections.with_nms(IOU_THRESHOLD)

        # Pass the filtered detections through the tracker
        detections = byte_track.update_with_detections(detections=detections)

        # Get the coordinates of the detections
        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        # Transform points based on the view (if necessary)
        points = view_transformer.transform_points(points=points).astype(int)

        # Store the coordinates of tracked detections
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        # Format the labels, including speed calculation
        labels = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # Calculate the speed of the object
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6  # Convert to km/h
                labels.append(f"#{tracker_id} {int(speed)} km/h")

        # Annotate the frame with traces, bounding boxes, and labels
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # Add the annotated frame to the target video
        sink.write_frame(annotated_frame)