import ultralytics
import supervision
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on the video with arguments
# model.predict(source=r"E:\Intern_FOE\TrafficMonitor\test.mp4", save=True, imgsz=320, conf=0.5)

results = model.track(source=r"E:\Intern_FOE\TrafficMonitor\video.mp4",conf=0.3, iou=0.5, save=True, tracker="bytetrack.yaml")