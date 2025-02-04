import ultralytics
import supervision
import torch
from ultralytics import YOLO

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Ensure the model runs on the appropriate device (CPU in this case)
model.to(device)

# Run inference on the video file
model.predict(
    source=r"E:\Intern_FOE\TrafficMonitor\test.mp4",  # Input video
    save=True,  # Save the output video
    imgsz=320,  # Resize images to 320x320
    conf=0.5,  # Confidence threshold
    device=device  # Specify the device
)
