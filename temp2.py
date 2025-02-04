from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Process video with frame-by-frame streaming
results = model.track(source=r"E:\Intern_FOE\TrafficMonitor\video.mp4", 
                      conf=0.3, iou=0.5, save=True, tracker="bytetrack.yaml", stream=True)

# Iterate through frames
for result in results:
    frame = result.plot()  # Annotated frame
