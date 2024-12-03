import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Define the video path
video_path = r'E:\Intern_FOE\TrafficMonitor\video.mp4'

# Open video capture
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Initialize tracker and counts
tracker = Tracker()
entry_count = 0

# Class IDs for vehicles in COCO dataset
vehicle_classes = [2, 3, 5, 7, 9]  # Car, Motorcycle, Bus, Truck, Bicycle

# Define the entry line for counting vehicles
line_start, line_end = (350, 200), (650, 200)  # Blue line (Entry)

# Tracking dictionary
entry_tracking = {}

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # Perform YOLOv8 inference on the frame
    results = model.predict(frame)
    detections = results[0].boxes.data.cpu().numpy()

    # Store detected vehicle boxes
    vehicle_boxes = []

    # Filter results to keep only vehicle classes
    for det in detections:
        x1, y1, x2, y2, conf, class_id = map(int, det[:6])
        if class_id in vehicle_classes:
            vehicle_boxes.append([x1, y1, x2, y2])

    # Update tracker and get tracked boxes with IDs
    tracked_boxes = tracker.update(vehicle_boxes)

    # Draw the entry line
    cv2.line(frame, line_start, line_end, (255, 0, 0), 3)  # Blue for entry

    # Track each vehicle and count based on crossing the line
    for bbox in tracked_boxes:
        x3, y3, x4, y4, vehicle_id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        # Define offset for movement tolerance
        offset = 7

        # Condition for vehicles crossing the entry line
        # Condition for vehicles crossing the entry line
        if (
            line_start[1] - offset <= cy <= line_start[1] + offset and
            line_start[0] <= cx <= line_end[0]
        ):
            if vehicle_id not in entry_tracking:
                entry_tracking[vehicle_id] = cy
                entry_count += 1
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, str(vehicle_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


    # Display entry count on the frame
    cv2.rectangle(frame, (10, 10), (200, 60), (0, 0, 0), -1)  # Black background for text
    cv2.putText(frame, f'Entries: {entry_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Display the frame with tracking annotations
    cv2.imshow("YOLOv8 Vehicle Detection with Tracking", frame)

    # Press 'q' to stop the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

print(f"Total vehicles entering: {entry_count}")
