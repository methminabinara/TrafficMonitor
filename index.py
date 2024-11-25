import cv2
import numpy as np
import pandas as pd
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
entry_count, exit_count, parked_count = 0, 0, 0

# Class IDs for vehicles in COCO dataset
vehicle_classes = [2, 3, 5, 7, 9]  # Car, Motorcycle, Bus, Truck, Bicycle

# Define the angled lines for counting vehicles (entry/exit lines)
line_start_red, line_end_red = (10,250),(1100,250)  # Red line (Exit)
line_start_blue, line_end_blue = (10,300),(1100,300)  # Blue line (Entry)

# Initialize tracking variables
exit_tracking, entry_tracking = {}, {}
exit_vehicles, entry_vehicles = [], []
frame_count = 0

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
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

    # Draw entry/exit lines
    cv2.line(frame, line_start_red, line_end_red, (0, 0, 255), 3)  # Red for exit
    cv2.line(frame, line_start_blue, line_end_blue, (255, 0, 0), 3)  # Blue for entry

    # Track each vehicle and count based on movement direction
    for bbox in tracked_boxes:
        x3, y3, x4, y4, vehicle_id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        # Define offset for movement tolerance
        offset = 7

        # Condition for vehicles moving upwards (entry)
        if line_start_blue[1] < (cy + offset) and line_start_blue[1] > (cy - offset):
            entry_tracking[vehicle_id] = cy
        if vehicle_id in entry_tracking:
            if line_start_red[1] < (cy + offset) and line_start_red[1] > (cy - offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(vehicle_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                entry_vehicles.append(vehicle_id)
                del entry_tracking[vehicle_id]

        # Condition for vehicles moving downwards (exit)
        if line_start_red[1] < (cy + offset) and line_start_red[1] > (cy - offset):
            exit_tracking[vehicle_id] = cy
        if vehicle_id in exit_tracking:
            if line_start_blue[1] < (cy + offset) and line_start_blue[1] > (cy - offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(vehicle_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                exit_vehicles.append(vehicle_id)
                del exit_tracking[vehicle_id]

    # Display entry, exit, and parked counts on the frame
    exits = len(set(exit_vehicles))
    entries = len(set(entry_vehicles))

    # Make text visible by adding a solid background behind text
    cv2.rectangle(frame, (10, 10), (200, 80), (0, 0, 0), -1)  # Black background for text
    cv2.putText(frame, f'Entries: {entries}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f'Exits: {exits}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Display the frame with tracking annotations
    cv2.imshow("YOLOv8 Vehicle Detection with Tracking", frame)

    # Press 'q' to stop the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

print(f"Total vehicles entering: {entries}")
print(f"Total vehicles exiting: {exits}")
