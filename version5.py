import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Video input path
video_path = r'E:\Intern_FOE\TrafficMonitor\video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Initialize Tracker
tracker = Tracker()

# Counters
maligawa_end_count = 0
dalada_veediya_end_count = 0
left_lane_count = 0
right_lane_count = 0
kcc_road_count = 0  # Counter for KCC Road

# Sets for unique vehicle IDs
maligawa_end_ids = set()
dalada_veediya_end_ids = set()
left_lane_ids = set()
right_lane_ids = set()
kcc_road_ids = set()  # Set for KCC Road

# Vehicle class IDs
vehicle_classes = [2, 3, 5, 7, 9]  # Exclude pedestrians (class 0)

offset = 7  # Tolerance for line crossing

# Line coordinates
maligawa_line_start, maligawa_line_end = (150, 250), (150, 600)
dalada_line_start, dalada_line_end = (900, 250), (900, 600)
left_lane_line_start, left_lane_line_end = (420, 160), (540, 160)
right_lane_start, right_lane_end = (550, 150), (650, 150)
kcc_road_start, kcc_road_end = (660, 220), (830, 220)  # KCC Road line

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame, stream=True)

    vehicle_boxes = []
    for result in results:
        for det in result.boxes:
            box = det.xyxy[0].cpu().numpy().astype(int)
            class_id = int(det.cls[0].cpu().numpy())
            if class_id in vehicle_classes:  # Filter only vehicles
                vehicle_boxes.append(box)

    tracked_boxes = tracker.update(vehicle_boxes)

    # Draw lines
    cv2.line(frame, maligawa_line_start, maligawa_line_end, (255, 0, 0), 3)  # Blue
    cv2.line(frame, dalada_line_start, dalada_line_end, (0, 255, 0), 3)  # Green
    cv2.line(frame, left_lane_line_start, left_lane_line_end, (0, 0, 255), 3)  # Red
    cv2.line(frame, right_lane_start, right_lane_end, (128, 0, 128), 3)  # Purple
    cv2.line(frame, kcc_road_start, kcc_road_end, (255, 255, 0), 3)  # Cyan

    for bbox in tracked_boxes:
        x3, y3, x4, y4, vehicle_id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        # Maligawa End (Vertical)
        if maligawa_line_start[0] - offset <= cx <= maligawa_line_start[0] + offset and maligawa_line_start[1] <= cy <= maligawa_line_end[1]:
            if vehicle_id not in maligawa_end_ids:
                maligawa_end_ids.add(vehicle_id)
                maligawa_end_count += 1

        # Dalada Veediya End (Vertical)
        if dalada_line_start[0] - offset <= cx <= dalada_line_start[0] + offset and dalada_line_start[1] <= cy <= dalada_line_end[1]:
            if vehicle_id not in dalada_veediya_end_ids:
                dalada_veediya_end_ids.add(vehicle_id)
                dalada_veediya_end_count += 1

        # Left Lane (Horizontal)
        if left_lane_line_start[1] - offset <= cy <= left_lane_line_start[1] + offset and left_lane_line_start[0] <= cx <= left_lane_line_end[0]:
            if vehicle_id not in left_lane_ids:
                left_lane_ids.add(vehicle_id)
                left_lane_count += 1

        # Right Lane (Horizontal)
        if right_lane_start[1] - offset <= cy <= right_lane_start[1] + offset and right_lane_start[0] <= cx <= right_lane_end[0]:
            if vehicle_id not in right_lane_ids:
                right_lane_ids.add(vehicle_id)
                right_lane_count += 1

        # KCC Road (Horizontal)
        if kcc_road_start[1] - offset <= cy <= kcc_road_start[1] + offset and kcc_road_start[0] <= cx <= kcc_road_end[0]:
            if vehicle_id not in kcc_road_ids:
                kcc_road_ids.add(vehicle_id)
                kcc_road_count += 1

    # Display counts
    cv2.rectangle(frame, (10, 10), (300, 180), (0, 0, 0), -1)
    cv2.putText(frame, f'Maligawa End: {maligawa_end_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Dalada Veediya End: {dalada_veediya_end_count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Left Lane: {left_lane_count}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Right Lane: {right_lane_count}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'KCC Road: {kcc_road_count}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLOv8 Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the output counts to a text file
output_file = "vehicle_counts.txt"
with open(output_file, "w") as file:
    file.write(f"Total vehicles at Maligawa End: {maligawa_end_count}\n")
    file.write(f"Total vehicles at Dalada Veediya End: {dalada_veediya_end_count}\n")
    file.write(f"Total vehicles on Left Lane: {left_lane_count}\n")
    file.write(f"Total vehicles on Right Lane: {right_lane_count}\n")
    file.write(f"Total vehicles on KCC Road: {kcc_road_count}\n")

print(f"Vehicle counts saved to {output_file}")

cap.release()
cv2.destroyAllWindows()

print(f"Total vehicles at Maligawa End: {maligawa_end_count}")
print(f"Total vehicles at Dalada Veediya End: {dalada_veediya_end_count}")
print(f"Total vehicles on Left Lane: {left_lane_count}")
print(f"Total vehicles on Right Lane: {right_lane_count}")
print(f"Total vehicles on KCC Road: {kcc_road_count}")