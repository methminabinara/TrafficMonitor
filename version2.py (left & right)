import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('yolov8n.pt')

video_path = r'E:\Intern_FOE\TrafficMonitor\video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

tracker = Tracker()

left_entry_count = 0
right_entry_count = 0
main_road_count = 0

left_entry_ids = set()
right_entry_ids = set()
main_road_ids = set()

vehicle_classes = [2, 3, 5, 7, 9]

offset = 7  # Tolerance

left_line_start, left_line_end = (80, 300), (500, 300)
right_line_start, right_line_end = (600, 300), (1100, 300)
main_line_start, main_line_end = (350, 200), (650, 200)

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
            if class_id in vehicle_classes:
                vehicle_boxes.append(box)

    tracked_boxes = tracker.update(vehicle_boxes)

    cv2.line(frame, left_line_start, left_line_end, (255, 0, 0), 3)
    cv2.line(frame, right_line_start, right_line_end, (0, 255, 0), 3)
    cv2.line(frame, main_line_start, main_line_end, (0, 0, 255), 3)

    for bbox in tracked_boxes:
        x3, y3, x4, y4, vehicle_id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        if left_line_start[1] - offset <= cy <= left_line_start[1] + offset and left_line_start[0] <= cx <= left_line_end[0]:
            if vehicle_id not in left_entry_ids:
                left_entry_ids.add(vehicle_id)
                left_entry_count += 1

        if right_line_start[1] - offset <= cy <= right_line_start[1] + offset and right_line_start[0] <= cx <= right_line_end[0]:
            if vehicle_id not in right_entry_ids:
                right_entry_ids.add(vehicle_id)
                right_entry_count += 1

        if main_line_start[1] - offset <= cy <= main_line_start[1] + offset and main_line_start[0] <= cx <= main_line_end[0]:
            if vehicle_id not in main_road_ids:
                main_road_ids.add(vehicle_id)
                main_road_count += 1

    cv2.rectangle(frame, (10, 10), (250, 100), (0, 0, 0), -1)
    cv2.putText(frame, f'Left Entries: {left_entry_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Right Entries: {right_entry_count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Main Road: {main_road_count}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLOv8 Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total vehicles on main road: {main_road_count}")
