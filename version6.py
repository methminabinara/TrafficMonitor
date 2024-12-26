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
kcc_road_count = 0

# Vehicle mapping for entry and exit points
entry_exit_map = {
    "Maligawa": {"entry": set(), "exit": {"Left Lane": 0, "Right Lane": 0, "KCC Road": 0}},
    "Dalada Veediya": {"entry": set(), "exit": {"Left Lane": 0, "Right Lane": 0, "KCC Road": 0}},
}

# Vehicle class IDs (e.g., cars, buses, motorcycles, etc.)
vehicle_classes = [2, 3, 5, 7, 9]

offset = 7  # Tolerance for line crossing

# Line coordinates
maligawa_line_start, maligawa_line_end = (150, 250), (150, 600)
dalada_line_start, dalada_line_end = (900, 250), (900, 600)
left_lane_line_start, left_lane_line_end = (420, 160), (540, 160)
right_lane_start, right_lane_end = (550, 150), (650, 150)
kcc_road_start, kcc_road_end = (720, 220), (830, 220)

# Per-frame tracking of vehicle positions
vehicle_positions = {}

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

    # Draw lines
    cv2.line(frame, maligawa_line_start, maligawa_line_end, (255, 0, 0), 3)  # Blue
    cv2.line(frame, dalada_line_start, dalada_line_end, (0, 255, 0), 3)  # Green
    cv2.line(frame, left_lane_line_start, left_lane_line_end, (0, 0, 255), 3)  # Red
    cv2.line(frame, right_lane_start, right_lane_end, (128, 0, 128), 3)  # Purple
    cv2.line(frame, kcc_road_start, kcc_road_end, (255, 255, 0), 3)  # Cyan

    for bbox in tracked_boxes:
        x3, y3, x4, y4, vehicle_id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        if vehicle_id not in vehicle_positions:
            vehicle_positions[vehicle_id] = {"entry": None, "exit": None}

        # Check entry points
        if maligawa_line_start[0] - offset <= cx <= maligawa_line_start[0] + offset and maligawa_line_start[1] <= cy <= maligawa_line_end[1]:
            if vehicle_positions[vehicle_id]["entry"] is None:
                vehicle_positions[vehicle_id]["entry"] = "Maligawa"
                entry_exit_map["Maligawa"]["entry"].add(vehicle_id)
                maligawa_end_count += 1

        if dalada_line_start[0] - offset <= cx <= dalada_line_start[0] + offset and dalada_line_start[1] <= cy <= dalada_line_end[1]:
            if vehicle_positions[vehicle_id]["entry"] is None:
                vehicle_positions[vehicle_id]["entry"] = "Dalada Veediya"
                entry_exit_map["Dalada Veediya"]["entry"].add(vehicle_id)
                dalada_veediya_end_count += 1

        # Check exit points
        if left_lane_line_start[1] - offset <= cy <= left_lane_line_start[1] + offset and left_lane_line_start[0] <= cx <= left_lane_line_end[0]:
            if vehicle_positions[vehicle_id]["exit"] is None:
                vehicle_positions[vehicle_id]["exit"] = "Left Lane"
                if vehicle_positions[vehicle_id]["entry"] == "Maligawa":
                    entry_exit_map["Maligawa"]["exit"]["Left Lane"] += 1
                if vehicle_positions[vehicle_id]["entry"] == "Dalada Veediya":
                    entry_exit_map["Dalada Veediya"]["exit"]["Left Lane"] += 1
                left_lane_count += 1

        if right_lane_start[1] - offset <= cy <= right_lane_start[1] + offset and right_lane_start[0] <= cx <= right_lane_end[0]:
            if vehicle_positions[vehicle_id]["exit"] is None:
                vehicle_positions[vehicle_id]["exit"] = "Right Lane"
                if vehicle_positions[vehicle_id]["entry"] == "Maligawa":
                    entry_exit_map["Maligawa"]["exit"]["Right Lane"] += 1
                if vehicle_positions[vehicle_id]["entry"] == "Dalada Veediya":
                    entry_exit_map["Dalada Veediya"]["exit"]["Right Lane"] += 1
                right_lane_count += 1

        if kcc_road_start[1] - offset <= cy <= kcc_road_start[1] + offset and kcc_road_start[0] <= cx <= kcc_road_end[0]:
            if vehicle_positions[vehicle_id]["exit"] is None:
                vehicle_positions[vehicle_id]["exit"] = "KCC Road"
                if vehicle_positions[vehicle_id]["entry"] == "Maligawa":
                    entry_exit_map["Maligawa"]["exit"]["KCC Road"] += 1
                if vehicle_positions[vehicle_id]["entry"] == "Dalada Veediya":
                    entry_exit_map["Dalada Veediya"]["exit"]["KCC Road"] += 1
                kcc_road_count += 1

    # Display counts
    cv2.rectangle(frame, (10, 10), (400, 200), (0, 0, 0), -1)
    cv2.putText(frame, f'Maligawa End: {maligawa_end_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Dalada Veediya End: {dalada_veediya_end_count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Left Lane: {left_lane_count}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Right Lane: {right_lane_count}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'KCC Road: {kcc_road_count}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLOv8 Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save results
output_file = "vehicle_entry_exit_map.txt"
with open(output_file, "w") as file:
    for entry_point, data in entry_exit_map.items():
        file.write(f"{entry_point}:\n")
        file.write(f"  Entry count: {len(data['entry'])}\n")
        for exit_point, count in data["exit"].items():
            file.write(f"  Vehicles to {exit_point}: {count}\n")
        file.write("\n")

print(f"Vehicle entry and exit data saved to {output_file}")

cap.release()
cv2.destroyAllWindows()