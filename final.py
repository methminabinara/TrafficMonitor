import cv2
import os
from ultralytics import YOLO
from collections import defaultdict
import supervision as sv
import torch
import numpy as np

# Check for GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load YOLOv8 model with specified device
model = YOLO('yolov8n.pt').to(device)

# Define vehicle classes in COCO dataset
# 2: car, 3: motorcycle, 5: bus, 7: truck, 8: boat
VEHICLE_CLASSES = [2, 3, 5, 7, 8]

# Updated Line coordinates
maligawa_line_start, maligawa_line_end = sv.Point(300, 750), sv.Point(300, 1500)  # Entrance 1
dalada_line_start, dalada_line_end = sv.Point(2300, 720), sv.Point(2300, 1500)  # Entrance 2
left_lane_line_start, left_lane_line_end = sv.Point(1100, 480), sv.Point(1350, 480)  # Exit 1
right_lane_start, right_lane_end = sv.Point(1360, 450), sv.Point(1550, 450)  # Exit 2
kcc_road_start, kcc_road_end = sv.Point(1700, 550), sv.Point(1990, 550)  # Exit 3

# Track vehicle paths
vehicle_paths = defaultdict(lambda: {"entry": None, "exit": None, "counted": False})
route_counts = {
    "Maligawa to Left Lane": 0,
    "Maligawa to Right Lane": 0,
    "Maligawa to KCC Road": 0,
    "Dalada to Left Lane": 0,
    "Dalada to Right Lane": 0,
    "Dalada to KCC Road": 0
}

# Open text file to save output
output_file = "vehicle_paths.txt"
if os.path.exists(output_file):
    os.remove(output_file)  # Clear previous file
output = open(output_file, "w")

# Video input and output
video_path = r"E:\Intern_FOE\TrafficMonitor\video.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video writer to save output
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Store the track history
track_history = defaultdict(lambda: [])

# Store the last known positions to help with consistent tracking
last_positions = {}

# Create a dictionary to keep track of objects that have crossed the lines
crossed_objects = defaultdict(lambda: {"entry": None, "exit": None})

# Improved tracking parameters
results = model.track(
    source=video_path,
    conf=0.25,  # Slightly lower confidence threshold
    iou=0.45,   # Adjusted IOU threshold
    persist=True,  # Maintain tracks even if object disappears briefly
    tracker="bytetrack.yaml",
    stream=True,
    show=False,
    device=device
)

frame_count = 0
MAX_FRAMES = 1000  # Process only first 1000 frames for testing

# Iterate through frames
for result in results:
    if frame_count >= MAX_FRAMES:
        break  # Stop after specified number of frames

    frame = result.orig_img  # Get the original frame

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xywh.cpu().numpy()  # Get bounding boxes in xywh format
        
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()  # Get track IDs
            classes = result.boxes.cls.cpu().numpy().astype(int)  # Get class indices
            
            # Filter for vehicle classes only
            vehicle_indices = [i for i, cls in enumerate(classes) if cls in VEHICLE_CLASSES]
            
            for i in vehicle_indices:
                if i >= len(boxes) or i >= len(track_ids):
                    continue  # Skip if index is out of range
                    
                box = boxes[i]
                track_id = track_ids[i]
                
                x, y, w, h = box  # Bounding box coordinates
                cx, cy = int(x), int(y)  # Center point of vehicle

                # Store this position as the last known position
                last_positions[track_id] = (cx, cy)

                # Draw bounding box and larger ID on frame
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), (int(y + h / 2))), (0, 255, 255), 3)
                cv2.putText(frame, f"ID: {track_id}", (int(x - w / 2), int(y - h / 2) - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                # Track history
                track = track_history[track_id]
                track.append((float(cx), float(cy)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)

                # Draw track
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, (0, 255, 255), 2)

                # Check if vehicle crosses an entry line
                if crossed_objects[track_id]["entry"] is None:
                    if maligawa_line_start.x - 10 <= cx <= maligawa_line_start.x + 10 and maligawa_line_start.y <= cy <= maligawa_line_end.y:
                        crossed_objects[track_id]["entry"] = "Maligawa"
                    elif dalada_line_start.x - 10 <= cx <= dalada_line_start.x + 10 and dalada_line_start.y <= cy <= dalada_line_end.y:
                        crossed_objects[track_id]["entry"] = "Dalada"

                # Check if vehicle crosses an exit line
                if crossed_objects[track_id]["entry"] is not None and crossed_objects[track_id]["exit"] is None:
                    if left_lane_line_start.y - 10 <= cy <= left_lane_line_end.y + 10 and left_lane_line_start.x <= cx <= left_lane_line_end.x:
                        crossed_objects[track_id]["exit"] = "Left Lane"
                    elif right_lane_start.y - 10 <= cy <= right_lane_end.y + 10 and right_lane_start.x <= cx <= right_lane_end.x:
                        crossed_objects[track_id]["exit"] = "Right Lane"
                    elif kcc_road_start.y - 10 <= cy <= kcc_road_end.y + 10 and kcc_road_start.x <= cx <= kcc_road_end.x:
                        crossed_objects[track_id]["exit"] = "KCC Road"

                # If both entry and exit are detected and not counted yet
                if crossed_objects[track_id]["entry"] and crossed_objects[track_id]["exit"] and not vehicle_paths[track_id]["counted"]:
                    route_key = f"{crossed_objects[track_id]['entry']} to {crossed_objects[track_id]['exit']}"
                    if route_key in route_counts:
                        route_counts[route_key] += 1
                    vehicle_paths[track_id]["counted"] = True  # Mark as counted

    # Draw entry and exit lines on the frame
    cv2.line(frame, (maligawa_line_start.x, maligawa_line_start.y), (maligawa_line_end.x, maligawa_line_end.y), (255, 0, 0), 5)  # Blue
    cv2.line(frame, (dalada_line_start.x, dalada_line_start.y), (dalada_line_end.x, dalada_line_end.y), (0, 255, 0), 5)  # Green
    cv2.line(frame, (left_lane_line_start.x, left_lane_line_start.y), (left_lane_line_end.x, left_lane_line_end.y), (0, 0, 255), 5)  # Red
    cv2.line(frame, (right_lane_start.x, right_lane_start.y), (right_lane_end.x, right_lane_end.y), (128, 0, 128), 5)  # Purple
    cv2.line(frame, (kcc_road_start.x, kcc_road_start.y), (kcc_road_end.x, kcc_road_end.y), (255, 255, 0), 5)  # Cyan

    # Display counts on the video
    cv2.rectangle(frame, (10, 10), (300, 220), (0, 0, 0), -1)
    cv2.putText(frame, f'Maligawa to Left Lane: {route_counts["Maligawa to Left Lane"]}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Maligawa to Right Lane: {route_counts["Maligawa to Right Lane"]}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Maligawa to KCC Road: {route_counts["Maligawa to KCC Road"]}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Dalada to Left Lane: {route_counts["Dalada to Left Lane"]}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Dalada to Right Lane: {route_counts["Dalada to Right Lane"]}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Dalada to KCC Road: {route_counts["Dalada to KCC Road"]}', (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Write frame to output video
    out.write(frame)
    frame_count += 1

# Write route summary to file
for route, count in route_counts.items():
    output.write(f"{route}: {count} vehicles\n")

# Close text file and video writer
output.close()
out.release()
cap.release()
cv2.destroyAllWindows()

print("Processing complete! Output saved to vehicle_paths.txt and output_video.mp4")