import cv2
import os
from ultralytics import YOLO
from collections import defaultdict
import supervision as sv

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Define entry and exit points
entries = {
    "Maligawa": (sv.Point(300, 750), sv.Point(300, 1500)),
    "Dalada": (sv.Point(2300, 720), sv.Point(2300, 1500))
}

exits = {
    "Left Lane": (sv.Point(1100, 480), sv.Point(1350, 480)),
    "Right Lane": (sv.Point(1360, 450), sv.Point(1550, 450)),
    "KCC Road": (sv.Point(1700, 550), sv.Point(1990, 550))
}

# Allowed paths (entry → exit)
valid_paths = {
    ("Maligawa", "Left Lane"),
    ("Maligawa", "Right Lane"),
    ("Maligawa", "KCC Road"),
    ("Dalada", "Left Lane"),
    ("Dalada", "Right Lane"),
    ("Dalada", "KCC Road")
}

# Track vehicle movements
vehicle_routes = defaultdict(lambda: {"entry": None, "exit": None, "counted": False})
route_counts = defaultdict(int)

# Clear previous output file
output_file = "vehicle_paths.txt"
if os.path.exists(output_file):
    os.remove(output_file)

# Video input/output setup
video_path = r"E:\Intern_FOE\TrafficMonitor\video.mp4"
cap = cv2.VideoCapture(video_path)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output_video2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Store tracking history
track_history = defaultdict(list)

# Process video
results = model.track(source=video_path, conf=0.3, iou=0.5, save=False, tracker="bytetrack.yaml", stream=True, persist=True)
frame_count = 0
MAX_FRAMES = 1500  # Limit for testing

for result in results:
    if frame_count >= MAX_FRAMES:
        break

    frame = result.orig_img
    if result.boxes is not None:
        boxes = result.boxes.xywh.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            cx, cy = int(x), int(y)

            # Draw bounding box
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 255), 3)
            cv2.putText(frame, f"ID: {track_id}", (int(x - w / 2), int(y - h / 2) - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            # Store track history
            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            # Detect entry points
            if vehicle_routes[track_id]["entry"] is None:
                for name, (start, end) in entries.items():
                    if start.x - 10 <= cx <= start.x + 10 and start.y <= cy <= end.y:
                        vehicle_routes[track_id]["entry"] = name

            # Detect exit points
            if vehicle_routes[track_id]["entry"] and vehicle_routes[track_id]["exit"] is None:
                for name, (start, end) in exits.items():
                    if end.y - 10 <= cy <= end.y + 10 and start.x <= cx <= end.x:
                        vehicle_routes[track_id]["exit"] = name

            # Count only valid paths
            entry = vehicle_routes[track_id]["entry"]
            exit_ = vehicle_routes[track_id]["exit"]
            if entry and exit_ and not vehicle_routes[track_id]["counted"] and (entry, exit_) in valid_paths:
                route_counts[f"{entry} → {exit_}"] += 1
                vehicle_routes[track_id]["counted"] = True

    # Draw entry and exit lines
    for name, (start, end) in {**entries, **exits}.items():
        color = (255, 0, 0) if name in entries else (0, 255, 0)
        cv2.line(frame, (start.x, start.y), (end.x, end.y), color, 5)

    # Display route counts
    cv2.rectangle(frame, (10, 10), (400, 220), (0, 0, 0), -1)
    y_offset = 40
    for route, count in route_counts.items():
        cv2.putText(frame, f"{route}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30

    # Save frame
    out.write(frame)
    frame_count += 1

# Save results to file with UTF-8 encoding
with open(output_file, "w", encoding="utf-8") as output:
    for route, count in route_counts.items():
        output.write(f"{route}: {count} vehicles\n")

# Release resources
out.release()
cap.release()
cv2.destroyAllWindows()
print("Processing complete! Output saved.")
