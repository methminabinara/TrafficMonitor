import cv2
import os
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Updated Line coordinates
maligawa_line_start, maligawa_line_end = (300, 750), (300, 1500)  # Entrance 1
dalada_line_start, dalada_line_end = (2300, 720), (2300, 1500)  # Entrance 2
left_lane_line_start, left_lane_line_end = (1100, 500), (1350, 500)  # Exit 1
right_lane_start, right_lane_end = (1360, 470), (1550, 470)  # Exit 2
kcc_road_start, kcc_road_end = (1700, 550), (1990, 550)  # Exit 3

# Track vehicle paths
vehicle_paths = defaultdict(lambda: {"entry": None, "exit": None})
route_counts = defaultdict(int)  # Count vehicles per route

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

# Process video with frame-by-frame streaming
results = model.track(source=video_path, conf=0.3, iou=0.5, save=False, tracker="bytetrack.yaml", stream=True)

frame_count = 0
MAX_FRAMES = 600  # Process only first 200 frames for testing

# Iterate through frames
for result in results:
    if frame_count >= MAX_FRAMES:
        break  # Stop after 200 frames

    frame = result.orig_img  # Get the original frame

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        track_ids = result.boxes.id.int().cpu().tolist()  # Get track IDs

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box  # Bounding box coordinates
            cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)  # Center point of vehicle

            # Draw bounding box and larger ID on frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # Check if vehicle crosses an entry line
            if maligawa_line_start[0] - 10 <= cx <= maligawa_line_start[0] + 10:
                vehicle_paths[track_id]["entry"] = "Maligawa End"

            if dalada_line_start[0] - 10 <= cx <= dalada_line_start[0] + 10:
                vehicle_paths[track_id]["entry"] = "Dalada Weediya End"

            # Check if vehicle crosses an exit line
            if left_lane_line_start[1] - 10 <= cy <= left_lane_line_end[1] + 10:
                vehicle_paths[track_id]["exit"] = "Left Lane"

            if right_lane_start[1] - 10 <= cy <= right_lane_end[1] + 10:
                vehicle_paths[track_id]["exit"] = "Right Lane"

            if kcc_road_start[1] - 10 <= cy <= kcc_road_end[1] + 10:
                vehicle_paths[track_id]["exit"] = "KCC Road"

            # If both entry and exit are detected, count the route
            if vehicle_paths[track_id]["entry"] and vehicle_paths[track_id]["exit"]:
                route_key = f"{vehicle_paths[track_id]['entry']} to {vehicle_paths[track_id]['exit']}"
                route_counts[route_key] += 1

    # Draw entry and exit lines on the frame
    cv2.line(frame, maligawa_line_start, maligawa_line_end, (255, 0, 0), 5)  # Blue
    cv2.line(frame, dalada_line_start, dalada_line_end, (0, 255, 0), 5)  # Green
    cv2.line(frame, left_lane_line_start, left_lane_line_end, (0, 0, 255), 5)  # Red
    cv2.line(frame, right_lane_start, right_lane_end, (128, 0, 128), 5)  # Purple
    cv2.line(frame, kcc_road_start, kcc_road_end, (255, 255, 0), 5)  # Cyan

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
