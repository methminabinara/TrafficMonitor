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
vehicle_paths = defaultdict(lambda: {"entry": None, "exit": None, "counted": False, "trajectory": []})
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

# Process video with frame-by-frame streaming
results = model.track(source=video_path, conf=0.3, iou=0.5, save=False, tracker="botsort.yaml", stream=True)

frame_count = 0
MAX_FRAMES = 600  # Process only first 600 frames for testing

# Function to check if a vehicle crosses a line
def is_crossing_line(prev_point, curr_point, line_start, line_end):
    prev_side = (prev_point[0] - line_start[0]) * (line_end[1] - line_start[1]) - (prev_point[1] - line_start[1]) * (line_end[0] - line_start[0])
    curr_side = (curr_point[0] - line_start[0]) * (line_end[1] - line_start[1]) - (curr_point[1] - line_start[1]) * (line_end[0] - line_start[0])
    return prev_side * curr_side < 0

# Iterate through frames
for result in results:
    if frame_count >= MAX_FRAMES:
        break  # Stop after MAX_FRAMES

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

            # Store trajectory
            if track_id in vehicle_paths:
                vehicle_paths[track_id]["trajectory"].append((cx, cy))
                if len(vehicle_paths[track_id]["trajectory"]) > 1:
                    prev_cx, prev_cy = vehicle_paths[track_id]["trajectory"][-2]  # Previous position

                    # Check if vehicle crosses an entry line
                    if vehicle_paths[track_id]["entry"] is None:
                        if is_crossing_line((prev_cx, prev_cy), (cx, cy), maligawa_line_start, maligawa_line_end):
                            vehicle_paths[track_id]["entry"] = "Maligawa"
                        elif is_crossing_line((prev_cx, prev_cy), (cx, cy), dalada_line_start, dalada_line_end):
                            vehicle_paths[track_id]["entry"] = "Dalada"

                    # Check if vehicle crosses an exit line
                    if vehicle_paths[track_id]["entry"] is not None and vehicle_paths[track_id]["exit"] is None:
                        if is_crossing_line((prev_cx, prev_cy), (cx, cy), left_lane_line_start, left_lane_line_end):
                            vehicle_paths[track_id]["exit"] = "Left Lane"
                        elif is_crossing_line((prev_cx, prev_cy), (cx, cy), right_lane_start, right_lane_end):
                            vehicle_paths[track_id]["exit"] = "Right Lane"
                        elif is_crossing_line((prev_cx, prev_cy), (cx, cy), kcc_road_start, kcc_road_end):
                            vehicle_paths[track_id]["exit"] = "KCC Road"

                    # If both entry and exit are detected and not counted yet
                    if vehicle_paths[track_id]["entry"] and vehicle_paths[track_id]["exit"] and not vehicle_paths[track_id]["counted"]:
                        route_key = f"{vehicle_paths[track_id]['entry']} to {vehicle_paths[track_id]['exit']}"
                        if route_key in route_counts:
                            route_counts[route_key] += 1
                        vehicle_paths[track_id]["counted"] = True  # Mark as counted

    # Draw entry and exit lines on the frame
    cv2.line(frame, maligawa_line_start, maligawa_line_end, (255, 0, 0), 5)  # Blue
    cv2.line(frame, dalada_line_start, dalada_line_end, (0, 255, 0), 5)  # Green
    cv2.line(frame, left_lane_line_start, left_lane_line_end, (0, 0, 255), 5)  # Red
    cv2.line(frame, right_lane_start, right_lane_end, (128, 0, 128), 5)  # Purple
    cv2.line(frame, kcc_road_start, kcc_road_end, (255, 255, 0), 5)  # Cyan

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