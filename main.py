import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
import os
import time

# Constants for tracking
MAX_TRACK_AGE = 60  # Maximum frames to keep a lost track
MIN_TRACK_VISIBILITY = 10  # Minimum frames visible to count a track
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4

# Vehicle classes in COCO dataset - removed boat (8)
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

class TrafficMonitor:
    def __init__(self, video_path, model_path='yolov8s.pt'):
        # Check for GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load YOLO model - using small version for better accuracy while still fast on GTX 1650
        self.model = YOLO(model_path).to(self.device)
        
        # Video settings
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Define entry and exit lines
        # Format: [(name, point1, point2), ...]
        self.entry_lines = [
            ("Maligawa", (300, 750), (300, 1500)),
            ("Dalada Weediya", (2300, 720), (2300, 1500))
        ]
        
        self.exit_lines = [
            ("Left Lane", (1100, 460), (1350, 460)),
            ("Right Lane", (1360, 460), (1550, 460)),
            ("KCC Road", (1700, 550), (1990, 550))
        ]
        
        # Tracking data structures
        self.active_tracks = {}  # {track_id: {data}}
        self.completed_tracks = []  # Tracks that have both entry and exit points
        self.track_history = defaultdict(list)  # For visualization
        
        # Route counting
        self.route_counts = {}
        self.vehicle_type_counts = {}  # Track counts by vehicle type
        self._initialize_counters()
        
        # For improved track prediction
        self.kalman_trackers = {}
        
        # Output settings
        self.output_video_path = "improved_traffic_output.mp4"
        self.output_stats_path = "improved_traffic_stats.txt"
        self.video_writer = None

    def _initialize_counters(self):
        """Initialize counters for all possible routes and vehicle types"""
        # Initialize route counters
        for entry in [line[0] for line in self.entry_lines]:
            for exit in [line[0] for line in self.exit_lines]:
                route = f"{entry} to {exit}"
                self.route_counts[route] = 0
                
                # Initialize vehicle type counters for each route
                self.vehicle_type_counts[route] = {v_type: 0 for v_type in VEHICLE_CLASSES.values()}

    def _setup_video_writer(self):
        """Initialize video writer"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_video_path, fourcc, self.fps, (self.width, self.height)
        )

    def _check_line_crossing(self, point, line):
        """Check if a point crosses a line with margin for more reliable detection"""
        line_start, line_end = line[1], line[2]
        line_name = line[0]
        
        # Different logic depending on horizontal or vertical line
        if abs(line_start[1] - line_end[1]) < abs(line_start[0] - line_end[0]):  # Horizontal line
            if line_start[1] - 15 <= point[1] <= line_end[1] + 15:
                if min(line_start[0], line_end[0]) <= point[0] <= max(line_start[0], line_end[0]):
                    return line_name
        else:  # Vertical line
            if line_start[0] - 15 <= point[0] <= line_end[0] + 15:
                if min(line_start[1], line_end[1]) <= point[1] <= max(line_start[1], line_end[1]):
                    return line_name
        return None

    def _init_kalman_tracker(self, track_id, x, y):
        """Initialize Kalman filter tracker for smoother prediction"""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]
        ], np.float32)
        kalman.processNoiseCov = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        ], np.float32) * 0.03
        
        kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
        kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
        
        self.kalman_trackers[track_id] = {
            'filter': kalman,
            'last_update': time.time(),
            'missed_frames': 0
        }

    def _update_kalman_tracker(self, track_id, x, y):
        """Update Kalman filter with new measurement"""
        if track_id not in self.kalman_trackers:
            self._init_kalman_tracker(track_id, x, y)
            return x, y
        
        kalman = self.kalman_trackers[track_id]['filter']
        self.kalman_trackers[track_id]['last_update'] = time.time()
        self.kalman_trackers[track_id]['missed_frames'] = 0
        
        # Correct and predict
        measurement = np.array([[x], [y]], np.float32)
        kalman.correct(measurement)
        prediction = kalman.predict()
        
        # Return smoothed position
        pred_x = int(prediction[0, 0])
        pred_y = int(prediction[1, 0])
        
        # Apply some weighted averaging between prediction and measurement
        smoothed_x = int(0.6 * x + 0.4 * pred_x)
        smoothed_y = int(0.6 * y + 0.4 * pred_y)
        
        return smoothed_x, smoothed_y

    def _predict_missing_position(self, track_id):
        """Predict position for temporarily missing track"""
        if track_id in self.kalman_trackers:
            kalman = self.kalman_trackers[track_id]['filter']
            self.kalman_trackers[track_id]['missed_frames'] += 1
            
            # Only predict for a limited number of frames
            if self.kalman_trackers[track_id]['missed_frames'] < 10:
                prediction = kalman.predict()
                pred_x = int(prediction[0, 0])
                pred_y = int(prediction[1, 0])
                return pred_x, pred_y
        return None

    def process_frame(self, frame, results):
        """Process detection results for a single frame"""
        # Draw entry and exit lines
        self._draw_monitoring_lines(frame)
        
        detected_ids = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xywh.cpu().numpy()
            
            if results.boxes.id is not None:
                track_ids = results.boxes.id.int().cpu().tolist()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                
                # Filter for vehicle classes
                vehicle_indices = [i for i, cls in enumerate(classes) if cls in VEHICLE_CLASSES]
                
                for i in vehicle_indices:
                    if i >= len(boxes) or i >= len(track_ids):
                        continue
                    
                    box = boxes[i]
                    track_id = track_ids[i]
                    vehicle_class = classes[i]
                    vehicle_type = VEHICLE_CLASSES[vehicle_class]
                    detected_ids.append(track_id)
                    
                    x, y, w, h = box
                    cx, cy = int(x), int(y)
                    
                    # Apply Kalman filtering for smoother tracking
                    smooth_cx, smooth_cy = self._update_kalman_tracker(track_id, cx, cy)
                    
                    # Initialize track if new
                    if track_id not in self.active_tracks:
                        self.active_tracks[track_id] = {
                            'entry': None,
                            'exit': None,
                            'positions': [],
                            'frames_visible': 1,
                            'last_seen': 0,
                            'vehicle_type': vehicle_type,
                            'vehicle_class': vehicle_class
                        }
                    else:
                        self.active_tracks[track_id]['frames_visible'] += 1
                        self.active_tracks[track_id]['last_seen'] = 0  # Reset counter
                        
                        # Update vehicle type if we have more detections
                        # This helps if initial detection was ambiguous
                        if self.active_tracks[track_id]['frames_visible'] < 10:
                            self.active_tracks[track_id]['vehicle_type'] = vehicle_type
                            self.active_tracks[track_id]['vehicle_class'] = vehicle_class
                    
                    # Store position history
                    self.active_tracks[track_id]['positions'].append((smooth_cx, smooth_cy))
                    
                    # Check entry lines
                    if self.active_tracks[track_id]['entry'] is None:
                        for line in self.entry_lines:
                            entry_point = self._check_line_crossing((smooth_cx, smooth_cy), line)
                            if entry_point:
                                self.active_tracks[track_id]['entry'] = entry_point
                                break
                    
                    # Check exit lines
                    if self.active_tracks[track_id]['entry'] is not None and self.active_tracks[track_id]['exit'] is None:
                        for line in self.exit_lines:
                            exit_point = self._check_line_crossing((smooth_cx, smooth_cy), line)
                            if exit_point:
                                self.active_tracks[track_id]['exit'] = exit_point
                                # Count completed route
                                if self.active_tracks[track_id]['frames_visible'] >= MIN_TRACK_VISIBILITY:
                                    route = f"{self.active_tracks[track_id]['entry']} to {exit_point}"
                                    v_type = self.active_tracks[track_id]['vehicle_type']
                                    
                                    # Update counters
                                    self.route_counts[route] += 1
                                    self.vehicle_type_counts[route][v_type] += 1
                                    self.completed_tracks.append(self.active_tracks[track_id])
                                break
                    
                    # Update track history for visualization (limit to last 60 positions)
                    self.track_history[track_id].append((smooth_cx, smooth_cy))
                    if len(self.track_history[track_id]) > 60:
                        self.track_history[track_id].pop(0)
                    
                    # Draw bounding box and ID
                    color = self._get_vehicle_color(vehicle_class)
                    
                    cv2.rectangle(frame, 
                                  (int(x - w/2), int(y - h/2)), 
                                  (int(x + w/2), int(y + h/2)), 
                                  color, 2)
                    
                    # Display ID and vehicle type
                    label = f"ID:{track_id} {vehicle_type}"
                    cv2.putText(frame, label, (int(x - w/2), int(y - h/2 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw track trail
                    if len(self.track_history[track_id]) > 1:
                        pts = np.array(self.track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, color, 2)
        
        # Handle tracks not detected in current frame
        tracks_to_remove = []
        for track_id, track_data in self.active_tracks.items():
            if track_id not in detected_ids:
                # Use Kalman prediction for temporarily missing tracks
                predicted_pos = self._predict_missing_position(track_id)
                if predicted_pos:
                    # Draw predicted position with dashed boundary
                    cx, cy = predicted_pos
                    cv2.circle(frame, (cx, cy), 5, (0, 165, 255), -1)
                    
                    # Update history with predicted position
                    self.track_history[track_id].append((cx, cy))
                    if len(self.track_history[track_id]) > 60:
                        self.track_history[track_id].pop(0)
                
                # Increment missing frames counter
                track_data['last_seen'] += 1
                
                # Remove tracks missing for too long
                if track_data['last_seen'] > MAX_TRACK_AGE:
                    # If track had entry but no exit and was visible enough, count towards nearest exit
                    if (track_data['entry'] is not None and 
                        track_data['exit'] is None and 
                        track_data['frames_visible'] >= MIN_TRACK_VISIBILITY):
                        
                        # Find nearest exit line to last position
                        if track_data['positions']:
                            last_pos = track_data['positions'][-1]
                            min_distance = float('inf')
                            nearest_exit = None
                            
                            for exit_line in self.exit_lines:
                                exit_center = ((exit_line[1][0] + exit_line[2][0])//2, 
                                              (exit_line[1][1] + exit_line[2][1])//2)
                                dist = np.sqrt((last_pos[0] - exit_center[0])**2 + 
                                             (last_pos[1] - exit_center[1])**2)
                                
                                if dist < min_distance:
                                    min_distance = dist
                                    nearest_exit = exit_line[0]
                            
                            # Only count if reasonably close to an exit
                            if min_distance < 300 and nearest_exit:
                                route = f"{track_data['entry']} to {nearest_exit}"
                                v_type = track_data['vehicle_type']
                                
                                # Update counters
                                self.route_counts[route] += 1
                                self.vehicle_type_counts[route][v_type] += 1
                    
                    tracks_to_remove.append(track_id)
        
        # Remove expired tracks
        for track_id in tracks_to_remove:
            if track_id in self.active_tracks:
                del self.active_tracks[track_id]
            if track_id in self.kalman_trackers:
                del self.kalman_trackers[track_id]
        
        # Draw counts on frame
        self._draw_stats(frame)
        
        return frame

    def _get_vehicle_color(self, vehicle_class):
        """Get color for vehicle type visualization"""
        # Different color for each vehicle type
        colors = {
            2: (0, 255, 0),    # Car - Green
            3: (255, 0, 0),    # Motorcycle - Blue
            5: (0, 165, 255),  # Bus - Orange
            7: (128, 0, 128),  # Truck - Purple
        }
        return colors.get(vehicle_class, (0, 255, 255))  # Default yellow

    def _draw_monitoring_lines(self, frame):
        """Draw entry and exit lines on the frame"""
        # Draw entry lines
        for i, line in enumerate(self.entry_lines):
            color = (255, 0, 0) if i == 0 else (0, 255, 0)  # Blue for first, Green for second
            cv2.line(frame, line[1], line[2], color, 3)
            cv2.putText(frame, f"Entry: {line[0]}", 
                      (line[1][0] - 50, line[1][1] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw exit lines
        colors = [(0, 0, 255), (128, 0, 128), (255, 255, 0)]  # Red, Purple, Cyan
        for i, line in enumerate(self.exit_lines):
            cv2.line(frame, line[1], line[2], colors[i], 3)
            cv2.putText(frame, f"Exit: {line[0]}", 
                      (line[1][0], line[1][1] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

    def _calculate_text_block_size(self, routes, vehicle_types):
        """Calculate required size for the text block"""
        # Base height for the header
        height = 60
        
        # Height for vehicle type counts
        height += 25 * len([c for c in vehicle_types.values() if c > 0]) + 25
        
        # Height for all routes
        height += 25 * sum(1 for r in routes.values() if r > 0) + 25
        
        # Add some padding
        height += 10
        
        # Width based on longest possible route name
        width = 410
        
        return width, height

    def _draw_stats(self, frame):
        """Draw route statistics and vehicle counts on frame with properly sized background"""
        # Calculate vehicle totals
        vehicle_totals = defaultdict(int)
        for route_data in self.vehicle_type_counts.values():
            for v_type, count in route_data.items():
                vehicle_totals[v_type] += count
        
        # Get actual routes with counts > 0
        active_routes = {r: c for r, c in self.route_counts.items() if c > 0}
        
        # Calculate background size based on content
        bg_width, bg_height = self._calculate_text_block_size(active_routes, vehicle_totals)
        
        # Draw background for text with calculated size
        cv2.rectangle(frame, (10, 10), (10 + bg_width, 10 + bg_height), (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, "Traffic Flow Statistics:", (20, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display vehicle type counts
        y_pos = 60
        cv2.putText(frame, "Vehicle Types:", (20, y_pos),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos += 25
        for v_type, count in vehicle_totals.items():
            if count > 0:
                cv2.putText(frame, f"{v_type}s: {count}", (30, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_pos += 25
        
        # Display all routes with counts > 0
        y_pos += 10
        cv2.putText(frame, "All Routes:", (20, y_pos),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos += 25
        # Sort routes by count for better readability
        sorted_routes = sorted(active_routes.items(), key=lambda x: x[1], reverse=True)
        for route, count in sorted_routes:
            if count > 0:
                cv2.putText(frame, f"{route}: {count}", (30, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_pos += 25

    def process_video(self):
        """Process the entire video"""
        print("Starting video processing...")
        self._setup_video_writer()
        
        # Set up tracker with more robust parameters
        results = self.model.track(
            source=self.video_path,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            stream=True,
            device=self.device,
            show=False
        )
        
        frame_count = 0
        for result in results:
            frame = result.orig_img
            processed_frame = self.process_frame(frame, result)
            
            self.video_writer.write(processed_frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        # Close video writer
        if self.video_writer:
            self.video_writer.release()
        
        # Save statistics
        self._save_statistics()
        
        print(f"Processing complete! Total frames: {frame_count}")
        print(f"Output saved to {self.output_video_path} and {self.output_stats_path}")

    def _save_statistics(self):
        """Save detailed route and vehicle type statistics to file"""
        with open(self.output_stats_path, 'w') as f:
            f.write("===== TRAFFIC FLOW STATISTICS =====\n\n")
            
            # Write summary of vehicle types
            f.write("VEHICLE TYPE SUMMARY\n")
            f.write("====================\n")
            
            # Calculate total vehicles by type
            vehicle_totals = defaultdict(int)
            for route_data in self.vehicle_type_counts.values():
                for v_type, count in route_data.items():
                    vehicle_totals[v_type] += count
            
            for v_type, count in vehicle_totals.items():
                f.write(f"{v_type}s: {count} vehicles\n")
            
            f.write(f"\nTotal Vehicles: {sum(vehicle_totals.values())}\n")
            f.write("\n===========================\n\n")
            
            # Write detailed route statistics with vehicle types
            f.write("DETAILED ROUTE STATISTICS\n")
            f.write("========================\n\n")
            
            for route, total in self.route_counts.items():
                if total > 0:
                    f.write(f"Route: {route} - Total: {total} vehicles\n")
                    f.write("Vehicle Types:\n")
                    
                    for v_type, count in self.vehicle_type_counts[route].items():
                        if count > 0:
                            percentage = (count / total) * 100 if total > 0 else 0
                            f.write(f"  - {v_type}s: {count} ({percentage:.1f}%)\n")
                    
                    f.write("\n")
            
            # Summary by entry point with vehicle types
            f.write("TRAFFIC BY ENTRY POINT\n")
            f.write("=====================\n\n")
            
            entry_vehicle_types = defaultdict(lambda: defaultdict(int))
            entry_totals = defaultdict(int)
            
            for route, route_totals in self.vehicle_type_counts.items():
                entry = route.split(" to ")[0]
                for v_type, count in route_totals.items():
                    entry_vehicle_types[entry][v_type] += count
                    entry_totals[entry] += count
            
            for entry, total in entry_totals.items():
                if total > 0:
                    f.write(f"Entry Point: {entry} - Total: {total} vehicles\n")
                    f.write("Vehicle Types:\n")
                    
                    for v_type, count in entry_vehicle_types[entry].items():
                        if count > 0:
                            percentage = (count / total) * 100 if total > 0 else 0
                            f.write(f"  - {v_type}s: {count} ({percentage:.1f}%)\n")
                    
                    f.write("\n")
            
            # Summary by exit point with vehicle types
            f.write("TRAFFIC BY EXIT POINT\n")
            f.write("====================\n\n")
            
            exit_vehicle_types = defaultdict(lambda: defaultdict(int))
            exit_totals = defaultdict(int)
            
            for route, route_totals in self.vehicle_type_counts.items():
                exit = route.split(" to ")[1]
                for v_type, count in route_totals.items():
                    exit_vehicle_types[exit][v_type] += count
                    exit_totals[exit] += count
            
            for exit, total in exit_totals.items():
                if total > 0:
                    f.write(f"Exit Point: {exit} - Total: {total} vehicles\n")
                    f.write("Vehicle Types:\n")
                    
                    for v_type, count in exit_vehicle_types[exit].items():
                        if count > 0:
                            percentage = (count / total) * 100 if total > 0 else 0
                            f.write(f"  - {v_type}s: {count} ({percentage:.1f}%)\n")
                    
                    f.write("\n")


# Main execution
if __name__ == "__main__":
    # Path to your video file
    video_path = "video.mp4"  # Update with your actual path
    
    # Initialize and run traffic monitor
    monitor = TrafficMonitor(video_path, model_path='yolov8s.pt')
    monitor.process_video()