import cv2
import json
import os
import numpy as np

# Global variables for annotation state
annotations_current_frame = []  # List of dicts: {"id": int, "map_point": [x, y]}
selected_point_idx = None

track_colors = [
    (0, 255, 0),    # Green 
    (255, 255, 0),  # Cyan 
    (0, 255, 255),  # Yellow 
    (128, 0, 0),    # Navy 
    (0, 128, 0),    # Dark Green 
    (0, 0, 128),    # Maroon 
    (128, 128, 0),  # Teal 
    (128, 0, 128),  # Purple 
    (0, 128, 128),  # Olive
]

def get_color_for_id(obj_id):
    """Fetches a consistent color for a given ID."""
    try:
        idx = int(obj_id) % len(track_colors)
        return track_colors[idx]
    except:
        return (0, 255, 0)

def map_mouse_callback(event, x, y, flags, param):
    """
    Handles mouse clicks on the field map:
    - Left-click on existing point: Selects it for editing/highlighting.
    - Left-click on empty space: Adds a new point with automatic sequential ID (0, 1, 2...).
    - Right-click: Deletes the closest point.
    """
    global annotations_current_frame, selected_point_idx
    
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pt = np.array([x, y])
        
        # Check if clicking near an existing point to select it
        if annotations_current_frame:
            pts = np.array([ann["map_point"] for ann in annotations_current_frame])
            distances = np.linalg.norm(pts - click_pt, axis=1)
            closest_idx = np.argmin(distances)
            
            if distances[closest_idx] < 15:
                selected_point_idx = closest_idx
                print(f"Selected point index {selected_point_idx} (ID: {annotations_current_frame[selected_point_idx]['id']})")
                return

        # Otherwise, add a new point with automatic ID (0, 1, 2, ...)
        new_id = len(annotations_current_frame)
        annotations_current_frame.append({
            "id": new_id,
            "map_point": [x, y]
        })
        selected_point_idx = len(annotations_current_frame) - 1
        print(f"Added point ID '{new_id}' at ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click: Delete the closest point within a 15-pixel radius
        if not annotations_current_frame:
            return
        
        pts = np.array([ann["map_point"] for ann in annotations_current_frame])
        click_pt = np.array([x, y])
        distances = np.linalg.norm(pts - click_pt, axis=1)
        closest_idx = np.argmin(distances)
        
        if distances[closest_idx] < 15:
            removed = annotations_current_frame.pop(closest_idx)
            print(f"Deleted point ID '{removed['id']}' at {removed['point']}")
            if selected_point_idx == closest_idx:
                selected_point_idx = None
            elif selected_point_idx is not None and selected_point_idx > closest_idx:
                selected_point_idx -= 1
        else:
            print("Right-click: No point close enough to delete.")

def annotate_video(video_path, map_path, start_sec, end_sec, frame_jump, output_file="annotations.json"):
    global annotations_current_frame, selected_point_idx

    if not os.path.exists(video_path):
        print(f"Error: Video path '{video_path}' not found.")
        return

    if not os.path.exists(map_path):
        print(f"Error: Map image path '{map_path}' not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    map_img = cv2.imread(map_path)
    if map_img is None:
        print(f"Error: Could not load map image from {map_path}")
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    start_frame = int(fps * start_sec)
    end_frame = int(fps * end_sec)

    print(f"Fast-forwarding to start time {start_sec}s (Frame {start_frame})...")
    for _ in range(start_frame):
        success, _ = cap.read()
        if not success:
            print("Error: Start time exceeds video duration.")
            cap.release()
            return

    current_frame_idx = start_frame
    all_annotations = []

    window_video = "Video Frame"
    window_map = "Field Map (Left-Click: Add/Select | Right-Click: Delete)"

    cv2.namedWindow(window_video)
    cv2.namedWindow(window_map)
    cv2.setMouseCallback(window_map, map_mouse_callback)

    print("\n--- Annotation Instructions ---")
    print("  [Left-Click Map]  : Add point (auto ID: 0, 1, 2...) or select existing point")
    print("  [Number Keys 0-9] : When a point is selected, press a number to change its ID in GUI")
    print("  [Right-Click Map] : Delete closest point")
    print("  [Space / Enter]   : Save current frame annotations and advance")
    print("  [n]               : Skip current frame without saving")
    print("  [q]               : Quit and save all annotations to file\n")

    should_exit = False

    while cap.isOpened() and current_frame_idx <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        success, frame = cap.read()
        if not success:
            print("End of video stream reached.")
            break

        frame = cv2.resize(frame, (1080, 720))

        frame_timestamp = current_frame_idx / fps
        annotations_current_frame = []  # Reset points for the new frame
        selected_point_idx = None

        while True:
            display_frame = frame.copy()
            display_map = map_img.copy()

            # Draw markers on the map for every annotation in this frame
            for idx, ann in enumerate(annotations_current_frame):
                pt = tuple(ann["map_point"])
                obj_id = ann["id"]
                color = get_color_for_id(obj_id)
                
                # Highlight selected point with a thicker red/white outer ring
                if idx == selected_point_idx:
                    cv2.circle(display_map, pt, 14, (0, 0, 255), 3)

                cv2.circle(display_map, pt, 8, color, -1)
                cv2.circle(display_map, pt, 10, (255, 255, 255), 2)  # White border
                cv2.putText(display_map, f"ID:{obj_id}", (pt[0] + 12, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Info Overlay on Video
            info_text = f"Frame: {current_frame_idx} | Time: {frame_timestamp:.2f}s | Points: {len(annotations_current_frame)}"
            cv2.putText(display_frame, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # GUI Status Overlay on Map
            status_y = 30
            cv2.putText(display_map, f"Frame: {current_frame_idx} ({frame_timestamp:.2f}s)", (20, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if selected_point_idx is not None and 0 <= selected_point_idx < len(annotations_current_frame):
                sel_id = annotations_current_frame[selected_point_idx]['id']
                cv2.putText(display_map, f"Selected Point #{selected_point_idx} | ID: {sel_id} (Press 0-9 to change)", (20, status_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(display_map, "Click point to select & edit ID", (20, status_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow(window_video, display_frame)
            cv2.imshow(window_map, display_map)

            key = cv2.waitKey(30) & 0xFF

            # If a number key (0-9) is pressed and a point is selected, update its ID in the GUI
            if ord('0') <= key <= ord('9'):
                if selected_point_idx is not None and 0 <= selected_point_idx < len(annotations_current_frame):
                    new_id = key - ord('0')
                    annotations_current_frame[selected_point_idx]['id'] = new_id
                    print(f"GUI Edit: Point #{selected_point_idx} ID changed to {new_id}")

            # Spacebar or Enter to finalize frame annotations
            elif key == 13 or key == 32:
                if annotations_current_frame:
                    all_annotations.append({
                        "frame": current_frame_idx,
                        "time_seconds": round(frame_timestamp, 3),
                        "detections": list(annotations_current_frame)
                    })
                    print(f"Saved {len(annotations_current_frame)} detections for Frame {current_frame_idx}.")
                else:
                    print(f"Skipped Frame {current_frame_idx} (no points added).")
                break

            # Press 'n' to skip frame without saving
            elif key == ord('n'):
                print(f"Skipped Frame {current_frame_idx}.")
                break

            # Press 'q' to quit early
            elif key == ord('q'):
                print("Exiting annotation loop...")
                should_exit = True
                break

        if should_exit:
            break

        current_frame_idx += frame_jump

    cap.release()
    cv2.destroyAllWindows()

    with open(output_file, "w") as f:
        json.dump(all_annotations, f, indent=4)
    print(f"\nSuccessfully dumped {len(all_annotations)} frame annotations to '{output_file}'.")

if __name__ == "__main__":
    VIDEO_PATH = "2026ontor_sf1m1.mp4"
    MAP_PATH = "assets/full_field.png"
    START_SECONDS = 8.0
    END_SECONDS = 28.0
    FRAME_JUMP = 30
    OUTPUT_JSON = "annotations.json"
    
    annotate_video(VIDEO_PATH, MAP_PATH, START_SECONDS, END_SECONDS, FRAME_JUMP, OUTPUT_JSON)