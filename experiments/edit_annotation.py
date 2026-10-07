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
    - Left-click on empty space: Adds a new point.
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

        # Otherwise, add a new point. Use max existing ID + 1, or 0 if empty.
        if annotations_current_frame:
            max_id = max(int(ann["id"]) for ann in annotations_current_frame)
            new_id = int(max_id) + 1
        else:
            new_id = 0

        annotations_current_frame.append({
            "id": new_id,
            "map_point": [x, y]
        })
        selected_point_idx = len(annotations_current_frame) - 1
        print(f"Added point ID '{new_id}' at ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if not annotations_current_frame:
            return
        
        pts = np.array([ann["map_point"] for ann in annotations_current_frame])
        click_pt = np.array([x, y])
        distances = np.linalg.norm(pts - click_pt, axis=1)
        closest_idx = np.argmin(distances)
        
        if distances[closest_idx] < 15:
            removed = annotations_current_frame.pop(closest_idx)
            print(f"Deleted point ID '{removed['id']}' at {removed['map_point']}")
            if selected_point_idx == closest_idx:
                selected_point_idx = None
            elif selected_point_idx is not None and selected_point_idx > closest_idx:
                selected_point_idx -= 1
        else:
            print("Right-click: No point close enough to delete.")

def change_id_globally(all_frames_dict, target_old_id, new_id, from_frame=None):
    """Changes an ID across all frames or from a specific frame onward."""
    count = 0
    for frame_idx, data in all_frames_dict.items():
        if from_frame is not None and frame_idx < from_frame:
            continue
        for det in data["detections"]:
            if det["id"] == target_old_id:
                det["id"] = new_id
                count += 1
    print(f"Updated ID {target_old_id} -> {new_id} in {count} instances.")

def edit_annotations(video_path, map_path, input_json, output_json):
    global annotations_current_frame, selected_point_idx

    if not os.path.exists(video_path):
        print(f"Error: Video path '{video_path}' not found.")
        return

    if not os.path.exists(map_path):
        print(f"Error: Map image path '{map_path}' not found.")
        return

    if not os.path.exists(input_json):
        print(f"Error: Input JSON path '{input_json}' not found.")
        return

    # Load input JSON into a dictionary keyed by frame index
    with open(input_json, "r") as f:
        json_data = json.load(f)

    # Convert list format to dictionary for easy random access/editing: {frame_idx: {"time_seconds": t, "detections": [...]}}
    all_frames_dict = {}
    for entry in json_data:
        frame_idx = entry["frame"]
        all_frames_dict[frame_idx] = {
            "time_seconds": entry.get("time_seconds", 0.0),
            "detections": entry.get("detections", [])
        }

    if not all_frames_dict:
        print("Error: Input JSON contains no frame entries.")
        return

    sorted_frames = sorted(all_frames_dict.keys())
    current_idx_pointer = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    map_img = cv2.imread(map_path)
    if map_img is None:
        print(f"Error: Could not load map image from {map_path}")
        cap.release()
        return

    window_video = "Video Frame"
    window_map = "Field Map Editor"

    cv2.namedWindow(window_video)
    cv2.namedWindow(window_map)
    cv2.setMouseCallback(window_map, map_mouse_callback)

    print("\n--- Annotation Editor Instructions ---")
    print("  [Left-Click Map]  : Add point or select existing point")
    print("  [Right-Click Map] : Delete closest point")
    print("  [Number Keys 0-9] : Change selected point's ID for current frame")
    print("  [g]               : Change selected point's ID globally (across all frames)")
    print("  [Space / Enter]   : Save current frame & go to NEXT frame")
    print("  [p]               : Go to PREVIOUS frame")
    print("  [d]               : Delete/clear current frame annotations")
    print("  [q]               : Quit and save all changes to output file\n")

    should_exit = False

    while 0 <= current_idx_pointer < len(sorted_frames):
        current_frame_idx = sorted_frames[current_idx_pointer]
        frame_data = all_frames_dict[current_frame_idx]
        frame_timestamp = frame_data["time_seconds"]
        
        # Load current frame annotations into global state copy
        annotations_current_frame = [dict(d) for d in frame_data["detections"]]
        selected_point_idx = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        success, frame = cap.read()
        if not success:
            print(f"Warning: Could not read frame {current_frame_idx} from video.")
            frame = np.zeros((720, 1080, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (1080, 720))

        while True:
            display_frame = frame.copy()
            display_map = map_img.copy()

            # Draw markers on map
            for idx, ann in enumerate(annotations_current_frame):
                pt = tuple(ann["map_point"])
                obj_id = ann["id"]
                color = get_color_for_id(obj_id)
                
                if idx == selected_point_idx:
                    cv2.circle(display_map, pt, 14, (0, 0, 255), 3)

                cv2.circle(display_map, pt, 8, color, -1)
                cv2.circle(display_map, pt, 10, (255, 255, 255), 2)
                cv2.putText(display_map, f"ID:{obj_id}", (pt[0] + 12, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Info Overlays
            info_text = f"Frame Index: {current_idx_pointer+1}/{len(sorted_frames)} (Frame: {current_frame_idx}) | Time: {frame_timestamp:.2f}s"
            cv2.putText(display_frame, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            status_y = 30
            cv2.putText(display_map, f"Frame: {current_frame_idx} | Points: {len(annotations_current_frame)}", (20, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if selected_point_idx is not None and 0 <= selected_point_idx < len(annotations_current_frame):
                sel_id = annotations_current_frame[selected_point_idx]['id']
                cv2.putText(display_map, f"Selected #{selected_point_idx} (ID: {sel_id}) [0-9: Edit | g: Global ID]", (20, status_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.putText(display_map, "Click point to select & edit ID", (20, status_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

            cv2.imshow(window_video, display_frame)
            cv2.imshow(window_map, display_map)

            key = cv2.waitKey(30) & 0xFF

            # Number keys (0-9) to change ID for current frame
            if ord('0') <= key <= ord('9'):
                if selected_point_idx is not None and 0 <= selected_point_idx < len(annotations_current_frame):
                    new_id = key - ord('0')
                    annotations_current_frame[selected_point_idx]['id'] = new_id
                    print(f"Frame {current_frame_idx}: Point #{selected_point_idx} ID changed to {new_id}")

            # Global ID change via 'g' key
            elif key == ord('g'):
                if selected_point_idx is not None and 0 <= selected_point_idx < len(annotations_current_frame):
                    old_id = annotations_current_frame[selected_point_idx]['id']
                    try:
                        new_id_str = input(f"\nEnter new ID to replace ALL instances of ID {old_id}: ")
                        new_id = int(new_id_str)
                        change_id_globally(all_frames_dict, old_id, new_id)
                        # Refresh current local frame view
                        annotations_current_frame = [dict(d) for d in all_frames_dict[current_frame_idx]["detections"]]
                    except ValueError:
                        print("Invalid ID entered. Aborted.")

            # Spacebar or Enter: Save current frame modifications and move forward
            elif key == 13 or key == 32:
                all_frames_dict[current_frame_idx]["detections"] = list(annotations_current_frame)
                print(f"Saved frame {current_frame_idx}. Moving to next frame.")
                current_idx_pointer += 1
                break

            # 'p' key: Move to previous frame
            elif key == ord('p'):
                all_frames_dict[current_frame_idx]["detections"] = list(annotations_current_frame)
                print(f"Saved frame {current_frame_idx}. Moving to previous frame.")
                current_idx_pointer -= 1
                break

            # 'd' key: Delete current frame from dataset
            elif key == ord('d'):
                annotations_current_frame = []
                all_frames_dict[current_frame_idx]["detections"] = []
                print(f"Cleared all points for frame {current_frame_idx}.")

            # 'q' key: Quit early
            elif key == ord('q'):
                all_frames_dict[current_frame_idx]["detections"] = list(annotations_current_frame)
                print("Exiting editor loop...")
                should_exit = True
                break

        if should_exit:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Reconstruct output list sorted by frame number
    final_output = []
    for f_idx in sorted(all_frames_dict.keys()):
        if all_frames_dict[f_idx]["detections"]:  # Only include frames with active points
            final_output.append({
                "frame": f_idx,
                "time_seconds": all_frames_dict[f_idx]["time_seconds"],
                "detections": all_frames_dict[f_idx]["detections"]
            })

    with open(output_json, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"\nSuccessfully saved updated annotations to '{output_json}'.")

if __name__ == "__main__":
    VIDEO_PATH = "videos/2026iri_sf2m1.mp4"
    MAP_PATH = "assets/full_field.png"
    INPUT_JSON = "annotations/2026iri_sf2m1/ground_truth.json"
    OUTPUT_JSON = "annotations/2026iri_sf2m1/ground_truth.json"
    
    edit_annotations(VIDEO_PATH, MAP_PATH, INPUT_JSON, OUTPUT_JSON)