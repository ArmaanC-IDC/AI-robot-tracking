import cv2
import json
import os
import numpy as np

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
        idx = abs(int(obj_id)) % len(track_colors)
        return track_colors[idx]
    except:
        # Fallback hash assignment if ID is a complex string
        idx = abs(hash(str(obj_id))) % len(track_colors)
        return track_colors[idx]

def view_annotations(annotations_file="annotations.json", map_path="assets/full_field.png"):
    if not os.path.exists(annotations_file):
        print(f"Error: Annotations file '{annotations_file}' not found.")
        return

    if not os.path.exists(map_path):
        print(f"Error: Map image path '{map_path}' not found.")
        return

    with open(annotations_file, "r") as f:
        annotations = json.load(f)

    if not annotations:
        print("Annotations file is empty.")
        return

    map_img = cv2.imread(map_path)
    if map_img is None:
        print(f"Error: Could not load map image from {map_path}")
        return

    window_name = "Annotation Viewer (Map)"
    cv2.namedWindow(window_name)

    print("\n--- Annotation Viewer Instructions ---")
    print("  [Spacebar] : Advance to the next annotated frame")
    print("  [q]        : Quit viewer\n")

    i = 0
    while i < len(annotations):
        entry = annotations[i]
        frame_idx = entry.get("frame", 0)
        time_sec = entry.get("time_seconds", 0.0)
        detections = entry.get("detections", [])

        display_map = map_img.copy()

        # Draw all recorded points/detections for this frame
        for det in detections:
            # Support both key naming conventions ("map_point" or "point")
            pt = det.get("map_point") or det.get("point")
            obj_id = det.get("id", "unknown")

            if pt and len(pt) == 2:
                x, y = int(pt[0]), int(pt[1])
                color = get_color_for_id(obj_id)

                # Draw solid colored circle with white outer border for readability
                cv2.circle(display_map, (x, y), 8, color, -1)
                cv2.circle(display_map, (x, y), 10, (255, 255, 255), 2)
                cv2.putText(display_map, f"ID:{obj_id}", (x + 12, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Overlay metadata on the map window
        info_text = f"Frame: {frame_idx} | Time: {time_sec:.2f}s | Objects: {len(detections)}"
        cv2.putText(display_map, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_map, "Press [Space] for next frame, [q] to quit", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(window_name, display_map)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('p'):
            i -= 2
        if key == ord('q'):
            print("Exiting annotation viewer...")
            break

        i += 1

    cv2.destroyAllWindows()
    print("Viewer session closed.")

if __name__ == "__main__":
    # Configure your paths here (matches your JSON output filename)
    ANNOTATIONS_PATH = "annotations/2026mil_sf9m1/initial.json"
    GROUND_TRUTH_PATH = "annotations/2026mil_sf9m1/initial.json"
    MAP_PATH = "assets/full_field.png"
    
    view_annotations(ANNOTATIONS_PATH, MAP_PATH)