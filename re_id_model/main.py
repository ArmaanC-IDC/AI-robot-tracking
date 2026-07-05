from frameToPoints import FrameToPoints
from track import Track
from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
import ast

videoPath = "./video-1.mp4"
cap = cv2.VideoCapture(videoPath)

map_img = cv2.imread("full_field.png")

start_seconds = 87
end_seconds = 128
frame_jump = 5

model_path = '../best.pt'
conf = 0.4
iom = 0.5
model = YOLO(model_path)

scouter = FrameToPoints(
    points_path='transferPoints.txt',
)

with open('transferPoints.txt', 'r') as f:
    lines = f.readlines()
    roi_coords = ast.literal_eval(lines[2].strip())
    roi_polygon = np.array(roi_coords, dtype=np.int32)

video_fps = cap.get(cv2.CAP_PROP_FPS)
count = int(video_fps * start_seconds)
end_frame = int(video_fps * end_seconds)

def IoM(boxes, confidences, threshhold):
    idxs = np.argsort(confidences)[::-1]

    for i, idx in enumerate(idxs):
        if idx==-1: continue

        x1, y1, x2, y2 = boxes[idx]
        size1 = (x2 - x1) * (y2 - y1)
        for j in range(i+1, len(idxs), 1):
            if idxs[j] == -1: continue
            xx1, yy1, xx2, yy2 = boxes[idxs[j]]

            overlap = (
                max(0, min(x2, xx2) - max(x1, xx1)) * max(0, min(y2, yy2) - max(y1, yy1))
            )

            size2 = (xx2 - xx1) * (yy2 - yy1)

            overlap_percent = overlap / max(min(size1, size2), 1)

            if overlap_percent > threshhold:
                idxs[j] = -1
    
    return [i for i in idxs if i != -1]


# UI Function for reviewing crops in a grid
def review_crops_ui(matches, original_frame, frame_idx):
    if not matches:
        return False
        
    CELL_W, CELL_H = 250, 300
    cols = 3
    rows = max(1, math.ceil(len(matches) / cols))
    
    ui_w = cols * CELL_W
    ui_h = rows * CELL_H + 80 
    
    window_name = "Review Crops"
    cv2.namedWindow(window_name)
    
    display_crops = []
    for track, box in matches:
        x1, y1, x2, y2 = map(int, box)
        h, w = original_frame.shape[:2]

        # Calculate the dimensions of the current box
        box_h = y2 - y1
        box_w = x2 - x1

        # Calculate 10% margins
        margin_y = int(box_h * 0.1)
        margin_x = int(box_w * 0.1)

        # Apply margins and constrain to image dimensions (0, h) and (0, w)
        y1, y2 = max(0, y1 - margin_y), min(h, y2 + margin_y)
        x1, x2 = max(0, x1 - margin_x), min(w, x2 + margin_x)

        if y2 > y1 and x2 > x1:
            crop = original_frame[y1:y2, x1:x2]
        else:
            crop = np.zeros((200, 200, 3), dtype=np.uint8)
            
        # crop_resized = cv2.resize(crop, (CELL_W - 20, CELL_H - 70))
        crop_resized = crop
        display_crops.append((track, crop_resized))

    state = {'done': False, 'should_exit': False, 'accepted': set()}

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if y > rows * CELL_H:
                if x < ui_w // 2: 
                    state['accepted'] = set(t.id for t, _ in matches)
                    state['done'] = True
                else: 
                    state['done'] = True
            else:
                c = x // CELL_W
                r = y // CELL_H
                idx = r * cols + c
                if idx < len(matches):
                    local_y = y - r * CELL_H
                    if local_y > CELL_H - 50:
                        t_id = matches[idx][0].id
                        if t_id in state['accepted']:
                            state['accepted'].remove(t_id)
                        else:
                            state['accepted'].add(t_id)

    cv2.setMouseCallback(window_name, mouse_cb)
    
    while not state['done'] and not state['should_exit']:
        canvas = np.ones((ui_h, ui_w, 3), dtype=np.uint8) * 40 
        
        for i, (track, crop_img) in enumerate(display_crops):
            r = i // cols
            c = i % cols
            cx = c * CELL_W
            cy = r * CELL_H
            
            canvas[cy+10:cy+10+crop_img.shape[0], cx+10:cx+10+crop_img.shape[1]] = crop_img
            
            btn_y1 = cy + CELL_H - 50
            btn_y2 = cy + CELL_H - 10
            btn_x1 = cx + 10
            btn_x2 = cx + CELL_W - 10
            
            is_accepted = track.id in state['accepted']
            color = (0, 200, 0) if is_accepted else (100, 100, 100)
            text = "ACCEPTED" if is_accepted else "ACCEPT"
            
            cv2.rectangle(canvas, (btn_x1, btn_y1), (btn_x2, btn_y2), color, -1)
            cv2.putText(canvas, text, (btn_x1 + 65, btn_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
        cv2.rectangle(canvas, (10, rows*CELL_H + 10), (ui_w//2 - 5, ui_h - 10), (0, 150, 0), -1)
        cv2.putText(canvas, "ACCEPT ALL", (ui_w//4 - 60, rows*CELL_H + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.rectangle(canvas, (ui_w//2 + 5, rows*CELL_H + 10), (ui_w - 10, ui_h - 10), (0, 0, 150), -1)
        cv2.putText(canvas, "DONE / SKIP", (ui_w*3//4 - 60, rows*CELL_H + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow(window_name, canvas)
        
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            state['should_exit'] = True
        elif key == 32: 
            state['accepted'] = set(t.id for t, _ in matches)
            state['done'] = True
            
    cv2.destroyWindow(window_name)
    
    for track, box in matches:
        if track.id in state['accepted']:
            os.makedirs(f"new_frames/{track.id}", exist_ok=True)
            x1, y1, x2, y2 = map(int, box)
            h, w = original_frame.shape[:2]

            box_h = y2 - y1
            box_w = x2 - x1

            margin_y = int(box_h * 0.1)
            margin_x = int(box_w * 0.1)

            y1, y2 = max(0, y1 - margin_y), min(h, y2 + margin_y)
            x1, x2 = max(0, x1 - margin_x), min(w, x2 + margin_x)

            if y2 > y1 and x2 > x1:
                crop = original_frame[y1:y2, x1:x2]
                cv2.imwrite(f"./new_frames/{track.id}/{frame_idx}.jpg", crop)
                print(f"Saved image: ./new_frames/{track.id}/{frame_idx}.jpg")
                
    return state['should_exit']


for _ in range(count):
    cap.read()

current_tracks = []
track_colors = [
    (0, 255, 0),    (255, 255, 0),  (255, 0, 255),  
    (0, 255, 255),  (128, 0, 0),    (0, 128, 0),    
    (0, 0, 128),    (128, 128, 0),  (128, 0, 128),  
    (0, 128, 128),  
]

while cap.isOpened():
    if count >= end_frame:
        print("Done")
        break

    success, frame = cap.read()
    if not success: 
        print("End of video stream reached.")
        break
    
    new_map_img = map_img.copy()
    map_points = []
    video_boxes = []
    confidences = []
    current_frame_matches = []
    
    predictions = model.predict(
        frame, 
        conf=float(conf), 
        verbose=False,
        agnostic_nms=True
    )
 
    for result in predictions:
        raw_boxes = result.boxes.xyxy.cpu().numpy().copy()
        raw_confs = result.boxes.conf.cpu().numpy().copy()

        valid_boxes, valid_confs = [], []
        for b, c in zip(raw_boxes, raw_confs):
            cx, cy = float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)
            if cv2.pointPolygonTest(roi_polygon, (cx, cy), measureDist=False) >= 0:
                valid_boxes.append(b)
                valid_confs.append(c)
                
        boxes = np.array(valid_boxes) if len(valid_boxes) > 0 else np.zeros((0,4))
        filtered_confidences = np.array(valid_confs) if len(valid_confs) > 0 else np.zeros((0,))

        if boxes.shape[0] > 0:
            indices = IoM(boxes, filtered_confidences.tolist(), iom)

            video_boxes = boxes
            
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                
                video_boxes = boxes[indices].reshape(-1, 4)
                final_confidences = filtered_confidences[indices]

            map_points = scouter.transform_box_to_position(video_boxes)
    
    if count == int(video_fps * start_seconds):
        current_tracks = [Track(video_fps, track_colors[i%len(track_colors)], i) for i in range(len(map_points))]

        for i in range(len(map_points)): 
            current_tracks[i].add_point(map_points[i], count)
            current_frame_matches.append((current_tracks[i], video_boxes[i]))
        
        for i in range(len(current_tracks)):
            print(f"Track {i} points: {len(current_tracks[i].prev_points)}")
    
    else:
        used_map_points = []
        used_tracks = []

        do_again = True
        while do_again:       
            do_again = False
            track_claims = {i: [] for i in range(len(current_tracks)) if i not in used_tracks}
            point_claims = {i: [] for i in range(len(map_points)) if i not in used_map_points}

            for idx in track_claims.keys():
                track = current_tracks[idx]
                for j in range(len(map_points)):
                    if j in used_map_points:
                        continue
                    if np.linalg.norm(map_points[j] - track.get_next_point(count)) < track.get_max_dist(count):
                        track_claims[idx].append(j)
                        point_claims[j].append(idx)

            for track_idx, points_claimed in track_claims.items():
                if len(points_claimed) == 0:
                    current_tracks[track_idx].mark_as_lost()
                    used_tracks.append(track_idx)

                elif len(points_claimed) == 1 and len(point_claims[points_claimed[0]]) == 1:
                    pt_idx = points_claimed[0]
                    current_tracks[track_idx].add_point(map_points[pt_idx], count)
                    used_tracks.append(track_idx)
                    used_map_points.append(pt_idx)
                    
                    current_frame_matches.append((current_tracks[track_idx], video_boxes[pt_idx]))
                    
                    do_again = True
                    break 
        
        for i in range(len(map_points)):
            if i in used_map_points:
                continue

            point = map_points[i]
            tracks = [current_tracks[t] for t, claims in track_claims.items() if i in claims]

            if len(tracks) > 0:
                most_recent_track = min(
                    tracks, 
                    key=lambda t: count - t.last_found_frame
                )
                
                most_recent_track.add_point(point, count)
                used_tracks.append(current_tracks.index(most_recent_track))
                used_map_points.append(i)
                
                current_frame_matches.append((most_recent_track, video_boxes[i]))
        
        for i in range(len(map_points)):
            if i not in used_map_points:
                new_track = Track(video_fps, track_colors[len(current_tracks)%len(track_colors)], len(current_tracks))
                new_track.add_point(map_points[i], count)
                current_tracks.append(new_track)
                
                current_frame_matches.append((new_track, video_boxes[i]))

        for i in range(len(current_tracks)):
            print(f"Track {i} points: {len(current_tracks[i].prev_points)}")


    for box, conf1 in zip(video_boxes, final_confidences):
        x1, y1, x2, y2 = map(int, box) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{conf1:.2f}"
        cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for track in current_tracks:
        for pt in track.prev_points:
            cv2.circle(new_map_img, (int(pt[0]), int(pt[1])), 10, track.color, -1)
        point = track.get_next_point(count)
        
        if not np.isnan(point[0]) and not np.isnan(point[1]):
            max_dist = track.get_max_dist(count)
            if not np.isnan(max_dist):
                cv2.circle(new_map_img, (int(point[0]), int(point[1])), int(max_dist), (255, 0, 0))
    
    for map_point in map_points:
        cv2.circle(new_map_img, (int(map_point[0]), int(map_point[1])), 5, (0, 0, 255), -1)

    cv2.imshow("Scouting", frame)
    cv2.imshow("Positions", new_map_img)

    should_exit = review_crops_ui(current_frame_matches, frame, count)
    
    if should_exit:
        break
        
    for _ in range(frame_jump - 1):
        cap.read()
        
    count += frame_jump

cap.release()
cv2.destroyAllWindows()