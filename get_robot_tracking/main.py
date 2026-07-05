from frameToPoints import FrameToPoints
from track import Track
from ultralytics import YOLO
import cv2
import numpy as np

video_path = '../re_id_model/video-1.mp4'
cap = cv2.VideoCapture(video_path)

map_img = cv2.imread("full_field.png")

start_seconds = 99
end_seconds = 128
frame_jump = 5

model_path = '../best.pt'
conf=0.4
iom=0.5
model = YOLO(model_path)

scouter = FrameToPoints(
    points_path='transferPoints.txt',
)

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
                max(0, min(x2, xx2) - max(x1, xx1)) * 
                max(0, min(y2, yy2) - max(y1, yy1))
            )

            size2 = (xx2 - xx1) * (yy2 - yy1)

            #max of min size and 1 to never divide by 0
            overlap_percent = overlap / max(min(size1, size2), 1)

            if overlap_percent > threshhold:
                idxs[j] = -1
    
    return [i for i in idxs if i != -1]


for _ in range(count):
    cap.read()

current_tracks = []
track_colors = [
    (0, 255, 0),    # Green
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 0),    # Navy
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Maroon
    (128, 128, 0),  # Teal
    (128, 0, 128),  # Purple
    (0, 128, 128),   # Olive
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
    
    predictions = model.predict(
        frame, 
        conf=float(conf), 
        verbose=False,
        agnostic_nms=True
    )
 
    for result in predictions:
        boxes = result.boxes.xyxy.cpu().numpy().copy()
        confidences = result.boxes.conf.cpu().numpy().copy()

        if boxes.shape[0] > 0:
            indices = IoM(boxes, confidences.tolist(), iom)

            video_boxes = boxes
            
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                
                video_boxes = boxes[indices].reshape(-1, 4)
                final_confidences = confidences[indices]

            map_points = scouter.transform_box_to_position(video_boxes)

            print(f"indices: {indices}")
    
    #update the tracks
    if count==int(video_fps * start_seconds):
        current_tracks = [Track(video_fps, track_colors[i%len(track_colors)], i) for i in range(len(map_points))]

        for i in range(len(map_points)): 
            current_tracks[i].add_point(map_points[i], count)
        
        for i in range(len(current_tracks)):
            print(f"Track {i} points: {len(current_tracks[i].prev_points)}")
    
    else:
        used_map_points = []
        used_tracks = []

        track_claims = {i: [] for i in range(len(current_tracks))}
        point_claims = {i: [] for i in range(len(map_points))}

        #get all points in range of all Tracks
        for i in range(len(current_tracks)):
            track = current_tracks[i]
            for j in range(len(map_points)):
                #if point in range
                if np.linalg.norm(map_points[j] - track.get_next_point(count)) < track.get_max_dist(count):
                    track_claims[i].append(j)
                    point_claims[j].append(i)

        #keep repeating logic until all tracks with only one possible point are satisfied
        do_again = True
        while do_again:       
            do_again = False

            for track_idx, points_claimed in track_claims.items():
                if track_idx in used_tracks:
                    continue

                track = current_tracks[track_idx]

                if len(points_claimed) == 0:
                    track.mark_as_lost()
                    used_tracks.append(track_idx)

                if len(points_claimed) == 1 and len(point_claims[points_claimed[0]]) == 1:
                    track.add_point(map_points[points_claimed[0]], count)
                    used_tracks.append(track_idx)
                    used_map_points.append(points_claimed[0])
                    do_again = True
        
        for i in range(len(map_points)):
            if i in used_map_points:
                continue

            point = map_points[i]
            tracks = [current_tracks[t] for t, point_claims in track_claims.items() if i in point_claims]

            if len(tracks) > 0:
                # Find the point with the minimum distance to the track's predicted next point
                most_recent_track = min(
                    tracks, 
                    key=lambda t: count - t.last_found_frame
                )
                
                most_recent_track.add_point(point, count)
                used_tracks.append(current_tracks.index(most_recent_track))
                used_map_points.append(i)
        
        for pt in [map_points[i] for i in range(len(map_points)) if i not in used_map_points]:
            current_tracks.append(Track(video_fps, track_colors[len(current_tracks)%len(track_colors)], i))
            current_tracks[-1].add_point(pt, count)

        for i in range(len(current_tracks)):
            print(f"Track {i} points: {len(current_tracks[i].prev_points)}")


                
    
    #draw boxes
    for box, conf1 in zip(video_boxes, final_confidences):
        x1, y1, x2, y2 = map(int, box) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{conf1:.2f}"
        cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    #draw map points
    for track in current_tracks:
        for pt in track.prev_points:
            cv2.circle(new_map_img, (pt[0], pt[1]), 4, track.color, -1)
        point = track.get_next_point(count)
        print(f"point: {point}, maxDist: {track.get_max_dist(count)}")
        cv2.circle(new_map_img, (int(point[0]), int(point[1])), int(track.get_max_dist(count)), (255, 0, 0))
    
    for map_point in map_points:
        cv2.circle(new_map_img, (map_point[0], map_point[1]), 2, (0, 0, 255), -1)

    cv2.imshow("Scouting", frame)
    cv2.imshow("Positions", new_map_img)

    should_exit = False
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 32:  #spacebar
            for _ in range(frame_jump - 1):
                cap.read()
                
            count += frame_jump
            break 
            
        elif key == ord("q"):
            should_exit = True
            break
    
    if should_exit:
        break
    


cap.release()
cv2.destroyAllWindows()