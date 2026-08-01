#TODO: Remove displaying and saving images

import cv2
import numpy as np
import sys
import os
from scipy.optimize import linear_sum_assignment

video_path = './2026cmptx_sf4m1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS)==0:
    print("Error: Could not open video.")
    sys.exit()

from helper_scripts.build_model import build_embedding_model
from helper_scripts.frame_to_points import FrameToPoints
from helper_scripts.track import Track
from ultralytics import YOLO

map_img = cv2.imread("assets/full_field.png")

start_seconds = 9
end_seconds = 128
frame_jump = 5

yolo_model_path = '../best.pt'
yolo_conf=0.4
yolo_iom=0.5
yolo_model = YOLO(yolo_model_path)
MODEL_FILEPATH = "./model.weights.h5"
re_id_model = build_embedding_model(MODEL_FILEPATH)
image_size = (128, 128)

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

def process_image(img, image_size=(128, 128)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ((img / 127.5) - 1.0).astype(np.float32)
    return img

for _ in range(count):
    cap.read()

current_tracks = []
track_colors = [
    (0, 255, 0),    # Green
    (255, 255, 0),  # Cyan
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
    
    #new map image to annotate (so as not to corrupt the original)
    new_map_img = map_img.copy()

    #map points is an array of points that robots occupy on the game map
    map_points = []

    #array of boxes
    video_boxes = []

    #array of confidences from the YOLO model where confidences[i] corresponds to the box at video_boxes[i]
    confidences = []
    
    predictions = yolo_model.predict(
        frame, 
        conf=float(yolo_conf), 
        verbose=False,
        agnostic_nms=True
    )
 
    result = predictions[0]
    boxes = result.boxes.xyxy.cpu().numpy().copy()
    confidences = result.boxes.conf.cpu().numpy().copy()

    #fallback in case IoM returns no indices
    video_boxes = boxes

    #list of indices that are valid boxes (do not overlap too much)
    indices = IoM(boxes, confidences.tolist(), yolo_iom)
    
    if len(indices) > 0:
        indices = np.array(indices).flatten()
        
        video_boxes = boxes[indices].reshape(-1, 4)
        confidences = confidences[indices]

    final_boxes, final_confidences = scouter.get_boxes_in_boundry(video_boxes, confidences)
    map_points = scouter.transform_box_to_position(final_boxes)

    #initialize tracks
    if count==int(video_fps * start_seconds):
        crops = np.array([cv2.resize(frame[y1:y2, x1:x2], image_size) for x1, y1, x2, y2 in np.array(final_boxes).astype(int)])
        embeddings = re_id_model.predict(np.array([process_image(crop) for crop in crops]), verbose=1)
        current_tracks = [Track(
            track_colors[i%len(track_colors)], 
            [embeddings[i]],
            i,
            video_fps
        ) for i in range(len(map_points))]

        for i in range(len(map_points)): 
            current_tracks[i].add_point(map_points[i], count, True, embeddings[i], crops[i])

    #update tracks----------------------------------------
    else:
        #region STEP 1: associate unambiguous cases
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
        
        #endregion

        #get unused points, tracks, and boxes (not already assigned in previous step)
        print(map_points)
        print(used_map_points)
        unused_map_points = [map_points[i] for i in range(len(map_points)) if i not in used_map_points]
        unused_tracks = [current_tracks[i] for i in range(len(current_tracks)) if i not in used_tracks]
        unused_boxes = [final_boxes[i] for i in range(len(map_points)) if i not in used_map_points]

        #region Visual association

        if len(unused_map_points) > 0 and len(unused_tracks) > 0:
            #new track/point claims with only unused points and tracks
            track_claims = {i: [] for i in range(len(unused_tracks))}
            point_claims = {i: [] for i in range(len(unused_map_points))}
            #get all points in range of all Tracks
            for i in range(len(unused_tracks)):
                track = unused_tracks[i]
                for j in range(len(unused_map_points)):
                    #if point in range
                    if np.linalg.norm(unused_map_points[j] - track.get_next_point(count)) < track.get_max_dist(count):
                        track_claims[i].append(j)
                        point_claims[j].append(i)

            #Build a matrix where  rows = new embeddings, cols = existing tracks
            num_points = len(unused_map_points)
            num_tracks = len(unused_tracks)
            cost_matrix = np.full((num_points, num_tracks), 1e9)

            crops = np.array([cv2.resize(frame[y1:y2, x1:x2], image_size) for x1, y1, x2, y2 in np.array(unused_boxes).astype(int)])
            embeddings = re_id_model.predict(np.array([process_image(crop) for crop in crops]), verbose=1)

            for e_idx, embedding in enumerate(embeddings):
                for t_idx, track in enumerate(unused_tracks):
                    if e_idx in track_claims[t_idx] and t_idx in point_claims[e_idx]:
                        min_dist = track.get_distance(embedding)   
                        cost_matrix[e_idx, t_idx] = min_dist

            #expand the cost matrix to have dummy rows and columns
            h_dummy_block = np.full((num_points, num_tracks), 50)
            v_dummy_block = np.hstack((np.full((num_points, num_tracks), 50), np.full((num_points, num_tracks), 100)))

            cost_matrix = np.hstack((cost_matrix, h_dummy_block))
            cost_matrix = np.vstack((cost_matrix, v_dummy_block))

            #get the best assignments
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_embeddings = set()

            # Apply the assignments
            for e_idx, t_idx in zip(row_ind, col_ind):
                distance = cost_matrix[e_idx, t_idx]

                assigned_embeddings.add(e_idx)

                if distance > 1.0: 
                    print(f"distance > 1.0: {distance}. Skipping")
                    continue

                print(f"Distance: {distance}")
                
                # Assign the embedding to the track
                current_tracks[t_idx].add_point(
                    map_points[e_idx], # the point to add
                    count,             # the frame number
                    True,              # include the embedding
                    embeddings[e_idx], # the embedding to include
                    crops[e_idx]       # crop image
                )
        
            #endregion

            # new tracks for new embeddings
            unassigned_embeddings = set(range(len(unused_map_points))) - assigned_embeddings

            for e_idx in unassigned_embeddings:
                new_track_idx = len(current_tracks)
                
                current_tracks.append(Track(
                    track_colors[new_track_idx % len(track_colors)],
                    [embeddings[e_idx]],
                    new_track_idx
                ))

    #draw boxes
    for box, conf1 in zip(final_boxes, final_confidences):
        x1, y1, x2, y2 = np.array(box).astype(int) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{conf1:.2f}"
        cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    #draw map points
    for track in current_tracks:
        for pt in track.get_points():
            cv2.circle(new_map_img, (pt[0], pt[1]), 4, track.color, -1)
    
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