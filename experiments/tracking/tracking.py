import cv2
import numpy as np
import sys
import os
import json

from scipy.optimize import linear_sum_assignment
from helper_scripts.build_model import build_embedding_model
from helper_scripts.frame_to_points import FrameToPoints
from helper_scripts.track import Track
from ultralytics import YOLO
import time

map_img = cv2.imread("assets/full_field.png")

start_seconds = 4
end_seconds = 24
frame_jump = 8

yolo_model_path = './tracking/best_int8_openvino_model/'
yolo_model = YOLO(yolo_model_path)

MODEL_FILEPATH = "./model.weights.h5"
re_id_model = build_embedding_model(MODEL_FILEPATH)
image_size = (128, 128)
num_images_to_save_per_track = 20
save_frequency = 16
save_to_disc_frequency = 32

scouter = FrameToPoints(
    points_path='transferPoints.txt',
)

# Container to store predictions for JSON export
all_predictions = []

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

def crop_with_padding(img, x1, x2, y1, y2, pad_ratio=0.10):
    img_h, img_w = img.shape[:2]
    
    pad_w = int((x2 - x1) * pad_ratio)
    pad_h = int((y2 - y1) * pad_ratio)
    
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(img_w, x2 + pad_w)
    ny2 = min(img_h, y2 + pad_h)
    
    return img[ny1:ny2, nx1:nx2]

def get_yolo_model(frame, yolo_conf, yolo_iom):
    map_points = []
    video_boxes = []
    confidences = []
    
    predictions = yolo_model(frame, device="cpu", conf=yolo_conf)
 
    result = predictions[0]
    boxes = result.boxes.xyxy.cpu().numpy().copy()
    confidences = result.boxes.conf.cpu().numpy().copy()

    video_boxes = boxes

    indices = IoM(boxes, confidences.tolist(), yolo_iom)
    
    if len(indices) > 0:
        indices = np.array(indices).flatten()
        video_boxes = boxes[indices].reshape(-1, 4)
        confidences = confidences[indices]

    final_boxes, final_confidences = scouter.get_boxes_in_boundry(video_boxes, confidences)
    map_points = scouter.transform_box_to_position(final_boxes)

    return final_boxes, map_points, final_confidences

def update_tracks(
        current_tracks, 
        map_points, 
        final_boxes, 
        count, 
        num_frames_considered_lost, 
        frame, 
        weight_re_id, 
        max_time_since_last_detection, 
        track_colors
    ):
    global all_predictions

    used_map_points = []
    used_tracks = []

    track_claims = {i: [] for i in range(len(current_tracks))}
    point_claims = {i: [] for i in range(len(map_points))}
    
    for i in range(len(current_tracks)):
        track = current_tracks[i]
        if count - track.get_point_times()[-1] > num_frames_considered_lost: continue
        for j in range(len(map_points)):
            if np.linalg.norm(map_points[j] - track.get_next_point(count)) < track.get_max_dist(count):
                track_claims[i].append(j)
                point_claims[j].append(i)

    do_again = True
    while do_again:       
        do_again = False

        for track_idx, points_claimed in track_claims.items():
            if track_idx in used_tracks:
                continue

            track = current_tracks[track_idx]

            if len(points_claimed) == 0: continue

            if len(points_claimed) == 1 and len(point_claims[points_claimed[0]]) == 1:
                print(f"Matched {track.id} or ({track.color}) with no re-id.")
                track.add_point(map_points[points_claimed[0]], count, "no_embedding", include_embedding=False)
                used_tracks.append(track_idx)
                used_map_points.append(points_claimed[0])
                do_again = True

    unused_map_points = [map_points[i] for i in range(len(map_points)) if i not in used_map_points]
    unused_tracks = [current_tracks[i] for i in range(len(current_tracks)) if i not in used_tracks]
    unused_boxes = [final_boxes[i] for i in range(len(map_points)) if i not in used_map_points]

    if len(unused_map_points) == 0: return

    assigned_embeddings = set()
    crops = np.array(
        [
            cv2.resize(
                crop_with_padding(frame, x1, x2, y1, y2), 
                image_size
            ) 
            for x1, y1, x2, y2 in np.array(unused_boxes).astype(int)
            ]
        )
    
    embeddings = re_id_model.predict(np.array([process_image(crop) for crop in crops]), verbose=1)
    print(f"calculated {len(embeddings)} embeddings (association)")

    if len(unused_tracks) > 0:
        track_claims = {i: [] for i in range(len(unused_tracks))}
        point_claims = {i: [] for i in range(len(unused_map_points))}
        
        for i in range(len(unused_tracks)):
            track = unused_tracks[i]
            for j in range(len(unused_map_points)):
                if np.linalg.norm(unused_map_points[j] - track.get_next_point(count)) < track.get_max_dist(count):
                    track_claims[i].append(j)
                    point_claims[j].append(i)

        num_points = len(unused_map_points)
        num_tracks = len(unused_tracks)
        cost_matrix = np.full((num_points, num_tracks), 1e9)

        for e_idx, embedding in enumerate(embeddings):
            for t_idx, track in enumerate(unused_tracks):
                if e_idx in track_claims[t_idx] and t_idx in point_claims[e_idx]:
                    min_dist = track.get_distance(embedding) 
                    time_since_last_found = count - track.point_times[-1]
                    cost_matrix[e_idx, t_idx] = (min_dist * weight_re_id) + min(time_since_last_found / max_time_since_last_detection, 1) * (1 - weight_re_id)

        h_dummy_block = np.full((num_points, num_points), 1e5)
        v_dummy_block = np.hstack((np.full((num_tracks, num_tracks), 1e5), np.full((num_tracks, num_points), 0)))

        cost_matrix = np.hstack((cost_matrix, h_dummy_block))
        cost_matrix = np.vstack((cost_matrix, v_dummy_block))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for e_idx, t_idx in zip(row_ind, col_ind):
            distance = cost_matrix[e_idx, t_idx]

            if e_idx >= num_points or t_idx >= num_tracks:
                continue

            time_cost = min((count - unused_tracks[t_idx].point_times[-1]) / max_time_since_last_detection, 1)
            embedding_score = (distance - time_cost*(1-weight_re_id)) / weight_re_id

            if distance > 0.8:
                print(f"Skipped {unused_tracks[t_idx].id}. Dist: {distance:.2f}.")
                continue
            else:
                print(f"Matched {unused_tracks[t_idx].id}. Dist: {distance:.2f}.")

            assigned_embeddings.add(e_idx)
            
            unused_tracks[t_idx].add_point(
                unused_map_points[e_idx],
                count,
                "embedding",
                include_embedding=True,
                embedding=embeddings[e_idx],
                embedding_score=embedding_score,
                time_score=time_cost,
                match_score=distance,
            )

    unassigned_embeddings = set(range(len(unused_map_points))) - assigned_embeddings

    for e_idx in unassigned_embeddings:
        new_track_idx = len(current_tracks)
        
        current_tracks.append(Track(
            track_colors[new_track_idx % len(track_colors)],
            [embeddings[e_idx]],
            new_track_idx
        ))

        current_tracks[-1].add_point(unused_map_points[e_idx], count, "init", include_embedding=False)

    for track in current_tracks[:]:
        if count - track.get_point_times()[-1] > 3 * frame_jump and len(track.get_point_times()) == 1:
            current_tracks.remove(track)
            all_predictions = [
                {
                    "frame": f["frame"],
                    "time_seconds": f["time_seconds"],
                    "detections": [d for d in f["detections"] if d["id"] != str(track.id)]
                }
                for f in all_predictions
            ]
            print(f"removed track {track.id}")
        else:
            print(f"kept track {str(track.id)}")

def track(video_path, yolo_conf, output_json_path, yolo_iom=0.1):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS)==0:
        print("Error: Could not open video.")
        sys.exit()

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(video_fps * start_seconds)
    count = start_frame
    end_frame = int(video_fps * end_seconds)

    num_frames_considered_lost = 0.5 * video_fps #number of frames track can go without detections before being considered lost
    weight_re_id = 0.5 #when matching using visual features, the weight assigned to the re-id model score (rest is assigned based on time since last detection)
    max_time_since_last_detection = 3 * video_fps

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

    for _ in range(count):
        cap.read()

    while cap.isOpened():
        frame_start_time = time.perf_counter()
        print(f"New {count/video_fps:.2f}--------------------------------")
        if count >= end_frame:
            print("Done")
            break

        success, frame = cap.read()
        if not success: 
            print("End of video stream reached.")
            break
        frame = cv2.resize(frame, (1280, 720))
        
        new_map_img = map_img.copy()

        final_boxes, map_points, final_confidences = get_yolo_model(frame, yolo_conf, yolo_iom)

        if len(final_boxes) == 0: 
            print("no detections")
            continue

        if count==int(video_fps * start_seconds):
            crops = np.array(
                [
                    cv2.resize(
                        crop_with_padding(frame, x1, x2, y1, y2), 
                        image_size
                    ) 
                    for x1, y1, x2, y2 in np.array(final_boxes).astype(int)
                ]
            )
            embeddings = re_id_model.predict(np.array([process_image(crop) for crop in crops]), verbose=1)
            print(f"calculated {len(embeddings)} embeddings (initialization)")
            current_tracks = [Track(
                track_colors[i%len(track_colors)], 
                [embeddings[i]],
                i,
                video_fps
            ) for i in range(len(map_points))]

            for i in range(len(map_points)): 
                current_tracks[i].add_point(map_points[i], count, "init", True, embeddings[i])

        else:                
            update_tracks(
                current_tracks, 
                map_points, 
                final_boxes, 
                count, 
                num_frames_considered_lost, 
                frame, 
                weight_re_id, 
                max_time_since_last_detection, 
                track_colors
            )       

            if (count - start_frame) % save_frequency == 0:
                crops = []
                tracks_needing_embeddings = []
                for i in range(len(current_tracks)):
                    if len(current_tracks[i].point_times)==0:
                        continue
                    if (current_tracks[i].get_num_embeddings() < num_images_to_save_per_track 
                        and not current_tracks[i].embedding_added_on_frame(count)
                        and current_tracks[i].point_times[-1]==count
                    ):
                        pt_idx = [idx for idx, pt in enumerate(map_points) if pt is current_tracks[i].get_points()[-1]][0]
                        x1, y1, x2, y2 = map(int, final_boxes[pt_idx])
                        crops.append(cv2.resize(crop_with_padding(frame, x1, x2, y1, y2), image_size))
                        tracks_needing_embeddings.append(i)

                if len(crops) > 0:
                    embeddings = re_id_model.predict(np.array([process_image(crop) for crop in crops]), verbose=1)
                    print(f"calculated {len(embeddings)} embeddings (storing)")

                    for i, t_idx in enumerate(tracks_needing_embeddings):
                        current_tracks[t_idx].add_embedding(embeddings[i], count)

        draw_start_time = time.perf_counter()

        display_frame = frame.copy()

        for box, conf1 in zip(final_boxes, final_confidences):
            x1, y1, x2, y2 = map(int, box) 
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{conf1:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for track in current_tracks:
            for pt in track.get_points()[-10:]:
                cv2.circle(new_map_img, (pt[0], pt[1]), 10, track.color, -1)
                cv2.putText(new_map_img, str(track.id), (pt[0]-5, pt[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            point = track.get_next_point(count)
            cv2.circle(new_map_img, (int(point[0]), int(point[1])), int(track.get_max_dist(count)), track.color)

        for track in current_tracks:
            for pt in track.get_points()[-1:]:
                cv2.circle(new_map_img, (pt[0], pt[1]), 10, track.color, -1)
                cv2.putText(new_map_img, str(track.id), (pt[0]-5, pt[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5)
        
        for map_point in map_points:
            cv2.circle(new_map_img, (map_point[0], map_point[1]), 2, (0, 0, 255), -1)

        cv2.imshow("Scouting", display_frame)
        cv2.imshow("Positions", new_map_img)

        # Collect detections for the current frame to export later
        frame_detections = []
        for track in current_tracks:
            if len(track.point_times) > 0 and track.point_times[-1] == count:
                pts = track.get_points()
                if len(pts) > 0:
                    latest_pt = pts[-1]
                    frame_detections.append({
                        "id": str(track.id),
                        "map_point": [int(latest_pt[0]), int(latest_pt[1])],
                        "last_point_calc_method": track.last_point_calc_method,
                        "embedding_score": track.last_embedding_score,
                        "time_score": track.last_time_score,
                        "match_score": track.last_match_score,
                        "calculated_embedding": track.embedding_times[-1]==count
                    })

        all_predictions.append({
            "frame": count,
            "time_seconds": round(count / video_fps, 3),
            "detections": frame_detections
        })

        should_exit = False

        # while True:
        #     key = cv2.waitKey(0) & 0xFF
        #     if key == 32:  # spacebar
        #         break 
        #     elif key == ord("q"):
        #         should_exit = True
        #         print(f"Quitting at frame {count}, or {count/video_fps}s")
        #         break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            should_exit = True
            print(f"Quitting at frame {count}, or {count/video_fps}s")
            break

        if should_exit:
            break

        for _ in range(frame_jump - 1):
            cap.read()
            
        count += frame_jump

    cap.release()
    cv2.destroyAllWindows()

    # Dump collected predictions to JSON file named after the video
    with open(output_json_path, "w") as f:
        json.dump(all_predictions, f, indent=4)
    print(f"Successfully exported predictions to '{output_json_path}'.")

conf = 0.2
ioms = [0.25]
video_path = "./videos/2026mil_sf9m1.mp4"
for iom in ioms:
    track(video_path, conf, f"annotations/2026mil_sf9m1/initial.json", yolo_iom=iom)