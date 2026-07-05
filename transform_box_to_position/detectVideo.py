import cv2
from ultralytics import YOLO
import numpy as np
import json
import os

class FRCFieldScouter:
    def __init__(self, model_path='best1.pt', video_path='video4.mp4', 
                 map_image_path='full_field.png', points_path='transferPoints.txt',
                 start_seconds=5, end_seconds=175, frame_jump=5, 
                 conf_thresh=0.5, iou_thresh=0.4):
        
        self.video_path = video_path
        self.frame_jump = frame_jump
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # 1. Initialize the YOLO model
        self.model = YOLO(model_path)
        
        # 2. Setup and process the 2D field map image
        self.map_img = cv2.imread(map_image_path)
        if self.map_img is None:
            raise FileNotFoundError(f"Could not load map image from: {map_image_path}")
        self.map_img = cv2.rotate(self.map_img, cv2.ROTATE_180)
        
        # 3. Initialize Video Capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")
            
        self.vid_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # 4. Handle start and end bounds tracking
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        
        self.count = int(self.video_fps * self.start_seconds)
        self.end_frame = int(self.video_fps * self.end_seconds)
        
        # Fast-forward the stream to our initial starting point
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.count)
        
        # 5. Load and compute the Homography matrix
        self.H = self._compute_homography(points_path)

    def _compute_homography(self, points_path):
        if not os.path.exists(points_path):
            raise FileNotFoundError(f"Points calibration file missing: {points_path}")
            
        with open(points_path, "r") as f:
            lines = f.read().strip().split("\n")

        pts_src = json.loads(lines[0])
        pts_dst = json.loads(lines[1])

        src_arr = np.array(pts_src, dtype=np.float32)
        dst_arr = np.array(pts_dst, dtype=np.float32)
            
        H, _ = cv2.findHomography(src_arr, dst_arr)
        return H

    def transform_box_to_position(self, boxes):
        result = []
        for box in boxes:

            x1, y1, x2, y2 = box
            xc = (x1 + x2) / 2
            bh = y2 - y1
            
            #y is 75% the way down the box (halway between the bottom and the center)
            target_y = y1 + (bh * 0.75) 
            
            target_pixel = np.array([[[xc, target_y]]], dtype='float32')
            map_pt = cv2.perspectiveTransform(target_pixel, self.H)[0][0]
            result.append([int(map_pt[0]), int(map_pt[1])])
        return result

    def run(self):        
        while self.cap.isOpened():
            if self.count >= self.end_frame:
                print("Done")
                break

            success, frame = self.cap.read()
            if not success: 
                print("End of video stream reached.")
                break
            
            new_map_img = self.map_img.copy()
            last_known_map_points = []
            last_known_video_boxes = []
            last_known_confidences = []

            ai_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            predictions = self.model.predict(
                ai_frame, 
                conf=self.conf_thresh, 
                iou=self.iou_thresh, 
                stream=True, 
                verbose=False
            )
            
            for result in predictions:
                boxes = result.boxes.xyxy.cpu().numpy().copy()
                confidences = result.boxes.conf.cpu().numpy().copy()
                
                if boxes.size > 0:
                    last_known_video_boxes = boxes 
                    last_known_confidences = confidences

                    map_points = self.transform_box_to_position(boxes)
                    last_known_map_points = map_points
            
            for box, conf in zip(last_known_video_boxes, last_known_confidences):
                x1, y1, x2, y2 = map(int, box) 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for map_point in last_known_map_points:
                cv2.circle(new_map_img, (map_point[0], map_point[1]), 10, (255, 0, 255), -1)

            cv2.imshow("Scouting", frame)
            cv2.imshow("Positions", new_map_img)

            should_exit = False
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == 32:  #spacebar
                    for _ in range(self.frame_jump - 1):
                        self.cap.read()
                        
                    self.count += self.frame_jump
                    break 
                    
                elif key == ord("q"):
                    should_exit = True
                    break
            
            if should_exit:
                break

        self.cap.release()
        cv2.destroyAllWindows()

scouter = FRCFieldScouter(
    model_path='best1.pt',
    video_path='video4.mp4',
    map_image_path='full_field.png',
    points_path='transferPoints.txt',
    start_seconds=6,      
    end_seconds=23,      
    frame_jump=3,
    conf_thresh=0.5,
    iou_thresh=0.4
)
scouter.run()