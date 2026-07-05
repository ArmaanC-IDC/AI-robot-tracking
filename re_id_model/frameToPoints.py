import cv2
from ultralytics import YOLO
import numpy as np
import json
import os

class FrameToPoints:
    def __init__(self, points_path='transferPoints.txt'):                
        self.H = self._compute_homography(points_path)

    def _compute_homography(self, points_path):
        if not os.path.exists(points_path):
            raise FileNotFoundError(f"Points calibration file missing: {points_path}")
            
        with open(points_path, "r") as f:
            lines = f.read().strip().split("\n")

        pts_src = json.loads(lines[0])
        pts_dst = json.loads(lines[1])
        pts_bnd = json.loads(lines[2])

        src_arr = np.array(pts_src, dtype=np.float32)
        dst_arr = np.array(pts_dst, dtype=np.float32)
        self.bnd_arr = np.array(pts_bnd, dtype=np.int32)
            
        H, _ = cv2.findHomography(src_arr, dst_arr)
        return H

    def transform_box_to_position(self, boxes):
        result = []
        for box in boxes:

            x1, y1, x2, y2 = box
            xc = (x1 + x2) / 2
            bh = y2 - y1
            
            #y is 90% the way down the box (halway between the bottom and the center)
            target_y = y1 + (bh * 0.9) 
            
            target_pixel = np.array([[[xc, target_y]]], dtype='float32')
            map_pt = cv2.perspectiveTransform(target_pixel, self.H)[0][0]
            result.append(np.array([int(map_pt[0]), int(map_pt[1] * 0.95)]))
        return result