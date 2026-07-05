import cv2
import numpy as np
import json
import math

img = cv2.imread('0001.jpg')
map_img = cv2.imread('full_field.png')
map_img = cv2.rotate(map_img, cv2.ROTATE_180)
img_h, img_w, _ = img.shape

with open("transferPoints.txt", "r") as f:
    pts_src, pts_dst, camera_pos = [json.loads(c) for c in (f.read().split("\n"))]

H, status = cv2.findHomography(np.array(pts_src), np.array(pts_dst))

with open('0001.txt', 'r') as f:
    for line in f: 
        _, xc, yc, bw, bh = map(float, line.split())
        foot_pos = [xc*img_w, (yc + bh/2)*img_h]
        center_pos = [xc * img_w, yc * img_h]

        feet = np.array([[foot_pos]], dtype='float32')
        feet_map_pt = cv2.perspectiveTransform(feet, H)[0][0]

        centers = np.array([[center_pos]], dtype='float32')
        center_map_pt = cv2.perspectiveTransform(centers, H)[0][0]

        x_dist_to_cam = camera_pos[0] / img_w - xc

        cv2.circle(map_img, 
            (int(feet_map_pt[0]), int(feet_map_pt[1])), 
            5, (255,0,255), -1)

        cv2.circle(map_img, 
            (int(center_map_pt[0]), int(center_map_pt[1])), 
            5, (255,255), -1)

        cv2.circle(map_img, 
            (
                int((center_map_pt[0] + feet_map_pt[0]) / 2),
                int((center_map_pt[1] + feet_map_pt[1]) / 2)
            ), 
            10, (0,255, 255), -1)

cv2.imshow("Done", map_img)
cv2.imshow("Image", img)
cv2.waitKey(0)