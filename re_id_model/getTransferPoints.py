import cv2
import numpy as np

TARGET_FRAME = 500
video_path = './video3.mp4'
map_img = cv2.imread('full_field.png')
map_img = cv2.rotate(map_img, cv2.ROTATE_180)

#source, destination, bounding
pts_src, pts_dst, pts_bnd = [], [], []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts, display_img = param
        pts.append([x, y])
        cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Window", display_img)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
_, frame = cap.read()
_, frame2 = cap.read()
cap.release()

cv2.imshow("Window", frame)
cv2.setMouseCallback("Window", click_event, (pts_src, frame))
cv2.waitKey(0)

cv2.imshow("Window", map_img)
cv2.setMouseCallback("Window", click_event, (pts_dst, map_img))
cv2.waitKey(0)
  
cv2.imshow("Window", frame2)
cv2.setMouseCallback("Window", click_event, (pts_bnd, frame2))
cv2.waitKey(0)

with open("transferPoints.txt", "w") as f:
    f.write(f"{str(pts_src)}\n{str(pts_dst)}\n{str(pts_bnd)}")

cv2.destroyAllWindows()