import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

video_path = 'video3.mp4' 
frameNum = 579
cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)

success, frame = cap.read()

results = model.predict(frame, conf=0.2, iou=0.1, verbose=False)

annotated_frame = results[0].plot()

cv2.imshow("Scouting", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()