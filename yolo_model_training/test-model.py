import cv2
from ultralytics import YOLO

model = YOLO('best (8).pt')

video_path = './2026oncmp1_sf3m1.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()

while cap.isOpened():
    for _ in range(10):
        cap.read()

    success, frame = cap.read()
    
    if success:
        results = model.predict(frame, conf=0.2, iou=0.4, verbose=False)

        annotated_frame = results[0].plot(labels=False, line_width=1, font_size=6)

        cv2.imshow("Scouting", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.waitKey(0)
    else:
        break

cap.release()
cv2.destroyAllWindows()