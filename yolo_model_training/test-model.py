import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

video_path = 'video3.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        results = model.predict(frame, conf=0.2, iou=0.4, verbose=False)

        annotated_frame = results[0].plot()

        cv2.imshow("Scouting", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()