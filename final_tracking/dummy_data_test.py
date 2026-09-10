from ultralytics import YOLO
import cv2

model = YOLO("best_int8_openvino_model/")

video_path = '././2026ontor_sf1m1.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    success, frame = cap.read()

    frame = cv2.resize(frame, (1280, 720))
    results = model(frame, device="cpu")

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy().copy()
    confidences = result.boxes.conf.cpu().numpy().copy()

    display_frame = frame.copy()

    for box, conf1 in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box) 
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{conf1:.2f}"
        cv2.putText(display_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Scouting", display_frame)

    cv2.waitKey(1)
