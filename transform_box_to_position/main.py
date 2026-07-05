from frameToPoints import FrameToPoints
from ultralytics import YOLO
import cv2

videoPath = "./video4.mp4"
cap = cv2.VideoCapture(videoPath)

map_img = cv2.imread("full_field.png")

start_seconds = 4
end_seconds = 24
frame_jump = 5

model_path = 'best1.pt'
conf=0.5
iou=0.4
model = YOLO(model_path)

scouter = FrameToPoints(
    model_path=model_path,
    points_path='transferPoints.txt',
    conf_thresh=conf,
    iou_thresh=iou
)

video_fps = cap.get(cv2.CAP_PROP_FPS)
count = int(video_fps * start_seconds)
end_frame = int(video_fps * end_seconds)

cap.set(cv2.CAP_PROP_POS_FRAMES, count)

count = 0

while cap.isOpened():
    if count >= end_frame:
        print("Done")
        break

    success, frame = cap.read()
    if not success: 
        print("End of video stream reached.")
        break
    
    new_map_img = map_img.copy()
    last_known_map_points = []
    last_known_video_boxes = []
    last_known_confidences = []

    ai_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    predictions = model.predict(
        ai_frame, 
        conf=float(0.5), 
        iou=float(conf), 
        stream=True, 
        verbose=False
    )
    
    for result in predictions:
        boxes = result.boxes.xyxy.cpu().numpy().copy()
        confidences = result.boxes.conf.cpu().numpy().copy()
        
        if boxes.size > 0:
            last_known_video_boxes = boxes 
            last_known_confidences = confidences

            map_points = scouter.transform_box_to_position(boxes)
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