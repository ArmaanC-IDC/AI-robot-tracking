from ultralytics import YOLO
import cv2
import numpy as np
import os
import ast

videoPath = "./2026cmptx_sf4m1.mp4"
cap = cv2.VideoCapture(videoPath)
video_fps = cap.get(cv2.CAP_PROP_FPS)

start_seconds = 65
frame_jump = 30

model_path = '../../best.pt'
conf = 0.4
iom = 0.5
model = YOLO(model_path)

teams = [4065, 4414, 1323, 6324, 4946, 2337]
num_img_per_team = []

# with open('transferPoints.txt', 'r') as f:
#     lines = f.readlines()
#     roi_coords = ast.literal_eval(lines[2].strip())
#     roi_polygon = np.array(roi_coords, dtype=np.int32)

for i in range(len(teams)):
    folder_name = f"{teams[i]}_{'r' if 0 <= i <= 2 else 'b'}"
    os.makedirs(f"./new_frames/{folder_name}", exist_ok=True)
    file_count = len([f for f in os.listdir(f"./new_frames/{folder_name}") if os.path.isfile(os.path.join(f"./new_frames/{folder_name}", f))])
    num_img_per_team.append(file_count)

for _ in range(int(video_fps * start_seconds)):
    cap.read()

should_continue = True
while should_continue:
    print("New Frame: ")
    success, frame = cap.read()
    if not success: 
        print("End of video stream reached.")
        break

    predictions = model.predict(
        frame, 
        conf=float(conf), 
        verbose=False,
        agnostic_nms=True
    )

    boxes = predictions[0].boxes.xyxy.cpu().numpy()
    del predictions
    for box in boxes:
        x1, y1, x2, y2 = box

        h, w = frame.shape[:2]

        cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
        # if cv2.pointPolygonTest(roi_polygon, (cx, cy), measureDist=False) >= 0 and y2 > y1 and x2 > x1:
        if y2 > y1 and x2 > x1:

            box_h = y2 - y1
            box_w = x2 - x1

            #0.1 bcs 10% margin on each side
            margin_y = int(box_h * 0.1)
            margin_x = int(box_w * 0.1)

            y1, y2 = int(max(0, y1 - margin_y)), int(min(h, y2 + margin_y))
            x1, x2 = int(max(0, x1 - margin_x)), int(min(w, x2 + margin_x))

            crop = frame[y1:y2, x1:x2]

            resized_crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)

            labeled_crop = cv2.copyMakeBorder(
                resized_crop, 0, 100, 0, 500, #50 padding
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            text = " | ".join([str(t) for t in teams])
            text1 = " | ".join([str(n) for n in num_img_per_team])
            
            cv2.putText(
                labeled_crop, text, (10, resized_crop.shape[0] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(
                labeled_crop, text1, (10, resized_crop.shape[0] + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Capture", labeled_crop)

            key = cv2.waitKey(0) & 0xFF 
            if key == ord('q'):
                print("The 'q' key was pressed. Exiting...")
                should_continue = False
                break
            elif ord('1') <= key <= ord('6'):
                number = key - ord('0')
                team = teams[number - 1]
                cv2.imwrite(f"./new_frames/{team}_{'r' if ord('1') <= key <= ord('3') else 'b'}/{num_img_per_team[number - 1]}.jpg", crop)
                print(f"Saved image: ./new_frames/{team}/{num_img_per_team[number - 1]}.jpg")
                num_img_per_team[number - 1] += 1

        for _ in range(frame_jump):
            cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break