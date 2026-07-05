import cv2
import os
from ultralytics import YOLO

fps_to_save = 0.5
video_path = "./video3.mp4"
output_root = "./unsorted_train"
images_folder = os.path.join(output_root, "images")
labels_folder = os.path.join(output_root, "labels")
video_name = "overcharge_e1_f2"
start_time = 50
end_time = 150

os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

model = YOLO('best.pt')

def save_frames():
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if video_fps == 0:
        print("Error: Could not read video.")
        return

    interval = int(video_fps / fps_to_save)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / video_fps
        
        if frame_count % interval == 0 and start_time < current_time < end_time:
            base_filename = f"frame_{saved_count:04d}_{video_name}"
            
            img_path = os.path.join(images_folder, f"{base_filename}.jpg")
            cv2.imwrite(img_path, frame)

            results = model.predict(frame, conf=0.25, verbose=False)
            for result in results:
                label_path = os.path.join(labels_folder, f"{base_filename}.txt")
                
                with open(label_path, 'w') as f:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        coords = box.xywhn[0].tolist() 
                        f.write(f"{cls} {' '.join(map(str, coords))}\n")

            saved_count += 1
            print(f"Saved to {images_folder} and {labels_folder}: {base_filename}")

        frame_count += 1

    cap.release()
    print("Processing complete.")

save_frames()