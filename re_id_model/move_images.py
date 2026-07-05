import os
import shutil

start_dir = "./sorted_frames"
end_dir = "frc_reid_dataset/bounding_box_train" 

os.makedirs(end_dir, exist_ok=True)

subfolders = [f for f in os.listdir(start_dir) if os.path.isdir(os.path.join(start_dir, f))]

for subfolder in subfolders:
    files = [f for f in os.listdir(os.path.join(start_dir, subfolder)) if os.path.isfile(os.path.join(start_dir, subfolder,  f))]
    count = 0
    for file in files:
        full_path = os.path.join(start_dir, subfolder, file)
        new_name = f"{subfolder}_c1_{count:04d}.jpg"
        shutil.move(full_path, os.path.join(end_dir, new_name))

        count += 1