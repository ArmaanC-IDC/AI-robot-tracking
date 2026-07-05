import os
import shutil

source_dir = "frc_reid_dataset/bounding_box_train" 
target_dir = "./sorted_frames"

files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

for file in files:
    team_id = file.split('_')[0]
    
    team_folder = os.path.join(target_dir, team_id)
    os.makedirs(team_folder, exist_ok=True)
    
    src_path = os.path.join(source_dir, file)
    dst_path = os.path.join(team_folder, file)
    
    shutil.copy2(src_path, dst_path)