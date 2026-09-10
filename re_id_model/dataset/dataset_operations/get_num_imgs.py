import os
import pprint

dirs = ["../dataset/test"]

total_imgs = 0

imgs_per_team = {}

for dir in dirs:
    for subdir in [d for d in os.listdir(dir) if os.path.exists(os.path.join(dir, d))]:
        if subdir not in imgs_per_team:
            imgs_per_team[subdir] = 0
        
        num_files = len([f for f in os.listdir(os.path.join(dir, subdir)) if f.endswith(".jpg")])
        imgs_per_team[subdir] += num_files
        total_imgs += num_files

pprint.pprint(imgs_per_team)
print(f"total dataset size: {total_imgs}")