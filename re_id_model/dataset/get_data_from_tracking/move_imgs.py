import os
import shutil

# start_dirs = ['../new_frames/0', '../new_frames/1', '../new_frames/2', '../new_frames/3', '../new_frames/4', '../new_frames/5', '../new_frames/6']
# end_dirs = ['../new_frames/2024_8729_b', '../new_frames/2024_9785_r', '../new_frames/2024_4039_b', '../new_frames/2024_4476_b', '../new_frames/2024_2056_r', '../new_frames/2024_4678_r', '../new_frames/2024_2056_b']

start_dirs = ['../new_frames/2024_2056_b']
end_dirs = ['../new_frames/2024_2056_r']

for src_dir, dest_dir in zip(start_dirs, end_dirs):
    # Skip if the source directory doesn't exist
    if not os.path.exists(src_dir):
        print(f"Skipping {src_dir}: Directory does not exist.")
        continue

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Find the current highest numbered .jpg in the destination directory
    existing_jpgs = [f for f in os.listdir(dest_dir) if f.lower().endswith('.jpg')]
    
    if existing_jpgs:
        # Extract the integer part before '.jpg' to find the true maximum
        highest_idx = max(int(f.split('.')[0]) for f in existing_jpgs if f.split('.')[0].isdigit())
        next_idx = highest_idx + 1
    else:
        next_idx = 0

    # Gather all .jpg files from the source directory
    src_jpgs = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpg')]
    
    # Sort them so they are moved in a predictable order
    for filename in sorted(src_jpgs):
        src_path = os.path.join(src_dir, filename)
        
        new_filename = f"{next_idx}.jpg"
        dest_path = os.path.join(dest_dir, new_filename)
        
        # Move and rename the file
        shutil.move(src_path, dest_path)
        print(f"Moved: {src_path} -> {dest_path}")
        
        # Increment for the next file
        next_idx += 1