import os
import shutil
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
raw_img_dir = "C:/Users/achandarana/Downloads/FRC Robot Detector.v9i.yolo26/train/images"   # Your current folder with only images
raw_lbl_dir = "C:/Users/achandarana/Downloads/FRC Robot Detector.v9i.yolo26/train/labels"   # Your current folder with only .txt files
output_root = "training_data"
split_ratio = (0.8, 0.1, 0.1) # Train, Val, Test

def organize_and_rename():
    # 1. Create the new YOLO structure
    for folder in ['train', 'val', 'test']:
        os.makedirs(f"{output_root}/{folder}/images", exist_ok=True)
        os.makedirs(f"{output_root}/{folder}/labels", exist_ok=True)

    # 2. Get image filenames and ensure matching labels exist
    image_files = sorted([f for f in os.listdir(raw_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    valid_pairs = []

    for img in image_files:
        stem = os.path.splitext(img)[0]
        label = f"{stem}.txt"
        if os.path.exists(os.path.join(raw_lbl_dir, label)):
            valid_pairs.append((img, label))
        else:
            print(f"Warning: No label found for {img}. Skipping.")

    if not valid_pairs:
        print("No valid image-label pairs found. Check your paths.")
        return

    # 3. Perform the Split (80/10/10)
    # 3. Perform the Split
    total_split = split_ratio[1] + split_ratio[2]

    if total_split == 0:
        # If val and test are 0, everything goes to train
        train_pairs = valid_pairs
        val_pairs = []
        test_pairs = []
    else:
        # Standard split logic
        train_pairs, temp_pairs = train_test_split(valid_pairs, test_size=total_split, random_state=42)
        
        # Calculate the internal ratio for the remaining temp_pairs
        if split_ratio[2] == 0:
            val_pairs = temp_pairs
            test_pairs = []
        else:
            val_pairs, test_pairs = train_test_split(temp_pairs, test_size=split_ratio[2] / total_split, random_state=42)

    # 4. Helper function to rename and move
    def move_files(pairs, set_name):
        # --- NEW LOGIC TO PREVENT OVERWRITE ---
        target_img_path = os.path.join(output_root, set_name, "images")
        
        # Count existing images in the destination folder
        existing_count = len([f for f in os.listdir(target_img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Start the counter after the existing files
        start_index = existing_count + 1
        # --------------------------------------

        for i, (img_name, lbl_name) in enumerate(pairs, start_index):
            new_stem = f"{i:04d}" # Formats to 0001, 0002... (or continues from 0051, etc)
            ext = os.path.splitext(img_name)[1]

            # Copy Image
            shutil.copy(os.path.join(raw_img_dir, img_name), 
                        os.path.join(output_root, set_name, "images", f"{new_stem}{ext}"))
            
            # Copy Label
            shutil.copy(os.path.join(raw_lbl_dir, lbl_name), 
                        os.path.join(output_root, set_name, "labels", f"{new_stem}.txt"))

    # 5. Execute
    move_files(train_pairs, "train")
    move_files(val_pairs, "val")
    move_files(test_pairs, "test")
    
    print(f"Successfully appended {len(valid_pairs)} pairs into '{output_root}'")

if __name__ == "__main__":
    organize_and_rename()