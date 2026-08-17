import os
import shutil

folders = ["./val", "./train"]

end_loc = "./all_data"

os.makedirs(end_loc, exist_ok=True)

for dir in folders:
    if not os.path.exists(dir):
        continue

    for subdir in os.listdir(dir):
        if not os.path.isdir(os.path.join(dir, subdir)):
            continue

        files = os.listdir(os.path.join(dir, subdir))

        os.makedirs(os.path.join(end_loc, subdir), exist_ok=True)

        print(f"copying {subdir} from {dir}")

        num_files_in_dir = len(os.listdir(os.path.join(end_loc, subdir)))

        for file in files:
            if os.path.isfile(os.path.join(dir, subdir, file)):
                shutil.copy(os.path.join(dir, subdir, file), os.path.join(end_loc, subdir, f"{num_files_in_dir}.jpg"))
                num_files_in_dir += 1