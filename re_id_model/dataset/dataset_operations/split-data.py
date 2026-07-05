import os
import shutil
import random

start = "./all_data"

end_locs = ["./train", "./val"]
split = [0.9, 0.1]

for team_folder in os.listdir(start):
    for end in end_locs:
        os.makedirs(os.path.join(end, team_folder))
    
    for file in os.listdir(os.path.join(start, team_folder)):
        rand = random.random()

        sum = 0
        for i in range(len(end_locs)):
            sum += split[i]
            if rand < sum:
                num_files_in_dir = len(os.listdir(os.path.join(end_locs[i], team_folder)))

                FILENAME = f"{num_files_in_dir}.jpg"
                shutil.copy(os.path.join(start, team_folder, file), os.path.join(end_locs[i], team_folder, FILENAME))

                break

    # for subdir in os.listdir(dir):
    #     if not os.path.isdir(os.path.join(dir, subdir)):
    #         continue

    #     files = os.listdir(os.path.join(dir, subdir))

    #     os.makedirs(os.path.join(end_loc, subdir), exist_ok=True)

    #     print(f"copying {subdir} from {dir}")

    #     num_files_in_dir = len(os.listdir(os.path.join(end_loc, subdir)))

    #     for file in files:
    #         if os.path.isfile(os.path.join(dir, subdir, file)):
    #             shutil.copy(os.path.join(dir, subdir, file), os.path.join(end_loc, subdir, f"{num_files_in_dir}.jpg"))
    #             num_files_in_dir += 1