import numpy as np
import random
import os
import cv2
import matplotlib.pyplot as plt

def get_data_generator(main_folder, batch_size=32, image_size=(128, 128)):
    
    image_per_robot = {}
    robots = []
    for robot in [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]:
        robot_folder_path = os.path.join(main_folder, robot)
        image_per_robot[robot] = [
            os.path.join(robot_folder_path, f)
            for f in os.listdir(os.path.join(main_folder, robot)) 
            if os.path.isfile(os.path.join(robot_folder_path, f))
        ]
        robots.append(robot)
    
    def generator():
        while True:


            data_a = []
            data_b = []
            labels = []

            for _ in range(batch_size):
                if random.random() < 0.5:
                    robot = random.choice(robots)
                    
                    img_a_path = random.choice(image_per_robot[robot])
                    img_b_path = random.choice(image_per_robot[robot])

                    label = 1.0
                else:
                    robot_a = random.choice(robots)
                    robot_b = random.choice(robots)
                    while robot_a == robot_b:
                        robot_b = random.choice(robots)

                    img_a_path = random.choice(image_per_robot[robot_a])

                    img_b_path = random.choice(image_per_robot[robot_b])

                    label = 0.0
                
                img_a = load_image(img_a_path, image_size)
                img_b = load_image(img_b_path, image_size)

                data_a.append(img_a)
                data_b.append(img_b)
                labels.append(label)
            
            print(labels[:5])
            
            yield (
                (np.array(data_a, dtype=np.float32), np.array(data_b, dtype=np.float32)), np.array(labels, dtype=np.float32)
            )
    
    return generator()

def load_image(path, image_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    img = ((img / 127.5) - 1.0).astype(np.float32)
    return img


folder_path = "sorted_frames"
gen = get_data_generator(folder_path, batch_size=8, image_size=(128, 128))

# 2. Pull exactly one batch from the generator using next()
(batch_a, batch_b), batch_labels = next(gen)

# 3. Setup Matplotlib figure
fig, axes = plt.subplots(8, 2, figsize=(8, 20))
plt.subplots_adjust(wspace=0.1, hspace=0.4)

for i in range(8):
    # A. The images are currently floats from -1.0 to 1.0. 
    # We must un-normalize them back to 0-255 uint8 integers to view them.
    img_a = ((batch_a[i] + 1.0) * 127.5).astype(np.uint8)
    img_b = ((batch_b[i] + 1.0) * 127.5).astype(np.uint8)

    # B. Plot Image A
    axes[i, 0].imshow(img_a)
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f"Image A (Index {i})")

    # C. Plot Image B and the Label
    axes[i, 1].imshow(img_b)
    axes[i, 1].axis('off')
    
    label_val = batch_labels[i]
    label_text = "MATCH (1.0)" if label_val == 1.0 else "DIFFERENT (0.0)"
    color = "green" if label_val == 1.0 else "red"
    
    axes[i, 1].set_title(f"Image B | {label_text}", color=color)

plt.show()