import numpy as np
import random
import os
import cv2
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import RandomFlip, RandomRotation, RandomBrightness, RandomZoom, RandomTranslation,  RandomContrast
from model.data_augmentations import augmenter
import tensorflow as tf

#note: selection method for data is different in real training

folder_path = "./dataset/train"

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
                    print(robot)
                    
                    img_a_path = random.choice(image_per_robot[robot])
                    img_b_path = random.choice(image_per_robot[robot])

                    label = 1.0
                else:
                    robot_a = random.choice(robots)
                    robot_b = random.choice(robots)
                    while robot_a == robot_b:
                        robot_b = random.choice(robots)
                    
                    print(f"{robot_a}, {robot_b}")

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

gen = get_data_generator(folder_path, batch_size=1, image_size=(128, 128))

while True:
    (batch_a, batch_b), batch_labels = next(gen)

    augmented_batch_a = augmenter(batch_a)
    augmented_batch_b = augmenter(batch_b)

    img_a = augmented_batch_a[0]
    img_b = augmented_batch_b[0]

    display_img_a = ((img_a + 1.0) * 127.5).numpy().astype(np.uint8)
    display_img_b = ((img_b + 1.0) * 127.5).numpy().astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(display_img_a, interpolation='nearest') 
    axes[1].imshow(display_img_b, interpolation='nearest') 
    plt.axis('off')
    # plt.suptitle(f"Label: {batch_labels[0]}", fontsize=16)

    plt.show()