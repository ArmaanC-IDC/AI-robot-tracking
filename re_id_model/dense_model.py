import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, RandomFlip, RandomRotation, RandomBrightness, RandomZoom
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import random
import cv2
import numpy as np
import os

augmenter = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomBrightness(0.2),
    RandomZoom(0.1)
])

filepath = "train/train4"

image_shape = (128, 128, 6)

inputs = Input(image_shape)

augmented = augmenter(inputs)

adapter = Conv2D(3, (1, 1), activation="linear")(augmented)

mobile_net = MobileNetV2(weights="imagenet", include_top=False)
for layer in mobile_net.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True 
    else:
        layer.trainable = False

global_avg_pooling = GlobalAveragePooling2D()(mobile_net(adapter))

dense1 = Dense(512, activation="relu", name="dense1")(global_avg_pooling)

dense2 = Dense(1, activation="sigmoid")(dense1)

model = Model(inputs=inputs, outputs=dense2)

class DataGen(tf.keras.utils.Sequence):
    def __init__(self, main_folder, batch_size=32, image_size=(128, 128)):
        self.batch_size = batch_size
        self.image_size = image_size
        self.main_folder = main_folder

        self.steps_per_epoch = 500 // self.batch_size

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
        
        self.image_per_robot = image_per_robot
        self.robots = robots
                
    
    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        data = []
        labels = []

        for _ in range(self.batch_size):
            if random.random() < 0.5:
                robot = random.choice(self.robots)
                
                img_a_path = random.choice(self.image_per_robot[robot])
                img_b_path = random.choice(self.image_per_robot[robot])

                label = 1.0
            else:
                robot_a = random.choice(self.robots)
                robot_b = random.choice(self.robots)
                while robot_a == robot_b:
                    robot_b = random.choice(self.robots)

                img_a_path = random.choice(self.image_per_robot[robot_a])

                img_b_path = random.choice(self.image_per_robot[robot_b])

                label = 0.0
            
            img_a = self.load_image(img_a_path)
            img_b = self.load_image(img_b_path)

            data.append(np.concatenate([img_a, img_b], -1))
            labels.append(label)
        
        return np.array(data), np.array(labels)
            
    
    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        img = (img / 127.5) - 1.0
        return img

csv_logger = CSVLogger(filepath + '/results.csv', append=True)

checkpoint = ModelCheckpoint(
    filepath=filepath + '/model.keras', 
    monitor='loss',
    save_best_only=True,
    verbose=1
)

model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
data = DataGen(main_folder="sorted_frames")
val_generator = DataGen(main_folder="test_data")
model.fit(data, epochs=20, validation_data=val_generator, callbacks=[csv_logger, checkpoint])