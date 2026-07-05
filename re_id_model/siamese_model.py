import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomBrightness, RandomZoom, RandomTranslation,  Lambda, Concatenate, RandomContrast
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow_similarity as tfsim
import keras
import random
import cv2
import numpy as np
import os

@keras.saving.register_keras_serializable()
def dist(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

augmenter = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.15),
    RandomBrightness(0.3),
    RandomContrast(0.3),
    RandomZoom(0.15),
    RandomTranslation(height_factor=0.15, width_factor=0.15)
])

filepath = "siamese_train/train12"

image_shape = (128, 128, 3)

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
        if len(image_per_robot[robot]) > 1:
            robots.append(robot)
    
    def generator():
        while True:


            a, p, n = [], [], []

            for _ in range(batch_size):
                robot_a = random.choice(robots)

                anchor_path, positive_path = random.sample(image_per_robot[robot_a], 2)

                robot_n = random.choice(robots)
                while robot_n==robot_a:
                    robot_n = random.choice(robots)
                
                negative_path = random.sample(image_per_robot[robot_n], 1)[0]

                a.append(load_image(anchor_path, image_size))
                p.append(load_image(positive_path, image_size))
                n.append(load_image(negative_path, image_size))
            
            dummy_labels = np.zeros((batch_size,), dtype=np.float32)
            
            yield (
                (np.array(a, dtype=np.float32), 
                 np.array(p, dtype=np.float32), 
                 np.array(n, dtype=np.float32)),
                 dummy_labels
            )
    
    return generator()
        

def load_image(path, image_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    img = ((img / 127.5) - 1.0).astype(np.float32)
    return img

@keras.saving.register_keras_serializable()
class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, label, prediction):
        d_pos = prediction[:, 0]
        d_neg = prediction[:, 1]

        loss = K.maximum(d_pos - d_neg + self.margin, 0.0)
        return K.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"margin": self.margin})
        return config

@keras.saving.register_keras_serializable()
def mean_dist_matches(label, prediction):
    return K.mean(prediction[:, 0])

@keras.saving.register_keras_serializable()
def mean_dist_non_matches(label, prediction):
    return K.mean(prediction[:, 1])

csv_logger = CSVLogger(filepath + '/results.csv', append=True)

checkpoint = ModelCheckpoint(
    filepath=filepath + '/model.keras', 
    monitor='loss',
    save_best_only=True,
    verbose=1
)

output_signature = (
    (
        tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32), 
        tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32)
    ),
    tf.TensorSpec(shape=(None,), dtype=tf.float32)
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: get_data_generator("dataset/train"),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: get_data_generator("dataset/val"),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

# loss_function = TripletLoss(margin=1.0)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=image_shape) #has 154 layers.
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = tf.keras.layers.UnitNormalization()(x)

embedding_model = Model(base.input, x, name="embedding")

input_a = Input(shape=(128, 128, 3), name="input_a")
input_p = Input(shape=(128, 128, 3), name="input_b")
input_n = Input(shape=(128, 128, 3), name="input_n")

augmented_a = augmenter(input_a)
augmented_p = augmenter(input_p)
augmented_n = augmenter(input_n)

embedding_a = embedding_model(augmented_a)
embedding_p = embedding_model(augmented_p)
embedding_n = embedding_model(augmented_n)

dist_p = Lambda(dist)([embedding_a, embedding_p])
dist_n = Lambda(dist)([embedding_a, embedding_n])

output = Concatenate(axis=1)([dist_p, dist_n])

model = Model(inputs=[input_a, input_p, input_n], outputs=output, name="final")

# model = load_model(filepath + "/model.keras", compile=False, custom_objects={
#     "dist": dist, 
# })

# embedding_model = model.get_layer("embedding")
# for layer in embedding_model.layers:
#     layer.trainable = False
# for layer in embedding_model.layers[-50:]:
#     layer.trainable = True

loss_function = tfsim.losses.TripletLoss(
    distance="l2", 
    margin=1.0, 
    mining="semi-hard"
)

model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=[mean_dist_matches, mean_dist_non_matches])
model.fit(
    train_dataset, 
    epochs=50, 
    validation_data=val_dataset, 
    callbacks=[csv_logger, checkpoint],
    steps_per_epoch=100,
    validation_steps=25,
)