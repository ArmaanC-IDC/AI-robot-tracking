import os
import sys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# from model.data_augmentations import augmenter

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow_similarity as tfsim
import keras

img1_path = "./image (2).png"
img2_paths = "./img2paths"
img3_paths = "./img3paths"

image_shape = (128, 128, 3)

MODEL_FILE = "../siamese_train/train16/model.weights.h5"

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=image_shape) #has 154 layers.
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.7)(x)
x = Dense(128, activation="relu")(x)
x = tf.keras.layers.UnitNormalization()(x)

embedding_model = Model(base.input, x, name="embedding")

input = Input(shape=(128, 128, 3), name="input")

# augmented = augmenter(input)

embedding = embedding_model(input)

model = tfsim.models.SimilarityModel(input, embedding)

model.load_weights(MODEL_FILE)

@keras.saving.register_keras_serializable()
def dist(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

@keras.saving.register_keras_serializable()
class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, **kwargs): super().__init__(**kwargs)
    def call(self, y_true, y_pred): return y_pred

@keras.saving.register_keras_serializable()
def mean_dist_positive(y_true, y_pred): return y_pred
@keras.saving.register_keras_serializable()
def mean_dist_negative(y_true, y_pred): return y_pred

def preprocess_image(image_path, target_size=(128, 128)):    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    
    img = ((img / 127.5) - 1.0).astype(np.float32)

    return np.expand_dims(img, axis=0)

embedding_extractor = model.get_layer("embedding")

img1 = preprocess_image(img1_path)
emb1 = embedding_extractor.predict(img1, verbose=0)

img2_embeddings = [embedding_extractor.predict(preprocess_image(os.path.join(img2_paths, f)))
                    for f in os.listdir(img2_paths) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

img2_dists = [np.linalg.norm(emb - emb1) for emb in img2_embeddings if np.linalg.norm(emb - emb1) != 0]

img3_embeddings = [embedding_extractor.predict(preprocess_image(os.path.join(img3_paths, f)))
                    for f in os.listdir(img3_paths) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

img3_dists = [np.linalg.norm(emb - emb1) for emb in img3_embeddings if np.linalg.norm(emb - emb1) != 0]

print("image 2: match-------------")
for i in range(len(img2_dists)):
    print(f"{i}: {img2_dists[i]}")

print("image 3: non-match-----------")
for i in range(len(img3_dists)):
    print(f"{i}: {img3_dists[i]}")

print(f"Mean 2: {np.mean(img2_dists)}. Mean 3: {np.mean(img3_dists)}")
# num_matches = 0

# for emb2 in img2_embeddings:
#     dist = np.linalg.norm(emb1 - emb2)
#     print(f"dist: {dist}")

#     if dist <= 0.45:
#         num_matches += 1

# print(f"Matched with {num_matches} of the {len(img2_embeddings)} images, or {((num_matches/len(img2_embeddings))*100):.2f}%")