import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import os
import cv2
from scipy.spatial.distance import cdist

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow_similarity as tfsim
from model.data_augmentations import augmenter

model_filepath = "./siamese_train/train16"
dataset_filepath = "./dataset/new_frames"

image_shape = (128, 128, 3)

# Load the model
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=image_shape) #has 154 layers.
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.7)(x)
x = Dense(128, activation="relu")(x)
x = tf.keras.layers.UnitNormalization()(x)

embedding_model = Model(base.input, x, name="embedding")

input = Input(shape=(128, 128, 3), name="input")

augmented = augmenter(input)

embedding = embedding_model(augmented)

model = tfsim.models.SimilarityModel(input, embedding)

model.load_weights(model_filepath + "/model.weights.h5")

def load(image_path, target_size=(128, 128)):    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    
    img = ((img / 127.5) - 1.0).astype(np.float32)

    return np.expand_dims(img, axis=0)

image_paths = [os.path.join(d, f) for d in os.listdir(dataset_filepath) for f in os.listdir(os.path.join(dataset_filepath, d))]

images = np.array([load(os.path.join(dataset_filepath, path)) for path in image_paths])

labels = [path.split("\\")[0] for path in image_paths]

embeddings = embedding_model.predict(np.squeeze(np.stack(images)))

robots = list(set(labels))
r_to_id = {r: idx for idx, r in enumerate(robots)}
id_to_r = {idx: r for idx, r in enumerate(robots)}

data = np.zeros((len(robots), len(robots)), dtype=np.float32)

robot_embeddings = {robot: [] for robot in robots}
for emb, label in zip(embeddings, labels):
    robot_embeddings[label].append(emb)

robot_matrices = {robot: np.array(embs) for robot, embs in robot_embeddings.items()}

for i in range(len(robots)):
    for j in range(len(robots)):
        if i < j: continue

        robot_i = robots[i]
        robot_j = robots[j]

        mat_i = robot_matrices[robot_i]
        mat_j = robot_matrices[robot_j]

        dist_matrix = cdist(mat_i, mat_j, metric='euclidean')

        if robot_i == robot_j:
            np.fill_diagonal(dist_matrix, np.nan)
            
            avg_dist = np.nanmean(dist_matrix)
        else:
            avg_dist = np.mean(dist_matrix)
        
        data[i][j] = min(1, avg_dist) #any distance greater than 1 is considered 100% different, so all distances above 1 are treated the same
        data[j][i] = data[i][j]



# Plot simple heatmap
sns.heatmap(data, 
            xticklabels=robots,
            yticklabels=robots,
            cmap='Blues_r')

plt.xticks(rotation=90)

plt.savefig("new_heatmap.pdf")

plt.show()