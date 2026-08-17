import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, UnitNormalization
from tensorflow.keras.models import Model
import tensorflow_similarity as tfsim

#Train got 68.8096% accuracy when measuring the mean, and 95.0930% when measuring the closest
#Validation got 59.5652% accuracy when measuring the mean, and 92.0308% when measuring the closest
#New frames got 48.9362% accuracy when measuring the mean, and 90.1639% accuracy when measuring the closest

MODEL_FILEPATH = "./siamese_train/train18/model.weights.h5"
DATASET_FILEPATH = "./dataset/new_frames"
IMAGE_SHAPE = (128, 128, 3)
BATCH_SIZE = 16
NUM_CLASSES_TO_TEST_AGAINST = 5 #test against the same class and X others
NUM_IMGS_PER_CLASS = 5

def build_embedding_model(input_shape):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.7)(x)
    x = Dense(128, activation="relu")(x)
    x = UnitNormalization()(x)

    embedding_model = Model(base.input, x, name="embedding")
    
    input_tensor = Input(shape=input_shape, name="input")
    embedding = embedding_model(input_tensor)
    
    model = tfsim.models.SimilarityModel(input_tensor, embedding)
    return model

model = build_embedding_model(IMAGE_SHAPE)
model.load_weights(MODEL_FILEPATH)

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = ((img / 127.5) - 1.0).astype(np.float32)
    return img

image_paths = []
labels = []
for root, _, files in os.walk(DATASET_FILEPATH):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            class_name = os.path.basename(root)
            image_paths.append(full_path)
            labels.append(class_name)

print(f"Found {len(image_paths)} images across {len(set(labels))} classes.")

images = np.array([load_and_preprocess_image(p, IMAGE_SHAPE[:2]) for p in image_paths])
embeddings = model.predict(images, batch_size=BATCH_SIZE, verbose=1)

class_to_indices = {}
for idx, label in enumerate(labels):
    class_to_indices.setdefault(label, []).append(idx)

all_classes = list(class_to_indices.keys())

correct_means = 0
total_means = 0

correct_mosts = 0
total_mosts = 0

for idx, base_emb in enumerate(embeddings):
    base_label = labels[idx]

    same_dists = []
    different_dists = {}

    same_indices = [i for i in class_to_indices[base_label] if i != idx]
    if len(same_indices) > 0:
        # Sample up to 5 (handles classes with fewer than 6 images safely)
        sampled_same = random.sample(same_indices, min(NUM_IMGS_PER_CLASS, len(same_indices)))
        for same_idx in sampled_same:
            dist = np.linalg.norm(base_emb - embeddings[same_idx]) # Euclidean distance
            same_dists.append(dist)
        
    other_classes = [c for c in all_classes if c != base_label]
    if len(other_classes) > 0:
        # Randomly pick 5 different classes (with replacement if < 5 other classes exist)
        sampled_diff_classes = random.sample(other_classes, NUM_CLASSES_TO_TEST_AGAINST)
        for diff_class in sampled_diff_classes:

            diff_idxs = random.sample(class_to_indices[diff_class], min(NUM_IMGS_PER_CLASS, len(class_to_indices[diff_class])))
            
            
            different_dists[diff_class] = []
            for diff_idx in diff_idxs:
                dist = np.linalg.norm(base_emb - embeddings[diff_idx])
                different_dists[diff_class].append(dist)

    #calculate if mean is correct
    same_mean = np.mean(same_dists)
    diff_means = {cls: np.mean(dists) for cls, dists in different_dists.items()}
    is_mean_correct = True
    for _, dist in diff_means.items():
        if dist < same_mean:
            is_mean_correct = False
    
    if is_mean_correct:
        correct_means += 1
    total_means += 1

    #calculate if lowest is correct
    lowest_same_idx = 0
    lowest_same_val = 3
    for idx, dist in enumerate(same_dists):
        if dist < lowest_same_val:
            lowest_same_idx = idx
            lowest_same_val = dist
    
    lowest_diff_val = 3
    for idx, dist1 in different_dists.items(): #dist1 is an array of distances, all belonging to the same class
        for dist2 in dist1:
            if dist2 < lowest_same_val:
                lowest_diff_val = dist
    
    if lowest_same_val < lowest_diff_val:
        correct_mosts += 1
    total_mosts += 1

print(f"Accuracy (measuring mean)     : {((correct_means / total_means) * 100):.4f}")
print(f"Accuracy (measuring closest)  : {((correct_mosts / total_mosts) * 100):.4f}")