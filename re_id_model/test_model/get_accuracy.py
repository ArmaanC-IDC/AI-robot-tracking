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

#With threshold 0.6
#TP, FP, FN, TN
#Train: 8122, 1833, 1774, 8181
#Val: 729, 257, 210, 949
#New Frames: 123, 107, 57, 178

MODEL_FILEPATH = "./siamese_train/train16/model.weights.h5"
DATASET_FILEPATH = "./dataset/new_frames"
IMAGE_SHAPE = (128, 128, 3)
BATCH_SIZE = 16

DISTANCE_THRESHOLD = 0.6

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

print("Loading model...")
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

print("Extracting embeddings...")
embeddings = model.predict(images, batch_size=BATCH_SIZE, verbose=1)

class_to_indices = {}
for idx, label in enumerate(labels):
    class_to_indices.setdefault(label, []).append(idx)

all_classes = list(class_to_indices.keys())

same_class_distances = []
diff_class_distances = []

same_class_correct = 0
same_class_total = 0
diff_class_correct = 0
diff_class_total = 0

print("Calculating 5-pos / 5-neg evaluation metrics...")

for idx, base_emb in enumerate(embeddings):
    base_label = labels[idx]
    
    same_indices = [i for i in class_to_indices[base_label] if i != idx]
    if len(same_indices) > 0:
        sampled_same = random.sample(same_indices, min(5, len(same_indices)))
        for same_idx in sampled_same:
            dist = np.linalg.norm(base_emb - embeddings[same_idx]) # Euclidean distance
            same_class_distances.append(dist)
            same_class_total += 1
            if dist <= DISTANCE_THRESHOLD:
                same_class_correct += 1

    other_classes = [c for c in all_classes if c != base_label]
    if len(other_classes) > 0:
        sampled_diff_classes = random.choices(other_classes, k=5)
        for diff_class in sampled_diff_classes:
            diff_idx = random.choice(class_to_indices[diff_class])
            dist = np.linalg.norm(base_emb - embeddings[diff_idx])
            diff_class_distances.append(dist)
            diff_class_total += 1
            if dist > DISTANCE_THRESHOLD:
                diff_class_correct += 1

same_acc = (same_class_correct / same_class_total) * 100 if same_class_total > 0 else 0
diff_acc = (diff_class_correct / diff_class_total) * 100 if diff_class_total > 0 else 0
total_acc = ((same_class_correct + diff_class_correct) / (same_class_total + diff_class_total)) * 100

print("\n" + "="*40)
print(f"EMBEDDING MODEL EVALUATION RESULTS")
print("="*40)
print(f"----------------------------------------")
print(f"Same-Class Accuracy     : {same_acc:.2f}% ({same_class_correct}/{same_class_total} below threshold)")
print(f"Same-Class Accuracy     : {(100 - same_acc):.2f}% ({(same_class_total - same_class_correct)}/{same_class_total} above threshold)")
print(f"----------------------------------------")
print(f"Different-Class Accuracy: {(100 - diff_acc):.2f}% ({diff_class_correct}/{diff_class_total} below threshold)")
print(f"Different-Class Accuracy: {diff_acc:.2f}% ({(diff_class_total - diff_class_correct)}/{diff_class_total} above threshold)")
print(f"----------------------------------------")
print(f"OVERALL ACCURACY        : {total_acc:.2f}%")
print("="*40)