import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow_similarity as tfsim
import keras
import random
import cv2
import numpy as np
import os
import csv
from data_augmentations import augmenter

filepath = "siamese_train/train15"

dataset_filepath = "dataset"

@keras.saving.register_keras_serializable()
def dist(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

image_shape = (128, 128, 3)

def get_data(main_folder, P=4, K=4, image_size=(128, 128)):
    x_train = []
    y_train = []

    robots = [r for r in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, r))]
    r_to_id = {r: idx for idx, r in enumerate(robots)}

    for r in robots:
        image_paths = [os.path.join(main_folder, r, f) for f in os.listdir(os.path.join(main_folder, r))]

        for path in image_paths:
            x_train.append(load_image(path, image_size))
            y_train.append(r_to_id[r])
    
    return x_train, y_train
        
def pk_generator(x_data, y_data, P, K):
    class_idxs = {}
    for idx, label in enumerate(y_data):
        if label not in class_idxs:
            class_idxs[label] = []
        class_idxs[label].append(idx)

    unique_classes = list(class_idxs.keys())

    while True:
        selected_robots = random.sample(unique_classes, P)

        imgs = []
        labels = []

        for robot in selected_robots:
            selected_idxs = random.sample(class_idxs[robot], K)

            for idx in selected_idxs:
                imgs.append(x_data[idx])
                labels.append(y_data[idx])
        
        yield np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.float32)

def load_image(path, image_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    img = ((img / 127.5) - 1.0).astype(np.float32)
    return img

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=image_shape) #has 154 layers.
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.7)(x)
x = Dense(128, activation="relu")(x)
#TODO: Add more layers (look into)
#TODO: Look into auto-encoder
x = tf.keras.layers.UnitNormalization()(x)

embedding_model = Model(base.input, x, name="embedding")

input = Input(shape=(128, 128, 3), name="input")

augmented = augmenter(input)

embedding = embedding_model(augmented)

model = tfsim.models.SimilarityModel(input, embedding)

# model.load_weights("siamese_train/train13" + "/model.weights.h5")

# model = load_model(filepath + "/model.keras", compile=False, custom_objects={
#     "dist": dist, 
# })

for layer in base.layers:
    layer.trainable = False
for layer in embedding_model.layers[-50:]:
    layer.trainable = True

margin = 1.0

P_VAL_TRAIN = 4
K_VAL_TRAIN = 4

x_train, y_train = get_data(dataset_filepath + "/train")
x_val, y_val = get_data(dataset_filepath + "/val")

dataset_train = tf.data.Dataset.from_generator(
    lambda: pk_generator(x_train, y_train, P=P_VAL_TRAIN, K=K_VAL_TRAIN),
    output_signature=(
        tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32), 
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

train_steps_per_epoch = len(x_train) // (P_VAL_TRAIN * K_VAL_TRAIN)

P_VAL_VAL = 8
K_VAL_VAL = 2
dataset_val = tf.data.Dataset.from_generator(
    lambda: pk_generator(x_val, y_val, P=P_VAL_VAL, K=K_VAL_VAL),
    output_signature=(
        tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32), 
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

val_steps_per_epoch = len(x_val) // (P_VAL_VAL * K_VAL_VAL)

# dataset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# dataset_val = (
#     dataset_val.shuffle(buffer_size=1000).batch(16).prefetch(tf.data.AUTOTUNE)
# )

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

def mine_semi_hard_triplets(embeddings, labels, margin=1.0):
    norms = np.sum(embeddings**2, axis=1, keepdims=True)
    dist_matrix = norms + tf.transpose(norms) - 2 * np.dot(embeddings, tf.transpose(embeddings))
    dist_matrix = np.maximum(dist_matrix, 0.0)

    triplets = []
    for i in range(len(labels)):
        pos_mask = (labels == labels[i])
        neg_mask = (labels != labels[i])
        
        pos_indices = np.where(pos_mask)[0]
        d_ap = np.max(dist_matrix[i, pos_indices])
        
        neg_indices = np.where(neg_mask)[0]
        semi_hard_negatives = [idx for idx in neg_indices 
                               if d_ap < dist_matrix[i, idx] < d_ap + margin]
        
        if semi_hard_negatives:
            n_idx = np.random.choice(semi_hard_negatives)
            triplets.append((i, np.random.choice(pos_indices), n_idx))
        
    if len(triplets) == 0:
        print("No semi-hard triplets")
        return np.empty((0, 3), dtype=np.int32)
    
    return np.array(triplets, dtype=np.int32)

def mine_triplets(images, labels, margin=1.0):
    labels = labels.numpy()
    labels = labels.reshape(-1, 1)
    
    is_same = (labels == labels.T)
    is_diff = (labels != labels.T)
    
    same_coords = np.argwhere(is_same) #all anchor/pos pairs
    diff_coords = np.argwhere(is_diff) #all anchor/neg pairs
    triplets = []
    
    a_to_p = {}
    for a, p in same_coords:
        if a != p:
            if a not in a_to_p: a_to_p[a] = []
            a_to_p[a].append(p)
            
    for a, n in diff_coords:
        if a in a_to_p:
            for p in a_to_p[a]:
                triplets.append((a, p, n))
                
    return np.array(triplets)

def calculate_triplet_loss(embeddings, triplet_indices):
    anchors = tf.gather(embeddings, triplet_indices[:, 0])
    positives = tf.gather(embeddings, triplet_indices[:, 1])
    negatives = tf.gather(embeddings, triplet_indices[:, 2])

    d_pos = tf.reduce_sum(tf.square(anchors - positives), axis=1)
    d_neg = tf.reduce_sum(tf.square(anchors - negatives), axis=1)

    loss = tf.maximum(d_pos - d_neg + margin, 0.0)
    return tf.reduce_mean(loss)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        embeddings = model(images, training=True)
        
        triplet_indices = tf.py_function(
            func=mine_triplets, 
            inp=[images, labels], 
            Tout=tf.int32 
        )

        triplet_indices.set_shape([None, 3])

        if tf.shape(triplet_indices)[0] == 0:
            loss = -1.0
        
        else:
            loss = calculate_triplet_loss(embeddings, triplet_indices)

    if loss >= 0.0:
        gradients = tape.gradient(loss, model.trainable_variables)
        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# @tf.function
def val_step(images, labels):
    embeddings = model(images, training=False)

    triplet_indices = tf.py_function(
        func=mine_triplets, 
        inp=[images, labels], 
        Tout=tf.int32 
    )

    if tf.shape(triplet_indices)[0] == 0:
        loss = np.nan
    
    else:
        loss = calculate_triplet_loss(embeddings, triplet_indices)

    mean_dist_matches = np.mean([np.linalg.norm(embeddings[i] - embeddings[j])
                                for i in range(len(embeddings)) 
                                for j in range(len(embeddings))
                                if i!=j and labels[i]==labels[j]])
    
    mean_dist_non_matches = np.mean([np.linalg.norm(embeddings[i] - embeddings[j])
                                for i in range(len(embeddings)) 
                                for j in range(len(embeddings))
                                if i!=j and labels[i]!=labels[j]])
    return loss, mean_dist_matches, mean_dist_non_matches

best_val_loss = float("inf")

file_exists = os.path.isfile(filepath + "/results.csv")

with open(filepath + "/results.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["epoch", "train_loss", "val_loss", "mean_dist_matches", "mean_dist_non_matches"])

    for epoch in range(50):
        print("starting epoch " + str(epoch) + " ---------------")

        train_iterator = iter(dataset_train)
        val_iterator = iter(dataset_val)

        train_losses = []
        for step in range(train_steps_per_epoch):

            images, labels = next(train_iterator)
            loss = train_step(images, labels)
            if loss >=0:
                train_losses.append(loss)

        val_losses, mean_dists_matches, mean_dists_non_matches = [], [], []
        for step in range(val_steps_per_epoch):
            images, labels = next(val_iterator)

            loss, mean_dist_matches, mean_dist_non_matches = val_step(images, labels)
            val_losses.append(loss)
            mean_dists_matches.append(mean_dist_matches)
            mean_dists_non_matches.append(mean_dist_non_matches)
            
        train_loss = np.nanmean(train_losses)
        val_loss = np.nanmean(val_losses)
        mean_dist_matches = np.nanmean(mean_dists_matches)
        mean_dist_non_matches = np.nanmean(mean_dists_non_matches)

        
        print(f"train loss: {train_loss:.4f}")
        print(f"val loss: {val_loss:.4f}")
        print(f"mean dist matches: {mean_dist_matches:.4f}")
        print(f"mean_dist_non_matches: {mean_dist_non_matches:.4f}")
        print()

        writer.writerow([epoch + 1, train_loss, val_loss, mean_dist_matches, mean_dist_non_matches])

        if val_loss < best_val_loss:
            print(f"Saving new model with a loss of {val_loss}, which is better than {best_val_loss}")
            best_val_loss = val_loss
            model.save_weights(filepath + "/model.weights.h5")