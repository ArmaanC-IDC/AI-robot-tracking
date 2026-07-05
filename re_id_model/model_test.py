import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("train1/model.keras")

def load(path):
    """Formats the image exactly like the Data Generator did."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img

# crop_A = load("test_data/4476/01.jpg")
# crop_B = load("test_data/8349/00.jpg")

crop_A = np.zeros((128, 128, 3))
crop_B = np.ones((128, 128, 3))

final_img = np.expand_dims(np.concatenate([crop_A, crop_B], -1), axis=0)

result = model.predict(final_img)

print(result)