from helper_scripts.build_model import build_embedding_model
import cv2
import numpy as np

base_path = "./images-3/444.png"
neg_path = "./images-3/459.png"
pos_path = "./images-0/459.png"

model = build_embedding_model('./model.weights.h5')

def load_image(path, image_size=(128, 128)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    img = ((img / 127.5) - 1.0).astype(np.float32)
    return img

base = load_image(base_path)
pos = load_image(pos_path)
neg = load_image(neg_path)

base_emb, pos_emb, neg_emb = model.predict(np.array([base, pos, neg]))

pos_dist = np.linalg.norm(base_emb - pos_emb)
neg_dist = np.linalg.norm(base_emb - neg_emb)

print(f"Correct: {pos_dist < neg_dist}")
print(f"Pos dist: {pos_dist}")
print(f"Neg dist {neg_dist}")