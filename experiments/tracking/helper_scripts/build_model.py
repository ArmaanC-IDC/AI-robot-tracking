import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Importing mobile net")
from tensorflow.keras.applications import MobileNetV2
print("importing layers")
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, UnitNormalization
print("importing model")
from tensorflow.keras.models import Model
print("importing similarity")
import tensorflow_similarity as tfsim

IMAGE_SHAPE = (128, 128, 3)

def build_embedding_model(model_filepath):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMAGE_SHAPE)
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.7)(x)
    x = Dense(128, activation="relu")(x)
    x = UnitNormalization()(x)

    embedding_model = Model(base.input, x, name="embedding")
    
    input_tensor = Input(shape=IMAGE_SHAPE, name="input")
    embedding = embedding_model(input_tensor)
    
    

    model = tfsim.models.SimilarityModel(input_tensor, embedding)

    model.load_weights(model_filepath)

    return model