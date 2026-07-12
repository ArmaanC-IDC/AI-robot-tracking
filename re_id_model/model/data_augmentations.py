from tensorflow.keras.layers import RandomHue, RandomErasing, RandomGaussianBlur, RandomFlip, RandomRotation, RandomBrightness, RandomZoom, RandomTranslation,  Lambda, Concatenate, RandomContrast
from tensorflow.keras.models import Sequential
import keras
import tensorflow as tf
    

#TODO: Review augmentations (ensure robots are still recognizable)
#Faisal was right: many images were fully unrecognizable
augmenter = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.15),
    RandomBrightness(0.15, value_range=(-1, 1)),
    RandomContrast(0.5, value_range=(-1, 1)),
    RandomZoom(0.15),
    RandomTranslation(height_factor=0.1, width_factor=0.1),
    RandomGaussianBlur(0.1, value_range=(-1, 1)),
    # RandomHue(factor=0.05),
    RandomErasing(0.3)
])