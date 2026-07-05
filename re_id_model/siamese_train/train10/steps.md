Training steps:
 - start with train9/model.keras (10 epochs at 1e-3 learning rate with the MobileNetV2 weights frozen)
 - do 50 epochs at 3e-5 learning rate with the last 50 layers of the embedding model unfrozen, the rest still frozen

Loss function: triplet loss (random sampling to get pairs)

Augmentations:
    RandomFlip("horizontal"),
    RandomRotation(0.15),
    RandomBrightness(0.4),
    RandomContrast(0.4),
    RandomZoom(0.2),
    RandomTranslation(height_factor=0.15, width_factor=0.15)

Total dataset: 751 images
 '10611': 33,
 '1114': 30,
 '1323': 21,
 '1360': 29,
 '1678': 29,
 '188': 30,
 '2056': 21,
 '2337': 38,
 '2481': 41,
 '254': 34,
 '4065': 15,
 '4414': 50,
 '4476': 34,
 '4915': 31,
 '4946': 31,
 '5885': 32,
 '6324': 30,
 '6800': 18,
 '7558': 30,
 '8349': 58,
 '8613': 54,
 '8884': 29,
 '919': 33

Train dataset: 626 images
 '10611': 33,
 '1114': 26,
 '1323': 21,
 '1360': 29,
 '1678': 25,
 '188': 26,
 '2056': 21,
 '2337': 38,
 '2481': 41,
 '4065': 15,
 '4414': 39,
 '4476': 30,
 '4915': 31,
 '4946': 31,
 '5885': 17,
 '6324': 30,
 '6800': 18,
 '7558': 30,
 '8349': 41,
 '8613': 43,
 '8884': 12,
 '919': 29

Val dataset: 125 images
 '1114': 4,
 '1678': 4,
 '188': 4,
 '254': 34,
 '4414': 11,
 '4476': 4,
 '5885': 15,
 '8349': 17,
 '8613': 11,
 '8884': 17,
 '919': 4
