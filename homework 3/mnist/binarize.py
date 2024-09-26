#!/usr/bin/env python

import numpy as np
import pickle

with open("mnist-train-images","rb") as f: train_images = pickle.load(f)
for image in train_images:
    image[image < 0.5] = 0
    image[image >= 0.5] = 1
with open("mnist-binary-train-images","wb") as f: pickle.dump(train_images.astype(np.bool_),f)
