"""
Tensorflow 2 version of animal detector
"""

import os
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import json
import pandas as pd
from pathlib import Path
####
# Create the input pipeline
####

#root_dir = Path("E:/wellington_pics/wct_images/images")
root_dir = "E:/wellington_pics/wct_images/images/"

root_dir = "E:\\wellington_pics\\wct_images\\images\\"
label_file = Path("E:/wellington_pics/wellington_camera_traps.json")

with open(label_file) as f:
    labels = json.load(f)

categories = pd.DataFrame(labels['categories'])
annotation = pd.DataFrame.from_dict(labels['annotations'])
images = pd.DataFrame.from_dict(labels['images'])

# Make dataframe conform for Keras
annotation['filename'] = [root_dir + x + ".JPG" for x in list(annotation.image_id)]
class_labels = [str(x) for x in list(annotation.category_id)]
annotation['class'] = np.array(class_labels)

# Create the generator
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_dataframe(dataframe=annotation, batch_size=BATCH_SIZE, x_col="filename",
                                                    shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), classes=list(categories.name), class_mode="sparse")

os.path.isfile(annotation.filename[0])