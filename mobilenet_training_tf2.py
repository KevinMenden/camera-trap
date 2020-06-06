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


#### Parameters
root_dir = "E:/wellington_pics/wct_images/images/"
root_dir = "E:\\wellington_pics\\wct_images\\images\\"
label_file = Path("E:/wellington_pics/wellington_camera_traps.json")
model_dir = Path("E:/wellington_pics/model_dir")
model_save_path = model_dir / "mobilenetv2_keras"
lite_model_path = model_dir / "mobilenetv2_tflite"
IMG_SHAPE = (244, 244, 3)
IMG_SIZE = (244, 244)
lr = 0.00001
batch_size = 32


with open(label_file) as f:
    labels = json.load(f)

categories = pd.DataFrame(labels['categories'])
annotation = pd.DataFrame.from_dict(labels['annotations'])
images = pd.DataFrame.from_dict(labels['images'])

# Make dataframe from anntations
annotation['filename'] = [root_dir + x + ".JPG" for x in list(annotation.image_id)]
class_labels = [str(x) for x in list(annotation.category_id)]
annotation['class'] = np.array(class_labels)

# Create the TF dataset (adjusted from https://stackoverflow.com/questions/44416764/loading-folders-of-images-in-tensorflow)
image_paths = list(annotation.filename)
labels = list(annotation.category_id)

epoch_size = len(image_paths)
image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
labels = tf.convert_to_tensor(labels)

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

dataset = dataset.repeat().shuffle(epoch_size)

def map_fn(path, label, IMG_SIZE=IMG_SIZE):
    # Load image, resize to (244, 244) and normalize, return label as is
    image = tf.image.decode_jpeg(tf.io.read_file(path))
    image = tf.image.resize(image, IMG_SIZE)
    image = (image/127.5) - 1
    return image, label



dataset = dataset.map(map_fn, num_parallel_calls=8)
dataset = dataset.batch(batch_size)
# try one of the following
AUTOTUNE = tf.data.experimental.AUTOTUNE
dataset = dataset.prefetch(buffer_size=AUTOTUNE)
# dataset = dataset.apply(
#            tf.contrib.data.prefetch_to_device('/gpu:0'))

images, labels = next(iter(dataset))


#==== end dataset creation ====#

IMG_SHAPE = (244, 244, 3)

# Create the model
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(17)
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])



load_pretrained = True
if load_pretrained:
    model = tf.keras.models.load_model(str(model_save_path))

# Compile the model
lr = 1e-8
model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# Perform the training
steps_per_epoch = 240000 / batch_size
model.fit(x=dataset, epochs=10, steps_per_epoch=steps_per_epoch)

# Save the trained model

model.save(str(model_save_path)) 

# Convert model to tensorflow-lite

converter = tf.lite.TFLiteConverter.from_saved_model(str(model_save_path))
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile(str(lite_model_path), 'wb') as f:
  f.write(tflite_model)
# Also save in this github repo
with tf.io.gfile.GFile("mobilenetv2_tflite", 'wb') as f:
    f.write(tflite_model)

