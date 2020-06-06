"""
Classify an image into animal-containing or empty
"""
import os
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import json
import pandas as pd
from pathlib import Path
"""
Helper functions
"""

def load_image(root_dir, id):
    img = root_dir / Path(id + ".JPG")
    image = tf.image.decode_jpeg(tf.io.read_file(str(img)))
    image = tf.image.resize(image, (244, 244))
    image = (image/127.5) - 1
    return image


# Load the model
model_dir = Path("E:/wellington_pics/model_dir")
model_save_path = model_dir / "resnet50_keras"
model = tf.keras.models.load_model(str(model_save_path))

# Format picture data
root_dir = Path("E:/wellington_pics/wct_images/images")
label_file = Path("E:/wellington_pics/wellington_camera_traps.json")

with open(label_file) as f:
    labels = json.load(f)
categories = pd.DataFrame(labels['categories'])
annotation = pd.DataFrame.from_dict(labels['annotations'])


config = tf.config
config.gpu_options.allow_growth = True

tf.config.C
# Load and classify image
idx = 4
tmp = annotation.loc[idx,]
cat = tmp.category_id
id = tmp.image_id
image = load_image(root_dir, id)
image = tf.keras.backend.expand_dims(image, 0)
res = np.argmax(model(image))


correct = 0
incorrect = 0
preds = []
cats = []
for i in range(5000):
    tmp = annotation.loc[i,]
    cat = tmp.category_id
    if not cat == 1:
        continue
    id = tmp.image_id
    image = load_image(root_dir, id)
    image = tf.keras.backend.expand_dims(image, 0)
    image = np.array(image)
    res = np.argmax(model(image))
    cats.append(cat)
    preds.append(res)
    if res == cat:
        correct += 1
    else:
        incorrect += 1

print(correct)
print(incorrect)
print(correct / (correct + incorrect))



## Test on unseen pictures from the internet
pic_path = Path("E:/wellington_pics/cat_test_pics/cat2.jpg")
image = tf.image.decode_jpeg(tf.io.read_file(str(pic_path)))
image = tf.image.resize(image, (244, 244))
image = (image/127.5) - 1
image = tf.keras.backend.expand_dims(image, 0)
res = np.argmax(model(image))
print(res)

