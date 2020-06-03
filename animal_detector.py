"""
Classify an image into animal-containing or empty
"""
import torch
import json
import pandas as pd
import torch.nn as nn
import torchvision
from PIL import Image
from pathlib import Path
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np

"""
Helper functions
"""
def load_model(model_dir=Path("E:/wellington_pics/model_dir/mobilenet"), n_classes=17):
    """ Load the pretrained model"""
    model = torchvision.models.mobilenet_v2()
    model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False), nn.Linear(1280, n_classes, bias=True))
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    return model

# perform transformation
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

def load_image(root_dir, id):
    img = id + ".JPG"
    image_pil = Image.open(root_dir / img)
    image = transform(image_pil)
    return image, image_pil


def perform_prediction(model, image):
    res = model(image.unsqueeze(0))
    #print(res)
    res = nn.functional.softmax(res, dim=1)
    res = res.detach().numpy()
    return np.argmax(res)



# Load the model
model = load_model()

# Format picture data
root_dir = Path("E:/wellington_pics/wct_images/images")
label_file = Path("E:/wellington_pics/wellington_camera_traps.json")

with open(label_file) as f:
    labels = json.load(f)
categories = pd.DataFrame(labels['categories'])
annotation = pd.DataFrame.from_dict(labels['annotations'])


# Load and classify image
idx = 4
tmp = annotation.loc[idx,]
cat = tmp.category_id
id = tmp.image_id
image, image_pil = load_image(root_dir, id)
#plt.imshow(image_pil)
#plt.show()
res = perform_prediction(model, image)

correct = 0
incorrect = 0
preds = []
cats = []
for i in range(1000):
    tmp = annotation.loc[i,]
    cat = tmp.category_id
    id = tmp.image_id
    image, image_pil = load_image(root_dir, id)
    res = perform_prediction(model, image)
    cats.append(cat)
    preds.append(res)
    if res == cat:
        correct += 1
    else:
        incorrect += 1

print(correct)
print(incorrect)

## Test on unseen pictures from the internet
pic_path = Path("E:/wellington_pics/cat_test_pics/cat1.jpg")
image = Image.open(pic_path)
image = transform(image)
