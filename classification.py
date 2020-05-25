import json
from pathlib import Path
import pandas as pd
import torch.nn as nn
from PIL import Image
import torchvision
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from dataset import WellingtonDataset


root_dir = Path("D:/wellington_pics/wct_images/images")
label_file = Path("D:/wellington_pics/wellington_camera_traps.json")

# perform transformation
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

def collate_fn(batch):
    imgs = [item['image'] for item in batch]
    targets = [item['label'] for item in batch]
    return imgs, targets

data = WellingtonDataset(label_file=label_file, root_dir=root_dir, transform=transform, transform_size=224)
data_loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate_fn)

model = torchvision.models.mobilenet_v2(pretrained=True)

batch = next(iter(data_loader))
images = torch.stack(batch[0])
labels = batch[1]

res = model(images)
res = res.detach().numpy()