"""
Classes and utility functions to create datasets usable with Pytorch
"""
import json
from pathlib import Path
import pandas as pd
import torch.nn as nn
from PIL import Image
import torchvision
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader


class WellingtonDataset(Dataset):
    """Wellington Camera Traps Dataset"""

    def __init__(self, label_file, root_dir, transform=None, transform_size=None):
        self.root_dir = root_dir
        self.transform = transform
        self.transform_size = transform_size

        # Load labels
        with open(label_file) as f:
            labels = json.load(f)
        self.labels = labels
        
        self.categories = pd.DataFrame(labels['categories'])
        self.images = pd.DataFrame.from_dict(labels['images'])
        self.annotation = pd.DataFrame.from_dict(labels['annotations'])

    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """ Get an image"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img = self.images.loc[idx].file_name
        image = Image.open(self.root_dir / img)
        label = self.annotation.loc[idx].category_id

        # Apply transform
        if self.transform is not None:
            # Transform image
            image = self.transform(image)

        return{'image': image, 'label': label}
    




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