from pathlib import Path
import torchvision
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import DataLoader
from dataset import WellingtonDataset
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def collate_fn(batch):
    imgs = [item['image'] for item in batch]
    targets = [item['label'] for item in batch]
    return imgs, targets

if __name__=="__main__":
        
    # file paths
    root_dir = Path("E:/wellington_pics/wct_images/images")
    label_file = Path("E:/wellington_pics/wellington_camera_traps.json")
    model_dir = Path("E:/wellington_pics/model_dir/mobilenet")
    writer_dir = Path("E:/wellington_pics/model_dir/")

    writer = SummaryWriter(logdir=writer_dir)
    save_freq = 100

    # perform transformation
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

    # create data loader
    data = WellingtonDataset(label_file=label_file, root_dir=root_dir, transform=transform, transform_size=224)
    data_loader = DataLoader(data, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=10, pin_memory=True)

    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create model
    n_classes = data.categories.shape[0]
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False), nn.Linear(1280, n_classes, bias=True))
    # load trained model
    model.load_state_dict(torch.load(model_dir))
    model.train()
    model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    criterion = nn.CrossEntropyLoss()
    loss_list = []

    n_epochs = 10
    for epoch in range(n_epochs):
        for i, batch in enumerate(data_loader):
            images = batch[0]
            labels = torch.tensor(batch[1])

            images = torch.stack([img.to(device) for img in images])
            labels = torch.stack([l.to(device) for l in labels])

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            writer.add_scalar('loss', loss.item(), i*(epoch+1))
            if i % 10 == 0:
                print(f"Ep: {epoch}, Step: {i}, loss: {loss}")
            if i % save_freq == 0:
                torch.save(model.state_dict(), model_dir)
        
        
    writer.close()


    plt.plot(loss_list)
    plt.show()


