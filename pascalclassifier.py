import numpy as np
import torch
import torchvision
from torchvision import models, transforms, datasets
from torch import optim

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

fc = torch.nn.Sequential(
                            torch.nn.Linear(1000, 256),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.5),
                         
                            torch.nn.Linear(256, 20),
                            torch.nn.LogSoftmax(dim=1)
)

model.fullconn = fc

criterion = torch.nn.NLLLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = optim.Adam(model.fullconn.parameters(), lr=0.002)
model.to(device)

train_data = torchvision.datasets.VOCSegmentation('./data', year='2012', image_set='train', download=False)

train_data = datasets.VOCDetection('./data', year='2012', image_set='train', download=True)
val_data = datasets.VOCDetection('./data', year='2012', image_set='val', download=True)

