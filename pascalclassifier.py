import numpy as np
import torch
import torchvision
import util
from util.VOCDetection import VOCDetection
from torchvision import models, transforms, datasets
from torch import optim
from torch.autograd import Variable

pascallabels = {}
pascallabels['person'] = 0
pascallabels['boat'] = 1
pascallabels['car'] = 2
pascallabels['bicycle'] = 3
pascallabels['bus'] = 4
pascallabels['motorbike'] = 5
pascallabels['train'] = 6
pascallabels['aeroplane'] = 7
pascallabels['bottle'] = 8
pascallabels['chair'] = 9
pascallabels['diningtable'] = 10
pascallabels['pottedplant'] = 11
pascallabels['tvmonitor'] = 12
pascallabels['sofa'] = 13
pascallabels['bird'] = 14
pascallabels['cat'] = 15
pascallabels['cow'] = 16
pascallabels['dog'] = 17
pascallabels['horse'] = 18
pascallabels['sheep'] = 19

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

optimizer = optim.Adam(model.fullconn.parameters(), lr=0.002)
model.to(device)

training_set = VOCDetection(pascallabels)
train_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)

epochs = 20

for epoch in range(epochs):

  model.train()

  for inputs, labels in train_loader:
        
    # Move input and label tensors to the default device
    inputs = Variable(inputs.type(torch.FloatTensor), requires_grad=True)
    labels = Variable(labels.type(torch.FloatTensor), requires_grad=True)
    inputs, labels = inputs.to(device), labels.to(device)
    labels = labels.type(torch.cuda.LongTensor)
    # inputs = inputs.type(torch.cuda.FloatTensor)

    optimizer.zero_grad()
    
    logps = model(inputs)
    loss = criterion(logps, labels)
    print(logps.shape)
    print('CLASS NUMBER: {}'.format(np.argmax(logps.detach().cpu())))
    loss.backward()
    optimizer.step()

    print('Current loss is (Epoch {}) : {}'.format(epoch, loss.item()))

# path_to_model = '/local/a/ksivaman/dataset-bias/models/pascal.pth'
# torch.save(model.state_dict(), path_to_model)
