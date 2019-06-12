import numpy as np
import torch
import torchvision
import util
from util.VOCDetection import VOCDetection
from torchvision import models, transforms, datasets
from torch import optim
from torch.autograd import Variable

# pascallabels = {}
# pascallabels['person'] = 0
# pascallabels['boat'] = 1
# pascallabels['car'] = 2
# pascallabels['bicycle'] = 3
# pascallabels['bus'] = 4
# pascallabels['motorbike'] = 5
# pascallabels['train'] = 6
# pascallabels['aeroplane'] = 7
# pascallabels['bottle'] = 8
# pascallabels['chair'] = 9
# pascallabels['diningtable'] = 10
# pascallabels['pottedplant'] = 11
# pascallabels['tvmonitor'] = 12
# pascallabels['sofa'] = 13
# pascallabels['bird'] = 14
# pascallabels['cat'] = 15
# pascallabels['cow'] = 16
# pascallabels['dog'] = 17
# pascallabels['horse'] = 18
# pascallabels['sheep'] = 19

pascallabels = {}
pascallabels['person'] = 0
pascallabels['noperson'] = 1

model = models.resnet18(pretrained=False)

for param in model.parameters():
    param.requires_grad = False

model = torch.nn.Sequential(
                            model,

                            torch.nn.Linear(1000, 256),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.5),
                         
                            torch.nn.Linear(256, 20),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.5),

                            torch.nn.Linear(20, 2),
                            torch.nn.LogSoftmax(dim=1)
)

# decay the learning rate of the optimizer after every lr_decay_epoch epochs
def exp_lr_scheduler(optimizer, epoch, lr_change_factor = 0.1, lr_decay_epoch=10):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_change_factor
    return optimizer

criterion = torch.nn.NLLLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)

training_set = VOCDetection(pascallabels)
train_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)

epochs = 200
save_every = 4

loss_by_epoch = []

for epoch in range(epochs):

  model.train()
  running_loss = 0.0

  print('epoch no: {}'.format(epoch + 1))

  for i, (inputs, labels) in enumerate(train_loader):
    # Move input and label tensors to the default device

    inputs = Variable(inputs.type(torch.FloatTensor), requires_grad=True)
    labels = Variable(labels.type(torch.FloatTensor), requires_grad=True)
    inputs, labels = inputs.to(device), labels.to(device)
    labels = labels.type(torch.cuda.LongTensor)

    optimizer.zero_grad()
    
    logps = model(inputs)
    loss = criterion(logps, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    # if i % 32 == 31:    # print every 32 mini-batches
    #   print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 32))
    #   running_loss = 0.0

  loss_by_epoch.append(running_loss)
  print('loss for epoch {} is {}'.format(epoch+1, running_loss))

  if epoch % save_every == 0:
    path_to_model = '/local/a/ksivaman/dataset-bias/models/pascal_scratch_binary_{}.pth'.format(epoch)
    torch.save(model.state_dict(), path_to_model)

  optimizer = exp_lr_scheduler(optimizer, epoch+1)
