import numpy as np
import torch
import torchvision
import util
from util.VOCDetection import VOCDetection
from torchvision import models, transforms, datasets
from torch import optim
from torch.autograd import Variable
from PIL import Image

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

path_to_model_dict = '/local/a/ksivaman/dataset-bias/models/pascal.pth'

model = models.resnet18(pretrained=True)

fc = torch.nn.Sequential(
                            torch.nn.Linear(1000, 256),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.5),
                         
                            torch.nn.Linear(256, 20),
                            torch.nn.LogSoftmax(dim=1)
)

model.fullconn = fc

for param in model.parameters():
    param.requires_grad = False

model_dict = torch.load(path_to_model_dict)
model.load_state_dict(model_dict)

# print(model)

root_img='/local/a/ksivaman/data/VOCdevkit/VOC2007/JPEGfew/000001.jpg'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

im = Image.open(root_img)
im = im.resize((224, 224))
img_np = np.array(im)
img_torch = torch.from_numpy(img_np)
img_torch = img_torch.view((3, 224, 224))

img_torch = Variable(img_torch.type(torch.FloatTensor), requires_grad=True)
img_torch = img_torch.to(device)

img_torch = img_torch.unsqueeze(dim=0)

model.to(device)

logps = model(img_torch)
print('Class num: ', np.argmax(logps.detach().cpu()))

