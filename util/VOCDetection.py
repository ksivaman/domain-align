import os
import torch
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image

class VOCDetection(Dataset):
    def __init__(self, labelstrtonum, height=224, width=224, root_img='/local/a/ksivaman/data/VOCdevkit/VOC2007/JPEGImages', root_label='/local/a/ksivaman/data/VOCdevkit/VOC2007/Annotations', transforms=None):
        """
        Args:
            root_img: path of folder with images
            root_label: path of folder with labels
            labelstrtonum: dictionary associating every python class wiht the corresponding number for it (for PASCALVOC 2012detection set)
        """
        self.data = []
        self.labels = []
        self.height = height
        self.width = width

        for img in os.listdir(root_img):
            im = Image.open(root_img + '/' + img)
            im = im.resize((224, 224))
            img_np = np.array(im)
            img_torch = torch.from_numpy(img_np)
            img_torch = img_torch.view((3, self.height, self.width))
            self.data.append(img_torch)

        for label in os.listdir(root_label):
            lab = open(root_label + '/' + label, 'r')
            tree = ET.parse(lab) 
            root = tree.getroot()
            gotLabel = False

            if gotLabel == False:
                for item in root.findall('./object'): 
                    if gotLabel == False:
                        for child in item:
                            if child.tag == 'name' and gotLabel == False:
                                if child.text == 'person':
                                    self.labels.append(labelstrtonum[str(child.text)])
                                    gotLabel = True
                                else:
                                    self.labels.append(labelstrtonum['noperson'])
                                    gotLabel = True

            lab.close()

    def __getitem__(self, index):
        img_label = self.labels[index]
        img = self.data[index]

        return img, img_label

    def __len__(self):
        return len(self.data)