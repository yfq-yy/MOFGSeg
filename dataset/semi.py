from dataset.transform import *
from copy import deepcopy
import math
import numpy as np
import os
import random
import cv2

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('/data/fyao309/MOFGSeg/splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        # print("*****************")
        # print(os.path.join(self.root, id.split(' ')[1]))
        # print(os.path.join(self.root, id.split(' ')[1].replace('SegmentationClass', 'edge')))
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))
        edge = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1].replace('train', 'edge')))))
        

        if self.mode == 'val':
            img, mask, edge = normalize(img, mask, edge)
            return img, mask, id

        img, mask, edge = resize(img, mask, edge, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask, edge = crop(img, mask, edge, self.size, ignore_value)
        img, mask, edge = hflip(img, mask, edge, p=0.5)
      
        if self.mode == 'train_u':
            return normalize(img)

        img_s1 = deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        img_s1, mask, edge = normalize(img_s1, mask, edge)
        return img_s1, mask, edge

    def __len__(self):
        return len(self.ids)