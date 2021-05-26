import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, img_path, transforms = None, mode = None):
        self.img_path = os.path.join(img_path, mode)
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        image = cv2.imread(self.img_path[index])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = self.transforms(image = image)["image"]
        image = np.transpose(image, (2,0,1)).astype(np.float32)

        if self.mode == "NORMAL":
            target = [1 for _ in range(len(self.img_path))]
        elif self.mode == "PNEUMONIA":
            target = [0 for _ in range(len(self.img_path))]
        
        
        return {
                "image" : torch.tensor(image), 
                "target" : torch.tensor(target)
                } 

if __name__ =="main":
    pass