import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from munge_data import mungeFile

class CustomDataset(Dataset):
    def __init__(self, img_path, df, target, transforms = None):
        self.img_path = img_path
        self.dataframe = df
        self.target = target
        self.transforms = transforms
        

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        image_path = os.path.join(self.img_path, str(self.dataframe["image_name"][index]))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image = image)
        image = augmented["image"]
        image = np.transpose(image, (2,0,1)).astype(np.float32)

        targets = self.target[index]
        
        
        return {
                "image" : torch.tensor(image), 
                "target" : torch.tensor(targets)
                } 


if __name__ == "__main__":
    def get_train_transforms():
        return A.Compose(
            [
                A.Resize(height=256, width=256, p=1),
                #ToTensorV2(p=1.0),
            ], 
            p=1.0)

    img_path = "C:\\Datasets\\chest_xray\\train"
    train_df = mungeFile(img_path)
    dataset = CustomDataset(img_path=img_path, df = train_df, target = train_df['target'], transforms=get_train_transforms())
    data_loader = DataLoader(dataset, shuffle=True, num_workers=0)
    image = next(iter(data_loader))["image"]
    image = image.squeeze()
    

