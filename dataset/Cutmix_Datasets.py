import os
import cv2
import random
import numpy as np


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from munge_data import mungeFile

class CustomDataset(Dataset):
    def __init__(self, img_path, df, target, transforms = None, test = False):
        self.img_path = img_path
        self.dataframe = df
        self.target = target
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index : int):
        image_path = os.path.join(self.img_path, str(self.dataframe["image_name"][index]))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        #image /= 255.0

        if self.test == False or random.random() > 0.5:
            image = self.cutmix(image, index)
        else:
            image = image

        augmented = self.transforms(image = image)
        image = augmented["image"]
        image = np.transpose(image, (2,0,1))

        targets = self.target[index]
        
        return {
                "image" : torch.tensor(image), 
                "target" : torch.tensor(targets)
                }

    def cutmix(self, img, index, imsize=512):
        w, h = imsize, imsize
        s = imsize // 2

        # 중앙값 랜덤하게 잡기
        xc, yc = [int(random.uniform(imsize*0.25, imsize*0.75)) for _ in range(2)] #256 ~ 768
        indexes = None
        if self.dataframe["target"][index] == 0 :
            indexes = [index] + [random.randint(0, len(self.dataframe[self.dataframe["target"] == 0])-1) for _ in range(3)]
        elif self.dataframe["target"][index] == 1:
            indexes = [index] + [random.randint(len(self.dataframe[self.dataframe["target"] == 0]), len(self.dataframe[self.dataframe["target"] == 1])-1) for _ in range(3)]
        
        #검은색 배경의 임의 이미지 생성 (여기다가 이미지들 붙여넣는 방식) 
        result_img = np.full((imsize, imsize, 3), 1, dtype=np.float32)

        for i, index in enumerate(indexes):
            image = img
            #top left
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            result_img[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

        return result_img 


if __name__ == "__main__":
    def get_train_transforms():
        return A.Compose(
            [
                A.Resize(height=512 , width=512, p=1),
                #ToTensorV2(p=1.0),
            ], 
            p=1.0)

    img_path = "C:\\Datasets\\chest_xray\\train"
    train_df = mungeFile(img_path)
    dataset = CustomDataset(img_path=img_path, df = train_df, target = train_df['target'], transforms=get_train_transforms(), test=False)
    data_loader = DataLoader(dataset, shuffle=True, num_workers=0)
    image = next(iter(data_loader))["image"]
    image = np.array(image.squeeze().permute(1,2,0))
    cv2.imwrite('cutmix_sample.jpg', image)
    
    

