import os
import cv2
import random
from tqdm import tqdm
import numpy as np


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from dataset.munge_data import mungeFile

def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

class CustomDataset(Dataset):
    def __init__(self, img_path, df, target, transforms = None, test = False, Cutmix = True,  cutmix_params={'alpha': 1,}):
        self.img_path = img_path
        self.dataframe = df
        self.target = target
        self.transforms = transforms
        self.test = test
        self.Cutmix = Cutmix
        self.cutmix_params = cutmix_params

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index : int):
        #image_path = os.path.join(self.img_path, str(self.dataframe["image_name"][index]))
        image = self.load_image(index)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        #image /= 255.0

        augmented = self.transforms(image = image)
        image = augmented["image"]
        target = self.target[index]

        if self.Cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            image = self.cutmix(index)
            image = torch.tensor(image)
        else:
            image = image


        return {
                "image" : image, 
                "target" : target
                }

    def load_image(self, index):
        image_path = os.path.join(self.img_path, str(self.dataframe["image_name"][index]))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_transform = A.Compose([A.Resize(height=128, width=128, p=1.0)], 
                                    p=1.0, 
                                    )

        resized = resize_transform(**{
                'image': image,
            })

        return resized["image"]

    
    def cutmix(self, index, imsize=128):
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
        result_img = np.full((imsize, imsize,3), 1, dtype=np.float32)

        for i, index in enumerate(indexes):
            cm_image = self.load_image(index)
            
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

            result_img[y1a:y2a, x1a:x2a] = cm_image[y1b:y2b, x1b:x2b]

        result_img = np.transpose(result_img, (2,0,1))
        return result_img 
        

if __name__ == "__main__":
    def get_train_transforms():
        return A.Compose(
            [
                A.Resize(height=128 , width=128, p=1),
                ToTensorV2(p=1.0),
            ], 
            p=1.0)

    img_path = "C:\\Datasets\\chest_xray\\train"
    train_dataframe = mungeFile(img_path)
    train_dataframe = train_dataframe.sample(frac=1).reset_index(drop=True)

    train_len = int(len(train_dataframe))
    tr_idx = int(train_len * 0.8)

    
    train_df = train_dataframe[:tr_idx].reset_index(drop=True)
    valid_df = train_dataframe[tr_idx:].reset_index(drop=True)

    dataset = CustomDataset(img_path=img_path, df = valid_df, target = train_df['target'], transforms=get_train_transforms(), test=False, Cutmix=True)
    data_loader = DataLoader(dataset,shuffle=True, num_workers=4)
    device = "cuda"
    
    
    
    image = next(iter(data_loader))["image"]
    print(image.shape)
    target = next(iter(data_loader))["target"]
    image = np.array(image.squeeze().permute(1,2,0))
    cv2.imwrite('cutmix_sample.jpg', image)
    
    
    
    

