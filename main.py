import os
import argparse
from random import shuffle

import torch
from albumentations.augmentations import transforms
from dataset import CustomDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



image_path = "../chest_xray"

#train, val
train_path = os.path.join(image_path, "train")
val_path = os.path.join(image_path, "val")

#Augmentation
def get_train_transforms():
    return A.Compose(
        [
            #A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0
    )
    






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Epochs", type=int, default = 50, help = "train Epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    opt = parser.parse_args()


    #dataset
    train_dataset = CustomDataset(train_path, transforms = get_train_transforms())
    valid_dataset = CustomDataset(val_path, transforms = get_valid_transforms())

    #dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = opt.batch_size, shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size = opt.batch_size, shuffle=True
    )