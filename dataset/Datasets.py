import torch
import torch.nn as nn
import torch.utils.data.dataloader
import cv2
import numpy as np

class CustomDataset:
    def __init__(self, img_path, transforms = None):
        self.img_path = img_path
        self.transforms = transforms

    def __len__(self):
        pass

    def __getitem__(self, index):
        