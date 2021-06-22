import torch
import torch.nn as nn

import os
from tqdm import tqdm
import numpy as np
import argparse

import Config as CFG
from CustomModel import ChestClassifier
from dataset.munge_data import mungeFile




from albumentations.augmentations import transforms
from dataset.CustomDataset import CustomDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def inference_one_epoch(model, data_loader, device):
    model.eval()

    img_preds_all = []
    pbar = tqdm(enumerate(data_loader), total = len(data_loader))
    for step, (imgs, labels) in pbar:
        imgs = imgs.to(device).float()
        labels = labels.to(device).long()

        pred = model(imgs)
        img_preds_all += [torch.softmax(pred, 1).detach().cpu().numpy()]
        

    img_preds_all = np.concatenate(img_preds_all, axis=0)

    return img_preds_all


def get_test_transforms():
    return A.Compose(
        [
            A.Resize(height=128, width=128, p=1),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Epochs", type=int, default = 10, help = "train Epochs")
    parser.add_argument("--Batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--Model", type=str, default = 'efficientnet_b0', help="Torch image Model")
    
    opt = parser.parse_args()

    device = CFG.DEVICE
    model = ChestClassifier(model_arch = opt.Model, n_class = 2, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = CFG.LEARNING_RATE, weight_decay = CFG.WEIGHT_DECAY)
    loss_test = nn.CrossEntropyLoss().to(device)

    image_path = "C:\\Datasets\\chest_xray\\"
    test_path = os.path.join(image_path, "test")
    test_dataframe = mungeFile(test_path)
    test_df = test_dataframe.sample(frac=1).reset_index(drop=True)

    test_dataset = CustomDataset(test_path, df = test_df, target = test_df['target'], transforms = get_test_transforms(),test=False, Cutmix=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.Batch_size, shuffle=False, num_workers = CFG.NUM_WORKERS)
    
    checkpoint = "checkpoints\efficientnet_b0_0.pth"
    model.load_state_dict(torch.load(checkpoint))
    
    for epoch in range(opt.Epochs):
        with torch.no_grad():
            inference_one_epoch(epoch, test_loader, device)
