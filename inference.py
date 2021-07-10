import torch
import torch.nn as nn

import os
from tqdm import tqdm
import numpy as np
import argparse
import random

import Config as CFG
from CustomModel import ChestClassifier
from dataset.munge_data import mungeFile


import matplotlib.pyplot as plt

from albumentations.augmentations import transforms
from dataset.CustomDataset import CustomDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



def inference_one_epoch(model, data_loader, device):
    model.eval()

    all_preds, all_labels = [], []
    

    pbar = tqdm(enumerate(data_loader), total = len(data_loader))
    for step, data in pbar:
        imgs = data["image"].to(device).float()
        labels = data["target"].to(device).long()

        pred = model(imgs)
        all_preds += [torch.argmax(pred, 1).detach().cpu().numpy()]
        all_labels += [labels.detach().cpu().numpy()]
        


    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print('Total Test accuracy = {:.4f}'.format((all_preds==all_labels).mean()))

def plot_result(data, data_df, model, i):
    classes = ["Normal", "PNEUMONIUA"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor='w')

    for ax in axes.ravel():
        idx = random.randint(0,len(data_df)-1)

        image, label = data[idx]["image"], data[idx]["target"]
        vis_image = np.array(image, dtype=np.uint8).transpose(1,2,0)
        
        pred = model(image.to(device).unsqueeze(dim=0))
        pred = torch.argmax(pred, 1).detach().cpu().numpy()
        
        ax.set_title(f"Ground truth : {classes[label]}, \n Predicted : {classes[int(pred)]}")
        ax.imshow(vis_image)

    plt.savefig(f'graph_result//GAN_CUTMIX_DA//result_{i}.png')

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
    
    for i in range(5):
        checkpoint = f"checkpoints\\GAN_CUTMIX_DA\\efficientnet_b0_{i}.pth"
        model.load_state_dict(torch.load(checkpoint))

        
        with torch.no_grad():
            inference_one_epoch(model, test_loader, device)

        plot_result(test_dataset, test_df, model, i)
    
