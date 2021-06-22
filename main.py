import os
import time
import argparse
import Config as CFG

import numpy as np
import CustomModel
from tqdm import tqdm
from random import shuffle
from dataset.munge_data import mungeFile

import torch
import torch.nn as nn
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler


from albumentations.augmentations import transforms
from dataset.CustomDataset import CustomDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2





#Augmentation
def get_train_transforms():
    return A.Compose(
        [
            #A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=128, width=128, p=1),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=128, width=128, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0
    )
    

def train_one_epoch(epoch, model, train_loader, optimizer , device, loss_fn, scheduler=None, schd_batch_update = None):
    model.train()

    t = time.time()
    running_loss = None
    progress_bar = tqdm(enumerate(train_loader), total = len(train_loader))
    for idx, data in progress_bar:
        x = data["image"].to(device).float() #img
        y = data["target"].to(device).long() #label
        
        #https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html
        with autocast():
            preds = model(x)
            loss = loss_fn(preds, y)

            scaler.scale(loss).backward()
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((idx + 1) %  CFG.ACCUM_ITER == 0) or ((idx + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

            if ((idx + 1) % CFG.VERBOSE_STEP == 0) or ((idx + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                progress_bar.set_description(description)


    if scheduler is not None and not schd_batch_update:
        scheduler.step()
                

def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler = None, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []


    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, data in pbar:
        x = data["image"].to(device).float()
        y = data["target"].to(device).long()
        
        image_preds = model(x)   #output = model(input)
        #print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [y.detach().cpu().numpy()]
        
        loss = loss_fn(image_preds, y)
        
        loss_sum += loss.item() * y.shape[0]
        sample_num += y.shape[0]  

        if ((step + 1) % CFG.VERBOSE_STEP == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))        

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Epochs", type=int, default = 10, help = "train Epochs")
    parser.add_argument("--Batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--Model", type=str, default = 'efficientnet_b0', help="Torch image Model")
    
    opt = parser.parse_args()

    image_path = "C:\\Datasets\\chest_xray\\"

    #train, val
    train_path = os.path.join(image_path, "train")
    train_dataframe = mungeFile(train_path)
    train_dataframe = train_dataframe.sample(frac=1).reset_index(drop=True)

    train_len = int(len(train_dataframe))
    tr_idx = int(train_len * 0.8)

    
    train_df = train_dataframe[:tr_idx].reset_index(drop=True)
    valid_df = train_dataframe[tr_idx:].reset_index(drop=True)
    
    
    device = CFG.DEVICE
    model = CustomModel.ChestClassifier(model_arch = opt.Model, n_class = 2, pretrained=True).to(device)
    scaler = GradScaler() 
    optimizer = torch.optim.Adam(model.parameters(), lr = CFG.LEARNING_RATE, weight_decay = CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult = 1, eta_min = 1e-6)



    #dataset
    train_dataset = CustomDataset(train_path, df = train_df, target = train_df['target'], transforms = get_train_transforms(),test=False, Cutmix=True)
    valid_dataset = CustomDataset(train_path, df = valid_df, target = valid_df['target'], transforms = get_valid_transforms(),test=False, Cutmix=False)

    #dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = opt.Batch_size, shuffle=True , num_workers = CFG.NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = opt.Batch_size, shuffle=True, num_workers = CFG.NUM_WORKERS)
    
    #loss function
    loss_Train = nn.CrossEntropyLoss().to(device)
    loss_Valid = nn.CrossEntropyLoss().to(device)
    
    for epoch in range(opt.Epochs):
        train_one_epoch(epoch, model = model, loss_fn = loss_Train, optimizer=optimizer, train_loader=train_loader, device = device, scheduler=scheduler, schd_batch_update=False)

        with torch.no_grad():
            valid_one_epoch(epoch, model = model, loss_fn = loss_Valid, val_loader= valid_loader, device=device, scheduler=None, schd_loss_update=False)

        torch.save(model.state_dict(), 'checkpoints/{}_{}.pth'.format(opt.Model, epoch))

    del model, optimizer, train_loader, valid_loader, scaler, scheduler
    torch.cuda.empty_cache()
    
    
