import os
import time
import argparse
import Config as CFG

import numpy as np
import matplotlib.pyplot as plt
import CustomModel
from tqdm import tqdm
from random import shuffle
from dataset.munge_data import mungeFile

import torch
import torch.nn as nn
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold


from albumentations.augmentations import transforms
from dataset.CustomDataset import CustomDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2





#Augmentation
if CFG.DATA_AUG:
    def get_train_transforms():
        return A.Compose(
            [
                #A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(height=128, width=128, p=1),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                ToTensorV2(p=1.0),
            ], 
            p=1.0, 
        )
else: 
    def get_train_transforms():
        return A.Compose(
            [
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

    return running_loss
                

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
            valid_loss = loss_sum/sample_num
            description = f'epoch {epoch} loss: {valid_loss:.4f}'
            pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    valid_Acc = (image_preds_all==image_targets_all).mean()
    print('validation accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))        

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()

    return valid_Acc, valid_loss

def plot_train(fold, tr_loss, val_loss, val_acc):
    plt.rcParams.update({'font.size': 22})

    fig = plt.figure(figsize=(18,14))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.plot(tr_loss, color="darkorange" , label = "Train Loss")
    ax1.plot(val_loss, color = "mediumslateblue", label = "Valid Loss")
    ax1.set_ylabel("LOSS")
    ax1.set_xlabel("Epoch")
    ax1.set_title(f"{fold} fold Loss")
    ax1.legend(ncol=2)
    
    ax2.plot(val_acc, color="skyblue")
    ax2.set_ylabel("Valid_Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_title(f"{fold} fold Accuracy")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(f'graph_result/GAN_CUTMIX_DA/{fold}_loss_Acc.png')



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

    folds = StratifiedKFold(n_splits=CFG.FOLD, shuffle=True, random_state=42).split(np.arange(train_dataframe.shape[0]), train_dataframe.target.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        
        print("{} fold Training".format(fold))
        train_df = train_dataframe.loc[trn_idx, :].reset_index(drop=True)
        valid_df = train_dataframe.loc[val_idx, :].reset_index(drop=True)

        #hyper params
        device = CFG.DEVICE
        model = CustomModel.ChestClassifier(model_arch = opt.Model, n_class = 2, pretrained=True).to(device)
        scaler = GradScaler() 
        optimizer = torch.optim.Adam(model.parameters(), lr = CFG.LEARNING_RATE, weight_decay = CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult = 1, eta_min = 1e-6)

        loss_Train = nn.CrossEntropyLoss().to(device)
        loss_Valid = nn.CrossEntropyLoss().to(device)
        
        #dataset
        train_dataset = CustomDataset(train_path, df = train_df, target = train_df['target'], transforms = get_train_transforms(),test=False, Cutmix=CFG.TR_CUTMIX)
        valid_dataset = CustomDataset(train_path, df = valid_df, target = valid_df['target'], transforms = get_valid_transforms(),test=False, Cutmix=CFG.VAL_CUTMIX)

        #dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = opt.Batch_size, shuffle=True , num_workers = CFG.NUM_WORKERS)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = opt.Batch_size, shuffle=True, num_workers = CFG.NUM_WORKERS)
        
        best_score = 0

        train_loss_list = []
        val_loss_list = []
        val_acc_list = []

        for epoch in range(opt.Epochs):
            train_loss = train_one_epoch(epoch, model = model, loss_fn = loss_Train, optimizer=optimizer, train_loader=train_loader, device = device, scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                val_score, valid_loss = valid_one_epoch(epoch, model = model, loss_fn = loss_Valid, val_loader= valid_loader, device=device, scheduler=None, schd_loss_update=False)

            train_loss_list.append(train_loss)
            val_loss_list.append(valid_loss)
            val_acc_list.append(val_score)

            if best_score < val_score:
                best_score = val_score
                torch.save(model.state_dict(), 'checkpoints/Gan_CUTMIX_DA/{}_{}.pth'.format(opt.Model, fold))

        plot_train(fold,train_loss_list, val_loss_list,val_acc_list )

        del model, optimizer, train_loader, valid_loader, scaler, scheduler
        torch.cuda.empty_cache()

        
            

