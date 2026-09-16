'''
train.py

for every run,

do dataset preparation.

next, check if the command was done to do a new run.

for new run, check if the folder 'runs' already existed, if not, create:
    inside the 'run' folder, create a new folder named [date&time]
    inside that folder will save the checkpoints, logs, and best model.

    /runs
        /[date&time]
            /checkpoints
            /logs
            /best_model

            
what to log:
1. date & time when the toolkit was run
2. save the setting/configurations used to run the toolkit.
3. save message error for the dataset code.
3.a log for missing expected folder.
3.b log for n specification mismatch
3.c log for n audio with no pairs
3.d log for not equal n_frames

4. date and time when the training will start, learning rate, optimizer, loss
5. log training per epoch, loss,
6. log best epoch


example:

[date&time] INFO DATASET PREPARATION
[date&time] INFO Dataset Path: ____, Total epoch run: ___, resume training: __
[date&time] INFO ....
[date&time] INFO clean wav: 
[date&time] INFO noisy wav:
[date&time] INFO audio with specification mismatch: 
[date&time] INFO audio with no pairs:
[date&time] INFO audio with no equal nframes:

[date&time] INFO Starts training:
[date&time] INFO optimizer: adam, loss: mseloss, learning rate: 
[date&time] INFO epoch 1: 
.
.
.

'''

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

import numpy as np
from torch import nn

from tqdm import tqdm

from src.preprocessing.speech_denoise_dataprep import prepare_dataset
from src.models.custom_denoising import CustomDenoiser
from src.datasets.custom_denoising_dataset import custom_denoiser_dataset

import torch.optim as optim

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("Dataset_path", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_keyword", type=str, default="train")
    parser.add_argument("--test_keyword", type=str, default="test")
    parser.add_argument("--resume_training", type=Path)
    parser.add_argument("--percentage", type=float, default=0.3)
    args = parser.parse_args()

    return args

def set_dataloader(train_dataset, test_dataset):
    load_train = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2,
                            pin_memory=torch.cuda.is_available())
    load_test = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2,
                           pin_memory=torch.cuda.is_available())
    
    return load_train, load_test

def train_model(train_dataloader, total_epochs, resume_path):
    model = CustomDenoiser()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model total parameters: ", total_params)
    print("Model trainable parameters: ", trainable_param)

    optimizer = optim.adam(model.parameters(), lr=0.01)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used for training: ", device)

    model.to(device)
    model.train()
    loss = nn.MSELoss()
    best_mse_loss = float('inf')
    start_epoch = -1

    for epoch in total_epochs:
        per_epoch_loss = 0
        
        for noisy, clean in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{total_epochs}]", leave=False):
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            model_loss = loss(output, clean)
            model_loss.backward()
            optimizer.step()
            per_epoch_loss += model_loss.item()

        avg_loss = per_epoch_loss / len(train_dataloader)


    pass


def main():
    args = setup_args()

    print("Preparing Train set.")
    prepare_train_dataset = prepare_dataset(args.Dataset_path, args.train_keyword)
    print("Preparing Test set")
    prepare_test_dataset = prepare_dataset(args.Dataset_path, args.test_keyword)

    train_dataset = custom_denoiser_dataset(prepare_train_dataset)
    test_dataset = custom_denoiser_dataset(prepare_test_dataset)
    train_dataloader, test_dataloader = set_dataloader(train_dataset, test_dataset)

    model = train_model(train_dataloader, args.n_epoch)



if __name__ == "__main__":
    main()