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


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("Dataset_path", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_keyword", type=str, default="train")
    parser.add_argument("--test_keyword", type=str, default="test")
    parser.add_argument("--resume_training", type=Path)
    args = parser.parse_args()

    return args