from distutils.command.config import config
import torch
import numpy as np
import os
from PIL import Image
from dataloader import DataLoader
from config import Config

if __name__ == "__main__":
    cfg = Config()
    dataset = DataLoader(cfg.path_dataset, cfg.mode)
    dataloader = dataset.torch_loader()
    w1 = []
    w2 = []
    w3 = []
    total_pixel = 480.0 * 720.0
    total_pixel = [total_pixel]
    total_pixel = torch.Tensor(total_pixel)

    for [_,y] in dataloader:
        y = y[0,0,:,:]
        count0 = torch.sum((y == 0))
        count1 = torch.sum((y == 1))
        count2 = torch.sum((y == 2))
        
        if count0 != 0:
            w1.append((total_pixel/count0).numpy()[0])
        if count1 != 0:
            w2.append((total_pixel/count1).numpy()[0])
        if count2 != 0:
            w3.append((total_pixel/count2).numpy()[0])
            
    W1 = sum(w1)/len(w1)
    W2 = sum(w2)/len(w2)
    W3 = sum(w3)/len(w3)
    print(W1 ,W2 ,W3)    