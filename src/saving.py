from config import Config
import torch
import numpy as np
from PIL import Image
import os
from config import Config

config = Config()

# Function to save predictions
def save_pred(image_tensor, epoch):
    img = (image_tensor[0].argmax(dim = 0)).cpu().detach().numpy()
    new_img = modify_lbl(img)
    img = new_img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(os.path.join(config.prediction,"masks_pred",str(epoch+1)+".jpg"))

# Function to save an image
def save_x(image_tensor, epoch):
    img = image_tensor[0].cpu().detach()
    img = img.numpy()
    img = img.transpose(1, 2, 0)*255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(os.path.join(config.prediction,"Images",str(epoch+1)+".jpg"))

# Function to save an ground truth
def save_y(image_tensor, epoch):
    img = image_tensor[0].cpu().detach()
    img = img.numpy()
    img = modify_lbl(img)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(os.path.join(config.prediction,"masks_grd",str(epoch+1)+".jpg"))

def modify_lbl(lbl):
        mask = np.zeros((lbl.shape[0],lbl.shape[1],3))
        class0 = (lbl == 0)
        class1 = (lbl == 1)
        class2 = (lbl == 2)
        mask[:,:,0] = class0
        mask[:,:,1] = class1
        mask[:,:,2] = class2
        return mask * 255

# Function to save the checkpoint
def save_checkpoint(checkpoint, epoch):
    torch.save(checkpoint, os.path.join(config.checkpoint,str(epoch)+'.pt'))
    print("Model saved successfully after {} epochs".format(checkpoint['epoch']))
