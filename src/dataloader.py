"""Loading and preparing the dataset"""

import torch
import torch.utils.data as Data
import numpy as np
from PIL import Image
import cv2
import os
from tqdm import tqdm
from config import Config

torch.manual_seed(17)

config = Config()
"""
original image size: (720, 1080, 3)
Resize image to size: (480, 720, 3)
"""

class DataLoader():
    def __init__(self, file_path, mode):
        self.file_path = file_path
        self.mode = mode

        if mode == "train" or mode == "test":
            self.path = os.path.join(self.file_path, mode)
        else:
            raise TypeError("Mode can only have value: train or test")

        # image_list: contains names of all the train or test images
        image_list = os.listdir(os.path.join(self.path, "Image"))

        # path_images: contains path to the images in the train/test folder
        # path_labels: contains path to the labels in the train/test folder
        path_images = []
        path_labels = []

        for name in image_list[:200]:
            path_images.append(os.path.join(self.path, "Image", name))
            path_labels.append(os.path.join(self.path, "masks", name))

        # self.images: contains all the images from the dataset
        # self.labels: contains all the labels from the dataset
        self.images = []
        self.labels = []

        print("Path to images has been prepared, Images will be loaded now!")

        for image,label in tqdm(zip(path_images,path_labels), desc = "Loading..."):
            img = Image.open(image)
            lbl = Image.open(label)
            img = np.array(img.resize((720,480),Image.BILINEAR))
            lbl = np.array(lbl.resize((720,480),Image.BILINEAR))
            self.images.append(img)
            self.labels.append(lbl)

        # permutes the shape of image from (480, 720, 3) to (1, 3, 480, 720)
        self.scale()
        print("Shape of loaded stack of Images: {}".format(self.images.shape))
        print("Shape of loaded stack of masks: {}".format(self.labels.shape))

        self.dataset = self.get_dataset()


    def scale(self):
        for i in tqdm(range(len(self.images)), desc = "Converting to array..."):
            image = self.images[i]
            label = self.labels[i]
            self.images[i] = np.stack((image[:,:,0],image[:,:,1],image[:,:,2]),axis = 0)
            self.labels[i] = np.stack((label[:,:,0],label[:,:,1],label[:,:,2]),axis = 0)
        
        self.images = np.stack(self.images,axis = 0) 
        self.labels = np.stack(self.labels,axis = 0)  
        print("Images and Labels are loaded successfully!!!") 

    def get_dataset(self):
        print("Creating the dataset.")
        dataset = (Data.TensorDataset(torch.from_numpy(self.images/256).float(),torch.from_numpy(self.labels).long())) 
        print("Dataset created Successfully!!!")
        return dataset

    def torch_loader(self):
        return Data.DataLoader(dataset=self.dataset,
                                                   batch_size = config.batch_size,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   )

