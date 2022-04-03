import torch
from config import Config
import os

config = Config()

def load_check(epoch):
    checkpoint = torch.load(os.path.join(config.checkpoint,epoch))
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['learning_rate'], checkpoint['model_state'], checkpoint['optimizer']

def fetch_lastepoch(path):
    ls = os.listdir(path)
    if len(ls) == 0:
        raise Exception("Training cannot be continued!")
    else:
        print("Training will begin from epoch {}...".format(ls[-1]))