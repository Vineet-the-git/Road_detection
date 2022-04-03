"""Code to run the model and train the model"""
from lmd import LMD
import torch
from dataloader import DataLoader
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import statistics
from config import Config
from load_chk import load_check, fetch_lastepoch
from saving import save_checkpoint, save_pred, save_x, save_y
import numpy as np

weights = np.array([1.3881783357262611, 4.187831141352653, 182.51125786141174])
weights = torch.Tensor(weights)
weights = weights.cuda()
lossFunc = CrossEntropyLoss( weight = weights )

# Define training operation
def train_op(model, x, y, optimizer):
    pred = model(x)
    loss = lossFunc(pred, y)  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return pred, loss

# Function to update learning rate
def update_lr(model, learning_rate):
    learning_rate = learning_rate * 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay=0.0005)
    return optimizer, learning_rate

# Function to start training
def main(resume):
    config = Config()
    dataset = DataLoader(config.path_dataset, config.mode)
    dataloader = dataset.torch_loader()
    print("Dataloader loader successfully!")

    if resume == True:
        #fetch last checkpoint
        last_epoch = fetch_lastepoch(config.checkpoint)

        #load checkpoint
        print("Preparing the model for training!!")
        model = LMD()
        start_epoch, loss_overall, learning_rate, model_state, optimizer_state = load_check(last_epoch)
        model.load_state_dict(model_state)
        print("Model prepared!!!")
        print("The loss after {} epoch was: {}".format(start_epoch,loss))
        print("The new starting learning rate is: {}".format(learning_rate))
    else:
        print("Preparing the model for training!!")
        model = LMD()
        start_epoch = 0
        loss_overall = []
        learning_rate = config.learning_rate
        print("Model prepared!!!")
    
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = config.momentum , weight_decay = config.weight_decay)
    if resume == True:
        optimizer.load_state_dict(optimizer_state)
    
    flag = False
    for epoch in range(start_epoch, config.max_iter):
        print("Epoch: ",str(epoch+1))
        if (epoch+1)%config.lr_decay_iter == 0:
            optimizer, learning_rate = update_lr(model, learning_rate)

        Loss = []
        if (epoch+1)%config.pred_interval == 0:
            flag = False

        for [x,y] in dataloader:
            x = x.cuda()
            y = y[:,0,:,:]
            y = y.cuda()

            pred, loss = train_op(model, x, y, optimizer)
            loss = loss.detach().cpu().numpy()
            loss = float(loss)
            Loss.append(loss)

            if flag == False:
                save_pred(pred, epoch)
                save_x(x, epoch)
                save_y(y, epoch)
                flag = True

        avg_loss = statistics.mean(Loss)
        print("Loss after epoch {} : {}".format(epoch+1, avg_loss))

        loss_overall.append(avg_loss)

        if (epoch+1)%config.check_interval == 0:
            checkpoint={
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'learning_rate': learning_rate,
                            'loss': loss_overall
                        }
            save_checkpoint(checkpoint, epoch+1)


if __name__ == "__main__":
    resume = False
    main(resume)
