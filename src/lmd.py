import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LMD(nn.Module):
    def __init__(self) -> None:
        super(LMD, self).__init__()

        batchNorm_momentum = 0.1

        # Block1
        self.econv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.ebn1_1 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.econv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.ebn1_2 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)


        # Block2
        self.econv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.ebn2_1 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.econv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.ebn2_2 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        # Block3
        self.econv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.ebn3_1 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.econv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.ebn3_2= nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.econv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.ebn3_3 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        # Block4
        self.econv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.ebn4_1 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.econv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.ebn4_2 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.econv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.ebn4_3 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        # Dilated convolution layers (to be modified)
        self.econv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, dilation=2)
        self.ebn5_1 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.econv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, dilation=2)
        self.ebn5_2 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.econv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, dilation=2)
        self.ebn5_3 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        # Layer for matching number of feature channels
        self.econv6 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.ebn6 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        # Decoder Block 1
        self.dconv1_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dbn1_1 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.dconv1_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dbn1_2 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.dconv1_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dbn1_3 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        # Decoder Block 2
        self.dconv2_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dbn2_1 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.dconv2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dbn2_2 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        # Decoder Block 3
        self.dconv3_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dbn3_1 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.dconv3_2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.dbn3_2 = nn.BatchNorm2d(3, momentum= batchNorm_momentum)

    def forward(self, x):

        # Encoder Block
        x = F.relu(self.ebn1_1(self.econv1_1(x)))
        x = F.relu(self.ebn1_2(self.econv1_2(x)))
        x, id1 = F.max_pool2d(x ,kernel_size=2, stride=2, return_indices=True)

        x = F.relu(self.ebn2_1(self.econv2_1(x)))
        x = F.relu(self.ebn2_2(self.econv2_2(x)))
        x, id2 = F.max_pool2d(x ,kernel_size=2, stride=2, return_indices=True)

        x = F.relu(self.ebn3_1(self.econv3_1(x)))
        x = F.relu(self.ebn3_2(self.econv3_2(x)))
        x = F.relu(self.ebn3_3(self.econv3_3(x)))
        x, id3 = F.max_pool2d(x ,kernel_size=2, stride=2, return_indices=True)

        x = F.relu(self.ebn4_1(self.econv4_1(x)))
        x = F.relu(self.ebn4_2(self.econv4_2(x)))
        x = F.relu(self.ebn4_3(self.econv4_3(x)))

        x = F.relu(self.ebn5_1(self.econv5_1(x)))
        x = F.relu(self.ebn5_2(self.econv5_2(x)))
        x = F.relu(self.ebn5_3(self.econv5_3(x)))

        x = F.relu(self.ebn6(self.econv6(x)))

        # Decoder Block
        x = F.max_unpool2d(x, id3, kernel_size=2, stride=2)
        x = F.relu(self.dbn1_1(self.dconv1_1(x)))
        x = F.relu(self.dbn1_2(self.dconv1_2(x)))
        x = F.relu(self.dbn1_3(self.dconv1_3(x)))

        x = F.max_unpool2d(x, id2, kernel_size=2, stride=2)
        x = F.relu(self.dbn2_1(self.dconv2_1(x)))
        x = F.relu(self.dbn2_2(self.dconv2_2(x)))

        x = F.max_unpool2d(x, id1, kernel_size=2, stride=2)
        x = F.relu(self.dbn3_1(self.dconv3_1(x)))
        x = F.relu(self.dbn3_2(self.dconv3_2(x)))

        x = F.softmax(x, dim = 1)

        return x

if __name__ == "__main__":
    model = LMD()
    x = np.ones((4, 3, 480, 720))
    x = torch.Tensor(x)
    model.eval()
    output = model(x)
    print(output.size())
