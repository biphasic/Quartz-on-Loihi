import torch
import torch.nn as nn
from torch.nn import functional as F
import ipdb


class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()
        drop = 0.3
        conv_bias = True
        l1_n_channels = 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=l1_n_channels, kernel_size=3, stride=1, bias=conv_bias)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout2d(0.5*drop)
        self.bn1 = nn.BatchNorm2d(l1_n_channels)
        
        l2_n_channels = 64
        self.conv2 = nn.Conv2d(in_channels=l1_n_channels, out_channels=l2_n_channels, kernel_size=4, stride=1, bias=conv_bias)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout2d(drop)
        self.bn2 = nn.BatchNorm2d(l2_n_channels)
        
#         l3_n_channels = 64
#         self.conv3 = nn.Conv2d(in_channels=l1_n_channels, out_channels=l2_n_channels, kernel_size=3, stride=1, bias=conv_bias)
#         self.pool3 = nn.MaxPool2d(kernel_size=2)
#         self.drop3 = nn.Dropout2d(drop)
#         self.bn2 = nn.BatchNorm2d(l2_n_channels)

        l3_n_channels = 200
        self.conv3 = nn.Conv2d(in_channels=l2_n_channels, out_channels=l3_n_channels, kernel_size=6, stride=1, bias=conv_bias)
        self.drop3 = nn.Dropout2d(drop)
        self.bn3 = nn.BatchNorm2d(l3_n_channels)
        
        l4_n_channels = 100
        self.fc1 = nn.Linear(in_features=l3_n_channels, out_features=l4_n_channels)
        self.fc2 = nn.Linear(in_features=l4_n_channels, out_features=n_classes)


    def forward(self, x):
        l1_out = self.conv1(x)
        #l1_out = self.bn1(l1_out)
        l1_out = F.relu(l1_out)
        l1_out = self.drop1(l1_out)
        l1_out = self.pool1(l1_out)

        l2_out = self.conv2(l1_out)
        #l2_out = self.bn2(l2_out)
        l2_out = F.relu(l2_out)
        l2_out = self.drop2(l2_out)
        l2_out = self.pool2(l2_out)

        l3_out = self.conv3(l2_out)
        #l3_out = self.bn3(l3_out)
        l3_out = F.relu(l3_out)
        l3_out = self.drop3(l3_out)
        l3_out = torch.flatten(l3_out, 1)

        l4_out = self.fc1(l3_out)
        l4_out = F.relu(l4_out)
        
        logits = self.fc2(l4_out)
        probs = F.softmax(logits, dim=1)
        return logits, probs, l1_out, l2_out, l3_out