import torch
import torch.nn as nn
from torch.nn import functional as F
import ipdb

    
class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        drop = 0.1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout2d(0.5*drop)
        
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout2d(drop)
        
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=120, kernel_size=5, stride=1)
        
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.drop4 = nn.Dropout(drop)
        self.fc2 = nn.Linear(in_features=84, out_features=n_classes)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        pool1_out = self.pool1(conv1_out)
        drop1_out = self.drop1(pool1_out)
        ipdb.set_trace()
        #return conv1_out, pool1_out, drop1_out
        
        conv2_out = F.relu(self.conv2(drop1_out))
        pool2_out = self.pool2(conv2_out)
        drop2_out = self.drop2(pool2_out)

        conv3_out = F.relu(self.conv3(drop2_out))
        conv3_out = torch.flatten(conv3_out, 1)

        fc1_out = F.relu(self.fc1(conv3_out))
        fc1_drop = self.drop4(fc1_out)
        logits = self.fc2(fc1_drop)
        probs = F.softmax(logits, dim=1)
        return logits, probs, pool1_out, pool2_out, conv3_out, fc1_out

    
class LeNet1(nn.Module):
    def __init__(self, n_classes):
        super(LeNet1, self).__init__()
        drop = 0.1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout2d(0.5*drop)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        pool1_out = self.pool1(conv1_out)
        drop1_out = self.drop1(pool1_out)
        
        return drop1_out
