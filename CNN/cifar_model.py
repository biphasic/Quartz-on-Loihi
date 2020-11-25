import torch
import torch.nn as nn
from torch.nn import functional as F
import ipdb

# class InvertedResidual(nn.Module):
#     def __init__(
#         self,
#         inp: int,
#         oup: int,
#         stride: int,
#         expand_ratio: int,
#     ) -> None:
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]

#         norm_layer = nn.BatchNorm2d

#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = self.stride == 1 and inp == oup

#         layers: List[nn.Module] = []
#         if expand_ratio != 1:
#             # pw
#             layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
#         layers.extend([
#             # dw
#             ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
#             # pw-linear
#             nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#             norm_layer(oup),
#         ])
#         self.conv = nn.Sequential(*layers)
        
class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()
        drop_conv = 0.4
        drop_dense = 0.4
        conv_bias = False
        l1_n_channels = 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=l1_n_channels, kernel_size=3, stride=1, bias=conv_bias)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout2d(0.5*drop_conv)
        self.bn1 = nn.BatchNorm2d(l1_n_channels)
        
        l2_n_channels = 64
        self.conv2 = nn.Conv2d(in_channels=l1_n_channels, out_channels=l2_n_channels, kernel_size=4, stride=1, bias=conv_bias)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout2d(drop_conv)
        self.bn2 = nn.BatchNorm2d(l2_n_channels)
        
        l3_n_channels = 200
        self.conv3 = nn.Conv2d(in_channels=l2_n_channels, out_channels=l3_n_channels, kernel_size=6, stride=1, bias=conv_bias)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.drop3 = nn.Dropout2d(drop_dense)
        self.bn3 = nn.BatchNorm2d(l3_n_channels)
        
#         l4_n_channels = 50
#         self.conv4 = nn.Conv2d(in_channels=l3_n_channels, out_channels=l4_n_channels, kernel_size=4, stride=1, bias=conv_bias)
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
#         self.drop4 = nn.Dropout2d(drop)
#         self.bn4 = nn.BatchNorm2d(l4_n_channels)

        #l5_n_channels = 200
        self.fc1 = nn.Linear(in_features=l3_n_channels, out_features=n_classes)
        #self.fc1 = nn.Linear(in_features=l3_n_channels, out_features=n_classes)


    def forward(self, x):
        l1_out = self.conv1(x)
        l1_out = F.relu(l1_out)
        l1_out = self.bn1(l1_out)
        l1_out = self.pool1(l1_out)
        l1_out = self.drop1(l1_out)

        l2_out = self.conv2(l1_out)
        l2_out = F.relu(l2_out)
        l2_out = self.bn2(l2_out)
        l2_out = self.pool2(l2_out)
        l2_out = self.drop2(l2_out)

        l3_out = self.conv3(l2_out)
        l3_out = F.relu(l3_out)
        l3_out = self.bn3(l3_out)
        l3_out = self.drop3(l3_out)
        
#         l4_out = self.conv4(l3_out)
#         l4_out = self.bn4(l4_out)
#         l4_out = F.relu(l4_out)
#         l4_out = self.drop4(l4_out)
#         #ipdb.set_trace()
        
        l4_out = torch.flatten(l3_out, 1)
        
        logits = self.fc1(l4_out)
        probs = F.softmax(logits, dim=1)
        return logits, probs, l1_out, l2_out, l3_out