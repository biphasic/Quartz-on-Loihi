import torch
import torch.nn as nn
from torch.nn import functional as F
import ipdb


class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()        
        self.features = nn.Sequential(
            ConvBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            Bottleneck(in_channels=32, out_channels=64, expansion_factor=4, stride=2),
            Bottleneck(in_channels=64, out_channels=64, expansion_factor=4, stride=1),
            Bottleneck(in_channels=64, out_channels=128, expansion_factor=4, stride=2),
        )
        self.classifier = nn.Sequential(
            ConvPool(in_channels=128, out_channels=256, conv_kernel_size=1, pool_kernel_size=4, stride=1),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=n_classes),
        )

    def forward(self, out):
        out = self.features(out)
        logits = self.classifier(out)
        probs = F.softmax(logits, dim=1)
        return logits, probs


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.4),
            nn.ReLU6(inplace=True)
        )


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, repeats=1, stride=1):
        super(Bottleneck, self).__init__()
        hidden_dim = int(round(in_channels * expansion_factor))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expansion_factor != 1:
            layers += ConvBNReLU(in_channels, hidden_dim, kernel_size=1, stride=1)
        layers += ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
        # even though the mobilenetv2 paper advocates against the use of ReLU for this layer, our SNN only supports ReLU output
        layers += ConvBNReLU(hidden_dim, out_channels, kernel_size=1, stride=1)        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvPool(nn.Sequential):
    drop_conv = 0.2
    conv_bias = False
    def __init__(self, in_channels, out_channels, conv_kernel_size, pool_kernel_size=2, stride=1):
        super(ConvPool, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=conv_kernel_size, stride=stride, bias=self.conv_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size),
            nn.Dropout2d(self.drop_conv),
        )


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        super(depthwise_separable_conv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )