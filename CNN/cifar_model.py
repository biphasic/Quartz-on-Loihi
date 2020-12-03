import torch
import torch.nn as nn
from torch.nn import functional as F
import ipdb


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

    
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, expansion_factor, output_channels, repeats, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)


class ConvPool(nn.Module):
    drop_conv = 0.4
    conv_bias = False
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvPool, self).__init__()
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=stride, bias=self.conv_bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(self.drop_conv),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        return self.conv_pool(x)

        
class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()
        drop_dense = 0.4
        
        features = []
        features.append(ConvPool(in_channels=3, out_channels=32, kernel_size=3, stride=1))
        features.append(ConvPool(in_channels=32, out_channels=64, kernel_size=4, stride=1))
        self.features = nn.Sequential(*features)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=200, kernel_size=6, stride=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(200),
            nn.Flatten(),
            nn.Linear(in_features=200, out_features=n_classes),
        )

    def forward(self, out):
        out = self.features(out)
        logits = self.classifier(out)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    
class Logger:
    def __init__(self):
        self.outputs = []
        self.n_layers = 0
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        del(self.outputs)
        self.outputs = []