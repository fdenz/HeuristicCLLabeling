# NOT FOR COMMERCIAL USE
# Implementation of an architecture to predict the CAD-RADS score 
# Context described in https://openreview.net/forum?id=vVPMifME8b


import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, depth=4, features=16, shared_linear=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.in_channels = in_channels
        for i in range(depth):
            self.layers.append(nn.Conv2d(self.in_channels, features * (2 ** i), kernel_size=3, stride=1, padding=1, bias=False))
            self.layers.append(nn.BatchNorm2d(features * (2 ** i)))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), padding=0))
            self.in_channels = features * (2 ** i)
        self.layers.append(nn.AdaptiveMaxPool2d((1)))
        if shared_linear:
            self.shared_fc = nn.Linear(self.in_channels,self.in_channels)
            self.shared_relu = nn.ReLU()
        self.shared_linear = shared_linear

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.shared_linear:
            x = self.shared_fc(x.view(-1,self.in_channels))
            x = self.shared_relu(x)
        return x


class seg_MLP(nn.Module):
    def __init__(self, hidden_size, le):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(hidden_size, 216))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(216, 54))
        self.layers.append(nn.ReLU())
        if le:
            self.layers.append(nn.Linear(54, 10))
        else:
            self.layers.append(nn.Linear(54, 1))

        self.sig = nn.Sigmoid()
        self.le = le

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        if self.le:
            x = self.sig(x)
        return x


class global_MLP(nn.Module):
    def __init__(self, hidden_size, le=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(hidden_size, 216))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(216, 54))
        self.layers.append(nn.ReLU())
        if le:
            self.layers.append(nn.Linear(54, 6))
        else:
            self.layers.append(nn.Linear(54, 1))

        self.sig = nn.Sigmoid()
        self.le = le

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        if self.le:
            x = self.sig(x)
        return x


class CADRADSNet(nn.Module):
    def __init__(self, *, in_channels, conv_blocks_depth, feature_maps, hidden_size,config=None):
        super().__init__()
        self.lens = 128
        self.cb = ConvBlock(in_channels, depth=conv_blocks_depth, features=feature_maps,shared_linear=config.shared_linear)
        self.maxpool = nn.AdaptiveMaxPool1d((1))
        self.seg = seg_MLP(hidden_size, config.LE)
        self.cadrads = global_MLP(hidden_size, config.LE)
        self.CS = global_MLP(hidden_size, config.LE)
        self.config = config

    def forward(self, img):
        x = self.cb(img[:, 0][..., :self.lens]).view(img.size(0), 1, self.hidden_size)
        for i in range(1, self.config.segments):
            tmp_x = self.conv_layer(img[:, i][..., :self.lens]).view(img.size(0), 1, self.hidden_size)
            x = torch.cat((x, tmp_x), dim=1)
        if self.config.LE:
            ret = self.seg(x[:, 0]).view(img.size(0), 1, 6)
        else:
            ret = self.seg(x[:, 0]).view(img.size(0), 1, 1)
        for i in range(1, self.config.segments):
            if self.config.LE:
                ret = torch.cat((ret, self.seg(x[:, i]).view(img.size(0), 1, 6)), dim=1)
            else:
                ret = torch.cat((ret, self.seg(x[:, i]).view(img.size(0), 1, 1)), dim=1)
        branches_combined = self.maxpool(x.transpose(2, 1)).view((x.size(0), -1))
        return ret, self.cadrads(branches_combined), self.CS(branches_combined)
        
 """
 Example config:
 class ConfigObj (object) :
    segments = 13  
    in_channels=1#or2
    conv_blocks_depth=5
    feature_maps=32
    hidden_size = 2**(conv_blocks_depth-1)*feature_maps
    shared_linear=True#or False
    LE = False#or True
config = ConfigObj()
 """
