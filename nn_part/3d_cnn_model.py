import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
import torchvision.models as models_res
import matplotlib.pyplot as plt
import dataProcess
import dataset  


cfg = {
    'VGG11': [16, 'M', 64, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_3D(nn.Module):
    def __init__(self, vgg_name,out_feature):
        super(VGG_3D, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        print(x.size(),x.type())
        out = self.features(x)
        print(out.size(),out.type())
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                #layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
                layers += []
                
            else:
                layers += [nn.Conv3d(in_channels, x, kernel_size=2, stride=3, padding=1),
                           nn.BatchNorm3d(x),
                           nn.ELU(inplace=True)]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    
class My_3d_cnn_model(nn.Module):
    def __init__(self,out_feature=4096):
        super(My_3d_cnn_model, self).__init__()
        self.input_conv1 = nn.Sequential(nn.Conv3d(in_channels=1,out_channels=16,kernel_size=2, stride=3),nn.ELU()  ) #,nn.BatchNorm3d(32))
        self.input_conv2 = nn.Sequential(nn.Conv3d(in_channels=16,out_channels=32,kernel_size=2, stride=3),nn.ELU() ) # ,nn.BatchNorm3d(64))
        self.input_conv3 = nn.Sequential(nn.Conv3d(in_channels=32,out_channels=32,kernel_size=2, stride=3),nn.ELU(), nn.BatchNorm3d(32))
        self.input_conv4 = nn.Sequential(nn.Conv3d(in_channels=32,out_channels=64,kernel_size=2, stride=3),nn.ELU(),nn.BatchNorm3d(64))
        self.input_conv5 = nn.Sequential(nn.Conv3d(in_channels=64,out_channels=64,kernel_size=2, stride=3),nn.ELU(),nn.BatchNorm3d(64))
        #self.input_conv6 = nn.Sequential(nn.Conv3d(in_channels=256,out_channels=256,kernel_size=2, stride=3),nn.ELU(),nn.BatchNorm3d(256))
        # self.input_conv1 = nn.Conv3d(in_channels=3,out_channels=3,kernel_size=3, stride=3) 
        self.fc1 = nn.Linear(4096,512)
        self.fc2 = nn.Linear(512,out_feature)
        
        #self.fc3 = nn.Linear(32768,4096)
        #self.fc1 = nn.Sequential(nn.Linear(),nn.BatchNorm1d(512),nn.ELU(),nn.Linear(512,out_feature))

    def forward(self,x):
        print(x.size(),x.type())
        exit(0)
        out = self.input_conv1(x)
        #print(out.size(),out.type())
        out = self.input_conv2(out) #.cuda()
        #print(out.size(),out.type())
        out = self.input_conv3(out)
        #print(out.size(),out.type())
        out = self.input_conv4(out)
        #print(out.size(),out.type())
        out = self.input_conv5(out)
        #print(out.size(),out.type())
      
        
        #print(x.size()[2:])
        #out = F.max_pool3d(x,kernel_size=x.size()[2:])
        out = out.view(out.size(0),-1)
        #print(out.size(),out.type())
       
        out = self.fc1(F.relu(out))
        #print(out.size(),out.type())
        
        out = self.fc2(out)
        #print(out.size(),out.type())
       
        return out

