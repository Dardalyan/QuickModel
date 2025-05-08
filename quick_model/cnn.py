import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from quick_model.base import BaseModel
from enum import Enum

class ConvDim(Enum):
    Conv2D = '2D'
    Conv3D = '3D'

class Channels(Enum):
    GrayScale = 1
    RGB       = 3
    RGBA      = 4

class CNNModel(BaseModel):

    def __init__(self, train_dataset: TensorDataset, test_dataset: TensorDataset, kernel_size = 3,stride = 1,conv_dim:ConvDim = ConvDim.Conv2D):
        super().__init__(train_dataset, test_dataset)

        self.image_height = train_dataset.tensors[0].shape[-2]
        self.image_weight = train_dataset.tensors[0].shape[-1]

        if len(train_dataset.tensors[0].shape) == 3:
            self.channel:int = Channels.GrayScale.value
        if len(train_dataset.tensors[0].shape) == 4:
            if train_dataset.tensors[0].shape[1] == Channels.RGB:
                self.channel:int = Channels.RGBA.value
            elif train_dataset.tensors[0].shape[1] == Channels.RGBA:
                self.channel:int = Channels.RGB.value
            else: raise Exception(
                "Uncommon or unknown chanell size ! "
                "\n The channel size mush be 1,3 or 4 which represent GrayScale, RGB and RGBA respectively."
            )

        if conv_dim == ConvDim.Conv2D:
            self.input_layer = nn.Conv2d(self.channel,10,kernel_size,stride)
            self.layers.append(self.input_layer)



        if conv_dim == ConvDim.Conv3D:
            pass

        #self.output_layer =nn.Linear(self.layers[-1].out_features,output_feature)
        #self.layers.append(self.output_layer)


    def forward(self,x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)