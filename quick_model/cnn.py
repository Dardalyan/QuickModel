import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from quick_model.base import BaseModel


class CNNModel(BaseModel):

    def __init__(self,input_feature:int,output_feature:int = 1,num_of_layer:int = 3):
        super().__init__()

        self.input_layer = nn.Linear(input_feature,16)
        self.layers.append(self.input_layer)


        for i in range(0,num_of_layer-2):
            lof:int = self.layers[-1].out_features
            self.layers.append(nn.Linear(lof,lof * 2))

        self.output_layer =nn.Linear(self.layers[-1].out_features,output_feature)
        self.layers.append(self.output_layer)


    def forward(self,x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)