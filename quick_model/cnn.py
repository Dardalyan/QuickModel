from typing import Tuple

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

    def __init__(self, train_dataset: TensorDataset, test_dataset: TensorDataset,
                 kernel_size = 3,stride = 1,
                 conv_dim:ConvDim = ConvDim.Conv2D,num_of_conv_layer:int=2,num_of_flattened_layer:int=3):
        super().__init__(train_dataset, test_dataset)

        if len(train_dataset.tensors[0].shape) == 3 :
            raise Exception("The size of tensors must be correct format -> [batch_size,channels,height,width] ")


        if train_dataset.tensors[1].ndim == 1 :
            # works if labels are in one dimension like [120]
            output_feature: int = len(set(train_dataset.tensors[1].tolist())) # how many different types available for in labels
        else:
            # works if labels are in more than one dimension like [120,10]
            output_feature: int = train_dataset.tensors[1].shape[1] # for example if labels :[120,10] so we get 10

        # H and W per Image
        self.image_height = train_dataset.tensors[0].shape[-2]
        self.image_width = train_dataset.tensors[0].shape[-1]

        #  h and w for c.o.s.f.
        h,w = self.image_height,self.image_width

        # number of  pooling layers
        self.num_of_pooling = num_of_conv_layer
        self.p_stride = 2
        self.p_kernel = 2

        # Channel Control: 1 for GrayScale | 3 for RGB | 4 for RGBA
        if train_dataset.tensors[0].shape[1] == Channels.GrayScale:
            self.channel:int = Channels.GrayScale.value
        elif train_dataset.tensors[0].shape[1] == Channels.RGB:
            self.channel:int = Channels.RGB.value
        elif train_dataset.tensors[0].shape[1] == Channels.RGBA:
            self.channel:int = Channels.RGBA.value
        else: raise Exception(
            "Uncommon or unknown chanell size ! "
            "\n The channel size mush be 1,3 or 4 which represent GrayScale, RGB and RGBA respectively."
        )

        # Convolutional 2D
        if conv_dim == ConvDim.Conv2D:

            # The first convolutional layer
            self.layers.append(nn.Conv2d(self.channel, 8, kernel_size, stride))

            # Remaining Convolutional Layers
            for i in range(num_of_conv_layer-1):
                self.layers.append(nn.Conv2d(self.channel, self.layers[-1].out_channels * 2, kernel_size, stride))

            # Calculate h and w after c.o.s.f. (n times for convolutional , n times for pooling->(after for each conv layer) )
            for index in range(self.num_of_pooling):
                h,w = self.__cosf(h,w,self.layers[index].stride,self.layers[index].kernel)
                h,w = self.__cosf(h,w,self.p_stride,self.p_kernel)


            # Fully Connected Layers

            # SET THE FIRST LINEAR LAYER (FLATENNED) ---------------------------------------------------
            if num_of_flattened_layer == 1 : # if we have just 1 flattened layer

                # The only Linear layer
                self.layers.append(nn.Linear(
                    # out_channel * the last HEIGHT * the last WIDTH
                    in_features=self.layers[-1].out_channels * h * w,
                    # Set output features based on the number of target classes which is taken from labels
                    out_features= output_feature
                ))

            else : # if we have more than 1 flattened layer
                # The first Linear layer
                self.layers.append(nn.Linear(
                    # out_channel * the last HEIGHT * the last WIDTH
                    in_features=self.layers[-1].out_channels * h * w,
                    # Set output features as the first Linear Layer
                    out_features=256
                ))
            #-----------------------------------------------------------------------------------------


            # THE REMAINING LAYERS (FLATENNED)--------------------------------------------------------
            for i in range(num_of_flattened_layer-1):

                # If it is the last layer
                if i == num_of_flattened_layer - 2:
                    self.layers.append(nn.Linear(
                        # out_channel * the last HEIGHT * the last WIDTH
                        in_features=self.layers[-1].out_channels * h * w,
                        # Set output features based on the number of target classes which is taken from labels
                        out_features=output_feature
                    ))

                # If it is NOT the last layer (we have at least 1 more layer before the last one)
                else:
                    if i == num_of_flattened_layer - 2:
                        self.layers.append(nn.Linear(
                            # out_channel * the last HEIGHT * the last WIDTH
                            in_features=self.layers[-1].out_channels * h * w,
                            # Set output features as half of the last layer's output features
                            out_features= self.layers[-1].out_features / 2
                        ))
            # -----------------------------------------------------------------------------------------




        # Convolutional 3D
        if conv_dim == ConvDim.Conv3D:
            pass

        #self.output_layer =nn.Linear(self.layers[-1].out_features,output_feature)
        #self.layers.append(self.output_layer)

    def __cosf(self, h: float, w: float,stride:int,kernel:int) -> Tuple[float, float]:

        """
        Convolutional Output Size Formula

        :param h: Height of the input image
        :param w: Width of the input image
        :param stride: How many pixels will be iterated at once
        :param kernel: The kernel (filter) size
        :return: Returns the result h,w: (h,w-kernel)/stride + 1
        """

        h = (h - kernel) / stride +1
        w = (w - kernel) / stride +1

        return h,w


    def forward(self,x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)