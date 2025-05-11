from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from quick_model.base import BaseModel
from enum import Enum

class Channels(Enum):
    GrayScale = 1
    RGB       = 3
    RGBA      = 4

class Conv2DModel(BaseModel):

    def __init__(self, train_dataset: TensorDataset,
                 test_dataset: TensorDataset,
                 kernel_size = 3,stride = 1,
                 num_of_conv_layer:int=2,
                 num_of_flattened_layer:int=3):
        """

        :param train_dataset:
        :param test_dataset:
        :param kernel_size:
        :param stride:
        :param num_of_conv_layer:
        :param num_of_flattened_layer:
        """
        super().__init__(train_dataset, test_dataset)

        self.conv_layers = nn.ModuleList()
        self.flattened_layers = nn.ModuleList()
        self.criterion = nn.CrossEntropyLoss()

        # Channel Control: 1 for GrayScale
        if len(train_dataset.tensors[0].shape) == 3 :
            raise Exception("Uncommon or unknown chanell size ! "
                    "\n The channel size mush be 1,3 or 4 which represent GrayScale, RGB and RGBA respectively.")

        # Channel Control: 3 for RGB | 4 for RGBA
        if train_dataset.tensors[0].shape[1] == Channels.GrayScale.value:
            self.channel: int = Channels.GrayScale.value
        elif train_dataset.tensors[0].shape[1] == Channels.RGB.value:
            self.channel: int = Channels.RGB.value
        elif train_dataset.tensors[0].shape[1] == Channels.RGBA.value:
            self.channel: int = Channels.RGBA.value
        else:
            raise Exception(
                "Uncommon or unknown chanell size ! "
                "\n The channel size mush be 1,3 or 4 which represent GrayScale, RGB and RGBA respectively."
            )

        if train_dataset.tensors[1].ndim == 1 :
            # works if labels are in one dimension like [120]
            self.output_feature: int = len(set(train_dataset.tensors[1].tolist())) # how many different types available for in labels
        else:
            # works if labels are in more than one dimension like [120,10]
            self.output_feature: int = train_dataset.tensors[1].shape[1] # for example if labels :[120,10] so we get 10

        # H and W per Image
        self.image_height = train_dataset.tensors[0].shape[-2]
        self.image_width = train_dataset.tensors[0].shape[-1]

        #  h and w for c.o.s.f.
        h,w = self.image_height,self.image_width

        # number of  pooling layers
        self.num_of_pooling = num_of_conv_layer
        self.p_stride:int = 2
        self.p_kernel:int = 2

        # The first convolutional layer
        self.conv_layers.append(nn.Conv2d(self.channel, 8, kernel_size, stride))

        # Remaining Convolutional Layers
        for i in range(num_of_conv_layer-1):
            self.conv_layers.append(nn.Conv2d(self.conv_layers[-1].out_channels, self.conv_layers[-1].out_channels * 2, kernel_size, stride))

        # Calculate h and w after c.o.s.f. (n times for convolutional , n times for pooling->(after for each conv layer) )
        for index in range(self.num_of_pooling):
            h,w = self.__cosf(h,w,self.conv_layers[index].stride[0],self.conv_layers[index].kernel_size[0])
            h,w = self.__cosf(h,w,self.p_stride,self.p_kernel)


        # Fully Connected Layers

        # SET THE FIRST LINEAR LAYER (FLATENNED) ---------------------------------------------------
        if num_of_flattened_layer == 1 : # if we have just 1 flattened layer

            # The only Linear layer
            self.flattened_layers.append(nn.Linear(
                # out_channel * the last HEIGHT * the last WIDTH
                in_features=self.conv_layers[-1].out_channels * h * w,
                # Set output features based on the number of target classes which is taken from labels
                out_features= self.output_feature
            ))

        else : # if we have more than 1 flattened layer
            # The first Linear layer
            self.flattened_layers.append(nn.Linear(
                # out_channel * the last HEIGHT * the last WIDTH
                in_features=self.conv_layers[-1].out_channels * h * w,
                # Set output features as the first Linear Layer
                out_features=256
            ))
        #-----------------------------------------------------------------------------------------


        # THE REMAINING LAYERS (FLATENNED) --------------------------------------------------------
        for i in range(num_of_flattened_layer-1):

            # If it is the last layer
            if i == num_of_flattened_layer - 2:
                self.flattened_layers.append(nn.Linear(
                    # out features of the last Linear Layer
                    in_features=self.flattened_layers[-1].out_features ,
                    # Set output features based on the number of target classes which is taken from labels
                    out_features=self.output_feature
                ))

            # If it is NOT the last layer (we have at least 1 more layer before the last one)
            else:
                self.flattened_layers.append(nn.Linear(
                    # out features of the last Linear Layer
                    in_features=self.flattened_layers[-1].out_features,
                    # Set output features as half of the last layer's output features
                    out_features= self.flattened_layers[-1].out_features // 2
                ))
        # -----------------------------------------------------------------------------------------

    def forward(self,x):


        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = F.max_pool2d(x, self.p_kernel, self.p_stride)

        # Make x one dimensional (flattened)
        x = x.view(-1,self.flattened_layers[0].in_features)

        for flattened in self.flattened_layers:

            # If the current layer is the last layer
            if flattened == self.flattened_layers[-1]:

                # for binary classification
                if self.output_feature == 2:
                    x = F.sigmoid(flattened(x))

                # for multiclass classification
                if self.output_feature > 2:
                    x = F.log_softmax(flattened(x),dim=1)
            else:
                x = F.relu(flattened(x))

        return x








    def __cosf(self, h: float, w: float,stride:int,kernel:int) -> Tuple[float, float]:

        """
        Convolutional Output Size Formula

        :param h: Height of the input image
        :param w: Width of the input image
        :param stride: How many pixels will be iterated at once
        :param kernel: The kernel (filter) size
        :return: Returns the result h,w: (h,w-kernel)/stride + 1
        """

        h = (h - kernel) // stride +1
        w = (w - kernel) // stride +1

        return h,w



    def _train(self,batch_size:int=10,shuffle:bool=True,epochs:int=1,optimizer:str='adam',lr:float=0.001):

        self.train() # set model train mode on.
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle) # convert Dataset to DataLoader
        criterion = self.criterion

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(),lr=lr)

        self.train_losses = []
        self.train_accuracy = []

        # number of trainings
        for epoch in range(epochs):

            # the total loss for this epoch
            epoch_loss = 0.0

            # The total number of correct predictions this epoch
            train_correct = 0

            # training here !
            for b, (x_train, y_train) in enumerate(train_loader):
                b += 1  # start batches at 1
                x_train = x_train.float()
                y_pred = self(x_train)  # get the predicted values from the training set
                # The loss in this batch
                loss = criterion(y_pred, y_train) # calculate loss for this batch


                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # We count the total loss 'till the end of this loop -> (the last batch == before the new epoch)
                epoch_loss += loss.item()

                # This is implemented to calculate the accuracy of the training
                predicted = torch.max(y_pred, dim=1)[1]
                train_correct += (predicted == y_train).sum().item()

                # This printing values are to see predicted, trained and the correct ones
                """
                print("Predicted",predicted)
                print("y_train",y_train)
                print(predicted == y_train)
                print("train_correct:",train_correct)
                """

            # calculate loss for this "epoch" -> the sum of losses for all batches at this epoch
            # calculated per the number of batch
            # divided by length of the train loader which is len(self.train_dataset) / batch_size = len(train_loader)
            avg_loss = epoch_loss / len(train_loader)

            # calculate accuracy for this "epoch" -> the sum of correct predictions for all batches at this epoch
            # calculated per the number of batch
            # divided by NOT length of the train loader, divided by the actual length of the train_dataset
            accuracy = train_correct / (len(train_loader) * batch_size)

            print(f"TRAINING: Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            self.train_losses.append(avg_loss)
            self.train_accuracy.append(accuracy) # to be used 'Training Accuracy'

            # At the end of the method we set model_trained -> True
            self.model_trained = True

    def _test(self,batch_size:int=10,shuffle:bool=False):

        self.eval()  # set evaluation mode on
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)
        criterion = self.criterion

        # the total loss
        test_loss = 0.0

        # test here !!
        with torch.no_grad():  # no gradient so don't update the weights and biases with test data.
            for b, (x_test, y_test) in enumerate(test_loader):
                b += 1  # start batches at 1

                y_val = self(x_test) # get validated Tensor

                loss = criterion(y_val, y_test) # Calculate test loss
                test_loss += loss.item()

        # the average loss
        avg_loss = test_loss / len(test_loader)

        print(f"TEST: Loss: {avg_loss:.4f}")



    def graph_loss(self):
        # Graph Loss Results
        train_losses = [train_loss for train_loss in self.train_losses]

        # Graph the loss and epochs
        plt.plot(train_losses, label="Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss At Epochs")
        plt.legend()
        plt.show()

    def graph_accuracy(self):
        # Graph Accuracy Results

        plt.plot([t for t in self.train_accuracy], label="Training Accuracy")
        plt.title("Accuracy at the end of Epochs")
        plt.legend()
        plt.show()