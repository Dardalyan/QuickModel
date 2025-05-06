import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from quick_model.base import BaseModel


class LinearModel(BaseModel):

    def __init__(self,input_feature:int,output_feature:int=1,num_of_layer:int = 3):
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
            if layer == self.layers[-1]:break
            x = F.relu(layer(x))  # Apply ReLU activation function for each layer
        return self.output_layer(x)

    def __set_dataset__(self,train_dataset:TensorDataset,test_dataset:TensorDataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.number_of_train_data = len(self.train_dataset)
        self.number_of_test_data = len(self.test_dataset)

    def _train(self,batch_size:int=10,shuffle:bool=True,epochs:int=1,optimizer:str='adam',lr:float=0.001):
        train_loader = DataLoader(self.train_dataset,batch_size=batch_size,shuffle=shuffle)
        criterion: nn.Module = nn.CrossEntropyLoss()

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(),lr=lr)

        self.train_losses = []

        self.train()

        # number of trainings
        for epoch in range(epochs):

            # the total loss for this epoch
            epoch_loss = 0.0

            # training here !
            for b, (x_train, y_train) in enumerate(train_loader):
                b += 1  # start batches at 1
                y_pred = self(x_train)  # get the predicted values from the training set
                # The loss in this batch
                loss = criterion(y_pred, y_train) # calculate loss for this batch

                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # We count the total loss 'till the end of this loop -> (the last batch == before the new epoch)
                epoch_loss += loss.item()

                # This printing values are to see predicted, trained and the correct ones
                """
                print("Predicted",predicted)
                print("y_train",y_train)
                print(predicted == y_train)
                """

            # calculate loss for this "epoch" -> the sum of losses for all batches at this epoch
            avg_loss = epoch_loss / len(train_loader)


            print(f"TRAINING: Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            self.train_losses.append(avg_loss)

            # At the end of the method we set model_trained -> True
            self.model_trained = True

    def _test(self,batch_size:int=10,shuffle:bool=False):
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)
        criterion: nn.Module = nn.CrossEntropyLoss()

        self.eval()

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
