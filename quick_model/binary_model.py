import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from quick_model.base_model import BaseModel


class BinaryModel(BaseModel):

    def __init__(self, train_dataset: TensorDataset, test_dataset: TensorDataset, num_of_hidden_layer: int = 1):
        """
        Initializes a neural network model for binary classification using BCEWithLogitsLoss
        as the default loss function. Accuracy is also calculated during training for performance evaluation.

        The input feature size is automatically determined from the shape of X in the given train_dataset,
        based on its last dimension. The output feature is fixed to 1, which is suitable for binary classification
        tasks (e.g., labels such as [0] or [1]).

        Parameters:
        - train_dataset (TensorDataset): The dataset used for training the model.
        - test_dataset (TensorDataset): The dataset used for evaluating the model.
        - num_of_hidden_layer (int): The number of hidden layers in the model.
                                     Default is 1.
        """
        super().__init__(train_dataset, test_dataset)

        input_feature:int = train_dataset.tensors[0].shape[-1] # get input feature, (e.g., dataset shape like [120,4], it takes '4' as the input feature)
        output_feature:int = 1

        self.input_layer = nn.Linear(input_feature,16)
        self.layers.append(self.input_layer)
        self.criterion = nn.BCEWithLogitsLoss()

        for i in range(0, num_of_hidden_layer):
            lof: int = self.layers[-1].out_features
            self.layers.append(nn.Linear(lof, lof * 2))

        self.output_layer = nn.Linear(self.layers[-1].out_features, output_feature)
        self.layers.append(self.output_layer)

    def forward(self, x):
        for layer in self.layers:
            if layer == self.layers[-1]: break
            x = F.relu(layer(x))  # Apply ReLU activation function for each layer except between the last and the layer before last
        x = F.sigmoid(self.output_layer(x))  # Apply sigmoid activation function to the last layer
        return x

    def _train(self, batch_size: int = 10, shuffle: bool = True, epochs: int = 1, optimizer: str = 'adam',
               lr: float = 0.001):

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle) # convert Dataset to DataLoader
        criterion = self.criterion

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

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
                y_pred = self(x_train)  # get the predicted values from the training set

                # The loss in this batch    -! y_train.float() is for 1 and 0 possibilities to 0. and 1. !-
                loss = criterion(y_pred, y_train.float())  # calculate loss for this batch

                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # We count the total loss 'till the end of this loop -> (the last batch == before the new epoch)
                epoch_loss += loss.item()

                # This is implemented to calculate the accuracy of the training
                predicted = (torch.sigmoid(y_pred) > 0.5).float()
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
            self.train_accuracy.append(accuracy)  # to be used 'Training Accuracy'

            # At the end of the method we set model_trained -> True
            self.model_trained = True

    def _test(self, batch_size: int = 10, shuffle: bool = False):

        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)
        criterion = self.criterion

        # the total loss
        test_loss = 0.0

        # test here !!
        with torch.no_grad():  # no gradient so don't update the weights and biases with test data.
            for b, (x_test, y_test) in enumerate(test_loader):
                b += 1  # start batches at 1

                y_val = self(x_test)  # get validated Tensor

                loss = criterion(y_val, y_test.float())  # Calculate test loss
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