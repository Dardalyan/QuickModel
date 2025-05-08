import torch
import torch.nn as nn
from torch.utils.data import  TensorDataset
from typing import Self


class BaseModel(nn.Module):

    def __init__ (self,train_dataset:TensorDataset,test_dataset:TensorDataset):
        super().__init__()
        self.layers:list[nn.Module] = []
        self.model_trained = False

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def forward (self,x):
        raise NotImplementedError()

    def save_model(self, path:str):
        """
        This method is used to save the model trained.
        :param path: Path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path:str) -> Self:
        """
        This method is used to load the model saved.
        :param path: Path to load the model from.
        """
        self.load_state_dict(torch.load(path))
        return self


    def train_model(self,batch_size:int=10,shuffle:bool=True,epochs:int=1,optimizer:str='adam',lr:float=0.001):
        """
         This method is used to train the model.

        :param batch_size: Batch size for DataLoader
        :param shuffle: Choose whether your data is mixed or not -> true of false
        :param epochs: Number of epochs
        :param optimizer: Optimizer is a string value and according to the given string -> 'adam' = torch.optim.Adam | 'sgd' =  torch.optim.SGD will be used.

        """
        self._train(batch_size, shuffle,epochs,optimizer,lr)

    def _train(self,batch_size:int=10,shuffle:bool=True,epochs:int=1,optimizer:str='adam',lr:float=0.001):
        """
        This method should not be used directly. This method is used in 'train_model()' method as a hook and it must be implemented in a subclass.

        According to the model type criterion will be chosen from the following options: CrossEntropyLoss, BCEWithLogitsLoss, MSELoss and L1Loss.

        !! NOTE !!
            You have to set 'model_trained' -> 'False' in order to validate the model via test_model() method.
            Unless you do, your test will raise an error.

        :param batch_size: Batch size for DataLoader
        :param shuffle: Choose whether your data is mixed or not -> true of false
        :param epochs: Number of epochs
        :param optimizer: Optimizer is a string value and according to the given -> torch.optim.Adam or torch.optim.SGD will be used.
        """
        raise NotImplementedError()


    def test_model(self,batch_size:int=10,shuffle:bool=False):
        """
        This method is used to validate the model with given test dataset via __set_dataset__() method.

        :param batch_size: Batch size for DataLoader
        :param shuffle: Choose whether your data is mixed or not -> true of false
        """

        if not self.model_trained: raise Exception("Model is not trained !")
        self._test(batch_size,shuffle)

    def _test(self,batch_size:int=10,shuffle:bool=False):
        """
        NOTE: This method should not be used directly.

        This method is used in 'test_model()' method as a hook and it must be implemented in a subclass.

        :param batch_size: Batch size for DataLoader
        :param shuffle: Choose whether your data is mixed or not -> true of false
        """
        raise NotImplementedError()

    def graph_loss(self):
        """
        Graphs the results of the training losses.
        """
        raise NotImplementedError()


    def graph_accuracy(self):
        """
        Graphs the results of the training accuracy.
        """
        raise NotImplementedError()
