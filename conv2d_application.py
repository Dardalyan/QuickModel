from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from quick_model import Conv2DModel

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data/cnn_data', train=True, download=False, transform=transform)
test_data = datasets.MNIST(root='./data/cnn_data', train=False, download=False, transform=transform)

# Make the size [batch,channel,h,w] of data to avoid from exception
train_data.data = train_data.data.unsqueeze(1)
test_data.data = test_data.data.unsqueeze(1)

# Put the training and test data into TensorDatasets
train_dataset = TensorDataset(train_data.data, train_data.targets)
test_dataset = TensorDataset(test_data.data, test_data.targets)

# Create Model
model = Conv2DModel(train_dataset,test_dataset,3,1,3,3)

# Train Model
model.train_model(batch_size=10,epochs=5,lr=0.001,optimizer='adam')

# Test Model
model.test_model(batch_size=10,shuffle=False)

# Graph Loss
model.graph_loss()

# Graph Accuracy
model.graph_accuracy()