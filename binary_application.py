from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import torch
from torch.utils.data import  TensorDataset
from quick_model import BinaryModel
import pandas as pd

data = load_breast_cancer(as_frame=True)

dataFrame = pd.DataFrame(data.data, columns=data.feature_names)
dataFrame['target'] = data.target

print(dataFrame.head())
"""
   mean radius  mean texture  ...  worst fractal dimension  target
0        17.99         10.38  ...                  0.11890       0
1        20.57         17.77  ...                  0.08902       0
2        19.69         21.25  ...                  0.08758       0
3        11.42         20.38  ...                  0.17300       0
4        20.29         14.34  ...                  0.07678       0
"""


X = dataFrame.drop('target',axis=1)
y = dataFrame['target']


X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41)

# Convert numpy arrays to torch tensors (TRAINING)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # y_train for classification should be long type

# Convert numpy arrays to torch tensors (TEST)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# Put the training and test data into TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create Model
model = BinaryModel(train_dataset,test_dataset,3)

# Train Model
model.train_model(batch_size=10,epochs=200,lr=0.001,optimizer='adam')

# Test Model
model.test_model(batch_size=10,shuffle=False)

# Graph Loss
model.graph_loss()

# Graph Accuracy
model.graph_accuracy()
