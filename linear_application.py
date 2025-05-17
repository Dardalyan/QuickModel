from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import torch
from torch.utils.data import  TensorDataset
from quick_model import LinearModel
import pandas as pd

data = fetch_california_housing(as_frame=True)

dataFrame = pd.DataFrame(data.data, columns=data.feature_names)
dataFrame['target'] = data.target

print(dataFrame.head())
"""
   MedInc  HouseAge  AveRooms  AveBedrms  ...  AveOccup  Latitude  Longitude  target
0  8.3252      41.0  6.984127   1.023810  ...  2.555556     37.88    -122.23   4.526
1  8.3014      21.0  6.238137   0.971880  ...  2.109842     37.86    -122.22   3.585
2  7.2574      52.0  8.288136   1.073446  ...  2.802260     37.85    -122.24   3.521
3  5.6431      52.0  5.817352   1.073059  ...  2.547945     37.85    -122.25   3.413
4  3.8462      52.0  6.281853   1.081081  ...  2.181467     37.85    -122.25   3.422
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
model = LinearModel(train_dataset,test_dataset,2)

# Train Model
model.train_model(batch_size=5,epochs=100,lr=0.001,optimizer='adam')

# Test Model
model.test_model(batch_size=5,shuffle=False)

# Graph Loss
model.graph_loss()

