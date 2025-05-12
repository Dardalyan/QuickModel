from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import  TensorDataset
from quick_model import MultiClassModel
import pandas as pd

dataFrame = pd.read_csv('data/iris.csv')
"""
   sepal.length  sepal.width  petal.length  petal.width variety
0           5.1          3.5           1.4          0.2  Setosa
1           4.9          3.0           1.4          0.2  Setosa
2           4.7          3.2           1.3          0.2  Setosa
3           4.6          3.1           1.5          0.2  Setosa
4           5.0          3.6           1.4          0.2  Setosa
"""

set_of_variety = set(dataFrame['variety'])
#print(set_of_variety) # -> output : {'Virginica', 'Setosa', 'Versicolor'}

# then we need to change these information with values instead of string values
dataFrame['variety'] = dataFrame['variety'].replace('Virginica',0)
dataFrame['variety'] = dataFrame['variety'].replace('Setosa',1)
dataFrame['variety'] = dataFrame['variety'].replace('Versicolor',2)


# train test split
X = dataFrame.drop('variety',axis=1) # inputs
y = dataFrame['variety'] # output

# Converting them into <class 'numpy.ndarray'>
X = X.values
y = y.values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41)

# Convert numpy arrays to torch tensors (TRAINING)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # y_train for classification should be long type

# Convert numpy arrays to torch tensors (TEST)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Put the training and test data into TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create Model
model = MultiClassModel(train_dataset,test_dataset,3)

# Train Model
model.train_model(batch_size=10,epochs=200,lr=0.001,optimizer='adam')

# Test Model
model.test_model(batch_size=10,shuffle=False)

# Graph Loss
model.graph_loss()

# Graph Accuracy
model.graph_accuracy()

