import sklearn.datasets
import matplotlib.pyplot as plt
import sklearn.model_selection
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available else "cpu"

X,Y = sklearn.datasets.make_circles(n_samples= 1000, random_state=42)

feature1 = X[:, 0]
feature2 = X[:, 1]

def modelData():
    print(f"X - {X.shape} Y - {Y.shape}")
    print(f"X-train - {X_train.shape} Y-train - {Y_train.shape}")
    print(f"X-test - {X_test.shape} Y-test - {Y_test.shape}")

def plotCircle(): 
    plt.figure(figsize=(10,7))
    plt.scatter(x=feature1, y=feature2, c=Y, cmap=plt.cm.RdYlBu, s=50)
    plt.xlabel("Feature 1 (x-coordinate)")
    plt.ylabel("Feature 2 (y-coordinate)")
    plt.show()

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state=42) 
modelData()

# Create a model

plotCircle()

# Method1
class CircleClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(in_features=1, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer1(self.layer2(x))

#Mehtod2

CircleClassifierPreBuilt = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)