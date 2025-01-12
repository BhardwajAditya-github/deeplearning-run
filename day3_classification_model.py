import sklearn.datasets
import matplotlib.pyplot as plt
import sklearn.model_selection
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available else "cpu"

X,Y = sklearn.datasets.make_circles(n_samples= 1000, noise=0.03, random_state=42)

X = torch.from_numpy(X).type(torch.float)
Y = torch.from_numpy(Y).type(torch.float)

feature1 = X[:, 0]
feature2 = X[:, 1]

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

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
# modelData()

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

# CircleClassifierPreBuilt = nn.Sequential(
#     nn.Linear(in_features=2, out_features=5),
#     nn.Linear(in_features=5, out_features=1)
# )

# X_test.to(device)
# Y_test.to(device)

# Non Linear Model

class NonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

CircleClassifierPreBuilt = NonLinearModel();

loss_fcn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(CircleClassifierPreBuilt.parameters(), lr=0.1)

# print(X_train.dtype)
# print(X_train[0][0].dtype)
# print(X_train[0][1].dtype)
modelData()
# print(X_test[:5], Y_test[:5])


# Converting logits into predictn data

# y_pred_labels = torch.round(torch.sigmoid(CircleClassifierPreBuilt(X_test)[:5]))


epochs = 1600

for epoch in range(epochs):

    CircleClassifierPreBuilt.train()

    y_logits = CircleClassifierPreBuilt(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fcn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   Y_train) 
    acc = accuracy_fn(y_true=Y_train, 
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step() 

    ### Testing
    CircleClassifierPreBuilt.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = CircleClassifierPreBuilt(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fcn(test_logits,
                            Y_test)
        test_acc = accuracy_fn(y_true=Y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

    # y_logits = CircleClassifierPreBuilt(X_train)
    # print(y_logits[:5])
    # y_pred_probs = torch.sigmoid(y_logits)
    # y_preds = torch.round(y_pred_probs)
    # print(y_preds[:100])


    # Train
    # predicts = CircleClassifierPreBuilt(X_train, Y_train)
    # loss = loss_fcn(Cir)

    # CircleClassifierPreBuilt.eval()
