import sklearn.model_selection
import torch
import torch.nn as nn
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def data_information(X, y):
    print("----------------x---------------")
    print(f"X - {X.shape}")
    print(f"Size - {X.size}")
    print(X[:5])
    print("----------------y---------------")
    print(f"Y - {y.shape}")
    print(f"Size - {y.size}")
    print(y[:5])

def visualizer(X, y):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    y_single_label = y.argmax(axis=1)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_single_label, cmap=plt.cm.RdYlBu, s=40)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("2D Visualization of Multilabel Classification Data")
    plt.colorbar()
    plt.show()

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

dataset_size = 1000

X,y = datasets.make_multilabel_classification(n_samples=dataset_size)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# data_information(X, y)
# visualizer(X, y)

device = "cuda" if torch.cuda.is_available else "cpu"

# Split Data

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
# data_information(X_train, X_test)
# data_information(y_train, y_test)

class MultiClassClassification(nn.Module):
    def __init__(self):
        super().__init__();
        self.layer1 = nn.Linear(in_features=20, out_features=160)
        self.layer2 = nn.Linear(in_features=160, out_features=160)
        self.layer3 = nn.Linear(in_features=160, out_features=5)
        self.relu = nn.ReLU()


    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))
    
MultiClassify = MultiClassClassification()

loss_fcn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(MultiClassify.parameters(), lr=0.01)

# Model Training

# print(MultiClassify(X_train).softmax(dim=1))

epochs = 100

for epoch in range(epochs):
    MultiClassify.train()

    train_pred = MultiClassify(X_train).softmax(dim=1)
    train_pred_not_in_logits = torch.round(torch.sigmoid(train_pred.squeeze()))
    train_loss = loss_fcn(y_train, train_pred)

    train_acc = accuracy_fn(y_true=y_train,
                               y_pred=train_pred_not_in_logits)

    optimizer.zero_grad()
    train_loss.backward()
    MultiClassify.eval()

    optimizer.step()

    with torch.inference_mode():
        test_pred = MultiClassify(X_test).softmax(dim=1)
        test_pred_not_in_logits = torch.round(torch.sigmoid(test_pred.squeeze()))
        test_loss = loss_fcn(test_pred, y_test)

        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred_not_in_logits)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {train_loss:.5f}, Accuracy: {train_acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")