import torch
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

X,y = datasets.make_multilabel_classification()
print("----------------x---------------")
print(f"X - {X.shape}")
print(X[:5])
print("----------------y---------------")
print(f"Y - {y.shape}")
print(y[:5])

visualizer(X, y)