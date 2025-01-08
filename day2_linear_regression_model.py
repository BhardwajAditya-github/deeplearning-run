import torch
import torch.nn as nn
import matplotlib.pyplot as plt

weight = 10
bias = 5
start = 0 
end = 1
step = .02

torch.manual_seed(42)

X = torch.arange(start,end,step).unsqueeze_(1)
Y = weight*X+bias

# fig, ax = plt.subplots()
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.legend()
# ax.plot(X, Y, label='Line')
# ax.plot(X, Y, 'o', markersize=5,color='red')
# plt.show()

# Splitting data into train and test

split_index = int(0.8 * len(X))

X_Train, Y_Train = X[:split_index], Y[:split_index]
X_Test, Y_Test = X[split_index:], Y[split_index:]

class LinearRegressionMode(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    
    # Forward method to define computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    

# print(X,Y,X_Train,Y_Train,X_Test,Y_Test)

# Creating model object

model = LinearRegressionMode()
model_params = list(model.parameters())
print(model_params)
print(model.state_dict())