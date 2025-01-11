import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# def plot_predictions(train_data=X_Train, 
#                      train_labels=X_Train, 
#                      test_data=X_Test, 
#                      test_labels=y_Test, 
#                      predictions=None):
#   """
#   Plots training data, test data and compares predictions.
#   """
#   plt.figure(figsize=(10, 7))

#   # Plot training data in blue
#   plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
#   # Plot test data in green
#   plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

#   if predictions is not None:
#     # Plot the predictions in red (predictions were made on the test data)
#     plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

#   # Show the legend
#   plt.legend(prop={"size": 14});

weight = 0.7
bias = 0.3
start = 0 
end = 2
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
print(model.state_dict())

with torch.inference_mode():
    y_preds = model(X_Test)

# print(y_preds)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.01)

print(f"Before Training - {list(model.parameters())}")

# building training loop
epochs = 85

for epoch in range(epochs):
  print(f"Epoch {epoch}")
  model.train() # Tracking gradient

  y_preds = model(X_Train)
  loss = loss_fn(y_preds, Y_Train)

  # Optimize zero grad
  optimizer.zero_grad()

  # Backpropagation on loss wrt to params of model
  loss.backward()

  # Progress the optimizer
  optimizer.step()

  model.eval() # Stop Tracking gradient

  with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model(X_Test)

      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, Y_Test.type(torch.float))

  print(f"After Training - {list(model.parameters())}")



test_predictn = model(4)
print(test_predictn)