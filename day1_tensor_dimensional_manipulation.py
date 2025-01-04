import torch

matA = torch.randint(1,9,(2,3))
print(matA)

reshaped_matA = torch.reshape(matA, (3,2))
print(reshaped_matA)

x = torch.zeros(2, 1, 2, 1, 2)
print(x)