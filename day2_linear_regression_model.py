import torch
import matplotlib.pyplot as plt

weight = 10
bias = 5
start = 0 
end = 51
step = 1

X = torch.arange(start,end,step).unsqueeze_(1)
Y = weight*X+bias

fig, ax = plt.subplots()
print(X,Y)
ax.plot(X,Y)
plt.show()