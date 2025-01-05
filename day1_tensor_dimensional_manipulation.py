import torch

matA = torch.randint(1,9,(2,3))
# print(matA)

reshaped_matA = torch.reshape(matA, (3,2))
# print(reshaped_matA)

x = torch.zeros(2, 1, 2, 1, 2)
matA_3d = torch.atleast_3d(matA)
# print(matA_3d)
# print(matA_3d.shape)

matB = torch.randint(1,5,(2,3))
matC = torch.randint(1,5,(2,3))
vstack_matb = torch.vstack([matB,matC])
hstack_matb = torch.hstack([matB,matC])
# print(vstack_matb)
# print(hstack_matb)

matD = torch.randint(1,3,(2,2,2,3))
matD_squeezed = torch.squeeze(matD)
# print(matD)
# print(matD_squeezed)

matE = torch.randint(1,5,(2,3,1))
matE_permuted = torch.permute(matE, (2,0,1))
print(matE)
print(matE_permuted)
