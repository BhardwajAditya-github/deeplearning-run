import torch

mat1 = torch.randint(1,10,(3,3),dtype=torch.float32)
print(mat1)
print(f"Mean - {torch.mean(mat1)}")
print(f"Max - {torch.max(mat1)}")
print(f"Min - {torch.min(mat1)}")
print(f"Sum - {torch.sum(mat1)}")
print(f"Max Element Position - {torch.argmax(mat1)}")
print(f"Min Element Position - {torch.argmin(mat1)}")


