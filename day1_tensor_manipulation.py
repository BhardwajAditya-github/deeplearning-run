import torch

mat1 = torch.randint(0,6,(3,3),dtype=torch.int64)
print(mat1)
scalar_add = 2+mat1
scalar_subtract = 2-mat1
scalar_divide = 2/mat1
scalar_multiply = 2*mat1
# print(f"Addition - {scalar_add}") 
# print(f"Subtraction - {scalar_subtract}")
# print(f"Division - {scalar_divide}")
# print(f"Multiplication - {scalar_multiply}")


matA = torch.randint(1,4,(2,3), dtype=torch.int64)
matB = torch.randint(1,4,(3,2), dtype=torch.int64)
print(matA)
print(matB)
matrix_multiplication = torch.matmul(matA,matB)
print(matrix_multiplication)