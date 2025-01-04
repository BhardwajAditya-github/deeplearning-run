import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

torch.set_printoptions(precision=10)

var = torch.tensor([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]]);
float_var = torch.tensor([[1.232332,2],[3,4]], dtype=torch.float16)
# print(float_var.shape)
zeros_copy = torch.zeros(float_var.shape, dtype=torch.int64);
ones_copy = torch.ones(var.shape, dtype=torch.int64)
# print(var)
# print(var.ndim)
# print(var.shape)

random_tensor = torch.rand(float_var.shape)

# print(float_var[0,0])
# print(zeros_copy)
# print(ones_copy)
# print(random_tensor)

start_time = time.time()
tensor_info = torch.rand(3, 3, dtype=torch.float32)
print(f"Shape: {tensor_info.shape}")
print(f"Dimensions: {tensor_info.ndim}")
print(f"Device: {tensor_info.device}")
print(f"Data Type: {tensor_info.dtype}")
end_time = time.time()

print(f"\nTime taken: {end_time - start_time:.6f} seconds")

ranged_tensor = torch.arange(1,10,1)
print(ranged_tensor)
