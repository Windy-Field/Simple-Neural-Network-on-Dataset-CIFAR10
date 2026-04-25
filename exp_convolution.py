import torch
from torch._inductor import kernel

"""
    卷积的作用是提取特征
"""

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 0, 1],
                       [0, 1, 0],
                       [1, 0, 1]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output_1 = torch.conv2d(input, kernel, stride=1)
output_2 = torch.conv2d(input, kernel, stride=2)
output_3 = torch.conv2d(input, kernel, stride=1, padding=1) # 注意：padding = 1 表示将 input 矩阵外加一圈 0，不是外加一圈 1（默认 padding = 0）
print(output_1)
print(output_2)
print(output_3)
