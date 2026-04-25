import torch
from torch import tensor, reshape
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = tensor([1, 2, 3], dtype=torch.float32)
targets = tensor([1, 2, 5], dtype=torch.float32)

loss_l1 = L1Loss(reduction="sum") # 绝对值损失函数
result = loss_l1(inputs, targets)
print(result)

loss_mse = MSELoss(reduction="mean") # 均方差损失函数（差的平方求和后取平均）
result = loss_mse(inputs, targets)
print(result)

loss_cross_entropy = CrossEntropyLoss(reduction="mean") # 交叉熵损失函数
x = tensor([0.1,0.2,0.3]) # 样本有 3 个类别，对于每个类别，有一个概率值
y = tensor([2]) # 样本的真实类别标签为 2
x = reshape(x, (1, 3)) # 参数：（批量大小，类数）
result = loss_cross_entropy(x, y)
print(result)
