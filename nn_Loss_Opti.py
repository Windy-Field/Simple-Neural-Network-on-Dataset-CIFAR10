import time

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, Module
from torch.utils.data import DataLoader
from torchvision import transforms

start_time = time.time()

dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=True, download=True,
                                       transform=transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=1)


class Model(Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # 参数设置可见官网公式计算得到
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # 可以简写成 MaxPool2d(2)，因为其余都是默认值
            Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )  # 使用 Sequential() 封装模型结构

    def forward(self, x):
        x = self.model1(x)
        return x


model = Model()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)

"""
    优化器根据梯度进行的优化，而梯度是通过损失函数得来的
"""

for epoch in range(10):
    running_loss = 0.0
    for data in data_loader:
        imgs, labels = data
        output = model(imgs)  # 前向传播
        result_loss = loss(output, labels)
        optim.zero_grad()
        result_loss.backward()  # 反向传播并计算梯度，但不更新参数（由优化器完成）
        optim.step()
        running_loss += result_loss.item()
    print(running_loss)

end_time = time.time()
print(f"程序耗时：{end_time - start_time:.4f} 秒")
