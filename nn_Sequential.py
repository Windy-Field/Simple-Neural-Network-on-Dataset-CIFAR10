import torch
import torchvision
from torch.nn import Sequential, MaxPool2d
from torch.nn import Module, Conv2d, Linear, Flatten
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=True, download=True,
                                        transform=transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=64)

class Model(Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, stride=1, padding=2), # 参数设置可见官网公式计算得到
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), # 可以简写成 MaxPool2d(2)，因为其余都是默认值
            Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Flatten (),
            Linear(1024, 64),
            Linear(64, 10),
        ) # 使用 Sequential() 封装模型结构
    def forward(self,x):
        x = self.model1(x)
        return x

model = Model()
input = torch.randn(64, 3, 32, 32) # 测试样本（随机数）
output = model(input)
print(output.shape) # [64, 10]

writer = SummaryWriter("./logs_sequential")
writer.add_graph(model, input) # 可视化模型结构
writer.close()
