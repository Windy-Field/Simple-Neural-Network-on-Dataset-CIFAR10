import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vgg16

train_data = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=True, download=True,
                                          transform=transforms.ToTensor())  # 该数据集一共有 10 类
vgg16_false = vgg16(pretrained=False)  # vgg16 能实现分类（1000类）
vgg16_true = vgg16(pretrained=True)

"""
    现在我们想用已有的 vgg16 网络来给 CIFAR10 分类
    但是 vgg16 默认会分成 1000 类，实际只需分 10 类
    所以我们需要修改 vgg16 的输出
"""

# 法一：新增 Linear
print(vgg16_true) # 查看模型结构
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10)) # 在 classifier 层添加
print(vgg16_true)

# 法二：原地修改最后一个 Linear
print(vgg16_false)  # 查看模型结构，发现最后一个（第 6 个）线性层的输出维度为 1000，将其修改为 10 即可
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
