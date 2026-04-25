import torch
from torch.nn import MaxPool2d
from torch.nn import Module, Conv2d, Linear
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=True, download=True,
                                       transform=transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=64, drop_last=True)


class MyModel(Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv2d = Conv2d(3, 6, kernel_size=3, stride=1, padding=0)  # 对于彩色图像，输入通道数为 3
        self.max_pool2d = MaxPool2d(kernel_size=2, stride=2,
                                    ceil_mode=True)  # 默认情况下，stride=kernel_size，这与卷积核不同；ceil_mode=True 表示不足部分照常处理
        self.linear = Linear(196608, 10) # 第一个参数是根据后文中的 output.shape 计算得到的，第二个表示输出类别数

    def forward(self, x):
        # x = self.conv2d(x)
        # x = self.max_pool2d(x)
        x = self.linear(x)
        return x

model = MyModel()
step = 0
for data in data_loader:
    imgs, labels = data
    output = torch.flatten(imgs)  # 拉平成行
    # print(output.shape) # [196608]，其中 196608 = 64 * 3 * 32 * 32，这里 drop_last=True 所以最后一批数据会被丢弃，保证不出现未满情况
    output = model(output)
    # print(output.shape)
    step += 1
