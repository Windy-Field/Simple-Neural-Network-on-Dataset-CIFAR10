import torch
from torch.nn import MaxPool2d
from torch.nn import Module, Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision


dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=True, download=True,
                                       transform=transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=64)

class MyModel(Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv2d = Conv2d(3, 6, kernel_size=3, stride=1, padding=0)  # 对于彩色图像，输入通道数为 3
        self.max_pool2d = MaxPool2d(kernel_size=2, stride=2,
                                    ceil_mode=True)  # 默认情况下，stride=kernel_size，这与卷积核不同；ceil_mode=True 表示不足部分照常处理

    def forward(self, x):
        # x = self.conv2d(x)
        x = self.max_pool2d(x)
        return x

writer = SummaryWriter("./logs_max_pool")
model = MyModel()
step = 0

for data in data_loader:
    imgs, labels = data
    output = model(imgs)
    writer.add_images("imgs", imgs, step)
    output = output[:, :3, :, :]  # 取前三个通道
    writer.add_images("imgs_max_pool", output, step)  # 只接受 3 通道图像
    step += 1

writer.close()
