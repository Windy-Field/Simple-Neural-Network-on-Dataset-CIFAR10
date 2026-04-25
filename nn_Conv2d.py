import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=True, download=True,
                                       transform=transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=64)


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义卷积层
        # out_channels 一般是 in_channels 的倍数，相当于从 6 个角度看图像
        self.conv2d = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0)  # 对于彩色图像，输入通道数为 3

    # 调用方法：实例化后，model(imgs) 即可
    # 前向传播逻辑
    def forward(self, x):
        x = self.conv2d(x)
        return x


model = MyModel()
writer = SummaryWriter("./logs_conv2d")
step = 0
for data in data_loader:
    imgs, labels = data
    output = model(imgs)
    step += 1
    writer.add_images("imgs", imgs, step)
    output = output[:, :3, :, :]  # 取前三个通道
    writer.add_images("imgs_conv2d", output, step)  # 只接受 3 通道图像
writer.close()