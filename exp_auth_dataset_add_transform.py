import torchvision
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 定义数据集的变换操作
datas_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomCrop(16), # 随机裁剪 16x16 的图像
])

# 设置清华镜像源（解决网络连接问题）
os.environ['TORCHVISION_CIFAR10_URL'] = 'https://mirrors.tuna.tsinghua.edu.cn/pytorch/cifar-10-python.tar.gz'

# 使用官方数据集 CIFAR10，包含了 60000 张彩色训练图像，每个图像的大小为 32x32，共 10 个类别
# 可去官网查询数据集 CIFAR10 的信息：https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html?highlight=cifar10#torchvision.datasets.CIFAR10
# 注意：train_set 和 test_set 是两个不同的数据集，分别用于训练和测试模型
train_set = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=True, transform=datas_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False, transform=datas_transform, download=True)

# 可视化数据集中的图像
writer = SummaryWriter("./logs")
for i in range(100):
    img, label = train_set[i] # 注意：train_set[i] 返回一个元组，包含图像和标签，而非只是图片
    print(test_set.classes[label])
    # img.show()
    writer.add_image("img_transform", img, i)
writer.close()