from torch.utils.data import DataLoader
import torchvision
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 定义数据集的变换操作
datas_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomCrop(16), # 随机裁剪 16x16 的图像
])

# 设置清华镜像源（解决网络连接问题）
os.environ['TORCHVISION_CIFAR10_URL'] = 'https://mirrors.tuna.tsinghua.edu.cn/pytorch/cifar-10-python.tar.gz'
train_set = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=True, transform=datas_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False, transform=datas_transform, download=True)
data_loader = DataLoader(test_set, batch_size=4, shuffle=True,
                         num_workers=0, drop_last=False)  # 一次从测试集加载 32 张图像（抓 32 张牌），抓完后随机打乱顺序（洗牌），使用 0 个线程并发行加载数据，不丢弃最后一个未满批次

# 单独查看测试集中的图像（返回图像及其标签）
# img, label = test_set[0]
# print(img.shape)
# print(label)

# 批量查看 DataLoader 打包后的图像与标签
# for data in data_loader:
#     imgs, labels = data
#     print(imgs.shape)  # 返回 (4, 3, 32, 32)，表示 4 张图像，每个图像的大小为 32x32，3 个通道（RGB，即彩图）的张量表示
#     print(labels.shape)  # 返回一个包含 4 个标签的张量，每个标签对应一张图片的类别

# 可视化数据集中的图像
writer = SummaryWriter("./logs_dataloader")
# 抓两轮数据
for epoch in range(2):
    step = 0
    for data in data_loader:
        imgs, labels = data
        writer.add_images(f"imgs_dataloader_{epoch}", imgs, step)  # 注意：这里用的是 add_images 方法，而不是 add_image
        step += 1
writer.close()