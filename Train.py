import torch
import torchvision
from torch import nn
from torch.nn import Module, Conv2d, MaxPool2d, Flatten, Linear
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# 3.准备神经网络
class Model(Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64 * 4 * 4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    # 智能设备检测（自动选择 GPU 或 CPU）
    def get_device():
        if torch.cuda.is_available():
            try:
                # 测试CUDA是否真的可用
                test_tensor = torch.tensor([1.0]).cuda()
                _ = test_tensor + 1
                print("CUDA 设备可用，使用 GPU")
                return torch.device("cuda")
            except RuntimeError as e:
                if "no kernel image is available" in str(e) or "sm_120" in str(e):
                    print("检测到兼容性问题，自动回退到 CPU")
                    return torch.device("cpu")
                else:
                    print(f"CUDA错误: {e}，回退到 CPU")
                    return torch.device("cpu")
        else:
            print("CUDA 不可用，使用 CPU")
            return torch.device("cpu")


    device = get_device()

    # 0.准备时间戳
    start_time = time.time()

    # 1.准备 Dataset
    train_data = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", transform=ToTensor(), train=True, download=True)
    test_data = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", transform=ToTensor(), train=False, download=True)

    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print(f"训练集大小：{train_data_size}；测试集大小：{test_data_size}")

    # 2.准备 DataLoader
    train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # 4.实例化模型、选择损失函数与优化器、设置训练属性、创建 TensorBoard 日志记录器
    model = Model()
    model = model.to(device)  # 将模型移动到设备中
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)  # 将损失函数移动到设备中
    total_train_step = 0
    total_test_step = 0
    total_epoch = 40
    writer = SummaryWriter("./logs_Train")

    for i in range(total_epoch):
        # 5.训练模型
        model.train()  # 切换到训练模式，建议常加
        for data in train_data_loader:
            # 前向传播，计算损失
            images, labels = data
            images = images.to(device)  # 将图片移动到设备中
            labels = labels.to(device)  # 将标签移动到设备中
            outputs = model(images)

            # 反向传播，优化模型
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 200 == 0:
                print(f"第 {i + 1} 轮，第 {total_train_step} 步，损失值为 {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), total_train_step)  # y 轴为训练集损失值，x 轴为训练步数

        # 6.测试模型（每轮训练结束后）
        model.eval()  # 切换到评估模式，建议常加
        total_test_loss = 0.0
        total_accuracy = 0
        with torch.no_grad():  # 测试集上不计算梯度，避免测试时偷偷优化参数
            for data in test_data_loader:
                images, labels = data
                images = images.to(device)  # 将图片移动到设备中
                labels = labels.to(device)  # 将标签移动到设备中
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_test_loss += loss.item()
                accuracy = (torch.argmax(outputs, dim=1) == labels).sum()  # 得到预测结果与真实标签相同的数量
                total_accuracy += accuracy.item()  # 累加预测结果与真实标签相同的数量
        print(f"第 {i + 1} 轮，测试集损失值为 {total_test_loss:.4f}")
        print(f"第 {i + 1} 轮，测试集准确率为 {total_accuracy / test_data_size:.4f}")
        writer.add_scalar("test_loss", total_test_loss, total_test_step)  # y 轴为测试集损失值，x 轴为测试步数
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)  # y 轴为测试集准确率，x 轴为测试步数
        total_test_step += 1

        # 7. 保存最后的模型参数
        if i == total_epoch - 1:
            torch.save(model.state_dict(), f"./model_{i + 1}.pth")

    # 8.结束操作
    writer.close()
    end_time = time.time()
    print(f"程序总耗时：{(end_time - start_time):.4f} 秒")
