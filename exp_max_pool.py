import torch
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

"""
    池化的作用是降低特征的数据量
"""

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
input = torch.reshape(input, (-1, 1, 5, 5))  # 相当于生成了一张虚拟图片。MaxPool2d 需要输入 4D 张量，第一个维度为批次大小，第二个维度为通道数，第三、四个维度分别为高度、宽度


class MyModel(Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.max_pool2d = MaxPool2d(kernel_size=2, stride=2,
                                    ceil_mode=True)  # 默认情况下，stride=kernel_size，这与卷积核不同；ceil_mode=True 表示不足部分照常处理

    def forward(self, x):
        x = self.max_pool2d(x)
        return x


model = MyModel()
output = model(input)

# writer = SummaryWriter("./logs_max_pool")
# # 注意：这里不能 reshape，因为元素个数不够
# input = input.repeat(1, 3, 1, 1) # 四个参数分别为：批次重复次数, 通道重复次数, 高度重复次数, 宽度重复次数
# output = output.repeat(1, 3, 1, 1)
# writer.add_images("input", input)
# writer.add_images("output", output)
# writer.close()
