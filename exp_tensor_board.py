from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")  # 实例化，且将其 add 的文件设置为：创建到 logs 文件夹中

#使用 tensorboard 显示函数图像
# # y = 2x
# for i in range(100):
#     writer.add_scalar("y = 3 * x", 3 * i, i)
# # terminal 命令行打开 tensorboard：tensorboard --logdir=logs --port=XXXX（端口号可自定义）

# 使用 numpy 数组添加图片，并在 tensorboard 中显示
image_path = "dataset/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
# print(type(img_array)) # 返回：<class 'numpy.ndarray'>
# print(img_array.shape) # 返回：(height, width, channel)
writer.add_image("ants_image", img_array, 1, dataformats="HWC") # 建议提前查看所需传入的参数类型

writer.close()
