import torch
from torchvision.models import vgg16

"""
    保存模型参数的一些方法，结合 exp_model_load 文件加载模型参数
    查看保存的文件的字节数的方法：终端输入：dir /a:-d
    对于自定义的网络，建议还是直接 import 文件
"""

vgg16_false = vgg16(pretrained=False)

# 保存方式 1 —— 保存模型结构
torch.save(vgg16_false, "./vgg16_false_1.pth")

# 保存方式 2【推荐】—— 保存模型参数
torch.save(vgg16_false.state_dict(), "./vgg16_false_2.pth")
