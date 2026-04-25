import torch
import torchvision

"""
    加载模型参数的一些方法，结合 exp_model_save 文件保存模型参数
    查看保存的文件的字节数的方法：终端输入：dir /a:-d
    对于自定义的网络，建议还是直接 import 文件
"""

# 加载方式 1（对应保存方式 1）
model = torch.load("./vgg16_false_1.pth", weights_only=False)
print(model) # 打印模型结构

# 加载方式 2（对应保存方式 2）
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("./vgg16_false_2.pth")) # 加载保存的参数到模型中
print(vgg16) # 此时已转变为模型结构，不再是保存的参数了
