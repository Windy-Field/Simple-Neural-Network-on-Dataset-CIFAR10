from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
"""
    Q: 为何需要 tensor（张量）数据类型？
    A: 1. 因为 torch 等框架需要 tensor 数据类型，而 PIL 图片是 numpy 数组，不能直接用于 torch 操作。
       2. 神经网络专用的数据类型，包含了许多神经网络需要的参数，如梯度、优化器等。
"""
# 实例化转换工具
tensor_trans_tool = transforms.ToTensor()

# 第一种图片形式：PIL 图片
from PIL import Image
img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path) # 此时 img 是 PIL 图片对象，打开方式为 open()
tensor_img1 = tensor_trans_tool(img) # ToTensor 支持常见的 jpg、png 等（PIL）图片格式

# 另一种图片形式：numpy 数组
import cv2
img_path = "dataset/train/ants_image/0013035.jpg"
cv_img = cv2.imread(img_path) # 此时 cv_img 是 numpy 数组，打开方式为 imread()
tensor_img2 = tensor_trans_tool(cv_img) # ToTensor 也支持 numpy 数组

# 使用 TensorBoard 可视化 tensor 图片
writer = SummaryWriter("logs")
writer.add_image("PIL Image", tensor_img1)
writer.add_image("numpy Image", tensor_img2)
writer.close()