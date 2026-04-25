import torch
from PIL import Image
import torchvision.transforms as transforms
from Train import Model

image_path = "imgs/dog8.png" # 请先在 imgs 目录下放一张狗的图片
img = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
img = transform(img)
print(img.shape)

model = Model()
model.load_state_dict(torch.load("./model_40.pth"))
# print(model)

img = img.unsqueeze(0) # 扩展维度，添加批量大小，等价于 img = img.reshape(1, 3, 32, 32)
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
# [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
print("预测结果是否正确：" + str(output.argmax(1).item() == 5))
