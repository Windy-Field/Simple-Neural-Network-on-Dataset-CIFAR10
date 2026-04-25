from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img_path = "dataset/train/ants_image/5650366_e22b7e1065.jpg"
img = Image.open(img_path)
writer = SummaryWriter("logs")

# 1.ToTensor() 的使用（将 PIL 图片转换为 tensor 类型）
trans_to_tensor = transforms.ToTensor()
img_to_tensor = trans_to_tensor(img)
print(type(img_to_tensor))

# 2.Normalize() 的使用（归一化到 [-1, 1] 范围，可去除奇异值影响）
# print(img_to_tensor[0][0][0])
trans_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                       std=[0.5, 0.5, 0.5])  # 默认 mean=[0.5, 0.5, 0.5]，std=[0.5, 0.5, 0.5]
img_normalize = trans_normalize(img_to_tensor)  # 必须传入 tensor 类型的图片
# print(img_normalize[0][0][0]) # 探究归一化后的输出变化


# 3.Resize() 的使用 1（调整为固定大小）
trans_resize_1 = transforms.Resize((512, 512))  # 调整为 224 x 224 大小
img_resize_1 = trans_resize_1(img_to_tensor)

# 4.Resize() 的使用 2（调整较小边，等比缩放）
trans_resize_2 = transforms.Resize(512)  # 将较小边调整为 512，保持宽高比（等比缩放）
img_resize_2 = trans_resize_2(img_to_tensor)

# 5.RandomCrop() 的使用（随机裁剪图片）
trans_random_crop = transforms.RandomCrop((224, 224))  # 随机裁剪 224 x 224 大小的图片
img_random_crop = trans_random_crop(img_to_tensor)

# 6.Compose() 的使用【不产生新的变换，只是多个转换工具的顺序组合】
trans_compose_1 = transforms.Compose(
    [trans_to_tensor, trans_normalize, trans_resize_1, trans_random_crop])  # 通过列表传入多个转换工具
img_compose = trans_compose_1(img)
print(type(img_compose))

# 7.多次随机裁剪，搭配 Compose() 使用
trans_compose_2 = transforms.Compose([trans_to_tensor, trans_random_crop])
for i in range(10):
    img_random_crop_10 = trans_compose_2(img)
    writer.add_image("RandomCrop 10 times Image", img_random_crop_10, i)  # 注意：每裁一次，放入 writer 一次

# 8.通过 TensorBoard 可视化图片变化
# 指令：tensorboard --logdir=logs
writer.add_image("Normalize Image", img_to_tensor, 0)
writer.add_image("Normalize Image", img_normalize, 1)

writer.add_image("Resize Image", img_to_tensor, 0)
writer.add_image("Resize Image", img_resize_1, 1)
writer.add_image("Resize Image", img_resize_2, 2)

writer.add_image("RandomCrop Image", img_to_tensor, 0)
writer.add_image("RandomCrop Image", img_random_crop, 1)

writer.add_image("Compose Image", img_to_tensor, 0)
writer.add_image("Compose Image", img_compose, 1)

writer.close()