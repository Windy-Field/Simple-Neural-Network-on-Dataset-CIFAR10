from sympy.physics.quantum import transforms
from torch.utils.data import Dataset
from PIL import Image
import os


# 想要继承 Dataset，必须重写 __getitem__() 和 __len__() 函数
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        # 根目录
        self.root_dir = root_dir
        # 标签文件夹路径
        self.label_dir = label_dir
        # 路径拼接函数
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 获得路径内所有图片的名称数组，如：['0013035.jpg', 'Ant_1.jpg']
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        # Image.open(img_item_path).show()
        img = Image.open(img_item_path)
        label = self.label_dir.split("_")[0]
        return img, label

    def __len__(self):
        return len(self.img_path)


def create_label_txt_by_image_path(root_dir, target_image_dir, out_dir):
    """
    根据图片文件夹路径生成标签文件
    Args:
        root_dir: 根目录（如：dataset/train）
        target_image_dir: 图片文件夹目录（如：ants_image）
        out_dir: 输出标签目录（如：mixed_label）
    """
    count = 0
    # 确保输出目录存在
    output_path = os.path.join(root_dir, out_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 对应每一张图片，生成其标签，以 .txt 文件生成到 ants_label 文件夹中
    image_path = os.path.join(root_dir, target_image_dir)
    label = target_image_dir.split("_")[0]  # target_image_dir.split("_")：将 "ants_image" 切成 [ants, image]

    # 根据扩展名过滤，获取所有图片数组，如：['0013035.jpg', 'Ant_1.jpg']
    image_files = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # 为每张图片生成标签文件
    for file_name in image_files:
        # 获取文件名（不含扩展名）
        file_name_splitext = os.path.splitext(file_name)[0]
        # 创建标签文件路径
        label_file_path = os.path.join(output_path, f"{file_name_splitext}.txt")

        # 写入标签内容
        with open(label_file_path, "w", encoding="utf-8") as f: # with 语句可以不用写 f.close()
            f.write(label)

        count += 1
    print(f"共生成 {count} 个标签文件")


def create_label_txt_by_data_set(root_dir, data_set, out_dir):
    """
    根据数据集对象生成标签文件
    支持 MyData 对象和 ConcatDataset 对象
    Args:
        root_dir: 根目录（如：dataset/train）
        data_set: 数据集对象（支持 MyData 或 ConcatDataset）
        out_dir: 输出标签目录（如：mixed_label）
    """
    count = 0
    # 确保输出目录存在
    output_path = os.path.join(root_dir, out_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for index in range(len(data_set)):
        # 获取（图片与）标签
        image, label = data_set[index]

        # 处理不同类型的 dataset
        # (bool) hasattr(data_set, 'img_path')：检查 data_set 是否包含 img_path 属性
        if hasattr(data_set, 'img_path'):
            # MyData 对象：直接获取文件名
            file_name = data_set.img_path[index]
        else:
            # 其它对象
            # 使用索引作为文件名
            file_name = f"image_{index:04d}"  # 序号补 0 补至四位

        # 获取文件名（不含扩展名）
        file_name_splitext = os.path.splitext(file_name)[0]  # 提取文件名，去除扩展名
        # 创建标签文件路径
        label_file_path = os.path.join(output_path, f"{file_name_splitext}.txt")

        # 写入标签内容
        with open(label_file_path, "w", encoding="utf-8") as f:
            f.write(label)

        count += 1

    print(f"共生成 {count} 个标签文件")


if __name__ == "__main__":
    # 样例
    root_dir = "dataset/train"
    ants_label_dir = "ants_image"
    ants_data_set = MyData(root_dir, ants_label_dir)
    ant_sample_image, ant_sample_label = ants_data_set[0]
    print(len(ants_data_set))

    bees_label_dir = "bees_image"
    bees_data_set = MyData(root_dir, bees_label_dir)
    bee_sample_image, bee_sample_label = bees_data_set[0]
    # bee_sample_image.show()
    print(len(bees_data_set))

    # 重写了 len() 函数后可以执行数据集相加操作，0~123 是蚂蚁，124~244 是蜜蜂
    train_data_set = ants_data_set + bees_data_set  # 注意：这是 ConcatDataset 对象，不含有 root_dir、label_dir、img_path 等属性
    print(len(train_data_set))

    # create_label_txt_b_image_path(root_dir, ants_label_dir, "mixed_label")
    # create_label_txt_by_image_path(root_dir, bees_label_dir, "mixed_label")
    # create_label_txt_by_data_set(root_dir, train_data_set, "mixed_label") # 建议不要使用，因为会破坏原先图片名称
    create_label_txt_by_data_set(root_dir, ants_data_set, "mixed_label")
    create_label_txt_by_data_set(root_dir, bees_data_set, "mixed_label")