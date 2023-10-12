import os
import cv2
import torch
from torch.utils.data import Dataset
from lxml import etree
import numpy as np


def image_pre_treat(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=300)
    height, width = edges.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    color_image[:, :, 0] = edges  # 复制灰度通道到三个通道
    color_image[:, :, 1] = edges
    color_image[:, :, 2] = edges
    return color_image


class Tower_Dataset(Dataset):
    def __init__(self, root_dir, type_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations_dir = os.path.join(root_dir, 'Annotations' + '/' + type_dir)
        self.images_dir = os.path.join(root_dir, 'JPEGImages' + '/' + type_dir)
        self.xml_files = [f for f in os.listdir(self.annotations_dir) if f.endswith('.xml')]

    def __len__(self):
        return len(self.xml_files)

    def __getitem__(self, idx):
        xml_file = os.path.join(self.annotations_dir, self.xml_files[idx])
        image_file = os.path.join(self.images_dir, os.path.splitext(self.xml_files[idx])[0] + '.jpg')
        # print(image_file)
        # 读取XML文件并解析标签信息
        tree = etree.parse(xml_file)
        root = tree.getroot()
        # 获取图像
        image = cv2.imread(image_file, 1)
        # image = image_pre_treat(image)
        # 获取标签信息
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            labels.append({
                'label': label,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        sample = {'image': torch.tensor(image), 'labels': labels}
        if self.transform is not None:
            sample['image'] = self.transform(sample['image'])
        return sample

# 使用示例
# if __name__ == "__main__":
#     # 定义数据集根目录
#     root_dir = 'data'
#     type_dir = '/train'
#     # 定义图像变换（可选）
#     # transform = transforms.Compose([
#     #     transforms.ToPILImage,
#     #     transforms.ToTensor(),
#     #     # 添加其他需要的变换
#     # ])
#
#     # 创建数据集
#     dataset = Tower_Dataset(root_dir, type_dir, transform=None)
#
#     # 获取数据集长度
#     print("Dataset Length:", len(dataset))
#
#     # 获取单个样本
#     sample = dataset[0]
#     image = sample['image']
#     labels = sample['labels']

# 在这里，你可以使用image和labels进行后续的处理
