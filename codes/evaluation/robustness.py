import os
import numpy as np
from PIL import Image

# 设置原始图像路径
original_image_path = "../results/Original image/Mona Lisa.jpg"

# 设置重构图像所在的目录路径
reconstructed_images_dir = "../results/robustness/yin-yangindel/indel0.10"

# 加载原始图像
original_image = Image.open(original_image_path)

# 将原始图像转换为 numpy 数组
original_image_np = np.array(original_image)

# 遍历重构图像的目录
for filename in os.listdir(reconstructed_images_dir):
    # 获取重构图像的路径
    reconstructed_image_path = os.path.join(reconstructed_images_dir, filename)

    # 加载重构图像
    reconstructed_image = Image.open(reconstructed_image_path)

    # 将重构图像转换为 numpy 数组
    reconstructed_image_np = np.array(reconstructed_image)

    # 计算相似元素的占比
    similarity_ratio = np.sum(original_image_np == reconstructed_image_np) / original_image_np.size

    # 打印相似元素的占比
    print(f"图像 {filename} 的相似元素占比为：{similarity_ratio * 100:.2f}%")