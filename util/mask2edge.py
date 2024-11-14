import cv2
import numpy as np
import os

# 输入和输出目录路径
input_dir = '/data/fyao309/MOFGSeg/oucuavseg/train/cologne'
output_dir = '/data/fyao309/MOFGSeg/oucuavseg/train/cologne'

# 检查输出目录是否存在，不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入目录中的所有文件
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):  # 假设掩膜图像是PNG格式
        file_path = os.path.join(input_dir, filename)

        # 读取掩膜图像
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Error: Unable to load mask image {filename}.")
            continue
        
        # 创建一个空白的图像用于存储边缘
        edges = np.zeros_like(mask, dtype=np.uint8)

        # 对每个类别进行边缘检测
        unique_classes = np.unique(mask)
        for class_id in unique_classes:
            class_mask = np.uint8(mask == class_id) * 255  # 创建当前类别的二值图像
            class_edges = cv2.Canny(class_mask, threshold1=100, threshold2=200)
            edges = np.maximum(edges, class_edges)

        # 保存边缘图到输出目录
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, edges)
        print(f"Edge image saved as '{output_path}'")

print("All edge images have been processed and saved.")
