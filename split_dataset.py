import os
import shutil
import random
import json
import cv2
from PIL import Image
import numpy as np

# 原始图像和标注文件所在目录
original_images_dir = 'D:/pycharm/yolov8fire/Buildingfire/images'
original_json_dir = 'D:/pycharm/yolov8fire/Buildingfire'  # 假设JSON文件在单独的json目录中

# 目标数据集目录
train_images_dir = 'D:/pycharm/yolov8fire/Buildingfire/train/images'
train_labels_dir = 'D:/pycharm/yolov8fire/Buildingfire/train/labels'
val_images_dir = 'D:/pycharm/yolov8fire/Buildingfire/val/images'
val_labels_dir = 'D:/pycharm/yolov8fire/Buildingfire/val/labels'
test_images_dir = 'D:/pycharm/yolov8fire/Buildingfire/test/images'
test_labels_dir = 'D:/pycharm/yolov8fire/Buildingfire/test/labels'

# 创建目录结构
for dir_path in [
    train_images_dir, train_labels_dir,
    val_images_dir, val_labels_dir,
    test_images_dir, test_labels_dir
]:
    os.makedirs(dir_path, exist_ok=True)

# 获取所有图像文件
image_files = [f for f in os.listdir(original_images_dir)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

# 数据集划分比例
train_ratio = 0.7
val_ratio = 0.15

# 计算各部分数量
train_count = int(len(image_files) * train_ratio)
val_count = int(len(image_files) * val_ratio)

# 类别映射 (根据您的JSON文件结构可能需要调整)
class_mapping = {"fire": 0}  # 将"fire"类别映射为0


def convert_json_to_yolo(json_path, image_path, output_path):
    """将JSON标注转换为YOLO格式"""
    try:
        # 获取图像尺寸
        img = Image.open(image_path)
        width, height = img.size

        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 解析JSON中的标注信息 (根据您的JSON结构可能需要调整)
        lines = []
        # 假设JSON格式包含一个"shapes"列表，每个shape包含"label"和"points"
        if 'shapes' in data:
            for shape in data['shapes']:
                label = shape['label']
                if label not in class_mapping:
                    continue  # 忽略未知类别

                class_id = class_mapping[label]

                # 提取边界框坐标 (假设为矩形框)
                points = shape['points']
                if len(points) < 2:
                    continue  # 无效框

                # 计算边界框的左上角和右下角坐标
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # 计算YOLO格式的中心点和宽高 (归一化)
                x_center = (x_min + x_max) / (2 * width)
                y_center = (y_min + y_max) / (2 * height)
                box_width = (x_max - x_min) / width
                box_height = (y_max - y_min) / height

                # 添加到输出行
                lines.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

        # 写入YOLO格式的标注文件
        if lines:
            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))
        else:
            print(f"警告: {json_path} 不包含有效标注")
            return False

        return True
    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {str(e)}")
        return False


# 遍历图像文件并划分数据集
for i, img_file in enumerate(image_files):
    img_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(original_images_dir, img_file)
    json_file = f"{img_name}.json"
    json_path = os.path.join(original_json_dir, json_file)

    # 确定目标目录
    if i < train_count:
        target_images_dir = train_images_dir
        target_labels_dir = train_labels_dir
    elif i < train_count + val_count:
        target_images_dir = val_images_dir
        target_labels_dir = val_labels_dir
    else:
        target_images_dir = test_images_dir
        target_labels_dir = test_labels_dir

    # 复制图像文件
    shutil.copy(img_path, os.path.join(target_images_dir, img_file))

    # 转换并保存标注文件
    if os.path.exists(json_path):
        txt_file = f"{img_name}.txt"
        txt_path = os.path.join(target_labels_dir, txt_file)
        convert_json_to_yolo(json_path, img_path, txt_path)
    else:
        print(f"警告: 找不到对应的JSON文件 {json_path}")

print("数据集划分和格式转换完成!")