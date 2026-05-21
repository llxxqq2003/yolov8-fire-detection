import json
import os
from glob import glob
import cv2  # 添加OpenCV库

def convert_labelme_to_yolo(labelme_json_dir, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob(os.path.join(labelme_json_dir, "*.json"))

    for json_path in json_files:
        image_file = os.path.basename(json_path).replace('.json', '.jpg')
        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            image_file = image_file.replace('.jpg', '.png')
            image_path = os.path.join(image_dir, image_file)
            if not os.path.exists(image_path):
                continue

        # 使用OpenCV读取图像并获取尺寸
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图像: {image_path}")
                continue
            img_height, img_width = img.shape[:2]
        except Exception as e:
            print(f"处理图像时出错 {image_path}: {e}")
            continue

        # 读取Labelme标注
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                label_file = json.load(f)
        except Exception as e:
            print(f"读取JSON文件时出错 {json_path}: {e}")
            continue

        # 生成YOLOv8标签（txt格式）
        label_txt = []
        for shape in label_file.get('shapes', []):
            if shape['label'] != 'fire':
                continue  # 仅保留"fire"类别
            points = shape['points']
            # 转换多边形标注为边界框（YOLOv8需要xyxy格式）
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            # 归一化坐标（YOLOv8要求0 - 1之间）
            x_center = (x_min + x_max) / (2 * img_width)
            y_center = (y_min + y_max) / (2 * img_height)
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            label_txt.append(f"0 {x_center} {y_center} {width} {height}")  # 类别索引0对应"fire"

        # 保存txt文件
        txt_filename = os.path.basename(image_path).replace(".jpg", ".txt").replace(".png", ".txt")
        with open(os.path.join(output_dir, txt_filename), "w") as f:
            f.write("\n".join(label_txt))

if __name__ == "__main__":
    labelme_json_dir = 'D:/pycharm/yolov8fire/Buildingfire'
    image_dir = 'D:/pycharm/yolov8fire/Buildingfire/images'
    yolo_output_dir = 'D:/pycharm/yolov8fire/Buildingfire/yolo_labels'
    convert_labelme_to_yolo(labelme_json_dir, image_dir, yolo_output_dir)