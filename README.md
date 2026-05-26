<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=180&section=header&text=🔥%20YOLOv8%20火焰检测&fontSize=50&fontColor=ffffff&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <strong>基于 YOLOv8 的火灾/火焰目标检测系统</strong>
</p>

---

## 📖 项目简介

本项目基于 Ultralytics YOLOv8 模型，实现了对火灾和火焰的实时检测。通过自定义数据集训练，能够准确识别图片和视频中的火焰目标，适用于消防安全监控、森林火灾预警等场景。

## ✨ 功能特性

- 🔥 高精度火焰/火灾目标检测
- 📷 支持图片、视频和实时摄像头输入
- 🏷️ 支持 Labelme 标注数据自动转换为 YOLO 格式
- 📊 数据集自动划分工具（训练集/验证集/测试集）
- 🎯 提供预测脚本，开箱即用

## 📁 项目结构

```
yolov8-fire-detection/
├── Buildingfire/          # 火焰数据集
├── labelme_to_yolo.py     # Labelme 标注转 YOLO 格式工具
├── predict.py             # 预测推理脚本
├── split_dataset.py       # 数据集划分工具
└── .gitignore
```

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 1.8
- Ultralytics

### 安装依赖

```bash
pip install ultralytics opencv-python pillow
```

### 数据准备

1. 使用 Labelme 标注火焰图片
2. 运行转换工具：
```bash
python labelme_to_yolo.py
```
3. 划分数据集：
```bash
python split_dataset.py
```

### 模型训练

```bash
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='fire.yaml', epochs=100, imgsz=640)
```

### 模型预测

```bash
python predict.py
```

## 📝 使用说明

| 脚本 | 功能 |
|------|------|
| `labelme_to_yolo.py` | 将 Labelme JSON 标注文件转换为 YOLO TXT 格式 |
| `split_dataset.py` | 按比例自动划分训练集、验证集和测试集 |
| `predict.py` | 加载训练好的模型进行推理预测 |

## 📌 注意事项

- 训练前请确保数据集路径配置正确
- 建议使用 GPU 进行训练以加速收敛
- 可根据实际需求调整模型大小（n/s/m/l/x）

---

<p align="center">
  <i>⭐ 如果这个项目对你有帮助，请给个 Star 支持一下！</i>
</p>
