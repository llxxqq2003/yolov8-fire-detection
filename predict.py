import os
from ultralytics import YOLO

# 训练参数配置
model_name = "yolov8n.pt"  # 可替换为 yolov8s.pt 等其他模型
data_yaml = "D:/pycharm/yolov8fire/Buildingfire/data.yaml"  # 数据集配置文件路径
epochs = 50
batch_size = 8
imgsz = 640
device = "0"  # 可改为 "0" 使用GPU，或 "cpu" 使用CPU
project_dir = "runs/train"  # 训练结果保存目录

# 1. 训练模型
model = YOLO(model_name)
train_results = model.train(
    data=data_yaml,
    epochs=epochs,
    batch=batch_size,
    imgsz=imgsz,
    device=device,
    project=project_dir,
    name="exp"  # 训练结果子目录名称
)

# 2. 训练完成后自动预测测试集
print("\n开始预测测试集...")
test_images_dir = "D:/pycharm/yolov8fire/Buildingfire/test/images"  # 测试集图像目录
predict_results = model.predict(
    source=test_images_dir,
    conf=0.5,  # 置信度阈值
    save=True,  # 自动保存预测结果到 runs/predict
    save_txt=True,  # 保存预测标签（txt格式）
    save_conf=True,  # 在图像上显示置信度
    project="runs",  # 预测结果保存到 runs 目录下
    name="predict"  # 预测结果子目录名称
)

print(f"\n预测完成！结果保存路径：{os.path.join('runs', 'predict')}")

# 3. 训练完成后进行验证并获取指标
val_results = model.val(
    data=data_yaml,  # 使用验证集配置
    save_json=True,  # 保存验证结果为json，方便后续查看详细信息
    project=project_dir,
    name="exp"
)

# 从验证结果中提取相关指标
P = val_results.results_dict.get('metrics/precision(B)', 0)
R = val_results.results_dict.get('metrics/recall(B)', 0)
mAP50 = val_results.results_dict.get('metrics/mAP50(B)', 0)
mAP50_95 = val_results.results_dict.get('metrics/mAP50-95(B)', 0)
Params = model.model.info.get('params', 0) / 1e6  # 转换为MB
GFLOPS = model.model.info.get('flops', 0) / 1e9  # 转换为G
Speed = val_results.speed['inference']  # 推理速度（FPS）

# 打印指标
print(f"模型: YOLOv8n")
print(f"P: {P}")
print(f"R: {R}")
print(f"mAP50(%): {mAP50}")
print(f"mAP50-95(%): {mAP50_95}")
print(f"Params(MB): {Params}")
print(f"GFLOPS(G): {GFLOPS}")
print(f"Speed(FPS): {Speed}")