""""
Accident and Non-accident label Image Dataset > Hai-s Augment attempt
https://universe.roboflow.com/accident-and-nonaccident/accident-and-non-accident-label-image-dataset
Provided by a Roboflow user
License: Public Domain
"""
# -----------------------------------------------------------------------------
# Copyright (c) 2025
#
# Authors:
#   Liruo Wang
#       School of Electrical Engineering and Computer Science,
#       University of Ottawa
#       lwang032@uottawa.ca
#
# All rights reserved.
# -----------------------------------------------------------------------------
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# ========= 路径配置：按你的实际情况修改 =========
# 数据集根目录（包含 train/valid/test/data.yaml）
ROOT = r"E:\Training\Acci_Dataset"
DATA_YAML = os.path.join(ROOT, "data.yaml")

# 训练输出目录
PROJECT_NAME = "accident_training"

# 建议把关键信息写进名字，方便后续实验对比
RUN_NAME = "yolov8m_accident_localbox_v1_20260314"

# 可根据机器情况修改
DEVICE = 0          # GPU 编号；无 GPU 可改成 "cpu"
WORKERS = 4
BATCH_SIZE = 16
IMG_SIZE = 640
EPOCHS = 120
PATIENCE = 40

# 随机种子，方便复现实验
SEED = 42
# ======================================================


def train_yolov8m_local_accident(data_yaml: str) -> str:
    """
    训练单类事故局部检测器（YOLOv8m）。

    任务定义：
    - 1 类：accident
    - 事故图：标注局部事故区域
    - 非事故图：无框（空标签）

    系统目标：
    - YOLO 提供“局部事故视觉证据”
    - 后续 detector/aggregator 基于逐帧 evidence 做事件级判断
    """
    print("=== 启动 YOLOv8m 训练：单类局部事故检测 ===")

    # 加载预训练权重
    model = YOLO("yolov8m.pt")

    results = model.train(
        # ---------- 数据 ----------
        data=data_yaml,

        # ---------- 基础训练配置 ----------
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        patience=PATIENCE,

        # ---------- 复现性 ----------
        seed=SEED,
        deterministic=True,

        # ---------- 优化器 ----------
        # AdamW 可以先作为 baseline 试验；
        # 后续可再对比 auto / SGD
        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,

        # ---------- 颜色增强 ----------
        # 保守一些，避免颜色扰动过强破坏事故局部语义
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.2,

        # ---------- 几何增强 ----------
        # 水平翻转保留一定比例，提升泛化
        # 垂直翻转对交通场景通常不合理，因此关闭
        fliplr=0.3,
        flipud=0.0,

        # 保持几何增强较弱，避免局部事故区域被过度扭曲
        degrees=0.0,
        scale=0.10,
        shear=0.0,
        translate=0.05,

        # ---------- 组合增强 ----------
        # 对事故检测任务，过强 mosaic 常会破坏自然场景结构
        # 这里保留很低比例，仅作为轻量增强
        mosaic=0.1,
        mixup=0.0,
        close_mosaic=20,

        # ---------- 其他 ----------
        pretrained=True,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=False,
        verbose=True,
        plots=True,
        val=True,
        save=True,
        save_period=-1,
    )

    save_dir = str(results.save_dir)
    best_ckpt = os.path.join(save_dir, "weights", "best.pt")

    print(f"=== 训练结束，best 模型路径: {best_ckpt} ===\n")
    return best_ckpt


def main():
    # ---------- 路径检查 ----------
    if not os.path.isdir(ROOT):
        raise RuntimeError(f"ROOT 目录不存在，请先修改脚本中的 ROOT: {ROOT}")
    if not os.path.isfile(DATA_YAML):
        raise RuntimeError(f"data.yaml 不存在，请检查路径: {DATA_YAML}")

    print("使用数据集根目录:", ROOT)
    print("使用配置文件:", DATA_YAML)
    print("训练输出目录:", PROJECT_NAME)
    print("运行名称:", RUN_NAME)
    print("-" * 60)

    # ---------- 启动训练 ----------
    best_model = train_yolov8m_local_accident(DATA_YAML)

    print("\n训练完成！")
    print(f"➡ 你后续可以在 detector 中加载这个模型: {best_model}")
    print("➡ 下一步建议：不要只看训练指标，还要接入你的 pipeline 做事件级回放测试。")


if __name__ == "__main__":
    main()

