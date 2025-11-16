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

# ========= 需要你根据实际情况修改的两个路径 =========
# 数据集根目录（包含 train/valid/test/data.yaml）
ROOT = r"E:\Training\CCD-DATA"
DATA_YAML = os.path.join(ROOT, "data.yaml")

# 训练输出目录前缀
PROJECT_NAME = "accident_training"
RUN_NAME = "yolov8m_accident_fullbbox_2"

# ======================================================
FULL_BOX_LINE = "0 0.5 0.5 1.0 1.0\n"
SUBSETS = ["train", "valid", "test"]



def convert_label_file(path: str) -> None:
    """将有内容的标签文件替换为全图 bbox，空文件保持为空。"""
    try:
        size = os.path.getsize(path)
    except OSError:
        return

    if size == 0:
        # 无事故：空文件不变
        return

    # 有事故：写成全图 bbox
    with open(path, "w", encoding="utf-8") as f:
        f.write(FULL_BOX_LINE)


def convert_all_labels(root: str) -> None:
    """遍历 train/valid/test，把所有有内容的标签改成全图 bbox。"""
    print("=== Step 1: 将所有事故标签转换为全图 bbox ===")
    for subset in SUBSETS:
        label_dir = os.path.join(root, subset, "labels")
        if not os.path.isdir(label_dir):
            print(f"[警告] labels 目录不存在，跳过: {label_dir}")
            continue

        print(f"处理目录: {label_dir}")
        count_accident = 0
        count_nonacc = 0

        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(label_dir, fname)

            size = os.path.getsize(path)
            if size == 0:
                # 无事故
                count_nonacc += 1
            else:
                # 有事故
                convert_label_file(path)
                count_accident += 1

        print(f"  ✓ 完成 {subset}: accident={count_accident}, non-accident={count_nonacc}")
    print("=== 标签转换完成 ===\n")


def sanity_check_dataset(root: str) -> None:
    """简单再统计一次，以确认标签状态。"""
    print("=== Step 2: 数据集 sanity check（确认 accident / non-accident 数量） ===")
    for subset in SUBSETS:
        label_dir = os.path.join(root, subset, "labels")
        if not os.path.isdir(label_dir):
            print(f"[警告] labels 目录不存在，跳过: {label_dir}")
            continue

        count_accident = 0
        count_nonacc = 0
        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(label_dir, fname)
            size = os.path.getsize(path)
            if size == 0:
                count_nonacc += 1
            else:
                count_accident += 1

        print(f"  {subset}: accident={count_accident}, non-accident={count_nonacc}")
    print("=== Sanity check 完成 ===\n")


def train_yolov8m(data_yaml: str) -> str:
    """启动 YOLOv8m 训练，返回 best.pt 路径。"""
    print("=== Step 3: 启动 YOLOv8m 训练 ===")

    # 加载预训练 YOLOv8m
    model = YOLO("yolov8m.pt")

    results = model.train(
        data=data_yaml,
        epochs=120,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=50,

        # ---------- 优化 Recall + 稳定 Precision ----------
        optimizer="AdamW",
        lr0=0.0015,  # ↑ 略提高初始学习率，有助 recall
        lrf=0.01,

        # ---------- 降低增强强度（避免模型学不到事故关键特征） ----------
        hsv_h=0.015,
        hsv_s=0.4,  # ↓ 原 0.7 → 改 0.4（减少颜色漂移）
        hsv_v=0.2,  # ↓ 原 0.4 → 改 0.2（减少亮度漂移）
        fliplr=0.3,  # ↓ 原 0.5
        flipud=0.0,
        degrees=0.0,
        scale=0.10,  # ↓ 原 0.2（避免缩放太大）
        shear=0.0,

        # ---------- 最关键：让模型更稳定地学“全图二分类” ----------
        mosaic=0.1,  # ↓ 原 1.0 → 0.1（强力减少噪声，提升 recall）
        mixup=0.0,
        close_mosaic=20,  # 删除早期 mosaic 用处不大，让模型晚点再复现真实分布

        # ---------- 更适合二分类 ----------
        pretrained=True,
        project=PROJECT_NAME,
        name=RUN_NAME,
    )

    # ultralytics 会在 results 中给出 run 的目录
    save_dir = results.save_dir  # e.g. accident_training/yolov8m_accident_fullbbox
    best_ckpt = os.path.join(save_dir, "weights", "best.pt")
    print(f"=== 训练结束，best 模型路径: {best_ckpt} ===\n")
    return best_ckpt


def main():
    # 路径检查
    if not os.path.isdir(ROOT):
        raise RuntimeError(f"ROOT 目录不存在，请先修改脚本中的 ROOT: {ROOT}")
    if not os.path.isfile(DATA_YAML):
        raise RuntimeError(f"data.yaml 不存在，请检查路径: {DATA_YAML}")

    print("使用数据集根目录:", ROOT)
    print("使用配置文件:", DATA_YAML)
    print("-" * 60)

    # Step 1: 标签转换
    convert_all_labels(ROOT)

    # Step 2: Sanity check
    sanity_check_dataset(ROOT)

    # Step 3: 训练 YOLOv8m
    best_model = train_yolov8m(DATA_YAML)

    print("\n全自动训练流程完成！")
    print(f"➡ 你可以在 detector 里加载这个模型权重: {best_model}")


if __name__ == "__main__":
    main()

