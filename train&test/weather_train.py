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
"""
@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
"""
import os
import json
import random
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image

# =============================
# 配置
# =============================
BDD_ROOT = r"E:\Training\weather\bdd100k"
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-4
VAL_RATIO = 0.1
MAX_CLEAR = 12000  # clear 最多保留这么多
BALANCE = True     # 是否对雨类进行过采样
# =============================


# ============================================
# 1) 解析 BDD100K → 只取 clear / rain
# ============================================
WEATHER_MAP = {
    "clear": "clear",
    "overcast": "clear",
    "cloudy": "clear",
    "partly cloudy": "clear",
    "rainy": "rain",
    "rain": "rain",
    "foggy": None,
    "snowy": None,
    "undefined": None,
    None: None,
}

def map_weather(raw):
    if raw is None: return None
    raw = raw.lower().strip()
    return WEATHER_MAP.get(raw, None)

def load_bdd_clear_rain(bdd_root, split="train"):
    img_dir = os.path.join(bdd_root, "bdd100k_images_100k", "100k", split)
    lbl_dir = os.path.join(bdd_root, "bdd100k_labels", "100k", split)

    samples = []

    for fname in os.listdir(img_dir):
        if not fname.endswith(".jpg"):
            continue

        stem = fname[:-4]
        json_path = os.path.join(lbl_dir, stem + ".json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        raw_weather = data.get("attributes", {}).get("weather")
        mapped = map_weather(raw_weather)

        if mapped is None:
            continue

        samples.append((os.path.join(img_dir, fname), mapped))

    print(f"[BDD] {split} loaded: {len(samples)} samples")
    return samples


# ============================================
# 2) Dataset（简单版）
# ============================================
LABEL2ID = {"clear":0, "rain":1}

class WeatherDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, LABEL2ID[label_str]


# ============================================
# 3) 在 train_weather 内部平衡 clear/rain
# ============================================
def build_balanced_dataset():
    samples = load_bdd_clear_rain(BDD_ROOT, split="train")

    clear = [s for s in samples if s[1] == "clear"]
    rain  = [s for s in samples if s[1] == "rain"]

    print(f"[INFO] raw clear={len(clear)}, rain={len(rain)}")

    # 1) clear 下采样（避免 clear 压死 rain）
    if len(clear) > MAX_CLEAR:
        clear = random.sample(clear, MAX_CLEAR)
        print(f"[INFO] downsample clear → {len(clear)}")

    # 2) rain 过采样到 clear 数量（自动平衡）
    if BALANCE:
        if len(rain) < len(clear):
            need = len(clear) - len(rain)
            rain = rain + random.choices(rain, k=need)
        print(f"[INFO] oversample rain → {len(rain)}")

    # 3) 合并
    all_samples = clear + rain
    random.shuffle(all_samples)

    # 4) 切分 train/val
    def split_by_ratio(lst):
        n_val = int(len(lst) * VAL_RATIO)
        return lst[n_val:], lst[:n_val]

    clear_tr, clear_val = split_by_ratio(clear)
    rain_tr,  rain_val  = split_by_ratio(rain)

    train_samples = clear_tr + rain_tr
    val_samples = clear_val + rain_val

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"[INFO] Train size: {len(train_samples)}, Val size: {len(val_samples)}")
    print("[STAT] Train:", Counter(x[1] for x in train_samples))
    print("[STAT] Val:  ", Counter(x[1] for x in val_samples))

    return train_samples, val_samples


# ============================================
# 4) 模型
# ============================================
def get_model():
    model = mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model


# ============================================
# 5) 训练主函数
# ============================================
def train():
    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    train_samples, val_samples = build_balanced_dataset()

    train_ds = WeatherDataset(train_samples, transform_train)
    val_ds   = WeatherDataset(val_samples, transform_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = get_model().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1)==labels).sum().item()

        train_acc = correct / len(train_ds)

        # ---- Val ----
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item()
                val_correct += (out.argmax(1)==labels).sum().item()

        val_acc = val_correct / len(val_ds)

        print(f"[Epoch {epoch}/{EPOCHS}] "
              f"TrainLoss={total_loss:.3f} TrainAcc={train_acc:.4f} | "
              f"ValLoss={val_loss:.3f} ValAcc={val_acc:.4f}")

    torch.save(model.state_dict(), "../events/pts/weather_cls_2class.pth")
    print("[DONE] Saved model → weather_cls_2class.pth")


if __name__ == "__main__":
    train()
