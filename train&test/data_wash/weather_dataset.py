# weather_dataset.py

from torch.utils.data import Dataset
from PIL import Image

# —— 2 类标签映射 ——
LABEL2ID = {
    "clear": 0,
    "rain": 1,
}

class WeatherDataset(Dataset):
    """
    将解析后的 samples:
        [(img_path, "clear"), (img_path, "rain"), ...]
    转换为模型训练需要的 (tensor, label_id)
    """

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

        label = LABEL2ID[label_str]
        return img, label
