# test_weather_single.py
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
import numpy as np

# ======================
# 配置（请修改路径）
# ======================
WEIGHTS_PATH = r"E:\PythonProject\DjangoTrafficAI\events\pts\weather_cls_2class.pth"
IMAGE_PATH = r"E:\Training\crash.png"
DEVICE = "cuda"  # 若无GPU -> 改为 "cpu"


# ======================
# Weather Detector（单图测试版）
# ======================
class WeatherDetectorSingle:
    def __init__(self, model_path):
        print(f"🔹 加载 MobileNetV3 state_dict: {model_path}")

        # 创建与你训练时一致的模型结构（很关键）
        self.model = mobilenet_v3_small(weights=None, num_classes=2)

        # 加载 state_dict
        state = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(state)

        self.model.to(DEVICE)
        self.model.eval()

        # 与训练保持一致
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

        # 类别映射
        self.id2label = {
            0: "clear",
            1: "rain",
        }

    @torch.no_grad()
    def predict_image(self, img_rgb: np.ndarray):
        """单张 RGB numpy 图像的推理"""
        pil = Image.fromarray(img_rgb)
        x = self.transform(pil).unsqueeze(0).to(DEVICE)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        cls = int(probs.argmax())
        label = self.id2label[cls]
        conf = float(probs[cls])

        return label, conf


# ======================
# 主测试逻辑
# ======================
if __name__ == "__main__":
    print("🚀 WeatherDetector 单图推理测试开始")

    # 1. 加载图片
    pil = Image.open(IMAGE_PATH).convert("RGB")
    img_rgb = np.array(pil)

    # 2. 加载模型
    detector = WeatherDetectorSingle(WEIGHTS_PATH)

    # 3. 推理
    label, conf = detector.predict_image(img_rgb)

    # 4. 打印结果
    print("================================")
    print(f"🌤️ 天气预测: {label}")
    print(f"📊 置信度: {conf:.4f}")
    print("================================")
