# detector_accident.py
"""
YOLOv8 单类事故检测（参数锁定版）

- 输入：订阅 'frames'（等时采样后的帧流）
- 输出：发布 'detections'（附带 frame_idx / pts_in_video）
- 所有模型与阈值参数均已锁定，不允许外部更改
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np

from events.bus import Frame, Detection, AsyncBus

# -------------------------------
# 固定配置区（不允许更改）
# -------------------------------
_MODEL_PATH = r"E:\PythonProject\DjangoTrafficAI\events\pts\best.pt"
_IMG_SIZE = 960
_YOLO_CONF = 0.05
_YOLO_IOU = 0.50
_DECISION_THRESH = 0.65
_DEVICE = 0  # GPU:0；如需CPU自行修改此文件，而不是外部参数
# -------------------------------


class AccidentDetector:
    """YOLOv8 单类事故检测引擎（参数锁定）"""

    def __init__(self):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("缺少 ultralytics，请先 pip install ultralytics") from e

        print(f"🔹 正在加载模型权重: {_MODEL_PATH}")
        self._yolo = YOLO(_MODEL_PATH)
        if hasattr(self._yolo, "overrides"):
            self._yolo.overrides["conf"] = _YOLO_CONF
            self._yolo.overrides["iou"] = _YOLO_IOU
            self._yolo.overrides["device"] = _DEVICE

        # GPU 预热
        try:
            dummy = np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
            _ = self._yolo.predict(
                dummy,
                imgsz=_IMG_SIZE,
                conf=_YOLO_CONF,
                iou=_YOLO_IOU,
                verbose=False,
                device=_DEVICE,
            )
        except Exception:
            pass

    # ---------------------------
    def infer_frame_conf(self, rgb: np.ndarray) -> float:
        """单帧推理 → 帧级置信度"""
        res = self._yolo.predict(
            rgb,
            imgsz=_IMG_SIZE,
            conf=_YOLO_CONF,
            iou=_YOLO_IOU,
            verbose=False,
            device=_DEVICE,
        )[0]

        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return 0.0
        confs = getattr(boxes, "conf", None)
        if confs is None or len(confs) == 0:
            return 0.0
        return float(confs.max().item())


# =========================================================
# 运行函数（外部唯一入口，不允许自定义参数）
# =========================================================
async def run_accident_detector(bus: AsyncBus, *, camera_id: Optional[str] = None):
    print(f"[{camera_id}] detector started, waiting frames")
    """
    内部固定参数版本：
      - 不接受外部阈值、尺寸、设备参数
      - 直接使用本文件预设的模型与阈值
    """
    engine = AccidentDetector()
    loop = asyncio.get_running_loop()

    async with bus.subscribe("frames") as q:
        while True:
            frame: Frame = await q.get()
            print(f"[{camera_id}] received frame {frame.frame_idx}")
            if camera_id and frame.camera_id != camera_id:
                continue

            # 在线程池中执行推理（防止阻塞事件循环）
            frame_conf = await loop.run_in_executor(None, engine.infer_frame_conf, frame.rgb)
            happened = frame_conf >= _DECISION_THRESH

            det = Detection(
                type="accident",
                camera_id=frame.camera_id,
                ts_unix=frame.ts_unix,
                happened=happened,
                confidence=frame_conf,
                frame_idx=getattr(frame, "frame_idx", 0),
                pts_in_video=getattr(frame, "pts_in_video", 0.0),
            )
            print(f"[{camera_id}] conf={frame_conf:.3f}, happened={happened}")
            await bus.publish("detections", det)
            await asyncio.sleep(0)
