
"""
YOLOv8 单实例多路推理（微批版）：
- 解决“两路/多路同时运行时重复加载模型”的问题；
- 仅加载一次模型，统一为多路 camera 做推理；
- 从各路 `frames:<camera_id>` 取帧，按轮询+微批方式拼成 batch 预测；
- 将结果按 camera_id 拆回并发布到 `accident:<camera_id>`；

使用：
  from events.detector_accident_multi import run_accident_detector_multi
  await run_accident_detector_multi(bus, camera_ids=["cam-1","cam-2"], batch_size=4, poll_ms=20)

注意：
  - 为了让聚合器能凑“连续3帧”，本推理器对每路使用 FIFO 消费（不使用 latest）。
  - 批组装策略：按相机轮询，每路最多取 1 帧入批，避免某一路独占；
  - poll_ms 控制最长等待时间；batch_size 达到或 poll 超时即触发一轮推理。
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from events.bus import AsyncBus, Frame, Detection, topic_for

# -------------------------------
# 固定配置（与单路版保持一致）
# -------------------------------
_MODEL_PATH: str = r"E:\PythonProject\DjangoTrafficAI\events\pts\best.pt"
_IMG_SIZE: int = 960
_YOLO_CONF: float = 0.05
_YOLO_IOU: float = 0.50
_DECISION_THRESH: float = 0.65
_DEVICE: int | str = 0      # GPU:0；若需 CPU 请改为 'cpu'
_FP16: bool = True          # 若 GPU 支持，可启用半精度
_LOG_BATCH: bool = True     # 打印批级日志


@dataclass(slots=True)
class _Item:
    cam: str
    frame: Frame


class _YOLOEngine:
    """封装 YOLO 单实例加载与推理"""
    def __init__(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("缺少 ultralytics，请先 pip install ultralytics") from e

        print(f"🔹 [multi-det] 正在加载模型权重: {_MODEL_PATH}")
        self.model = YOLO(_MODEL_PATH)
        if hasattr(self.model, "overrides") and isinstance(self.model.overrides, dict):
            self.model.overrides["conf"] = _YOLO_CONF
            self.model.overrides["iou"] = _YOLO_IOU
            self.model.overrides["device"] = _DEVICE
            # 注意：半精度在部分设备上由内部自动处理
        # 预热（静默）
        try:
            dummy = np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy,
                imgsz=_IMG_SIZE,
                conf=_YOLO_CONF,
                iou=_YOLO_IOU,
                verbose=False,
                device=_DEVICE,
                half=_FP16,
                workers=0,
                stream=False,
            )
        except Exception:
            pass

    def infer_batch(self, images: List[np.ndarray]):
        # Ultralytics 支持 list[np.ndarray]
        return self.model.predict(
            images,
            imgsz=_IMG_SIZE,
            conf=_YOLO_CONF,
            iou=_YOLO_IOU,
            verbose=False,
            device=_DEVICE,
            half=_FP16,
            workers=0,
            stream=False,
        )


async def run_accident_detector_multi(
    bus: AsyncBus,
    *,
    camera_ids: List[str],
    batch_size: int = 4,
    poll_ms: int = 20,
) -> None:
    """多路推理主入口：
    - 为每个相机订阅 `frames:<cam>`（FIFO, maxsize=64），入各自队列；
    - 定时/达批后做一次批推理；
    - 结果按相机发布到 `accident:<cam>`。
    """
    engine = _YOLOEngine()
    loop = asyncio.get_running_loop()

    # 每路一个本地缓冲队列（FIFO）
    bufs: Dict[str, deque[Frame]] = {cam: deque(maxlen=128) for cam in camera_ids}

    async def _collector(cam: str):
        topic_in = topic_for("frames", cam)
        async with bus.subscribe(topic_in, mode="fifo", maxsize=64) as q:
            while True:
                f: Frame = await q.get()
                bufs[cam].append(f)
                # 逐帧日志（可按需关闭）
                # print(f"[in ][{cam}] idx={f.frame_idx} pts={f.pts_in_video:.3f}")

    collectors = [asyncio.create_task(_collector(cam)) for cam in camera_ids]

    # 推理循环
    try:
        while True:
            batch_items: List[_Item] = []
            cams_round = list(camera_ids)

            # 轮询各路，每路取最多 1 帧，直到凑满 batch 或队列都空
            while len(batch_items) < batch_size and cams_round:
                cam = cams_round.pop(0)
                q = bufs[cam]
                if q:
                    frm = q.popleft()
                    batch_items.append(_Item(cam=cam, frame=frm))
                # 把该路放回末尾，形成简单的轮询
                cams_round.append(cam)
                # 若所有队列都空，会在下面 sleep
                if all(len(bufs[c]) == 0 for c in camera_ids):
                    break

            if not batch_items:
                # 无数据：小睡等待或下一轮
                await asyncio.sleep(poll_ms / 1000.0)
                continue

            # 组装 batch
            images = [it.frame.rgb for it in batch_items]
            # 执行推理放在线程池，避免阻塞事件循环
            results = await loop.run_in_executor(None, engine.infer_batch, images)

            if _LOG_BATCH:
                cams = ",".join([it.cam for it in batch_items])
                print(f"[infer] batch={len(batch_items)} cams=[{cams}]")

            # 拆分结果并发布
            for it, res in zip(batch_items, results):
                boxes = getattr(res, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    conf = 0.0
                else:
                    confs = getattr(boxes, "conf", None)
                    conf = float(confs.max().item()) if confs is not None and len(confs) > 0 else 0.0
                happened = conf >= _DECISION_THRESH

                det = Detection(
                    type="accident",
                    camera_id=it.cam,
                    ts_unix=it.frame.ts_unix,
                    happened=happened,
                    confidence=conf,
                    frame_idx=it.frame.frame_idx,
                    pts_in_video=it.frame.pts_in_video,
                )
                await bus.publish(topic_for("accident", it.cam), det)

                # 逐帧日志（可按需打开）
                print(
                    f"[out ][{it.cam}] idx={det.frame_idx:05d} pts={det.pts_in_video:7.3f} "
                    f"conf={det.confidence:5.3f} happened={det.happened}"
                )

    finally:
        for t in collectors:
            t.cancel()
            try:
                await t
            except Exception:
                pass
