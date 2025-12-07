"""
YOLOv8 single-instance multi-stream inference (micro-batch version):
- Solves the issue of “model being loaded repeatedly for two or more concurrent streams”;
- Loads the model only once and performs unified inference for multiple cameras;
- Pulls frames from each `frames:<camera_id>` topic and assembles them into batches via round-robin + micro-batching;
- Splits predictions by camera_id and publishes them to `accident:<camera_id>`.

Usage:
    from events.detector_accident_multi import run_accident_detector_multi
    await run_accident_detector_multi(bus, camera_ids=["cam-1","cam-2"], batch_size=4, poll_ms=20)

Notes:
  - To allow the aggregator to accumulate “3 consecutive frames”, this inference module uses FIFO consumption per camera (not `latest`).
  - Batch assembly strategy: round-robin polling, taking at most 1 frame per camera to avoid any single camera dominating;
  - poll_ms controls the maximum wait time; inference is triggered when batch_size is reached or poll timeout occurs.
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
# -----------------------------------------------------------------------------
# YOLOv8 single-instance multi-stream inference (micro-batch version, with vts)
#
# - 只加载一次 YOLO 模型
# - 从 frames:<camera_id>（采样后统一 fps 的流）拉帧
# - Round-robin 组 batch，做一次前向
# - 结果按 camera_id 拆分，发布到 accident:<camera_id>
# -----------------------------------------------------------------------------
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from events.bus import AsyncBus, Frame, Detection, topic_for

#this has come to an elbow,no need to find better one

_MODEL_PATH: str = r"E:\PythonProject\DjangoTrafficAI\events\pts\best-4.pt"
_IMG_SIZE: int = 640

# YOLO 保持低阈值用于捕捉弱事故
_YOLO_CONF: float = 0.03          # 从 0.05 降低 → 捕获更多弱框
_YOLO_IOU: float = 0.50

# 决策阈值适当提高（避免 TN 掉光）
_DECISION_THRESH: float = 0.18    # 比 0.10 更稳，不会太敏感

_DEVICE: int | str = 0
_FP16: bool = True
_LOG_BATCH: bool = True
_LOG_PER_FRAME: bool = False





@dataclass(slots=True)
class _Item:
    cam: str
    frame: Frame


class _YOLOEngine:
    """Wrapper for single-instance YOLO loading and inference."""
    def __init__(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("no ultralytics，please first pip install ultralytics") from e

        print(f" [multi-det] loading weight: {_MODEL_PATH}")
        self.model = YOLO(_MODEL_PATH)
        if hasattr(self.model, "overrides") and isinstance(self.model.overrides, dict):
            self.model.overrides["conf"] = _YOLO_CONF
            self.model.overrides["iou"] = _YOLO_IOU
            self.model.overrides["device"] = _DEVICE

        # Warm-up (silent)
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
    batch_size: int = 8,
    poll_ms: int = 20,
) -> None:
    """
    Multi-stream accident detector (YOLOv8):
    - 订阅 frames:<cam>（已经统一 fps 的流）
    - 以 round-robin 方式从各个摄像头缓冲区取帧，组装 batch
    - 批量推理，结果按 camera_id 拆分并发布到 accident:<cam>
    """

    engine = _YOLOEngine()
    loop = asyncio.get_running_loop()

    # 每个摄像头一个缓冲区
    bufs: Dict[str, deque[Frame]] = {cam: deque(maxlen=128) for cam in camera_ids}

    async def _collector(cam: str):
        topic_in = topic_for("frames", cam)
        async with bus.subscribe(topic_in, mode="fifo", maxsize=64) as q:
            while True:
                f: Frame = await q.get()
                bufs[cam].append(f)

    collectors = [asyncio.create_task(_collector(cam)) for cam in camera_ids]

    try:
        while True:
            batch_items: List[_Item] = []
            cams_round = list(camera_ids)

            # round-robin 取帧组 batch
            while len(batch_items) < batch_size and cams_round:
                cam = cams_round.pop(0)
                q = bufs[cam]
                if q:
                    frm = q.popleft()
                    batch_items.append(_Item(cam=cam, frame=frm))
                # 放回队尾，继续轮询
                cams_round.append(cam)

                # 所有缓冲都空了，就别死循环
                if all(len(bufs[c]) == 0 for c in camera_ids):
                    break

            if not batch_items:
                await asyncio.sleep(poll_ms / 1000.0)
                continue

            images = [it.frame.rgb for it in batch_items]

            # 子线程里跑 YOLO，避免卡 event loop
            results = await loop.run_in_executor(None, engine.infer_batch, images)

            if _LOG_BATCH:
                cams = ",".join([it.cam for it in batch_items])
                # print(f"[DET-BATCH] cams={cams}, size={len(batch_items)}")

            # 按帧拆结果
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
                    pts_in_video=it.frame.pts_in_video,  # 给前端伪同步
                    vts=it.frame.vts,                     # 给聚合器做时间线
                )

                if _LOG_PER_FRAME:
                    print(
                        f"[DET-DEBUG {it.cam}] "
                        f"frame={it.frame.frame_idx:04d}, "
                        f"pts={it.frame.pts_in_video:.2f}, "
                        f"vts={it.frame.vts:.2f}, "
                        f"conf={conf:.3f}, happened={happened}"
                    )

                await bus.publish(topic_for("accident", it.cam), det)

    finally:
        for t in collectors:
            t.cancel()
            try:
                await t
            except Exception:
                pass
