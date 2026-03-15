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
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from events.bus import AsyncBus, Frame, Detection, topic_for

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
_MODEL_PATH: str = r"E:\PythonProject\DjangoTrafficAI\train&test\accident_training\yolov8m_accident_localbox_v1_20260314\weights\best.pt"
_IMG_SIZE: int = 640

# YOLO 内部候选框阈值：尽量低一些，保留弱事故证据
_YOLO_CONF: float = 0.05
_YOLO_IOU: float = 0.45

# detector 输出 happened 的阈值：比 YOLO conf 更高
_DECISION_THRESH: float = 0.22

_DEVICE: int | str = 0
_FP16: bool = True

# subscriber / local buffer
_SUB_QUEUE_SIZE: int = 128
_LOCAL_BUF_SIZE: int = 128

# logging
_LOG_BATCH: bool = False
_LOG_PER_FRAME: bool = False
_LOG_BUFFER: bool = False


@dataclass(slots=True)
class _Item:
    cam: str
    frame: Frame


class _YOLOEngine:
    """Single YOLO instance wrapper."""
    def __init__(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("Ultralytics is not installed. Please run: pip install ultralytics") from e

        print(f"[multi-det] loading weights: {_MODEL_PATH}")
        self.model = YOLO(_MODEL_PATH)

        if hasattr(self.model, "overrides") and isinstance(self.model.overrides, dict):
            self.model.overrides["conf"] = _YOLO_CONF
            self.model.overrides["iou"] = _YOLO_IOU
            self.model.overrides["device"] = _DEVICE

        # warmup
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
    Multi-camera accident detector:
    - subscribe to frames:<cam>
    - use per-camera FIFO buffering
    - build micro-batches in round-robin manner
    - run one shared YOLO instance
    - publish Detection to accident:<cam>
    """
    engine = _YOLOEngine()
    loop = asyncio.get_running_loop()

    # per-camera local buffers
    bufs: Dict[str, deque[Frame]] = {
        cam: deque(maxlen=_LOCAL_BUF_SIZE) for cam in camera_ids
    }

    async def _collector(cam: str):
        topic_in = topic_for("frames", cam)
        async with bus.subscribe(topic_in, mode="fifo", maxsize=_SUB_QUEUE_SIZE) as q:
            while True:
                f: Frame = await q.get()
                bufs[cam].append(f)

    collectors = [asyncio.create_task(_collector(cam)) for cam in camera_ids]

    try:
        while True:
            batch_items: List[_Item] = []

            # 如果全空，先等一小会儿
            if all(len(bufs[c]) == 0 for c in camera_ids):
                await asyncio.sleep(poll_ms / 1000.0)
                continue

            # round-robin: 每轮每个 camera 最多取 1 帧
            while len(batch_items) < batch_size:
                took_any = False

                for cam in camera_ids:
                    if len(batch_items) >= batch_size:
                        break
                    if bufs[cam]:
                        frm = bufs[cam].popleft()
                        batch_items.append(_Item(cam=cam, frame=frm))
                        took_any = True

                # 本轮一个都没取到，说明缓冲空了
                if not took_any:
                    break

            if not batch_items:
                await asyncio.sleep(poll_ms / 1000.0)
                continue

            if _LOG_BUFFER:
                buf_state = ", ".join(f"{cam}:{len(bufs[cam])}" for cam in camera_ids)
                print(f"[DET-BUF] {buf_state}")

            images = [it.frame.rgb for it in batch_items]

            # run YOLO off the event loop
            results = await loop.run_in_executor(None, engine.infer_batch, images)

            if _LOG_BATCH:
                cams = ",".join(it.cam for it in batch_items)
                print(f"[DET-BATCH] cams={cams}, size={len(batch_items)}")

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
                    vts=it.frame.vts,
                )

                if _LOG_PER_FRAME:
                    print(
                        f"[DET {it.cam}] "
                        f"frame={it.frame.frame_idx:04d} "
                        f"pts={it.frame.pts_in_video:.2f} "
                        f"vts={it.frame.vts:.2f} "
                        f"conf={conf:.3f} "
                        f"happened={happened}"
                    )

                await bus.publish(topic_for("accident", it.cam), det)

    finally:
        for t in collectors:
            t.cancel()
        for t in collectors:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                pass