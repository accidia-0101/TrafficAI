# -*- coding: utf-8 -*-
"""
Motion Score Stage
------------------
Subscribe:
    frames:<camera_id>

Publish:
    motion.score:<camera_id>

This stage keeps heavy processing internal:
- grayscale conversion
- repeated-sample check
- dense optical flow
- spatial smoothing
- temporal EMA
- running anomaly score

Published payload is a LIGHTWEIGHT dict only.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Optional, Deque, List, Dict, Any

import cv2
import numpy as np

from events.bus import AsyncBus, Frame, topic_for

_TOPIC_IN_BASE = "frames"
_TOPIC_OUT_BASE = "motion.score"

# ---------------------------------------------------------------------
# Optical flow config
# ---------------------------------------------------------------------
_FLOW_PYR_SCALE = 0.5
_FLOW_LEVELS = 3
_FLOW_WINSIZE = 15
_FLOW_ITERATIONS = 3
_FLOW_POLY_N = 5
_FLOW_POLY_SIGMA = 1.2
_FLOW_FLAGS = 0

# ---------------------------------------------------------------------
# Spatial processing
# ---------------------------------------------------------------------
_GAUSS_KSIZE = 7
_GAUSS_SIGMA = 1.5

_USE_CENTER_ROI = True
_ROI_X1 = 0.15
_ROI_X2 = 0.85
_ROI_Y1 = 0.20
_ROI_Y2 = 0.90

# ---------------------------------------------------------------------
# Pooling / temporal processing
# ---------------------------------------------------------------------
_TOP_PERCENT = 0.08
_EMA_ALPHA = 0.25

# ---------------------------------------------------------------------
# Baseline / anomaly criterion
# ---------------------------------------------------------------------
_BASELINE_WIN = 45
_MIN_BASELINE_COUNT = 8
_EPS = 1e-6
_Z_CLIP_LOW = 0.0
_Z_CLIP_HIGH = 3.0

# ---------------------------------------------------------------------
# Validity / repeated-sample guards
# ---------------------------------------------------------------------
_MIN_DT_VTS = 1e-6
_REPEAT_MSE_THR = 1.0

# ---------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------
_SUB_QUEUE_SIZE = 128
_LOG_PER_FRAME = False


def _to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _center_roi(arr: np.ndarray) -> np.ndarray:
    if not _USE_CENTER_ROI:
        return arr

    h, w = arr.shape[:2]
    x1 = int(w * _ROI_X1)
    x2 = int(w * _ROI_X2)
    y1 = int(h * _ROI_Y1)
    y2 = int(h * _ROI_Y2)

    if x2 <= x1 or y2 <= y1:
        return arr
    return arr[y1:y2, x1:x2]


def _top_percent_mean(arr: np.ndarray, top_percent: float) -> float:
    flat = arr.reshape(-1)
    if flat.size == 0:
        return 0.0
    k = max(1, int(flat.size * top_percent))
    topk = np.partition(flat, -k)[-k:]
    return float(np.mean(topk))


class MotionScoreProducer:
    """
    One producer per camera_id.

    Publishes lightweight dict payload:
    {
        "type": "motion_score",
        "camera_id": ...,
        "ts_unix": ...,
        "frame_idx": ...,
        "pts_in_video": ...,
        "vts": ...,
        "dt_vts": ...,
        "valid": ...,
        "repeated_sample": ...,
        "score_motion": ...,
        "score_mag": ...,
        "score_ori": ...
    }
    """

    def __init__(self, camera_id: str, bus: AsyncBus) -> None:
        self.camera_id = camera_id
        self.bus = bus

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_vts: Optional[float] = None

        self._ema_motion: float = 0.0
        self._baseline: Deque[float] = deque(maxlen=_BASELINE_WIN)

    def _compute_flow(self, prev_gray: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            _FLOW_PYR_SCALE,
            _FLOW_LEVELS,
            _FLOW_WINSIZE,
            _FLOW_ITERATIONS,
            _FLOW_POLY_N,
            _FLOW_POLY_SIGMA,
            _FLOW_FLAGS,
        )
        u = flow[..., 0]
        v = flow[..., 1]
        mag = np.sqrt(u * u + v * v).astype(np.float32)
        ori = np.arctan2(v, u).astype(np.float32)
        return mag, ori

    def _spatial_process(self, mag: np.ndarray, ori: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mag = cv2.GaussianBlur(mag, (_GAUSS_KSIZE, _GAUSS_KSIZE), _GAUSS_SIGMA)

        cos_o = cv2.GaussianBlur(np.cos(ori), (_GAUSS_KSIZE, _GAUSS_KSIZE), _GAUSS_SIGMA)
        sin_o = cv2.GaussianBlur(np.sin(ori), (_GAUSS_KSIZE, _GAUSS_KSIZE), _GAUSS_SIGMA)
        ori = np.arctan2(sin_o, cos_o).astype(np.float32)

        mag = _center_roi(mag)
        ori = _center_roi(ori)
        return mag, ori

    def _score_magnitude(self, mag: np.ndarray) -> float:
        return _top_percent_mean(mag, _TOP_PERCENT)

    def _score_orientation_dispersion(self, ori: np.ndarray, mag: np.ndarray) -> float:
        if ori.size == 0:
            return 0.0

        w = np.maximum(mag, 0.0).reshape(-1)
        if float(np.sum(w)) < _EPS:
            return 0.0

        ang = ori.reshape(-1)
        c = float(np.sum(w * np.cos(ang)) / (np.sum(w) + _EPS))
        s = float(np.sum(w * np.sin(ang)) / (np.sum(w) + _EPS))
        r = np.sqrt(c * c + s * s)  # direction coherence in [0,1]
        dispersion = 1.0 - r
        return float(max(0.0, min(1.0, dispersion)))

    def _ema(self, x: float) -> float:
        self._ema_motion = _EMA_ALPHA * x + (1.0 - _EMA_ALPHA) * self._ema_motion
        return self._ema_motion

    def _running_anomaly(self, x: float) -> float:
        if len(self._baseline) < _MIN_BASELINE_COUNT:
            self._baseline.append(x)
            return 0.0

        mu = float(np.mean(self._baseline))
        sigma = float(np.std(self._baseline))
        z = (x - mu) / (sigma + _EPS)
        z = min(max(z, _Z_CLIP_LOW), _Z_CLIP_HIGH)
        score = z / _Z_CLIP_HIGH

        self._baseline.append(x)
        return float(score)

    def _is_repeated_sample(self, prev_gray: np.ndarray, gray: np.ndarray) -> bool:
        diff = prev_gray.astype(np.float32) - gray.astype(np.float32)
        mse = float(np.mean(diff * diff))
        return mse <= _REPEAT_MSE_THR

    async def process_frame(self, frame: Frame) -> Optional[Dict[str, Any]]:
        gray = _to_gray_u8(frame.rgb)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_vts = float(frame.vts)
            return None

        dt_vts = float(frame.vts) - float(self._prev_vts if self._prev_vts is not None else frame.vts)
        repeated_sample = self._is_repeated_sample(self._prev_gray, gray)
        valid = dt_vts > _MIN_DT_VTS

        score_mag = 0.0
        score_ori = 0.0
        score_motion = 0.0

        if valid and not repeated_sample:
            mag, ori = self._compute_flow(self._prev_gray, gray)
            mag, ori = self._spatial_process(mag, ori)

            score_mag = self._score_magnitude(mag)
            score_ori = self._score_orientation_dispersion(ori, mag)

            raw_motion = 0.7 * score_mag + 0.3 * score_ori
            ema_motion = self._ema(raw_motion)
            score_motion = self._running_anomaly(ema_motion)

        out = {
            "type": "motion_score",
            "camera_id": self.camera_id,
            "ts_unix": float(frame.ts_unix),
            "frame_idx": int(frame.frame_idx),
            "pts_in_video": float(frame.pts_in_video),
            "vts": float(frame.vts),
            "dt_vts": float(max(0.0, dt_vts)),
            "valid": bool(valid),
            "repeated_sample": bool(repeated_sample),
            "score_motion": float(score_motion),
            "score_mag": float(score_mag),
            "score_ori": float(score_ori),
        }

        self._prev_gray = gray
        self._prev_vts = float(frame.vts)

        return out

    async def run(self) -> None:
        topic_in = topic_for(_TOPIC_IN_BASE, self.camera_id)

        async with self.bus.subscribe(topic_in, mode="fifo", maxsize=_SUB_QUEUE_SIZE) as q:
            while True:
                frame: Frame = await q.get()
                out = await self.process_frame(frame)
                if out is None:
                    continue

                if _LOG_PER_FRAME:
                    print(
                        f"[MOTION-SCORE {self.camera_id}] "
                        f"frame={out['frame_idx']:04d} "
                        f"vts={out['vts']:.2f} "
                        f"dt={out['dt_vts']:.3f} "
                        f"valid={out['valid']} "
                        f"repeat={out['repeated_sample']} "
                        f"motion={out['score_motion']:.3f} "
                        f"mag={out['score_mag']:.3f} "
                        f"ori={out['score_ori']:.3f}"
                    )

                await self.bus.publish_partitioned(_TOPIC_OUT_BASE, self.camera_id, out)


async def run_motion_score_stage(
    bus: AsyncBus,
    *,
    camera_id: str,
) -> None:
    producer = MotionScoreProducer(camera_id=camera_id, bus=bus)
    await producer.run()


async def run_motion_score_stage_multi(
    bus: AsyncBus,
    *,
    camera_ids: List[str],
) -> None:
    tasks = [
        asyncio.create_task(run_motion_score_stage(bus, camera_id=cam))
        for cam in camera_ids
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                pass