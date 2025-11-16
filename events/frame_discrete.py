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

from events.bus import Frame, AsyncBus, topic_for
import cv2, time, asyncio, os

# async def run_frame_source_raw(bus: AsyncBus, camera_id: str, url_or_path: str):
#     """
#     Non-downsampled video source: decode frames sequentially and publish to frames_raw:<camera_id>
#     - Each frame includes frame_idx and pts_in_video (video timestamp in seconds)
#     """
#
#     cap = cv2.VideoCapture(url_or_path, cv2.CAP_FFMPEG)
#     try:
#         is_file = os.path.exists(url_or_path)
#         src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
#         src_fps = src_fps if src_fps and src_fps < 1000 else 0.0
#         start_mono = time.monotonic()
#         frame_idx = 0
#
#         while True:
#             ok, bgr = cap.read()
#             if not ok:
#                 # Exit when the file ends; for live sources, wait briefly
#                 if is_file:
#                     break
#                 await asyncio.sleep(0.01)
#                 continue
#
#             rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#             pts = frame_idx / src_fps if src_fps > 0 else (time.monotonic() - start_mono)
#
#             f = Frame(
#                 camera_id=camera_id,
#                 ts_unix=time.time(),
#                 rgb=rgb,
#                 frame_idx=frame_idx,
#                 pts_in_video=pts,
#             )
#
#             # Partitioned publish: frames_raw:<camera_id>
#             await bus.publish(topic_for("frames_raw", camera_id), f)
#             frame_idx += 1
#
#             # Prevent blocking the event loop
#             await asyncio.sleep(0)
#
#     finally:
#         print(f"[frame_source] {camera_id} finished, releasing video")
#         cap.release()
#         try:
#             cv2.destroyAllWindows()
#         except Exception:
#             pass
#
#
# async def run_sampler_equal_time(bus: AsyncBus, camera_id: str, target_fps: float = 60.0, jitter_epsilon: float = 1e-4):
#     """
#     Equal-interval sampling: pull frames from frames_raw:<camera_id>, sample them uniformly
#     according to the target FPS, and publish to frames:<camera_id>.
#     """
#
#     step = 1.0 / max(1e-3, target_fps)
#     next_t = None
#
#     topic_in = topic_for("frames_raw", camera_id)
#     topic_out = topic_for("frames", camera_id)
#
#     async with bus.subscribe(topic_in, mode="fifo", maxsize=64) as sub:
#         while True:
#             f: Frame = await sub.get()
#
#             # Initialize sampling clock
#             if next_t is None:
#                 next_t = f.pts_in_video
#
#             emitted = False
#             while f.pts_in_video + jitter_epsilon >= next_t:
#                 newf = Frame(
#                     camera_id=f.camera_id,
#                     ts_unix=f.ts_unix,
#                     rgb=f.rgb,
#                     frame_idx=f.frame_idx,
#                     pts_in_video=f.pts_in_video,
#                 )
#                 await bus.publish(topic_for("frames", camera_id), newf)
#                 next_t += step
#
#             if not emitted:
#                 await asyncio.sleep(0)
# -----------------------------------------------------------------------------
# Copyright (c) 2025
# Author: Liruo Wang (University of Ottawa)
# -----------------------------------------------------------------------------

import cv2
import time
import os
import asyncio

from events.bus import Frame, AsyncBus, topic_for


async def run_frame_source_raw(bus: AsyncBus, camera_id: str, url_or_path: str):
    """
    Non-downsampled video source: decode frames sequentially and publish to frames_raw:<camera_id>
    - Each frame includes frame_idx and pts_in_video (video timestamp in seconds)
    """

    cap = cv2.VideoCapture(url_or_path, cv2.CAP_FFMPEG)
    try:
        is_file = os.path.exists(url_or_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        src_fps = src_fps if src_fps and src_fps < 1000 else 0.0
        start_mono = time.monotonic()
        frame_idx = 0

        while True:
            ok, bgr = cap.read()
            if not ok:
                # Exit when the file ends; for live sources, wait briefly
                if is_file:
                    break
                await asyncio.sleep(0.01)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # 原始 pts：保留给前端伪同步和 debug
            pts = frame_idx / src_fps if src_fps > 0 else (time.monotonic() - start_mono)

            f = Frame(
                camera_id=camera_id,
                ts_unix=time.time(),
                rgb=rgb,
                frame_idx=frame_idx,
                pts_in_video=pts,
                vts=pts,   # 这里先给一个初值，真正的 vts 由 sampler 重写
            )

            await bus.publish(topic_for("frames_raw", camera_id), f)
            frame_idx += 1

            # 防止阻塞事件循环
            await asyncio.sleep(0)

    finally:
        print(f"[frame_source] {camera_id} finished, releasing video")
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


async def run_sampler_equal_time_vts(
    bus: AsyncBus,
    camera_id: str,
    target_fps: float = 15.0,
    jitter_epsilon: float = 1e-4,
):
    """
    Equal-interval sampling (with virtual time vts):
    - 从 frames_raw:<camera_id> 读取原始帧（可能帧率不稳）
    - 按 target_fps 在“虚拟时间轴” vts 上均匀采样
    - 输出到 frames:<camera_id> 的帧：
        * pts_in_video = 来自原视频，用于前端进度条
        * vts         = 均匀时间轴，用于 detector / aggregator 多路对齐
    """

    step = 1.0 / max(1e-3, target_fps)
    next_vts = None      # 下一次应该采样的虚拟时间点
    sample_idx = 0

    topic_in = topic_for("frames_raw", camera_id)
    topic_out = topic_for("frames", camera_id)

    async with bus.subscribe(topic_in, mode="fifo", maxsize=64) as sub:
        while True:
            f: Frame = await sub.get()

            # 初始化 vts 时钟：从 0 开始更稳定一点
            if next_vts is None:
                next_vts = 0.0

            # 允许根据真实 pts 做一点 guard：
            # 如果视频时间已经走到 t_real，就允许我们把若干 vts 落在当前帧上
            t_real = f.pts_in_video

            emitted = False
            # 当真实时间 + jitter 已经追上 next_vts，就在这个位置“捡一帧”
            while t_real + jitter_epsilon >= next_vts:
                newf = Frame(
                    camera_id=f.camera_id,
                    ts_unix=f.ts_unix,
                    rgb=f.rgb,
                    frame_idx=sample_idx,     # 采样后的索引
                    pts_in_video=f.pts_in_video,  # 保留原视频时间
                    vts=next_vts,             # 统一虚拟时间
                )
                await bus.publish(topic_out, newf)
                emitted = True

                sample_idx += 1
                next_vts += step

            if not emitted:
                await asyncio.sleep(0)
