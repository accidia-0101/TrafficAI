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

import cv2
import time
import os
import asyncio

from events.bus import Frame, AsyncBus, topic_for



async def run_frame_source_raw(
    bus: AsyncBus,
    camera_id: str,
    url_or_path: str,
    *,
    simulate_realtime: bool = False,     # 是否模拟监控真实推帧
):
    """
    Non-downsampled video source:
    - decode frames sequentially
    - publish to frames_raw:<camera_id>
    - If simulate_realtime=True, frames are pushed according to original FPS timing.
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
                # 文件结束 → 真退出
                if is_file:
                    break

                # 直播流 → 等待下一帧
                await asyncio.sleep(0.01)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # 真实视频 pts
            if src_fps > 0:
                pts = frame_idx / src_fps
            else:
                pts = time.monotonic() - start_mono

            #  模拟真实摄像头按 FPS 推帧
            if simulate_realtime and src_fps > 0:
                now = time.monotonic()
                expected = start_mono + pts
                delay = expected - now
                if delay > 0:
                    await asyncio.sleep(delay)

            f = Frame(
                camera_id=camera_id,
                ts_unix=time.time(),   # 摄像头看到的真实世界时间戳
                rgb=rgb,
                frame_idx=frame_idx,
                pts_in_video=pts,
                vts=pts,               # sampler 会重写
            )

            await bus.publish(topic_for("frames_raw", camera_id), f)
            frame_idx += 1

            await asyncio.sleep(0)  # 不阻塞事件循环

    finally:
        await bus.publish(topic_for("frames_raw", camera_id), None)
        print(f"[frame_source] {camera_id} finished, releasing video")
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


async def run_sampler_equal_time_vts(
    bus: AsyncBus,
    camera_id: str,
    target_fps: float = 15,
    jitter_epsilon: float = 1e-4,
):
    """
    Equal-interval sampling (virtual time vts):
    - 从 frames_raw:<camera_id> 读取原始帧（可能帧率不稳）
    - 按 target_fps 在统一虚拟时间轴 vts 上均匀采样
    - 输出到 frames:<camera_id>

    修复点（新增）：
    1. 使用 asyncio.wait_for 解决 raw 帧延迟导致 sampler 停止的问题。
    2. 支持 frame_source 发送 None，作为“真正结束标志”。
    3. 在 simulate_realtime 下不会提前退出。
    """

    step = 1.0 / max(1e-3, target_fps)
    next_vts = None
    sample_idx = 0

    topic_in = topic_for("frames_raw", camera_id)
    topic_out = topic_for("frames", camera_id)

    async with bus.subscribe(topic_in, mode="fifo", maxsize=64) as sub:
        while True:

            # 关键修复：等待 raw 帧直到出现或超时
            try:
                f = await asyncio.wait_for(sub.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # raw 帧暂未到达（simulate_realtime 情况下正常）
                continue

            #  真正的结束信号（来自 frame_source_raw）
            if f is None:
                break

            # ---------- 原采样逻辑（不动） ----------
            if next_vts is None:
                next_vts = 0.0

            t_real = f.pts_in_video
            emitted = False

            # 均匀采样
            while t_real + jitter_epsilon >= next_vts:
                newf = Frame(
                    camera_id=f.camera_id,
                    ts_unix=f.ts_unix,
                    rgb=f.rgb,
                    frame_idx=sample_idx,
                    pts_in_video=f.pts_in_video,
                    vts=next_vts,
                )

                await bus.publish(topic_out, newf)
                emitted = True

                sample_idx += 1
                next_vts += step

            if not emitted:
                await asyncio.sleep(0)

