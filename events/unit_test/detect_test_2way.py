# -*- coding: utf-8 -*-
"""
通用多路测试脚本：根据 (camera_id → video_path) 映射自动启动

管线：
  frames_raw:<cam> → frames:<cam> → accident:<cam> → accidents.open/close:<cam>

特性：
- 支持任意数量的相机/视频（列表或映射）
- 仅加载一次 YOLO（使用 detector_accident_multi 微批版）
- 逐相机聚合，打印开案/结案事件
- Ctrl+C 优雅退出并 flush，避免丢失结案

用法示例：
  直接在脚本里配置 CAM_TO_VIDEO，或通过命令行：
    python multi_video_detect_test.py \
      --cam cam-1 E:\\Videos\\a.mp4 \
      --cam cam-2 E:\\Videos\\b.mp4 \
      --fps 15 --batch 4 --poll 20

参数：
  --cam <camera_id> <video_path>   可多次传入，构建映射
  --fps <float>                    采样帧率（默认 15.0；60.0 需谨慎）
  --batch <int>                    推理微批大小（默认 4）
  --poll <ms>                      微批最长等待毫秒（默认 20）
  --log-every <n>                  每 N 帧打印一次检测日志（默认 30；设为 1 则逐帧）
"""
from __future__ import annotations

import argparse
import asyncio
import signal
from typing import Dict, List, Tuple

from events.bus import AsyncBus, topic_for
from events.frame_discrete import run_frame_source_raw, run_sampler_equal_time
from events.Accident_detect.accident_detector import run_accident_detector_multi
from events.Accident_detect.incident_aggregator import AccidentAggregator

# ============== 默认映射（可在此直接编辑） ==============
CAM_TO_VIDEO_DEFAULT: Dict[str, str] = {
    "cam-1": r"E:\Training\Recording 2025-10-30 172929.mp4",
    "cam-2": r"E:\Training\Recording 2025-11-02 172630.mp4",
}


async def _print_events(bus: AsyncBus, cam: str) -> None:
    """订阅聚合后的开/结案事件并打印。"""
    topic_open = topic_for("accidents.open", cam)
    topic_close = topic_for("accidents.close", cam)

    async def _listen(topic: str):
        async with bus.subscribe(topic, mode="fifo", maxsize=256) as q:
            while True:
                ev = await q.get()
                print(f"[EVENT][{cam}] {topic.split(':')[0]} → {ev}")

    await asyncio.gather(_listen(topic_open), _listen(topic_close))


async def _launch_one_cam(
    bus: AsyncBus,
    cam: str,
    video_path: str,
    sampler_fps: float,
) -> Tuple[AccidentAggregator, List[asyncio.Task]]:
    tasks: List[asyncio.Task] = []
    # 源 + 采样
    tasks.append(asyncio.create_task(run_frame_source_raw(bus, cam, video_path)))
    tasks.append(asyncio.create_task(run_sampler_equal_time(bus, cam, target_fps=sampler_fps)))
    # 聚合
    aggregator = AccidentAggregator(camera_id=cam, bus=bus)
    tasks.append(asyncio.create_task(aggregator.run()))
    # 事件打印
    tasks.append(asyncio.create_task(_print_events(bus, cam)))
    return aggregator, tasks


async def main_async(
    cam_to_video: Dict[str, str],
    sampler_fps: float,
    batch_size: int,
    poll_ms: int,
    log_every: int,
) -> None:
    if not cam_to_video:
        print("[ERR] 未提供 camera → video 映射；请使用 --cam cam-id path 或在脚本中设置 CAM_TO_VIDEO_DEFAULT。")
        return

    bus = AsyncBus()

    # 启动各路：源/采样/聚合/事件打印
    aggregators: List[AccidentAggregator] = []
    tasks: List[asyncio.Task] = []
    for cam, path in cam_to_video.items():
        agg, tlist = await _launch_one_cam(bus, cam, path, sampler_fps)
        aggregators.append(agg)
        tasks.extend(tlist)

    # 单实例多路推理（只启动一次）
    det_task = asyncio.create_task(
        run_accident_detector_multi(
            bus,
            camera_ids=list(cam_to_video.keys()),
            batch_size=batch_size,
            poll_ms=poll_ms,
        )
    )
    tasks.append(det_task)

    # 优雅退出
    stop_event = asyncio.Event()

    def _on_sig(*_):
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _on_sig)
        except NotImplementedError:
            pass

    await stop_event.wait()

    # 退出前 flush，确保结案不丢
    try:
        await asyncio.gather(*(agg.flush() for agg in aggregators))
    except Exception:
        pass

    # 取消所有任务
    for t in tasks:
        t.cancel()
        try:
            await t
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="通用多路视频测试")
    p.add_argument(
        "--cam",
        dest="cam_pairs",
        nargs=2,
        action="append",
        metavar=("CAMERA_ID", "VIDEO_PATH"),
        help="相机与视频路径（可重复传入多次）",
    )
    p.add_argument("--fps", type=float, default=60.0, help="采样帧率（默认 60.0）")
    p.add_argument("--batch", type=int, default=4, help="推理微批大小（默认 4）")
    p.add_argument("--poll", type=int, default=20, help="微批最长等待毫秒（默认 20ms）")
    p.add_argument("--log-every", type=int, default=30, help="每 N 帧打印一次检测日志（默认 30）")
    return p.parse_args()


def build_mapping(ns: argparse.Namespace) -> Dict[str, str]:
    mapping = dict(CAM_TO_VIDEO_DEFAULT)
    if ns.cam_pairs:
        for cam, path in ns.cam_pairs:
            mapping[cam] = path
    return mapping


def main() -> None:
    ns = parse_args()
    cam_to_video = build_mapping(ns)
    asyncio.run(
        main_async(
            cam_to_video=cam_to_video,
            sampler_fps=ns.fps,
            batch_size=ns.batch,
            poll_ms=ns.poll,
            log_every=ns.log_every,
        )
    )


if __name__ == "__main__":
    main()
