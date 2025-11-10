# -*- coding: utf-8 -*-
"""
本地视频一键检测脚本（多路并发友好，按相机分区主题）
- 源：frames_raw:<camera_id>
- 采样：frames:<camera_id>
- 检测：accident:<camera_id>
- 聚合：accidents.open:<camera_id> / accidents.close:<camera_id>

使用方法：
1) 修改 VIDEO_PATH 指向本地视频文件。
2) python -m events.unit_test.local_video_detect  （或直接 python local_video_detect.py）
3) 观察终端中开案/结案日志。
"""
from __future__ import annotations

import asyncio
import signal

from events.Accident_detect.accident_detector import run_accident_detector_multi  # 需确保该模块发布到 "accident:<camera_id>"
from events.Accident_detect.incident_aggregator import AccidentAggregator  # 若路径不同，请按实际项目修改导入
from events.bus import AsyncBus, topic_for
from events.frame_discrete import run_frame_source_raw, run_sampler_equal_time

# ====== 配置 ======
CAMERA_ID = "cam-1"
VIDEO_PATH = r"E:\Training\Recording 2025-10-30 172929.mp4"   # ← 改成你的本地视频
SAMPLER_FPS = 30.0                      # 等时采样帧率（检测用）


async def _print_open_close(bus: AsyncBus, camera_id: str) -> None:
    """订阅聚合后的开/结案事件并打印。"""
    topic_open = topic_for("accidents.open", camera_id)
    topic_close = topic_for("accidents.close", camera_id)

    async def _listen(topic: str):
        async with bus.subscribe(topic, mode="fifo", maxsize=128) as q:
            while True:
                ev = await q.get()
                print(f"[EVENT] {topic} → {ev}")

    await asyncio.gather(_listen(topic_open), _listen(topic_close))


async def main() -> None:
    bus = AsyncBus()

    # 启动帧源（不降帧）与等时采样（检测帧流）
    t_src = asyncio.create_task(run_frame_source_raw(bus, CAMERA_ID, VIDEO_PATH))
    t_smp = asyncio.create_task(run_sampler_equal_time(bus, CAMERA_ID, target_fps=SAMPLER_FPS))

    # 启动检测（订 frames:<cam>，发 accident:<cam>）
    t_det = asyncio.create_task(run_accident_detector_multi(bus,camera_ids=[CAMERA_ID]))

    # 启动聚合（订 accident:<cam>，发 open/close）
    aggregator = AccidentAggregator(camera_id=CAMERA_ID, bus=bus)
    t_agg = asyncio.create_task(aggregator.run())

    # 监听聚合后的开/结案事件
    t_evt = asyncio.create_task(_print_open_close(bus, CAMERA_ID))

    # 优雅退出（Ctrl+C）
    stop_event = asyncio.Event()

    def _on_sig(*_):
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _on_sig)
        except NotImplementedError:
            # Windows 下 add_signal_handler 对 SIGTERM 不可用，忽略
            pass

    await stop_event.wait()

    # Flush 聚合并取消任务
    try:
        await aggregator.flush()
    except Exception:
        pass

    for t in (t_evt, t_agg, t_det, t_smp, t_src):
        t.cancel()
        try:
            await t
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
