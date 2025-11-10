# -*- coding: utf-8 -*-
"""
运行时全局状态（逻辑分区 + 单实例多路检测模式）
"""
from events.bus import AsyncBus

BUS = AsyncBus()          # 全局唯一事件总线（逻辑分区）
SESSIONS = {}             # camera_id -> SingleFileSession 实例
INTENDED = {}             # camera_id -> 视频源路径（已登记但未启动）
DETECTOR_TASK = None      # asyncio.Task：全局 run_accident_detector_multi 任务


# 进程级后台事件循环
import asyncio, threading
_BG_LOOP = None
_BG_THREAD = None

def ensure_bg_loop():
    global _BG_LOOP, _BG_THREAD
    if _BG_LOOP is None or _BG_LOOP.is_closed():
        loop = asyncio.new_event_loop()
        def runner():
            asyncio.set_event_loop(loop)
            loop.run_forever()
        t = threading.Thread(target=runner, name="trafficai-bg-loop", daemon=True)
        t.start()
        _BG_LOOP, _BG_THREAD = loop, t
    return _BG_LOOP
