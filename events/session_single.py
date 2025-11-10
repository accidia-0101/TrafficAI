# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio
from typing import List

from events.bus import AsyncBus
from events.frame_discrete import run_frame_source_raw, run_sampler_equal_time
from events.Accident_detect.incident_aggregator import AccidentAggregator

class SingleFileSession:
    """单路视频检测会话（仅帧源+采样+聚合，共用全局 BUS）"""

    def __init__(self, *, camera_id: str, file_path: str,
                 bus: AsyncBus, sampler_fps: float = 15.0) -> None:
        self.camera_id = camera_id
        self.file_path = file_path
        self.bus = bus
        self.sampler_fps = sampler_fps
        self._tasks: List[asyncio.Task] = []
        self._running = False

    # def start(self) -> None:
    #     """启动该相机帧源、采样、聚合"""
    #     if self._running:
    #         return
    #     loop = asyncio.get_event_loop()
    #     self._running = True
    #     self._tasks = [
    #         loop.create_task(run_frame_source_raw(self.bus, self.camera_id, self.file_path)),
    #         loop.create_task(run_sampler_equal_time(self.bus, self.camera_id)),
    #         loop.create_task(AccidentAggregator(camera_id=self.camera_id, bus=self.bus).run()),
    #     ]
    #     print(f"[session] started {self.camera_id}")
    def start(self, *, loop) -> None:
        if self._running: return

        async def _start():
            self._running = True
            t1 = asyncio.create_task(run_frame_source_raw(self.bus, self.camera_id, self.file_path))
            t2 = asyncio.create_task(run_sampler_equal_time(self.bus, self.camera_id))
            t3 = asyncio.create_task(AccidentAggregator(camera_id=self.camera_id, bus=self.bus).run())
            self._tasks = [t1, t2, t3]
            print(f"[session] started {self.camera_id}")

        asyncio.run_coroutine_threadsafe(_start(), loop)

    def stop(self, *, loop) -> None:
        if not self._running: return

        async def _stop():
            for t in self._tasks: t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
            self._running = False
            print(f"[session] stopped {self.camera_id}")

        asyncio.run_coroutine_threadsafe(_stop(), loop)

