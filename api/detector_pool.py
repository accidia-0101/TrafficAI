# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from typing import Set, Optional, List

from events.Accident_detect.accident_detector import run_accident_detector_multi  # 你给的新文件
from events.bus import AsyncBus


class MultiDetectorPool:
    def __init__(self, *, bus: AsyncBus, batch_size: int = 4, poll_ms: int = 20):
        self.bus = bus
        self.batch_size = batch_size
        self.poll_ms = poll_ms
        self._active: Set[str] = set()
        self._task: Optional[asyncio.Task] = None

    def cameras(self) -> List[str]:
        return sorted(self._active)

    async def _restart(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try: await self._task
            except Exception: pass
        cams = self.cameras()
        if cams:
            self._task = asyncio.create_task(
                run_accident_detector_multi(
                    self.bus, camera_ids=cams,
                    batch_size=self.batch_size, poll_ms=self.poll_ms
                )
            )
        else:
            self._task = None

    async def ensure_join(self, camera_id: str):
        if camera_id in self._active:
            return
        self._active.add(camera_id)
        await self._restart()

    async def ensure_leave(self, camera_id: str):
        if camera_id not in self._active:
            return
        self._active.remove(camera_id)
        await self._restart()
