"""
Asynchronous Event Bus (AsyncBus)
- Partitioned topics: topic_for("frames", "cam-1") -> "frames:cam-1"
- Subscribe:
    async with bus.subscribe(topic, mode="fifo"|"latest", maxsize=64) as q:
        item = await q.get()
- Publish: await bus.publish(topic, item)
- Partitioned publish: await bus.publish_partitioned(base, camera_id, item)
- Publishers never block; slow subscribers drop old items and keep the latest; subscriptions auto-clean on exit
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
from dataclasses import dataclass
from typing import Any, Optional, Dict, List
import asyncio
from contextlib import asynccontextmanager
import numpy as np

# Partitioned Topic Utilities
def topic_for(base: str, camera_id: Optional[str] = None) -> str:
    """Construct a partitioned topic name, e.g., 'frames_raw:cam-1'."""
    return f"{base}:{camera_id}" if camera_id else base


# Data Structures
@dataclass(slots=True)
class Frame:
    camera_id: str
    ts_unix: float
    rgb: np.ndarray
    frame_idx: int = 0
    pts_in_video: float = 0.0
    vts: float = 0.0

@dataclass(slots=True)
class Detection:
    type: str
    camera_id: str
    ts_unix: float
    happened: bool
    confidence: float
    frame_idx: int = 0
    pts_in_video: float = 0.0
    vts: float = 0.0


# Subscriber Side
class _Subscriber:
    __slots__ = ("queue", "mode")

    def __init__(self, mode: str, maxsize: int):
        self.mode = "latest" if mode == "latest" else "fifo"
        cap = 1 if self.mode == "latest" else max(1, int(maxsize))
        self.queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=cap)

    def deliver(self, item: Any) -> None:
        if self.mode == "latest":
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull:
                try:
                    while True:
                        self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self.queue.put_nowait(item)
                except asyncio.QueueFull:
                    pass
        else:
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull:
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self.queue.put_nowait(item)
                except asyncio.QueueFull:
                    pass


# Asynchronous Event Bus
class AsyncBus:
    """
    Pub/Sub:
    - subscribe(topic, mode='fifo'|'latest', maxsize=64)
    - publish(topic, item) is non-blocking
    - publish_partitioned(base, camera_id, item)
    - subscribe_many(topics) to merge multiple topic subscriptions
    """

    __slots__ = ("_topics", "_lock")

    def __init__(self):
        self._topics: Dict[str, List[_Subscriber]] = {}
        self._lock = asyncio.Lock()

    # Single-topic subscription
    @asynccontextmanager
    async def subscribe(self, topic: str, *, mode: str = "fifo", maxsize: int = 64):
        sub = _Subscriber(mode=mode, maxsize=maxsize)
        async with self._lock:
            self._topics.setdefault(topic, []).append(sub)
            print(f"[bus] subscribe -> {topic}")
        try:
            yield sub.queue
        finally:
            async with self._lock:
                lst = self._topics.get(topic)
                if lst is not None:
                    try:
                        lst.remove(sub)
                    except ValueError:
                        pass
                    if not lst:
                        self._topics.pop(topic, None)
            print(f"[bus] unsubscribe -> {topic}")

    # Multi-topic subscription (merged)
    @asynccontextmanager
    async def subscribe_many(self, topics: List[str], *, mode: str = "fifo", maxsize: int = 64):
        """Subscribe to multiple topics at once and merge their output queues."""
        subs: List[_Subscriber] = []
        merged_q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        forward_tasks: List[asyncio.Task] = []

        async def forward(sub: _Subscriber):
            try:
                while True:
                    item = await sub.queue.get()
                    await merged_q.put(item)
            except asyncio.CancelledError:
                # 任务被取消时优雅退出
                return

        # 注册 subscribers
        async with self._lock:
            for t in topics:
                s = _Subscriber(mode=mode, maxsize=maxsize)
                self._topics.setdefault(t, []).append(s)
                subs.append(s)
                print(f"[bus] subscribe_many -> {t}")
                # 创建任务并保存引用
                forward_tasks.append(asyncio.create_task(forward(s)))

        try:
            yield merged_q

        finally:
            # ★ 取消 forward tasks
            for task in forward_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # ★ 注销 subscriber
            async with self._lock:
                for t, s in zip(topics, subs):
                    lst = self._topics.get(t)
                    if lst and s in lst:
                        lst.remove(s)
                    if lst == []:
                        self._topics.pop(t, None)
                    print(f"[bus] unsubscribe_many -> {t}")

    async def publish(self, topic: str, item: Any) -> None:
        async with self._lock:
            subs = list(self._topics.get(topic, []))
        if not subs:
            return
        for sub in subs:
            try:
                sub.deliver(item)
            except Exception:
                continue


    async def publish_partitioned(self, base: str, camera_id: str, item: Any) -> None:
        await self.publish(topic_for(base, camera_id), item)

    async def close_topic(self, topic: str) -> None:
        async with self._lock:
            self._topics.pop(topic, None)
