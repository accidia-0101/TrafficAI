"""
简洁稳定的异步事件总线（AsyncBus）
- 分区主题：topic_for("frames", "cam-1") -> "frames:cam-1"
- 订阅：async with bus.subscribe(topic, mode="fifo"|"latest", maxsize=64) as q: item = await q.get()
- 发布：await bus.publish(topic, item)  /  await bus.publish_partitioned(base, camera_id, item)
- 特点：发布端永不阻塞；慢订阅者丢旧保新；订阅退出自动清理
"""

from dataclasses import dataclass
from typing import Any, Optional, Dict, List
import asyncio
from contextlib import asynccontextmanager
import numpy as np  # 仅用于类型注解

# ---------- 分区主题工具 ----------
def topic_for(base: str, camera_id: Optional[str] = None) -> str:
    """构造分区主题名，如 'frames_raw:cam-1'。"""
    return f"{base}:{camera_id}" if camera_id else base

# ---------- 数据结构 ----------
@dataclass(slots=True)
class Frame:
    camera_id: str
    ts_unix: float
    rgb: np.ndarray               # HxWx3, uint8, RGB
    frame_idx: int = 0            # 帧号（从 0 递增）
    pts_in_video: float = 0.0     # 相对视频起点（秒）

@dataclass(slots=True)
class Detection:
    type: str                     # accident/weather/...
    camera_id: str
    ts_unix: float
    happened: bool
    confidence: float
    frame_idx: int = 0
    pts_in_video: float = 0.0

# ---------- 订阅端（非阻塞投递） ----------
class _Subscriber:
    __slots__ = ("queue", "mode")

    def __init__(self, mode: str, maxsize: int):
        self.mode = "latest" if mode == "latest" else "fifo"
        cap = 1 if self.mode == "latest" else max(1, int(maxsize))
        self.queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=cap)

    # 非阻塞投递：发布端永不 await
    def deliver(self, item: Any) -> None:
        if self.mode == "latest":
            # 只保留最新一条
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull:
                # 极端竞态：清空后再放一次
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
            # FIFO：尽量不丢，但保护发布端不阻塞
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull:
                # 丢最旧一条，腾位后重试一次
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self.queue.put_nowait(item)
                except asyncio.QueueFull:
                    # 仍满则放弃本条，保护全链路
                    pass

# ---------- 异步总线 ----------
class AsyncBus:
    """
    简单稳定的 Pub/Sub：
      - subscribe(topic, mode='fifo'|'latest', maxsize=64) -> async context yielding Queue
      - publish(topic, item) 非阻塞，慢订阅者不会拖垮发布端
      - publish_partitioned(base, camera_id, item) 分区发布
      - close_topic(topic) 主动清理
    """
    __slots__ = ("_topics", "_lock")

    def __init__(self):
        self._topics: Dict[str, List[_Subscriber]] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def subscribe(self, topic: str, *, mode: str = "fifo", maxsize: int = 64):
        sub = _Subscriber(mode=mode, maxsize=maxsize)
        async with self._lock:
            self._topics.setdefault(topic, []).append(sub)
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

    async def publish(self, topic: str, item: Any) -> None:
        # 取快照避免持锁期间执行用户代码
        async with self._lock:
            subs = list(self._topics.get(topic, []))
        if not subs:
            return
        for sub in subs:
            try:
                sub.deliver(item)
            except Exception:
                # 保护发布路径：单个订阅者异常不影响其他订阅者
                continue

    async def publish_partitioned(self, base: str, camera_id: str, item: Any) -> None:
        await self.publish(topic_for(base, camera_id), item)

    async def close_topic(self, topic: str) -> None:
        """主动关闭/清理某主题（例如会话结束时）。"""
        async with self._lock:
            self._topics.pop(topic, None)
