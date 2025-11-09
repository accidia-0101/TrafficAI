
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Final
import numpy as np
import asyncio
from contextlib import asynccontextmanager

# ----------- 主题分区工具（建议：按 camera_id 分区，减少消费端过滤负担） -----------
def topic_for(base: str, camera_id: Optional[str] = None) -> str:
    """构造分区主题名，如 detections:cam-1 / frames_raw:cam-2；无 id 则返回 base"""
    return f"{base}:{camera_id}" if camera_id else base

# ----------- 数据结构（slots 节省内存 & 提升属性访问性能） -----------
@dataclass(slots=True)
class Frame:
    camera_id: str
    ts_unix: float
    rgb: np.ndarray               # HxWx3, uint8, RGB
    frame_idx: int = 0            # 帧号（从 0 递增）
    pts_in_video: float = 0.0     # 相对视频起点秒

@dataclass(slots=True)
class Detection:
    type: str                     # accident/weather/...
    camera_id: str
    ts_unix: float
    happened: bool
    confidence: float
    frame_idx: int = 0            # 对齐帧号（可选但推荐）
    pts_in_video: float = 0.0     # 对齐媒体时间轴（秒）

# ----------- 订阅端适配器（非阻塞投递） -----------
class _Subscriber:
    __slots__ = ("queue", "mode")
    def __init__(self, mode: str, maxsize: int):
        self.mode = mode  # "fifo" or "latest"
        # latest 模式固定容量 1（只保留最新）
        cap = 1 if mode == "latest" else max(1, int(maxsize))
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=cap)

    # 改为同步方法：不 await，不阻塞发布者
    def deliver(self, item: Any) -> None:
        if self.mode == "latest":
            # 始终保存“最新”，丢弃旧元素，不阻塞
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull:
                # 极端竞态下再次满：清空后再放（仍然非阻塞）
                try:
                    while True:
                        self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self.queue.put_nowait(item)
                except asyncio.QueueFull:
                    # 理论上不应发生；保守忽略
                    pass
        else:
            # FIFO：尽量不丢，但**确保发布端不被阻塞**
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull:
                # 丢最旧，腾位再放；这会让慢订阅者“不影响”其他链路
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self.queue.put_nowait(item)
                except asyncio.QueueFull:
                    # 仍然满（极端竞态）：放弃本条，保护全链路
                    pass

# ----------- 异步总线 -----------
class AsyncBus:
    __slots__ = ("_topics", "_lock")
    def __init__(self):
        self._topics: Dict[str, List[_Subscriber]] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def subscribe(self, topic: str, *, mode: str = "fifo", maxsize: int = 64):
        """
        订阅主题：
          - mode="fifo"   ：按序处理（分析/检测链路）
          - mode="latest" ：只关心最新（预览/诊断）
        注意：请优先使用分区主题，如 topic_for("detections", camera_id)
        """
        sub = _Subscriber(mode=mode, maxsize=maxsize)
        async with self._lock:
            self._topics.setdefault(topic, []).append(sub)
        try:
            yield sub.queue
        finally:
            async with self._lock:
                lst = self._topics.get(topic, [])
                if sub in lst:
                    lst.remove(sub)
                if not lst:
                    # 清理空列表，防内存增长
                    self._topics.pop(topic, None)
