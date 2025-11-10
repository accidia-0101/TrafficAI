# """

"""
AccidentAggregator：逐帧检测结果聚合为稳定事故事件（分区主题版・重写）
--------------------------------------------------------------------
订阅：accident:<camera_id>               # 单帧检测结果（Detection）
发布：accidents.open:<camera_id>         # 开案事件（一次）
      accidents.close:<camera_id>        # 结案事件（可被合并窗口延迟）

设计要点：
- 纯 AsyncBus：按相机分区订阅/发布，避免多路干扰。
- 稳定判定：EMA 平滑 + 严格三连帧开案 + 退出阈 + 连续阴性关案。
- 遮挡宽限：短时断帧不立刻关案（_OCCLUSION_GRACE_SEC）。
- 合并窗口：结案后 _MERGE_GAP_SEC 内若再开案，合并为同一事故（不发布两次 open/close）。
- flush()：文件/会话结束时强制结案并清空待合并事件。

注意：
- 仅关注事故流，不依赖天气/HLS/DB。
- Detection 结构来自 events.bus：{type, camera_id, ts_unix, happened, confidence, frame_idx, pts_in_video}
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from events.bus import AsyncBus, Detection, topic_for

# ==================== 固定参数 ====================
_ALPHA = 0.25                   # EMA 平滑系数
_EXIT_THR = 0.40                # 关案 EMA 阈值（EMA 低于该值才允许进入关案通道）
_REQUIRED_HAPPENED_CONSEC = 3   # 严格开案：必须连续 N 帧 happened=True
_MIN_END_NEG_FRAMES = 8         # 关案：至少连续 N 帧“阴性”（基于 EMA 与 happened 共同驱动）
_OCCLUSION_GRACE_SEC = 1.0      # 遮挡宽限：本帧与上一帧的 pts_in_video 间隔若超过该值，才允许计为“负面演化”
_MERGE_GAP_SEC = 5.0            # 合并窗口：距上次关案 ≤ 该秒，再开案则并入前一事故

_TOPIC_IN_BASE   = "accident"          # 单帧事故检测结果流
_TOPIC_OPEN_BASE = "accidents.open"    # 开案输出主题基名
_TOPIC_CLOSE_BASE= "accidents.close"   # 结案输出主题基名

# ==================== 内部结构 ====================
@dataclass
class _Incident:
    id: str
    camera_id: str
    start_ts: float
    end_ts: float
    start_idx: int
    end_idx: int
    peak_conf: float = 0.0
    pos_frames: int = 0


class AccidentAggregator:
    """基于分区主题的事故聚合器：只关心事故检测结果，不依赖检测器实现。"""

    def __init__(self, camera_id: str, bus: AsyncBus, *, session_id: Optional[str] = None) -> None:
        self.camera_id = camera_id
        self.bus = bus
        self.session_id = session_id or str(int(time.time()))
        self._counter = 0

        # 聚合状态
        self.ema: float = 0.0
        self._hap_streak: int = 0         # 连续 happened=True 计数（用于开案）
        self._neg_streak: int = 0         # 连续“阴性”帧计数（用于关案）
        self._open: Optional[_Incident] = None
        self._last_seen_pts: Optional[float] = None

        # 合并窗口：延迟发布的 close 事件
        self._pending_close: Optional[Dict[str, Any]] = None   # 结构与 close 事件一致
        self._pending_close_time: Optional[float] = None       # 上次 close 形成时刻（以 pts_in_video 计）

    # ---------- 工具 ----------
    def _new_id(self) -> str:
        self._counter += 1
        return f"{self.session_id}-{self.camera_id}-{self._counter:06d}"

    # async def _emit_open(self, inc: _Incident) -> None:
    #     ev = {
    #         "type": "accident_open",
    #         "session_id": self.session_id,
    #         "incident_id": inc.id,
    #         "camera_id": self.camera_id,
    #         "start_ts": inc.start_ts,
    #         "start_frame_idx": inc.start_idx,
    #         "peak_confidence": inc.peak_conf,
    #     }
    #     await self.bus.publish(topic_for(_TOPIC_OPEN_BASE, self.camera_id), ev)
    #     print(f"🚨 OPEN {ev}")
    async def _emit_open(self, inc: _Incident, det: Detection | None = None) -> None:
        ev = {
            "type": "accident_open",
            "session_id": self.session_id,
            "incident_id": inc.id,
            "camera_id": self.camera_id,
            "start_ts": inc.start_ts,
            "start_frame_idx": inc.start_idx,
            "peak_confidence": inc.peak_conf,
        }
        # ==== 新增帧信息 ====
        if det is not None:
            ev["frame_idx"] = getattr(det, "frame_idx", None)
            ev["pts_in_video"] = getattr(det, "pts_in_video", None)
            ev["confidence"] = getattr(det, "confidence", None)
        await self.bus.publish(topic_for(_TOPIC_OPEN_BASE, self.camera_id), ev)
        print(f"🚨 OPEN {ev}")
    # async def _schedule_close(self, inc: _Incident) -> None:
    #     """结案并进入合并观察窗口：先缓存在 _pending_close；
    #     若窗口内无再开案，则真正发布 close；若窗口内再开案则合并。
    #     """
    #     close_ev = {
    #         "type": "accident_close",
    #         "session_id": self.session_id,
    #         "incident_id": inc.id,
    #         "camera_id": self.camera_id,
    #         "start_ts": inc.start_ts,
    #         "end_ts": inc.end_ts,
    #         "duration_sec": max(0.0, inc.end_ts - inc.start_ts),
    #         "peak_confidence": inc.peak_conf,
    #         "pos_frames": inc.pos_frames,
    #     }
    #     self._pending_close = close_ev
    #     self._pending_close_time = inc.end_ts
    #     print(f"⏳ CLOSE (pending merge) {close_ev}")

    async def _schedule_close(self, inc: _Incident, det: Detection | None = None) -> None:
        """结案并进入合并观察窗口"""
        close_ev = {
            "type": "accident_close",
            "session_id": self.session_id,
            "incident_id": inc.id,
            "camera_id": self.camera_id,
            "start_ts": inc.start_ts,
            "end_ts": inc.end_ts,
            "duration_sec": max(0.0, inc.end_ts - inc.start_ts),
            "peak_confidence": inc.peak_conf,
            "pos_frames": inc.pos_frames,
        }
        # ==== 新增帧信息 ====
        if det is not None:
            close_ev["frame_idx"] = getattr(det, "frame_idx", None)
            close_ev["pts_in_video"] = getattr(det, "pts_in_video", None)
            close_ev["confidence"] = getattr(det, "confidence", None)

        self._pending_close = close_ev
        self._pending_close_time = inc.end_ts
        print(f"⏳ CLOSE (pending merge) {close_ev}")

    async def _flush_pending_close_if_expired(self, now_pts: float) -> None:
        if self._pending_close is None:
            return
        if self._pending_close_time is None:
            return
        if now_pts - self._pending_close_time > _MERGE_GAP_SEC:
            # 发布并清空
            await self.bus.publish(topic_for(_TOPIC_CLOSE_BASE, self.camera_id), self._pending_close)
            print(f"✅ CLOSE (emit) {self._pending_close}")
            self._pending_close = None
            self._pending_close_time = None

    def _merge_reopen_into_pending(self, new_start_ts: float, new_end_ts: float, new_peak: float, new_pos_frames: int, new_end_idx: int) -> None:
        """在合并窗口内再次开案：把 reopen 并入待发布的 close 事件，扩大时窗与峰值。"""
        assert self._pending_close is not None
        pc = self._pending_close
        pc["end_ts"] = new_end_ts
        pc["duration_sec"] = max(0.0, pc["end_ts"] - pc["start_ts"])
        pc["peak_confidence"] = max(float(pc["peak_confidence"]), float(new_peak))
        pc["pos_frames"] = int(pc.get("pos_frames", 0)) + int(new_pos_frames)
        # end_idx 仅用于日志/调试（保留在内部，不上报）

    # ---------- 主循环 ----------
    async def run(self) -> None:
        topic_in = topic_for(_TOPIC_IN_BASE, self.camera_id)
        async with self.bus.subscribe(topic_in, mode="fifo", maxsize=128) as sub:
            while True:
                det: Detection = await sub.get()
                await self._process(det)

    async def _process(self, det: Detection) -> None:
        ts = float(getattr(det, "pts_in_video", 0.0))
        conf = float(getattr(det, "confidence", 0.0))
        happened = bool(getattr(det, "happened", False))
        fidx = int(getattr(det, "frame_idx", 0))

        # 先处理 pending close 的超时发布
        await self._flush_pending_close_if_expired(ts)

        # 计算遮挡/断帧
        prev_pts = self._last_seen_pts
        self._last_seen_pts = ts
        occlusion_ok = True
        if prev_pts is not None and (ts - prev_pts) > _OCCLUSION_GRACE_SEC:
            # 超出宽限，认为中间存在空档；仅在关案计数上更谨慎
            occlusion_ok = False

        # EMA 平滑
        self.ema = _ALPHA * conf + (1.0 - _ALPHA) * self.ema

        # 连续阳性 streak（用于“严格开案”）
        if happened:
            self._hap_streak += 1
        else:
            self._hap_streak = 0

        # 阴性演化计数（EMA 低于阈值才增长；遮挡异常则不增长）
        if self.ema <= _EXIT_THR and occlusion_ok:
            self._neg_streak += 1
        else:
            self._neg_streak = 0

        # ========== 开案判定 ==========
        if self._open is None and self._hap_streak >= _REQUIRED_HAPPENED_CONSEC:
            # 若存在待发布的 close 且仍在合并窗口内 → 合并 reopen
            if self._pending_close is not None and self._pending_close_time is not None:
                if (ts - self._pending_close_time) <= _MERGE_GAP_SEC:
                    # 将 reopen 并入之前的事故：更新待 close 的 end_ts/peak/pos_frames
                    new_peak = conf
                    new_pos = 1  # 本帧记入正帧
                    self._merge_reopen_into_pending(
                        new_start_ts=ts, new_end_ts=ts, new_peak=new_peak, new_pos_frames=new_pos, new_end_idx=fidx
                    )
                    # 合并后相当于“仍在进行中”：把 open 状态恢复
                    inc = _Incident(
                        id=self._pending_close["incident_id"],
                        camera_id=self.camera_id,
                        start_ts=self._pending_close["start_ts"],
                        end_ts=ts,
                        start_idx=int(getattr(det, "frame_idx", fidx)),
                        end_idx=fidx,
                        peak_conf=float(self._pending_close["peak_confidence"]),
                        pos_frames=int(self._pending_close.get("pos_frames", 0)),
                    )
                    # 清空 pending close，恢复开案状态
                    self._pending_close = None
                    self._pending_close_time = None
                    self._open = inc
                    self._hap_streak = 0
                    return

            # 正常新开案
            inc = _Incident(
                id=self._new_id(),
                camera_id=self.camera_id,
                start_ts=ts,
                end_ts=ts,
                start_idx=fidx,
                end_idx=fidx,
                peak_conf=conf,
                pos_frames=1,
            )
            self._open = inc
            self._hap_streak = 0
            await self._emit_open(inc)
            return

        # ========== 进行时更新 ==========
        if self._open is not None:
            inc = self._open
            inc.end_ts = ts
            inc.end_idx = fidx
            inc.peak_conf = max(inc.peak_conf, conf)
            if happened:
                inc.pos_frames += 1

            # 关案判定：EMA 持续低于阈值 + 连续阴性帧达到下限
            if self.ema <= _EXIT_THR and self._neg_streak >= _MIN_END_NEG_FRAMES:
                # 进入合并窗口：不立即发布，等待 _MERGE_GAP_SEC 以捕捉可能的复燃
                await self._schedule_close(inc)
                self._open = None
                self.ema = 0.0
                self._neg_streak = 0

    # ---------- flush ----------
    async def flush(self) -> None:
        """视频/会话结束时：
        - 若仍开案：直接形成 close 并发布（不再合并）。
        - 若有 pending close：直接发布并清空。
        """
        # 发布 pending close
        if self._pending_close is not None:
            await self.bus.publish(topic_for(_TOPIC_CLOSE_BASE, self.camera_id), self._pending_close)
            print(f"✅ CLOSE (emit pending) {self._pending_close}")
            self._pending_close = None
            self._pending_close_time = None

        # 强制结案
        if self._open is None:
            print("ℹ️ [Aggregator] flush(): 无需结案。")
            return

        inc = self._open
        ev = {
            "type": "accident_close",
            "session_id": self.session_id,
            "incident_id": inc.id,
            "camera_id": self.camera_id,
            "start_ts": inc.start_ts,
            "end_ts": inc.end_ts,
            "duration_sec": max(0.0, inc.end_ts - inc.start_ts),
            "peak_confidence": inc.peak_conf,
            "pos_frames": inc.pos_frames,
            "reason": "flush_open"
        }
        await self.bus.publish(topic_for(_TOPIC_CLOSE_BASE, self.camera_id), ev)
        print(f"✅ [Aggregator] flush_close {ev}")
        self._open = None
