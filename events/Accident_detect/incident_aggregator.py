"""
AccidentAggregator: aggregates per-frame detection results into stable accident events
(partitioned-topic version, rewritten, dual-stage decision)
--------------------------------------------------------------------
Subscribe: accident:<camera_id>              # Per-frame detection results (Detection)
Publish:   accidents.open:<camera_id>        # Accident-open event (once)
           accidents.close:<camera_id>       # Accident-close event (may be delayed due to merge window)

Dual-stage decision:
- Stage 1 (Suspicion): 使用低置信度阈值 + EMA 累积“怀疑”，避免单帧噪声直接触发开案。
- Stage 2 (Validation): 在进入“怀疑期”后，再要求若干帧高置信度 happened 才真正开案。

外部接口保持不变：
- 订阅/发布 topic 名不变
- open/close 事件字段不变
- __init__ / run / flush 签名不变
"""

# # -----------------------------------------------------------------------------
# # Copyright (c) 2025
# #
# # Authors:
# #   Liruo Wang
# #       School of Electrical Engineering and Computer Science,
# #       University of Ottawa
# #       lwang032@uottawa.ca
# #
# # All rights reserved.
# # -----------------------------------------------------------------------------



from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from events.bus import AsyncBus, Detection, topic_for



_TOPIC_IN_BASE    = "accident"
_TOPIC_OPEN_BASE  = "accidents.open"
_TOPIC_CLOSE_BASE = "accidents.close"

# ---------- EMA + 关案逻辑（保持高 precision） ----------
_ALPHA = 0.25                     # EMA 平滑
_EXIT_THR = 0.40                  # EMA 低于此阈值才允许进入关案计数
_MIN_END_NEG_FRAMES = 8          # 关案严格：连续8帧负面才关案
_MIN_DURATION_SEC = 0.3           # 最短事故时间，避免一闪而过误报
_OCCLUSION_GRACE_SEC = 1.0        # 遮挡窗口（基于 vts）
_MERGE_GAP_SEC = 5.0              # merge reopen 窗口

# ---------- 开案参数（重点优化 recall + 保持高 precision） ----------
_OPEN_CONF_MIN = 0.25             # ↓ 原 0.50 → 0.25（显著提高 recall）
_SOFT_INC = 1.0                   # 正样本 soft_score 增量
_SOFT_DEC = 0.28                  # ↓ 原 0.4 → 0.28（略提升 recall）
_OPEN_SCORE_THR = 1.35            # ↓ 原 1.8 → 1.6 → 1.35（稍微易开案）


@dataclass
class _Incident:
    id: str
    camera_id: str
    start_ts: float     # 使用 vts 作为内部时间轴
    end_ts: float       # 使用 vts
    start_idx: int
    end_idx: int
    peak_conf: float = 0.0
    pos_frames: int = 0


class AccidentAggregator:
    """
    事故聚合器（偏向 precision）：
    - 输入：Detection(type='accident', ..., vts, pts_in_video, confidence, happened)
    - 输出事件：
        * accidents.open:<cam>
        * accidents.close:<cam>
      字段保持与旧版兼容。
    """

    def __init__(self, camera_id: str, bus: AsyncBus, *, session_id: Optional[str] = None) -> None:
        self.camera_id = camera_id
        self.bus = bus
        self.session_id = session_id or str(int(time.time()))
        self._counter = 0

        # 主 EMA（用于关案）
        self.ema: float = 0.0
        self._neg_streak: int = 0          # 连续“负面演化”帧计数

        # 当前是否有 open 的事故
        self._open: Optional[_Incident] = None

        # vts 时间线
        self._last_vts: Optional[float] = None

        # 软连续计数（开案逻辑）
        self._soft_score: float = 0.0

        # merge window: pending close 事件（等待是否要 merge reopen）
        self._pending_close: Optional[Dict[str, Any]] = None
        self._pending_close_time: Optional[float] = None  # 用 vts 记录

    # ---------- 工具 ----------

    def _new_id(self) -> str:
        self._counter += 1
        return f"{self.session_id}-{self.camera_id}-{self._counter:06d}"

    async def _emit_open(self, inc: _Incident) -> None:
        """
        开案事件输出格式保持不变：
        - pts_in_video 字段仍然使用 start_ts（这里用 vts，对前端只是一个时间轴）。
        """
        ev = {
            "type": "accident_open",
            "camera_id": self.camera_id,
            "frame_idx": inc.start_idx,
            "pts_in_video": inc.start_ts,
            "confidence": inc.peak_conf,
            "session_id": self.session_id,
            "incident_id": inc.id,
            "peak_confidence": inc.peak_conf,
        }
        await self.bus.publish(topic_for(_TOPIC_OPEN_BASE, self.camera_id), ev)
        print(f" OPEN {ev}")

    async def _schedule_close(self, inc: _Incident) -> None:
        """
        关案暂存为 pending close，保留 merge window。
        """
        close_ev = {
            "type": "accident_close",
            "camera_id": self.camera_id,
            "frame_idx": inc.end_idx,
            "pts_in_video": inc.end_ts,  # 同样用 vts 输出，保持与 open 一致
            "confidence": inc.peak_conf,
            "session_id": self.session_id,
            "incident_id": inc.id,
            # merge / 分析所需字段：
            "start_ts": inc.start_ts,
            "end_ts": inc.end_ts,
            "duration_sec": max(0.0, inc.end_ts - inc.start_ts),
            "peak_confidence": inc.peak_conf,
            "pos_frames": inc.pos_frames,
        }
        self._pending_close = close_ev
        self._pending_close_time = inc.end_ts
        print(f" CLOSE (pending merge) {close_ev}")

    async def _flush_pending_close_if_expired(self, now_vts: float) -> None:
        """
        若 merge window 已经过期，则真正发出 close 事件。
        """
        if self._pending_close is None or self._pending_close_time is None:
            return

        if now_vts - self._pending_close_time > _MERGE_GAP_SEC:
            await self.bus.publish(topic_for(_TOPIC_CLOSE_BASE, self.camera_id), self._pending_close)
            print(f" CLOSE (emit pending) {self._pending_close}")
            self._pending_close = None
            self._pending_close_time = None

    def _merge_reopen_into_pending(
        self,
        new_vts: float,
        new_peak: float,
        new_pos_frames: int,
        new_end_idx: int,
    ) -> None:
        """
        merge window 内 reopen：合并到 pending close 中。
        """
        assert self._pending_close is not None
        pc = self._pending_close
        pc["end_ts"] = new_vts
        pc["duration_sec"] = max(0.0, pc["end_ts"] - pc["start_ts"])
        pc["peak_confidence"] = max(float(pc["peak_confidence"]), float(new_peak))
        pc["pos_frames"] = int(pc.get("pos_frames", 0)) + int(new_pos_frames)
        # frame_idx 只用于 debug，不存到事件里

    # ---------- 主循环 ----------

    async def run(self) -> None:
        topic_in = topic_for(_TOPIC_IN_BASE, self.camera_id)
        async with self.bus.subscribe(topic_in, mode="fifo", maxsize=128) as sub:
            while True:
                det: Detection = await sub.get()
                await self._process(det)

    async def _process(self, det: Detection) -> None:
        # vts 为聚合时间轴；pts_in_video 仍保留给前端用（若需要）
        vts = float(getattr(det, "vts", getattr(det, "pts_in_video", 0.0)))
        conf = float(getattr(det, "confidence", 0.0))
        happened = bool(getattr(det, "happened", False))
        fidx = int(getattr(det, "frame_idx", 0))

        # 1) merge window 过期检测
        await self._flush_pending_close_if_expired(vts)

        # 2) occlusion / frame gap（用 vts，看是否跳太多）
        prev_vts = self._last_vts
        self._last_vts = vts
        if prev_vts is not None and (vts - prev_vts) > _OCCLUSION_GRACE_SEC:
            occlusion_ok = False
        else:
            occlusion_ok = True

        # 3) 更新 EMA（用于关案）
        self.ema = _ALPHA * conf + (1.0 - _ALPHA) * self.ema

        # 4) 开案软计数（soft_score）
        if happened and conf >= _OPEN_CONF_MIN:
            self._soft_score += _SOFT_INC
        else:
            self._soft_score -= _SOFT_DEC

        if self._soft_score < 0.0:
            self._soft_score = 0.0

        # ---------- 开案判定（偏向 precision 的旧逻辑替代） ----------
        if self._open is None and self._soft_score >= _OPEN_SCORE_THR:
            # 4.1 merge window 内的 reopen
            if self._pending_close is not None and self._pending_close_time is not None:
                if (vts - self._pending_close_time) <= _MERGE_GAP_SEC:
                    # 合并 reopen 信息
                    self._merge_reopen_into_pending(
                        new_vts=vts,
                        new_peak=conf,
                        new_pos_frames=1,
                        new_end_idx=fidx,
                    )
                    pc = self._pending_close
                    inc = _Incident(
                        id=pc["incident_id"],
                        camera_id=self.camera_id,
                        start_ts=pc["start_ts"],
                        end_ts=vts,
                        start_idx=fidx,  # 这里 start_idx 仅用于 debug，前端不依赖
                        end_idx=fidx,
                        peak_conf=float(pc["peak_confidence"]),
                        pos_frames=int(pc.get("pos_frames", 0)),
                    )
                    self._pending_close = None
                    self._pending_close_time = None
                    self._open = inc
                    # 开案之后 soft_score 不清零，这样后续还有一定鲁棒性
                    return

            # 4.2 正常新开案
            inc = _Incident(
                id=self._new_id(),
                camera_id=self.camera_id,
                start_ts=vts,
                end_ts=vts,
                start_idx=fidx,
                end_idx=fidx,
                peak_conf=conf,
                pos_frames=1 if happened else 0,
            )
            self._open = inc
            await self._emit_open(inc)
            # soft_score 不归零，略微提高后续对持续事故的容忍
            return

        # ---------- 事件已 open：更新 + 关案判定 ----------
        if self._open is not None:
            inc = self._open
            inc.end_ts = vts
            inc.end_idx = fidx
            inc.peak_conf = max(inc.peak_conf, conf)
            if happened:
                inc.pos_frames += 1

            # 关案条件：EMA 低 + 连续负帧
            if self.ema <= _EXIT_THR and occlusion_ok:
                self._neg_streak += 1
            else:
                self._neg_streak = 0

            duration = inc.end_ts - inc.start_ts

            if self._neg_streak >= _MIN_END_NEG_FRAMES and duration >= _MIN_DURATION_SEC:
                await self._schedule_close(inc)
                # 重置状态
                self._open = None
                self.ema = 0.0
                self._neg_streak = 0
                # soft_score 可以适当保留一点记忆，也可以归零：
                self._soft_score = 0.0

    # ---------- flush 收尾 ----------

    async def flush(self) -> None:
        """
        Session 结束时调用：
        - 若有 pending close：直接发送。
        - 若还有 open：立即发送一个 close。
        """
        did_close = False

        if self._pending_close is not None:
            await self.bus.publish(topic_for(_TOPIC_CLOSE_BASE, self.camera_id), self._pending_close)
            print(f" CLOSE (emit pending) {self._pending_close}")
            self._pending_close = None
            self._pending_close_time = None
            did_close = True

        if self._open is not None:
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
                "reason": "flush_open",
            }
            await self.bus.publish(topic_for(_TOPIC_CLOSE_BASE, self.camera_id), ev)
            print(f"[Aggregator] flush_close {ev}")
            self._open = None
            did_close = True

        if not did_close:
            print("[Aggregator] flush(): 无需结案。")

