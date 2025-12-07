"""
AccidentAggregator: aggregates per-frame detection results into stable accident events
(partitioned-topic version, rewritten, dual-stage decision, continuous-confidence version)
--------------------------------------------------------------------
Subscribe: accident:<camera_id>              # Per-frame detection results (Detection)
Publish:   accidents.open:<camera_id>        # Accident-open event (once)
           accidents.close:<camera_id>       # Accident-close event (may be delayed due to merge window)

Dual-stage decision:
- Stage 1 (Suspicion): 使用连续的置信度证据 + soft_score 累积“怀疑”，而不是对单帧做硬二元划分。
- Stage 2 (Validation): 在进入“怀疑期”后，再结合 EMA 和连续负帧判断，决定关案时机。

外部接口保持不变：
- 订阅/发布 topic 名不变
- open/close 事件字段不变
- __init__ / run / flush 签名不变
"""
from __future__ import annotations

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



import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from events.bus import AsyncBus, Detection, topic_for
print(">>> LOADED Aggregator: CUSTOM VERSION <<<")


_TOPIC_IN_BASE    = "accident"
_TOPIC_OPEN_BASE  = "accidents.open"
_TOPIC_CLOSE_BASE = "accidents.close"


#this has come to an elbow,no need to find better one

# ---------- EMA + 关案逻辑 ----------
_ALPHA = 0.22                     # EMA 更快响应下降
_EXIT_THR = 0.38                  # 更容易满足关案条件（并不影响开案）
_MIN_END_NEG_FRAMES = 8
_MIN_DURATION_SEC = 0.15          # 缩短最短事故秒数（对小事故有效）
_OCCLUSION_GRACE_SEC = 1.2
_MERGE_GAP_SEC = 4.0              # 合并窗口缩短，更精准

# ---------- 连续证据版开案参数 ----------
_EVIDENCE_BASELINE = 0.10         # 原先 0.12 → 降低，捕获更多“弱事故”
_EVIDENCE_MIN_CONF = 0.08         # 原先 0.10 → 降低，增加 soft_score 积累
_SOFT_GAIN = 3.0                  # 原先 2.5 → 提升事故累积分数速度
_SOFT_DECAY = 0.05                # 维持不变，避免爆炸式误报

# ---------- 开案瞬间条件 ----------
_OPEN_SCORE_THR = 0.75            # 原先 0.9 → 降低，更容易开案
_MIN_OPEN_CONF = 0.15             # 原先 0.18 → 降低，更容易开案





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
    事故聚合器（连续证据版，偏向高 precision，兼顾 recall）：
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

        # soft_score：连续证据累积（开案逻辑）
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
        happened = bool(getattr(det, "happened", False))  # 暂时保留，便于日后扩展
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

        # 4) 连续证据版 soft_score 更新：
        #    - conf 低于 EVIDENCE_MIN_CONF 视为纯背景，不加分
        #    - conf 高于 EVIDENCE_BASELINE 部分视为“正向证据”，按比例放大累加
        #    - 每帧有固定衰减，避免长时间积累导致误报
        if conf >= _EVIDENCE_MIN_CONF:
            # 只取高于 baseline 的部分作为正向证据
            pos_part = max(0.0, conf - _EVIDENCE_BASELINE)
            inc = pos_part * _SOFT_GAIN
        else:
            inc = 0.0

        self._soft_score += inc
        self._soft_score -= _SOFT_DECAY
        if self._soft_score < 0.0:
            self._soft_score = 0.0

        # ---------- 开案判定（连续证据版） ----------
        if self._open is None and self._soft_score >= _OPEN_SCORE_THR and conf >= _MIN_OPEN_CONF:
            # 4.1 merge window 内 reopen：合并到 pending close
            if self._pending_close is not None and self._pending_close_time is not None:
                if (vts - self._pending_close_time) <= _MERGE_GAP_SEC:
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
                pos_frames=1 if (happened or conf >= _MIN_OPEN_CONF) else 0,
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
            if happened or conf >= _MIN_OPEN_CONF:
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
                # soft_score 重置，重新进入观察期
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
            print("[Aggregator] flush(): no need to close")


