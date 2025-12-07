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
print(">>> LOADED Aggregator: CUSTOM VERSION + WARMUP <<<")

_TOPIC_IN_BASE    = "accident"
_TOPIC_OPEN_BASE  = "accidents.open"
_TOPIC_CLOSE_BASE = "accidents.close"

# --------------------- NEW: warmup frame count ---------------------
_WARMUP_FRAMES = 20
# -------------------------------------------------------------------

# ---------- EMA + close logic ----------
_ALPHA = 0.22
_EXIT_THR = 0.38
_MIN_END_NEG_FRAMES = 8
_MIN_DURATION_SEC = 0.15
_OCCLUSION_GRACE_SEC = 1.2
_MERGE_GAP_SEC = 4.0

# ---------- soft evidence ----------
_EVIDENCE_BASELINE = 0.10
_EVIDENCE_MIN_CONF = 0.08
_SOFT_GAIN = 3.0
_SOFT_DECAY = 0.05

# ---------- open conditions ----------
_OPEN_SCORE_THR = 0.75
_MIN_OPEN_CONF = 0.15


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

    def __init__(self, camera_id: str, bus: AsyncBus, *, session_id: Optional[str] = None) -> None:
        self.camera_id = camera_id
        self.bus = bus
        self.session_id = session_id or str(int(time.time()))
        self._counter = 0

        # internal states
        self.ema = 0.0
        self._neg_streak = 0
        self._open: Optional[_Incident] = None

        self._last_vts: Optional[float] = None
        self._soft_score: float = 0.0

        # merge window
        self._pending_close: Optional[Dict[str, Any]] = None
        self._pending_close_time: Optional[float] = None

        # ------------------ NEW: warmup counter ------------------
        self._warmup_left = _WARMUP_FRAMES
        # ---------------------------------------------------------

    # utils
    def _new_id(self) -> str:
        self._counter += 1
        return f"{self.session_id}-{self.camera_id}-{self._counter:06d}"

    async def _emit_open(self, inc: _Incident) -> None:
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
        close_ev = {
            "type": "accident_close",
            "camera_id": self.camera_id,
            "frame_idx": inc.end_idx,
            "pts_in_video": inc.end_ts,
            "confidence": inc.peak_conf,
            "session_id": self.session_id,
            "incident_id": inc.id,
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
        if self._pending_close is None:
            return
        if now_vts - self._pending_close_time > _MERGE_GAP_SEC:
            await self.bus.publish(topic_for(_TOPIC_CLOSE_BASE, self.camera_id), self._pending_close)
            print(f" CLOSE (emit pending) {self._pending_close}")
            self._pending_close = None
            self._pending_close_time = None

    def _merge_reopen_into_pending(self, new_vts, new_peak, new_pos_frames, new_end_idx):
        pc = self._pending_close
        pc["end_ts"] = new_vts
        pc["duration_sec"] = max(0.0, pc["end_ts"] - pc["start_ts"])
        pc["peak_confidence"] = max(float(pc["peak_confidence"]), float(new_peak))
        pc["pos_frames"] += int(new_pos_frames)

    # main loop
    async def run(self) -> None:
        topic_in = topic_for(_TOPIC_IN_BASE, self.camera_id)
        async with self.bus.subscribe(topic_in, mode="fifo", maxsize=128) as sub:
            while True:
                det: Detection = await sub.get()
                await self._process(det)

    async def _process(self, det: Detection) -> None:

        vts = float(det.vts)
        conf = float(det.confidence)
        happened = bool(det.happened)
        fidx = int(det.frame_idx)

        # flush pending close
        await self._flush_pending_close_if_expired(vts)

        # occlusion check
        prev_vts = self._last_vts
        self._last_vts = vts
        occlusion_ok = not (prev_vts is not None and (vts - prev_vts) > _OCCLUSION_GRACE_SEC)

        # EMA update
        self.ema = _ALPHA * conf + (1 - _ALPHA) * self.ema

        # soft evidence update
        if conf >= _EVIDENCE_MIN_CONF:
            pos_part = max(0.0, conf - _EVIDENCE_BASELINE)
            inc = pos_part * _SOFT_GAIN
        else:
            inc = 0.0
        self._soft_score += inc
        self._soft_score -= _SOFT_DECAY
        self._soft_score = max(self._soft_score, 0.0)

        # ----------------------- NEW: warmup block -----------------------
        if self._open is None and self._warmup_left > 0:
            self._warmup_left -= 1
            return
        # ----------------------------------------------------------------

        # -------- open decision --------
        if self._open is None and self._soft_score >= _OPEN_SCORE_THR and conf >= _MIN_OPEN_CONF:

            # merge reopen case
            if self._pending_close is not None and (vts - self._pending_close_time) <= _MERGE_GAP_SEC:
                pc = self._pending_close
                self._merge_reopen_into_pending(vts, conf, 1, fidx)
                inc = _Incident(
                    id=pc["incident_id"],
                    camera_id=self.camera_id,
                    start_ts=pc["start_ts"],
                    end_ts=vts,
                    start_idx=fidx,
                    end_idx=fidx,
                    peak_conf=float(pc["peak_confidence"]),
                    pos_frames=int(pc.get("pos_frames", 0)),
                )
                self._pending_close = None
                self._pending_close_time = None
                self._open = inc
                return

            # normal open
            inc = _Incident(
                id=self._new_id(),
                camera_id=self.camera_id,
                start_ts=vts,
                end_ts=vts,
                start_idx=fidx,
                end_idx=fidx,
                peak_conf=conf,
                pos_frames=1 if happened or conf >= _MIN_OPEN_CONF else 0,
            )
            self._open = inc
            await self._emit_open(inc)
            return

        # -------- update during open --------
        if self._open is not None:
            inc = self._open
            inc.end_ts = vts
            inc.end_idx = fidx
            inc.peak_conf = max(inc.peak_conf, conf)
            if happened or conf >= _MIN_OPEN_CONF:
                inc.pos_frames += 1

            # close logic
            if self.ema <= _EXIT_THR and occlusion_ok:
                self._neg_streak += 1
            else:
                self._neg_streak = 0

            duration = inc.end_ts - inc.start_ts
            if self._neg_streak >= _MIN_END_NEG_FRAMES and duration >= _MIN_DURATION_SEC:
                await self._schedule_close(inc)
                self._open = None
                self.ema = 0.0
                self._neg_streak = 0
                self._soft_score = 0.0

    async def flush(self) -> None:
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
