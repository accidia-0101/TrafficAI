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

print(">>> LOADED Aggregator: V2 STATE-MACHINE + FUTURE MOTION SLOT <<<")

_TOPIC_IN_BASE = "accident"
_TOPIC_OPEN_BASE = "accidents.open"
_TOPIC_CLOSE_BASE = "accidents.close"

# ---------------------------------------------------------------------
# State names
# ---------------------------------------------------------------------
_IDLE = "IDLE"
_SUSPECT = "SUSPECT"
_OPEN = "OPEN"

# ---------------------------------------------------------------------
# Compatibility / startup protection
# ---------------------------------------------------------------------
_WARMUP_FRAMES = 12  # 比旧版更短；主要避免一启动就被噪声点亮

# ---------------------------------------------------------------------
# Evidence smoothing
# ---------------------------------------------------------------------
_EVIDENCE_EMA_ALPHA = 0.30

# 当前只有 YOLO，所以 fused_evidence = yolo_evidence
# 以后接 motion 时，在 _fuse_evidence() 里加入 motion 即可
_YOLO_EVIDENCE_FLOOR = 0.05  # 很弱的 detector conf 直接视为接近无证据
_YOLO_EVIDENCE_SCALE = 1.0   # 当前单源，保持 1.0

# ---------------------------------------------------------------------
# SUSPECT -> OPEN
# ---------------------------------------------------------------------
# 用积分 + 连续高证据共同判断
_SUSPECT_ENTER_THR = 0.22        # 进入怀疑态的单帧/平滑证据阈值
_SUSPECT_SCORE_GAIN = 1.00
_SUSPECT_SCORE_DECAY = 0.18
_SUSPECT_OPEN_SCORE_THR = 2   # 达到该积分才正式 open
_SUSPECT_MIN_HIGH_FRAMES = 5      # 至少积累若干个高证据帧
_SUSPECT_TIMEOUT_SEC = 2.0        # 怀疑太久没成案就回到 IDLE

# ---------------------------------------------------------------------
# OPEN maintenance / close
# ---------------------------------------------------------------------
_OPEN_HOLD_HIGH_THR = 0.16       # 高于这个值认为仍有正向证据
_CLOSE_LOW_THR = 0.10            # 低于这个值开始记“低证据帧”
_MIN_CLOSE_LOW_FRAMES = 10       # 持续低证据达到这么多帧才允许 close
_MIN_DURATION_SEC = 0.25         # 太短的事件不合理，避免刚开就关
_OCCLUSION_GRACE_SEC = 1.2       # 时间断裂过大时，不立刻强制 close

# ---------------------------------------------------------------------
# Merge / reopen
# ---------------------------------------------------------------------
_MERGE_GAP_SEC = 4.0

# ---------------------------------------------------------------------
# Future extension slot
# ---------------------------------------------------------------------
_ENABLE_MOTION_SLOT = False
# 以后你接 motion 时，把 _last_motion_score 填起来，然后在 _fuse_evidence() 里融合即可


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

        # ------------ external-compatible incident states ------------
        self._open: Optional[_Incident] = None

        # ------------ explicit state machine ------------
        self._state: str = _IDLE
        self._warmup_left = _WARMUP_FRAMES

        # ------------ timing ------------
        self._last_vts: Optional[float] = None
        self._state_enter_vts: Optional[float] = None

        # ------------ evidence memory ------------
        self._ema_evidence: float = 0.0
        self._suspect_score: float = 0.0
        self._high_streak: int = 0
        self._low_streak: int = 0

        # ------------ latest source evidence ------------
        self._last_yolo_conf: float = 0.0
        self._last_motion_score: float = 0.0  # future slot
        self._last_fused_evidence: float = 0.0

        # ------------ merge window ------------
        self._pending_close: Optional[Dict[str, Any]] = None
        self._pending_close_time: Optional[float] = None

    # -----------------------------------------------------------------
    # ID helpers
    # -----------------------------------------------------------------
    def _new_id(self) -> str:
        self._counter += 1
        return f"{self.session_id}-{self.camera_id}-{self._counter:06d}"

    # -----------------------------------------------------------------
    # External event emitters (payload kept compatible)
    # -----------------------------------------------------------------
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

    def _merge_reopen_into_pending(self, new_vts: float, new_peak: float, new_pos_frames: int) -> None:
        pc = self._pending_close
        pc["end_ts"] = new_vts
        pc["duration_sec"] = max(0.0, pc["end_ts"] - pc["start_ts"])
        pc["peak_confidence"] = max(float(pc["peak_confidence"]), float(new_peak))
        pc["pos_frames"] += int(new_pos_frames)

    # -----------------------------------------------------------------
    # Future-proof evidence layer
    # -----------------------------------------------------------------
    def _yolo_to_evidence(self, conf: float) -> float:
        # 把 detector conf 转成更平滑、更适合事件级聚合的 evidence
        # floor 以下直接弱化，避免全是微弱噪声
        if conf <= _YOLO_EVIDENCE_FLOOR:
            return 0.0
        return min(1.0, (conf - _YOLO_EVIDENCE_FLOOR) / (1.0 - _YOLO_EVIDENCE_FLOOR)) * _YOLO_EVIDENCE_SCALE

    def _fuse_evidence(self, yolo_evidence: float) -> float:
        # 未来 motion 进来时，在这里加：
        # motion_evidence = ...
        # agreement_bonus = ...
        # return ...
        if not _ENABLE_MOTION_SLOT:
            return yolo_evidence
        # placeholder
        return yolo_evidence

    # -----------------------------------------------------------------
    # State helpers
    # -----------------------------------------------------------------
    def _enter_state(self, state: str, vts: float) -> None:
        self._state = state
        self._state_enter_vts = vts

    def _reset_suspect(self, vts: float) -> None:
        self._suspect_score = 0.0
        self._high_streak = 0
        self._enter_state(_IDLE, vts)

    def _reset_after_close(self, vts: float) -> None:
        self._open = None
        self._suspect_score = 0.0
        self._high_streak = 0
        self._low_streak = 0
        self._ema_evidence = 0.0
        self._enter_state(_IDLE, vts)

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
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

        await self._flush_pending_close_if_expired(vts)

        prev_vts = self._last_vts
        self._last_vts = vts

        # 允许时间断裂，但不把它直接当成 close
        occlusion_ok = not (prev_vts is not None and (vts - prev_vts) > _OCCLUSION_GRACE_SEC)

        # -------------------------------------------------------------
        # Warmup
        # -------------------------------------------------------------
        if self._open is None and self._warmup_left > 0:
            self._warmup_left -= 1
            return

        # -------------------------------------------------------------
        # Evidence update
        # -------------------------------------------------------------
        self._last_yolo_conf = conf
        yolo_evidence = self._yolo_to_evidence(conf)
        fused = self._fuse_evidence(yolo_evidence)
        self._last_fused_evidence = fused

        self._ema_evidence = _EVIDENCE_EMA_ALPHA * fused + (1.0 - _EVIDENCE_EMA_ALPHA) * self._ema_evidence

        current_signal = max(fused, self._ema_evidence)

        # -------------------------------------------------------------
        # IDLE -> SUSPECT
        # -------------------------------------------------------------
        if self._state == _IDLE and self._open is None:
            if current_signal >= _SUSPECT_ENTER_THR:
                self._suspect_score = current_signal * _SUSPECT_SCORE_GAIN
                self._high_streak = 1
                self._low_streak = 0
                self._enter_state(_SUSPECT, vts)
            return

        # -------------------------------------------------------------
        # SUSPECT state
        # -------------------------------------------------------------
        if self._state == _SUSPECT and self._open is None:
            if current_signal >= _SUSPECT_ENTER_THR:
                self._suspect_score += current_signal * _SUSPECT_SCORE_GAIN
                self._high_streak += 1
            else:
                self._suspect_score -= _SUSPECT_SCORE_DECAY
                self._suspect_score = max(0.0, self._suspect_score)

            # timeout: 怀疑太久没成案，回到 IDLE
            if self._state_enter_vts is not None and (vts - self._state_enter_vts) > _SUSPECT_TIMEOUT_SEC:
                self._reset_suspect(vts)
                return

            # 如果掉太低，也回到 IDLE
            if self._suspect_score <= 0.0:
                self._reset_suspect(vts)
                return

            # 满足开案条件：积分够 + 连续高证据够
            if self._suspect_score >= _SUSPECT_OPEN_SCORE_THR and self._high_streak >= _SUSPECT_MIN_HIGH_FRAMES:
                # merge reopen case
                if self._pending_close is not None and (vts - self._pending_close_time) <= _MERGE_GAP_SEC:
                    pc = self._pending_close
                    self._merge_reopen_into_pending(vts, conf, 1)
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
                    self._low_streak = 0
                    self._enter_state(_OPEN, vts)
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
                    pos_frames=1 if (happened or current_signal >= _SUSPECT_ENTER_THR) else 0,
                )
                self._open = inc
                self._low_streak = 0
                self._enter_state(_OPEN, vts)
                await self._emit_open(inc)
            return

        # -------------------------------------------------------------
        # OPEN state
        # -------------------------------------------------------------
        if self._state == _OPEN and self._open is not None:
            inc = self._open
            inc.end_ts = vts
            inc.end_idx = fidx
            inc.peak_conf = max(inc.peak_conf, conf)

            if happened or current_signal >= _OPEN_HOLD_HIGH_THR:
                inc.pos_frames += 1

            # close logic
            if current_signal <= _CLOSE_LOW_THR and occlusion_ok:
                self._low_streak += 1
            else:
                self._low_streak = 0

            duration = inc.end_ts - inc.start_ts
            if self._low_streak >= _MIN_CLOSE_LOW_FRAMES and duration >= _MIN_DURATION_SEC:
                await self._schedule_close(inc)
                self._reset_after_close(vts)
            return

    # -----------------------------------------------------------------
    # Flush: keep external behavior compatible
    # -----------------------------------------------------------------
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
