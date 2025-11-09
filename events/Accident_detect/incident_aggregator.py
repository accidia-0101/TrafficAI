# events/logic/accident_aggregator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple

# ==============================
# 固定配置（不对外暴露）
# ==============================
_ALPHA = 0.25                   # EMA 平滑系数（越大越敏感）
_ENTER_THR = 0.65               # 进入阈值（仅用于可选宽松通道；本版不开）
_EXIT_THR = 0.40                # 退出阈值（EMA 降到此以下才允许关案）
_REQUIRED_HAPPENED_CONSEC = 3   # 严格开案：必须连续 N 帧 happened=True
_MIN_END_NEG_FRAMES = 8         # 关案：至少连续 N 帧“阴性”
_OCCLUSION_GRACE_SEC = 1.0      # 遮挡宽限：超时未见到帧则不允许关案
_MERGE_GAP_SEC = 5.0            # 合并窗口：距上次关案 ≤ 该秒，再开案则视为同一事故
_USE_RELAXED_OPEN = False       # 关闭“EMA 开案”通道，纯严格

@dataclass
class IncidentState:
    incident_id: str
    camera_id: str
    start_ts: float            # 视频内 PTS 秒
    end_ts: float
    peak_conf: float
    pos_frames: int
    start_fidx: Optional[int] = None
    end_fidx: Optional[int] = None

class AccidentAggregator:
    """
    帧级检测 → 事故(open/close) 事件。
    设计目标：简单可证、行为可预期、和前端 currentTime 天然对齐。
    - 开案：必须连续 N 帧 happened=True（严格通道）
    - 关案：EMA <= EXIT_THR 且连续 MIN_END_NEG_FRAMES 阴性；且未超出遮挡宽限
    - 合并窗口：MERGE_GAP_SEC 内复开 → 合并为同一事故（起点复用上次 start_ts）
    - ts 一律使用 pts_in_video；frame_idx 仅用于对齐和日志
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id

        # 滤波与节拍
        self.ema: float = 0.0
        self._pos_streak: int = 0        # EMA ≥ 阈值的连续帧（仅用于关案阶段计数）
        self._neg_streak: int = 0        # EMA < 阈值的连续帧
        self._hap_streak: int = 0        # happened=True 的连续帧（严格开案通道）

        # 严格通道回溯起点
        self._first_hap_ts: Optional[float] = None
        self._first_hap_fidx: Optional[int] = None

        # 可见性
        self._last_seen_ts: Optional[float] = None

        # 事件状态
        self._open: Optional[IncidentState] = None
        self._last_closed_start_ts: Optional[float] = None
        self._last_end_ts: Optional[float] = None

        # 仅内部使用的自增 ID
        self._counter: int = 0

    # ---------- 工具 ----------
    def _new_id(self) -> str:
        self._counter += 1
        return f"{self.camera_id}-{self._counter:06d}"

    # ---------- 外部喂入（推荐） ----------
    def push_detection(self, det) -> Tuple[Optional[dict], List[dict]]:
        """
        输入：Detection(type='accident', happened:bool, confidence:float,
                       pts_in_video:float, frame_idx:int, frame_ok:bool?)
        返回：(open_event_or_None, close_events_list)
        """
        return self.update(
            ts=float(getattr(det, "pts_in_video", 0.0)),
            conf=float(getattr(det, "confidence", 0.0)),
            frame_ok=bool(getattr(det, "frame_ok", True)),
            happened=bool(getattr(det, "happened", False)),
            frame_idx=getattr(det, "frame_idx", None),
        )

    # ---------- 外部喂入（逐字段，兼容旧代码） ----------
    def update(
        self,
        ts: float,
        conf: float,
        *,
        frame_ok: bool = True,
        happened: bool = False,
        frame_idx: Optional[int] = None,
    ) -> Tuple[Optional[dict], List[dict]]:
        """
        返回：
          open_event: {type='accident_open', incident_id, camera_id, ts, start_frame_idx, confidence}
          close_events: [{type='accident_close', incident_id, camera_id, start_ts, end_ts, ...}, ...]
        """
        open_event: Optional[dict] = None
        close_events: List[dict] = []

        # 0) 可见性更新
        if frame_ok:
            self._last_seen_ts = ts

        # 1) EMA 更新
        self.ema = _ALPHA * conf + (1.0 - _ALPHA) * self.ema

        # 2) “严格开案通道”：统计 happened 连击
        if self._open is None:
            if happened:
                if self._hap_streak == 0:
                    self._first_hap_ts = ts
                    self._first_hap_fidx = frame_idx
                self._hap_streak += 1
            else:
                self._hap_streak = 0
                self._first_hap_ts = None
                self._first_hap_fidx = None

        # 3) 阴/阳性 streak（仅用于关案阶段的稳定判定）
        is_pos = (self.ema >= _EXIT_THR)  # 关案阶段用 EXIT_THR 来区分阴/阳
        if is_pos:
            self._pos_streak += 1
            self._neg_streak = 0
        else:
            self._neg_streak += 1
            self._pos_streak = 0

        # 4) 开案判定
        if self._open is None:
            strict_open = (self._hap_streak >= _REQUIRED_HAPPENED_CONSEC)
            relaxed_open = _USE_RELAXED_OPEN and (self.ema >= _ENTER_THR)

            if strict_open or relaxed_open:
                # 是否与刚关闭的事件合并
                reuse_last_start = (
                    self._last_end_ts is not None
                    and (ts - self._last_end_ts) <= _MERGE_GAP_SEC
                    and self._last_closed_start_ts is not None
                )

                start_ts = (self._first_hap_ts if strict_open else ts) or ts
                start_fidx = (self._first_hap_fidx if strict_open else None)

                if reuse_last_start:
                    # 合并：起点复用上次事故的 start_ts；frame_idx 不再向前回溯
                    start_ts = self._last_closed_start_ts
                    start_fidx = None  # 合并时不再承诺精确帧号

                inc = IncidentState(
                    incident_id=self._new_id(),
                    camera_id=self.camera_id,
                    start_ts=start_ts,
                    end_ts=ts,
                    peak_conf=conf,
                    pos_frames=1 if (is_pos or happened) else 0,
                    start_fidx=start_fidx,
                    end_fidx=frame_idx,
                )
                self._open = inc

                open_event = {
                    "type": "accident_open",
                    "incident_id": inc.incident_id,
                    "camera_id": inc.camera_id,
                    "ts": inc.start_ts,
                    "start_frame_idx": inc.start_fidx,
                    "confidence": float(conf),
                }

                # 重置严格通道缓存
                self._hap_streak = 0
                self._first_hap_ts = None
                self._first_hap_fidx = None

        else:
            # 5) 已开案：更新尾部、峰值、正样本计数
            self._open.end_ts = ts
            self._open.end_fidx = frame_idx
            if conf > self._open.peak_conf:
                self._open.peak_conf = conf
            if is_pos or happened:
                self._open.pos_frames += 1

            # 6) 关案判定：阴性持续 + EMA 低 + 可见性未超时
            grace_ok = True
            if self._last_seen_ts is not None:
                grace_ok = (ts - self._last_seen_ts) <= _OCCLUSION_GRACE_SEC

            if grace_ok and (self.ema <= _EXIT_THR) and (self._neg_streak >= _MIN_END_NEG_FRAMES):
                inc = self._open
                close_events.append({
                    "type": "accident_close",
                    "incident_id": inc.incident_id,
                    "camera_id": inc.camera_id,
                    "start_ts": inc.start_ts,
                    "end_ts": inc.end_ts,
                    "start_frame_idx": inc.start_fidx,
                    "end_frame_idx": inc.end_fidx,
                    "duration_sec": max(0.0, inc.end_ts - inc.start_ts),
                    "peak_confidence": float(inc.peak_conf),
                    "pos_frames": int(inc.pos_frames),
                })

                # 更新“上一次结案”的时间用于合并窗口
                self._last_closed_start_ts = inc.start_ts
                self._last_end_ts = inc.end_ts

                # 复位
                self._open = None
                self._pos_streak = 0
                self._neg_streak = 0
                self._hap_streak = 0
                self._first_hap_ts = None
                self._first_hap_fidx = None

        return open_event, close_events

    def flush(self) -> List[dict]:
        """
        文件结束/切源前强制结案；返回 0~1 个 close 事件。
        """
        if self._open is None:
            return []
        inc = self._open
        self._open = None

        self._last_closed_start_ts = inc.start_ts
        self._last_end_ts = inc.end_ts

        # 复位 streak
        self._pos_streak = 0
        self._neg_streak = 0
        self._hap_streak = 0
        self._first_hap_ts = None
        self._first_hap_fidx = None

        return [{
            "type": "accident_close",
            "incident_id": inc.incident_id,
            "camera_id": inc.camera_id,
            "start_ts": inc.start_ts,
            "end_ts": inc.end_ts,
            "start_frame_idx": inc.start_fidx,
            "end_frame_idx": inc.end_fidx,
            "duration_sec": max(0.0, inc.end_ts - inc.start_ts),
            "peak_confidence": float(inc.peak_conf),
            "pos_frames": int(inc.pos_frames),
        }]
