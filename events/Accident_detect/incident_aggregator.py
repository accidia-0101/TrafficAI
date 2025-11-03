# events/logic/accident_aggregator.py
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class IncidentState:
    incident_id: str
    camera_id: str
    start_ts: float
    end_ts: float
    peak_conf: float
    pos_frames: int

class AccidentAggregator:
    """
    逐帧 -> 事件(open/close)
    - 开案(严格模式)：必须连续 N 帧 happened=True 才开（默认 N=3）
    - 关案：EMA <= exit_thr 且连续 min_end_frames 阴性
    - 其余：遮挡宽限、合并窗口保持
    可选：use_ema_open=True 时，允许 EMA+持续帧 作为“备用开案”通道
    """
    def __init__(
        self,
        camera_id: str,
        alpha: float = 0.25,
        enter_thr: float = 0.65,      # 仅用于 EMA 备用通道 & 关案时的阈值选择
        exit_thr: float = 0.40,
        min_persistence_frames: int = 3,  # EMA 备用通道所需正向帧数
        min_end_frames: int = 8,
        occlusion_grace_sec: float = 1.0,
        merge_gap_sec: float = 5.0,
        required_happened_consecutive: int = 3,  # ✅ 连续 N 帧 happened 才开
        use_ema_open: bool = False,              # ✅ 关闭：只靠 happened 连击开案
    ):
        self.camera_id = camera_id
        self.alpha = alpha
        self.enter_thr = enter_thr
        self.exit_thr = exit_thr
        self.min_persistence_frames = min_persistence_frames
        self.min_end_frames = min_end_frames
        self.occlusion_grace_sec = occlusion_grace_sec
        self.merge_gap_sec = merge_gap_sec
        self.required_happened_consecutive = max(1, required_happened_consecutive)
        self.use_ema_open = use_ema_open

        # 状态
        self.ema = 0.0
        self._pos_streak = 0          # 基于 EMA 的正向计数（仅用于备用通道/关案）
        self._neg_streak = 0
        self._hap_streak = 0          # ✅ happened 连击计数（严格开案用）
        self._first_pos_ts: Optional[float] = None
        self._first_hap_ts: Optional[float] = None  # ✅ 连击首帧时间
        self._open: Optional[IncidentState] = None
        self._last_seen_ts: Optional[float] = None
        self._last_end_ts: Optional[float] = None
        self._counter = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"{self.camera_id}-{self._counter:06d}"

    def update(
        self,
        ts: float,
        conf: float,
        frame_ok: bool = True,
        happened: bool = False,  # ✅ 上游逐帧是否报事故
    ) -> Tuple[Optional[dict], List[dict]]:
        open_event = None
        close_events: List[dict] = []

        # 1) EMA
        self.ema = self.alpha * conf + (1 - self.alpha) * self.ema

        # 2) 取帧/遮挡
        if frame_ok:
            self._last_seen_ts = ts

        # 3) 维护计数
        # 3.1 happened 连击（严格开案）
        if self._open is None:
            if happened:
                if self._hap_streak == 0:
                    self._first_hap_ts = ts
                self._hap_streak += 1
            else:
                self._hap_streak = 0
                self._first_hap_ts = None

        # 3.2 EMA 正/负（用于备用开案与关案）
        thr = self.exit_thr if self._open is not None else self.enter_thr
        is_pos = (self.ema >= thr)
        if is_pos:
            if self._pos_streak == 0:
                self._first_pos_ts = ts
            self._pos_streak += 1
            self._neg_streak = 0
        else:
            self._neg_streak += 1
            if self._open is None:
                self._first_pos_ts = None
                self._pos_streak = 0

        # 4) 进入判定
        if self._open is None:
            strict_open = (self._hap_streak >= self.required_happened_consecutive)  # ✅ 必须 N 连击
            ema_open = self.use_ema_open and (self.ema >= self.enter_thr) and (self._pos_streak >= self.min_persistence_frames)

            if strict_open or ema_open:
                reuse_last_start = (self._last_end_ts is not None and (ts - self._last_end_ts) <= self.merge_gap_sec)
                incident_id = self._new_id()
                # 回溯开案时间：严格模式用 first_hap_ts；备用通道用 first_pos_ts
                base_start = (self._first_hap_ts if strict_open else self._first_pos_ts) or ts
                start_ts = self._last_end_ts if reuse_last_start else base_start
                self._open = IncidentState(
                    incident_id=incident_id,
                    camera_id=self.camera_id,
                    start_ts=start_ts,
                    end_ts=ts,
                    peak_conf=conf,
                    pos_frames=1,
                )
                open_event = {
                    "type": "accident_open",
                    "incident_id": incident_id,
                    "camera_id": self.camera_id,
                    "ts_unix": start_ts,
                    "confidence": float(conf),
                }
                # 开案后清理连击起点（仅用于开案时的回溯）
                self._first_hap_ts = None
                self._first_pos_ts = None
                self._hap_streak = 0  # 防止马上又把下一帧当作新连击开第二次

        else:
            # 5) 事件已打开：更新尾部与峰值
            self._open.end_ts = ts
            if conf > self._open.peak_conf:
                self._open.peak_conf = conf
            if is_pos or happened:
                self._open.pos_frames += 1

            # 6) 结束判定（保守）
            grace_ok = True
            if self._last_seen_ts is not None and (ts - self._last_seen_ts) > self.occlusion_grace_sec:
                grace_ok = False
            if (self._neg_streak >= self.min_end_frames) and grace_ok:
                close_events.append({
                    "type": "accident_close",
                    "incident_id": self._open.incident_id,
                    "camera_id": self.camera_id,
                    "start_ts": self._open.start_ts,
                    "end_ts": self._open.end_ts,
                    "duration_sec": max(0.0, self._open.end_ts - self._open.start_ts),
                    "peak_confidence": float(self._open.peak_conf),
                    "pos_frames": int(self._open.pos_frames),
                })
                self._last_end_ts = self._open.end_ts
                self._open = None
                self._pos_streak = 0
                self._neg_streak = 0
                self._first_pos_ts = None
                self._first_hap_ts = None
                self._hap_streak = 0

        return open_event, close_events

    def flush(self) -> List[dict]:
        if self._open is None:
            return []
        inc = self._open
        self._open = None
        self._last_end_ts = inc.end_ts
        self._pos_streak = 0
        self._neg_streak = 0
        self._first_pos_ts = None
        self._first_hap_ts = None
        self._hap_streak = 0
        return [{
            "type": "accident_close",
            "incident_id": inc.incident_id,
            "camera_id": self.camera_id,
            "start_ts": inc.start_ts,
            "end_ts": inc.end_ts,
            "duration_sec": max(0.0, inc.end_ts - inc.start_ts),
            "peak_confidence": float(inc.peak_conf),
            "pos_frames": int(inc.pos_frames),
        }]
