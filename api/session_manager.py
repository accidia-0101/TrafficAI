# -*- coding: utf-8 -*-
import os, threading, time, uuid
from dataclasses import dataclass
from typing import Optional

from .camera_map import CAMERA_VIDEO_MAP

@dataclass
class SourceMeta:
    fps: float | None
    width: int | None
    height: int | None
    duration_sec: float | None
    has_true_pts: bool

@dataclass
class PlaySession:
    session_id: str
    camera_id: str
    video_id: str
    video_path: str
    server_ack_ts: float
    source_meta: SourceMeta
    sample_fps: float = 15.0
    status: str = "running"           # "running" | "stopping" | "stopped"
    stop_reason: str | None = None
    stop_ts: float | None = None

class SessionManager:
    """极简内存会话管理"""
    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: dict[str, PlaySession] = {}
        self._camera_lock: dict[str, str] = {}     # camera_id -> session_id
        self._idem_cache: dict[str, str] = {}      # idempotency_key -> session_id

    def get_session(self, session_id: str) -> Optional[PlaySession]:
        with self._lock:
            return self._sessions.get(session_id)

    def get_by_idempotency(self, idem_key: str) -> Optional[PlaySession]:
        if not idem_key: return None
        with self._lock:
            sid = self._idem_cache.get(idem_key)
            return self._sessions.get(sid) if sid else None

    def create_session(self, camera_id: str, *, idem_key: Optional[str]=None, allow_parallel: bool=False) -> PlaySession:
        if idem_key:
            s = self.get_by_idempotency(idem_key)
            if s: return s

        if camera_id not in CAMERA_VIDEO_MAP:
            raise FileNotFoundError("camera_unmapped")
        video_path = CAMERA_VIDEO_MAP[camera_id]
        if not (video_path.startswith("rtsp://") or os.path.exists(video_path)):
            raise FileNotFoundError("video_not_found")

        with self._lock:
            if (not allow_parallel) and (camera_id in self._camera_lock):
                sid_busy = self._camera_lock[camera_id]
                raise RuntimeError(f"camera_locked:{sid_busy}")

            meta = self._probe_meta(video_path)
            sid = f"sess_{uuid.uuid4().hex[:8]}"
            video_id = self._make_video_id(camera_id, video_path)
            ps = PlaySession(
                session_id=sid,
                camera_id=camera_id,
                video_id=video_id,
                video_path=video_path,
                server_ack_ts=time.time(),
                source_meta=meta,
            )
            self._sessions[sid] = ps
            self._camera_lock[camera_id] = sid
            if idem_key:
                self._idem_cache[idem_key] = sid
            return ps

    def stop_session(self, session_id: str, reason: str = "stopped") -> tuple[bool, Optional[str], Optional[PlaySession]]:
        """
        幂等停止：返回 (ok, released_camera_id, session_obj_or_none)
        - ok: True 表示本次调用已成功进入/完成关停流程；不存在则 False
        - released_camera_id: 成功时返回释放的 camera_id；否则 None
        """
        with self._lock:
            ps = self._sessions.get(session_id)
            if not ps:
                return (False, None, None)

            # 已经在停止或已停止：幂等返回
            if ps.status in ("stopping", "stopped"):
                # 确保 camera 锁已释放（如果之前没释放，这里补释放一次）
                self._camera_lock.pop(ps.camera_id, None)
                return (True, ps.camera_id, ps)

            # 标记停止（这里不涉及实际任务取消，仅做会话层状态与资源释放）
            ps.status = "stopping"
            ps.stop_reason = reason
            ps.stop_ts = time.time()

            # 释放 camera 占用（允许新的会话建立）
            self._camera_lock.pop(ps.camera_id, None)

            # 最终标记为 stopped
            ps.status = "stopped"
            return (True, ps.camera_id, ps)

    def release_camera(self, camera_id: str) -> None:
        with self._lock:
            self._camera_lock.pop(camera_id, None)

    # ---- helpers ----
    def _make_video_id(self, camera_id: str, path: str) -> str:
        base = os.path.basename(path.rstrip("/\\")) if "://" not in path else path
        return f"{camera_id}:{base}"

    def _probe_meta(self, path: str) -> SourceMeta:
        fps = width = height = duration = None
        has_true_pts = False
        try:
            import cv2
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
            try:
                _fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                _w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                _h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                _n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(_fps) if _fps and _fps < 1000 else None
                width  = _w or None
                height = _h or None
                if fps and _n and fps > 0:
                    duration = float(_n) / fps
                has_true_pts = bool(fps and fps > 0)
            finally:
                cap.release()
        except Exception:
            pass
        return SourceMeta(fps=fps, width=width, height=height, duration_sec=duration, has_true_pts=has_true_pts)

# 单例
SESSION_MANAGER = SessionManager()
