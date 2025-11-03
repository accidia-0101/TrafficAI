# frame_discrete.py
"""
视频切帧

功能：
- 发布 'frames'（Frame.rgb: HxWx3, uint8, RGB）
- 对视频切帧，降低输出的帧率
"""
import cv2, time, asyncio, os
from events.bus import Frame, AsyncBus

async def run_frame_source(bus: AsyncBus, camera_id: str, url_or_path: str, target_fps: float = 60.0):
    cap = cv2.VideoCapture(url_or_path)
    try:
        interval = 1.0 / max(1e-3, target_fps)
        last_emit = 0.0
        is_file = os.path.exists(url_or_path)
        while True:
            ok, bgr = cap.read()
            if not ok:
                if is_file:
                    break  # 文件结束
                await asyncio.sleep(0.02)  # RTSP/摄像头临时断帧，稍后重试
                continue

            now = time.time()
            if now - last_emit < interval:
                await asyncio.sleep(0)
                # 主动丢帧控速，避免后端积压
                continue
            last_emit = now
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame = Frame(camera_id=camera_id, ts_unix=now, rgb=rgb)
            await bus.publish("frames", frame)
            await asyncio.sleep(0)  # 让出事件循环
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass  # 无 GUI 环境时忽略
