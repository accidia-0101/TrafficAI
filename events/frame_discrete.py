#
# from events.bus import Frame, AsyncBus, topic_for
# import cv2, time, asyncio, os
#
# async def run_frame_source_raw(bus: AsyncBus, camera_id: str, url_or_path: str):
#     cap = cv2.VideoCapture(url_or_path, cv2.CAP_FFMPEG)
#     try:
#         is_file = os.path.exists(url_or_path)
#         src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
#         src_fps = src_fps if src_fps and src_fps < 1000 else 0.0
#         start_mono = time.monotonic()
#         frame_idx = 0
#
#         while True:
#             ok, bgr = cap.read()
#             if not ok:
#                 if is_file:
#                     await asyncio.sleep(0.2); continue
#                 await asyncio.sleep(0.01); continue
#             rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#
#             pts = (frame_idx / src_fps) if src_fps > 0 else (time.monotonic() - start_mono)
#             f = Frame(camera_id=camera_id, ts_unix=time.time(), rgb=rgb, frame_idx=frame_idx, pts_in_video=pts)
#
#             # ✅ 分区发布：frames_raw:<camera_id>
#             await bus.publish_partitioned("frames_raw", camera_id, f)
#             frame_idx += 1
#             await asyncio.sleep(0)
#     finally:
#         cap.release()
#         try: cv2.destroyAllWindows()
#         except Exception: pass
#
#
# async def run_sampler_equal_time(bus: AsyncBus, camera_id: str, target_fps: float = 60.0, jitter_epsilon: float = 1e-4):
#     step = 1.0 / max(1e-3, target_fps)
#     next_t = None
#
#     # ✅ 订阅分区源：frames_raw:<camera_id>
#     topic_in = topic_for("frames_raw", camera_id)
#     async with bus.subscribe(topic_in, mode="fifo", maxsize=64) as sub:
#         while True:
#             f: Frame = await sub.get()
#
#             if next_t is None:
#                 next_t = f.pts_in_video
#
#             emitted = False
#             while f.pts_in_video + jitter_epsilon >= next_t:
#                 # ✅ 分区发布：frames:<camera_id>
#                 await bus.publish_partitioned("frames", camera_id, f)
#                 next_t += step
#                 emitted = True
#             if not emitted:
#                 await asyncio.sleep(0)
from events.bus import Frame, AsyncBus, topic_for
import cv2, time, asyncio, os


async def run_frame_source_raw(bus: AsyncBus, camera_id: str, url_or_path: str):
    """
    不降帧视频源：逐帧解码并发布到 frames_raw:<camera_id>
    - 每帧包含 frame_idx 和 pts_in_video（视频时间秒）
    """
    cap = cv2.VideoCapture(url_or_path, cv2.CAP_FFMPEG)
    try:
        is_file = os.path.exists(url_or_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        src_fps = src_fps if src_fps and src_fps < 1000 else 0.0
        start_mono = time.monotonic()
        frame_idx = 0

        while True:
            ok, bgr = cap.read()
            if not ok:
                # 文件播完则退出，直播源则短暂等待
                if is_file:
                    break
                await asyncio.sleep(0.01)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pts = frame_idx / src_fps if src_fps > 0 else (time.monotonic() - start_mono)

            f = Frame(
                camera_id=camera_id,
                ts_unix=time.time(),
                rgb=rgb,
                frame_idx=frame_idx,
                pts_in_video=pts,
            )

            # ✅ 分区发布：frames_raw:<camera_id>
            await bus.publish(topic_for("frames_raw", camera_id), f)
            frame_idx += 1

            # 防止阻塞事件循环
            await asyncio.sleep(0)

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


async def run_sampler_equal_time(bus: AsyncBus, camera_id: str, target_fps: float = 60.0, jitter_epsilon: float = 1e-4):
    """
    等时采样：从 frames_raw:<camera_id> 取帧，按目标 FPS 均匀采样，
    发布到 frames:<camera_id>。
    """
    step = 1.0 / max(1e-3, target_fps)
    next_t = None

    topic_in = topic_for("frames_raw", camera_id)
    topic_out = topic_for("frames", camera_id)

    async with bus.subscribe(topic_in, mode="fifo", maxsize=64) as sub:
        while True:
            f: Frame = await sub.get()

            # 初始化采样时钟
            if next_t is None:
                next_t = f.pts_in_video

            emitted = False
            while f.pts_in_video + jitter_epsilon >= next_t:
                newf = Frame(
                    camera_id=f.camera_id,
                    ts_unix=f.ts_unix,
                    rgb=f.rgb,
                    frame_idx=f.frame_idx,
                    pts_in_video=f.pts_in_video,
                )
                await bus.publish(topic_for("frames", camera_id), newf)
                next_t += step

            if not emitted:
                await asyncio.sleep(0)
