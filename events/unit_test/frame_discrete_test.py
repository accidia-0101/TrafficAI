import asyncio
import time
import os
import cv2
import numpy as np

# -------- 你的原函数 --------
async def run_frame_source(bus, camera_id: str, url_or_path: str, target_fps: float = 60):
    cap = cv2.VideoCapture(url_or_path)  # read from file or web-camera
    try:
        interval = 1.0 / max(1e-3, target_fps)
        last = 0.0
        is_file = os.path.exists(url_or_path)
        while True:
            ok, bgr = cap.read()
            if not ok:
                if is_file:
                    break  # 文件播放完毕退出
                await asyncio.sleep(0.02)
                continue  # 摄像头暂时没帧则重试

            now = time.time()
            if now - last < interval:
                continue  # 控制帧率上限

            last = now
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame = Frame(camera_id=camera_id, ts_unix=now, rgb=rgb)
            await bus.publish("frames", frame)
            await asyncio.sleep(0)
    finally:
        cap.release()


# -------- Frame 类 --------
class Frame:
    def __init__(self, camera_id, ts_unix, rgb):
        self.camera_id = camera_id
        self.ts_unix = ts_unix
        self.rgb = rgb


# -------- 假总线（实时显示帧） --------
class DisplayBus:
    def __init__(self):
        self.frames = []
        self.start_time = time.time()
        self.last_show = self.start_time

    async def publish(self, topic, frame):
        """接收帧并直接显示"""
        self.frames.append(frame)

        # 计算实时 FPS
        now = time.time()
        elapsed = now - self.start_time
        fps = len(self.frames) / elapsed if elapsed > 0 else 0.0

        # 还原 BGR 显示
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = bgr.shape
        text = f"Frame {len(self.frames)} | {fps:.1f} FPS"
        cv2.putText(bgr, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Video Output", bgr)

        # 这里设 waitKey(1) 就能流畅播放
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):  # ESC 或 Q 键退出
            raise StopAsyncIteration


# -------- 实际测试函数 --------
async def main():
    video_path = r"E:\Training\Recording 2025-10-30 172929.mp4"
    if not os.path.exists(video_path):
        print(f"❌ 文件不存在: {video_path}")
        return

    bus = DisplayBus()
    start = time.time()

    print(f"▶ 开始读取视频: {video_path}")
    try:
        await run_frame_source(bus, "CAM_TEST", video_path, target_fps=60)
    except StopAsyncIteration:
        print("⏹ 手动退出播放")
    finally:
        end = time.time()
        frames = bus.frames
        cv2.destroyAllWindows()

        if len(frames) > 1:
            intervals = np.diff([f.ts_unix for f in frames])
            mean_interval = np.mean(intervals)
            real_fps = 1 / mean_interval
            print(f"✅ 完成！输出 {len(frames)} 帧，用时 {end - start:.2f}s，平均 FPS ≈ {real_fps:.2f}")
        else:
            print(f"⚠️ 仅输出 {len(frames)} 帧，无法计算 FPS")


if __name__ == "__main__":
    asyncio.run(main())
