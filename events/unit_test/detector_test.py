# detector_test.py
import asyncio
import time

from events.Accident_detect.accident_detector import run_accident_detector
from events.bus import AsyncBus, Detection

# ==== 可调参数（集中管理）====
VIDEO_PATH1 = r"E:\Training\Recording 2025-11-02 152123.mp4"
VIDEO_PATH2 = r"E:\Training\Recording 2025-10-30 172929.mp4"
VIDEO_PATH3 = r"E:\Training\Recording 2025-11-02 151143.mp4"
VIDEO_PATH4 = r"E:\Training\Recording 2025-11-02 172630.mp4"
CAMERA_ID = "cam-1"
TARGET_FPS = 60

# 模型判定阈值（仅影响 run_accident_detector 内部 det.happened 判定）
DECISION_THRESH = 0.65

# 设备：0 为首块 GPU；
DEVICE = 0

# 事件聚合参数（严格要求 3 连击 happened 才开案）
AGG_ALPHA = 0.25
AGG_ENTER_THR = 0.65       # 备用 EMA 通道的进入阈值（默认关闭备用通道）
AGG_EXIT_THR = 0.40        # 备用 EMA 通道退出阈值
AGG_MIN_PERSIST_FRAMES = 3  # 备用 EMA 通道所需正向帧（默认关闭，不生效）
AGG_MIN_END_FRAMES = 8      # 结束判定需要的连续阴性帧
AGG_OCCLUSION_GRACE = 3.0   # 遮挡宽限（秒）
AGG_MERGE_GAP = 5.0         # 合并窗口（秒）
AGG_REQUIRED_HAP = 3        # ✅ 连续 N 帧 happened=True 才开案
AGG_USE_EMA_OPEN = False     # ✅ 仅靠 happened 连击开案（更稳）
# ============================

# 正确导入你保存的聚合器文件（路径一定要和你的文件一致）
from events.Accident_detect.incident_aggregator import AccidentAggregator


# ------------------ 事件聚合消费者：detections -> events ------------------
async def run_event_aggregator(bus: AsyncBus, camera_id: str):
    """
    订阅 'detections'，将逐帧 Detection 聚合成事故事件：
    - 开案：连续 AGG_REQUIRED_HAP 帧 det.happened=True（严格）
      * open 的 ts 会回溯到这段连续的第1帧时间
    - 关案：EMA <= exit_thr 且连续 AGG_MIN_END_FRAMES 阴性
    - 合并窗口：结束后 AGG_MERGE_GAP 秒内再触发并入同一条
    """
    q = bus.subscribe("detections")
    agg = AccidentAggregator(
        camera_id=camera_id,
        alpha=AGG_ALPHA,
        enter_thr=AGG_ENTER_THR,
        exit_thr=AGG_EXIT_THR,
        min_persistence_frames=AGG_MIN_PERSIST_FRAMES,
        min_end_frames=AGG_MIN_END_FRAMES,
        occlusion_grace_sec=AGG_OCCLUSION_GRACE,
        merge_gap_sec=AGG_MERGE_GAP,
        required_happened_consecutive=AGG_REQUIRED_HAP,
        use_ema_open=AGG_USE_EMA_OPEN,
    )

    try:
        while True:
            det: Detection = await q.get()
            # 逐帧聚合（务必传 happened）
            open_ev, close_evs = agg.update(
                ts=det.ts_unix,
                conf=det.confidence,
                frame_ok=True,
                happened=det.happened,  # 关键：严格 3 连击依赖它
            )

            if open_ev is not None:
                await bus.publish("events", open_ev)
            for ev in close_evs:
                await bus.publish("events", ev)

            await asyncio.sleep(0)
    finally:
        # 视频结束时做一次收尾，输出未闭合事件
        for ev in agg.flush():
            await bus.publish("events", ev)


# ------------------ 逐帧打印（保留你的原版） ------------------
async def run_print_detections(bus: AsyncBus):
    q = bus.subscribe("detections")
    counter = 0
    while True:
        det: Detection = await q.get()
        counter += 1
        if counter % 5 == 0:
            print(f"[检测日志] 已收到 {counter} 次检测结果")
        if det.type == "accident" and det.happened:
            print(f"[!!!] 检测到疑似事件 | 摄像头={det.camera_id} | 置信度={det.confidence:.3f} | 时间戳={det.ts_unix:.3f}")
        else:
            print(f"🔹 正常帧 | conf={det.confidence:.3f}")
        await asyncio.sleep(0)


# ------------------ 事件打印（open/close） ------------------
async def run_print_events(bus: AsyncBus):
    q = bus.subscribe("events")
    while True:
        ev = await q.get()
        if ev["type"] == "accident_open":
            print(f"🚨 事故开始 | cam={ev['camera_id']} | id={ev['incident_id']} | ts={ev['ts_unix']:.3f} | conf≈{ev.get('confidence',0):.3f}")
        elif ev["type"] == "accident_close":
            print(f"✅ 事故结束 | cam={ev['camera_id']} | id={ev['incident_id']} | 持续={ev.get('duration_sec',0):.2f}s | 峰值={ev.get('peak_confidence',0):.3f} | 阳性帧={ev.get('pos_frames',0)}")
        await asyncio.sleep(0)


# ------------------ 帧源（加了防忙等） ------------------
async def run_frame_source_debug(bus: AsyncBus, camera_id: str, url_or_path: str, target_fps: float = 60.0):
    import cv2, os
    print(f"🎥 打开视频源: {url_or_path}")
    cap = cv2.VideoCapture(url_or_path)
    if not cap.isOpened():
        print("❌ 无法打开视频源！")
        return

    interval = 1.0 / max(1e-3, target_fps)
    last_emit = 0.0
    is_file = os.path.exists(url_or_path)
    frame_count = 0
    start = time.time()

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                if is_file:
                    print("🔚 视频读取完毕。")
                    break
                await asyncio.sleep(0.02)
                continue

            now = time.time()
            if now - last_emit < interval:
                # 防忙等：给其他协程让出调度
                await asyncio.sleep(0)
                continue
            last_emit = now

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"[取帧日志] 已读取 {frame_count} 帧")

            import numpy as np, cv2
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            from events.bus import Frame
            frame = Frame(camera_id=camera_id, ts_unix=now, rgb=rgb)
            await bus.publish("frames", frame)
            await asyncio.sleep(0)

    finally:
        cap.release()
        dur = time.time() - start
        print(f"✅ 视频结束，共读取 {frame_count} 帧，用时 {dur:.1f} 秒")


# ------------------ 主函数 ------------------
async def main():
    print("🚀 启动 TrafficAI 检测调试")
    bus = AsyncBus()

    tasks = [
        asyncio.create_task(run_frame_source_debug(bus, CAMERA_ID, VIDEO_PATH1, target_fps=TARGET_FPS)),
        asyncio.create_task(run_accident_detector(
            bus,
            decision_thresh=DECISION_THRESH,
            device=DEVICE,
        )),
        asyncio.create_task(run_event_aggregator(bus, CAMERA_ID)),  # ← 聚合层
        asyncio.create_task(run_print_events(bus)),                 # ← 先打印事件
        asyncio.create_task(run_print_detections(bus)),             # ← 再打印逐帧
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n手动中止。")
