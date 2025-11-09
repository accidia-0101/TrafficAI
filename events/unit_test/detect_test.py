from __future__ import annotations
import asyncio, os, sys, time
from events.bus import AsyncBus
from events.frame_discrete import run_frame_source_raw, run_sampler_equal_time
from events.Accident_detect.accident_detector import run_accident_detector
from events.Accident_detect.incident_aggregator import AccidentAggregator


CAMERA_VIDEO = {
    "cam-1": r"E:\Training\Recording 2025-10-30 172929.mp4",   # ← 修改为你的测试视频路径
}

# Windows: 修复 asyncio Proactor 写入 bug
if os.name == "nt":
    try:
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    except Exception:
        pass


# ---------- 消费检测结果并聚合 ----------
async def consume_detections(bus: AsyncBus, camera_id: str, agg: AccidentAggregator):
    """消费 detections → 调用聚合器 → 打印检测与聚合日志"""
    async with bus.subscribe("detections") as sub:
        frame_count = 0
        last_print = 0.0
        while True:
            det = await sub.get()
            if getattr(det, "camera_id", None) != camera_id:
                continue

            frame_count += 1
            ts = getattr(det, "pts_in_video", 0.0)
            conf = getattr(det, "confidence", 0.0)
            happened = getattr(det, "happened", False)

            # 每隔一定时间打印一次检测日志
            if time.time() - last_print > 0.5:
                state = "⚠️事故" if happened else "✅正常"
                print(f"[DET] ts={ts:7.3f}s | conf={conf:5.3f} | {state}")
                last_print = time.time()

            # 聚合器处理
            open_event, close_events = agg.push_detection(det)

            # 开案
            if open_event:
                print(f"\n=== 🚨 [OPEN] 事故开始 ===")
                print(f"ID={open_event['incident_id']} cam={open_event['camera_id']}")
                print(f"  start_ts={open_event['ts']:.3f}s frame={open_event['start_frame_idx']} "
                      f"conf≈{open_event['confidence']:.3f}\n")

            # 关案
            for ev in close_events:
                print(f"=== ✅ [CLOSE] 事故结束 ===")
                print(f"ID={ev['incident_id']} cam={ev['camera_id']}")
                print(f"  {ev['start_ts']:.3f}s → {ev['end_ts']:.3f}s "
                      f"dur={ev['duration_sec']:.3f}s "
                      f"peak={ev['peak_confidence']:.3f} pos_frames={ev['pos_frames']}\n")


# ---------- 主流程 ----------
async def main():
    camera_id = "cam-1"
    if len(sys.argv) >= 2:
        camera_id = sys.argv[1]

    video_path = CAMERA_VIDEO.get(camera_id)
    if len(sys.argv) >= 3:
        video_path = sys.argv[2]

    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到视频：{video_path!r}，请修改 CAMERA_VIDEO 或命令行参数。")

    print(f"🎬 启动本地检测 | camera_id={camera_id} | file={video_path}")
    bus = AsyncBus()
    agg = AccidentAggregator(camera_id)

    # 任务链：源 → 采样 → 检测 → 聚合
    producer_task = asyncio.create_task(
        run_frame_source_raw(bus, camera_id=camera_id, url_or_path=video_path)
    )
    sampler_task = asyncio.create_task(
        run_sampler_equal_time(bus, camera_id=camera_id, target_fps=60.0)
    )
    detector_task = asyncio.create_task(
        run_accident_detector(bus, camera_id=camera_id)
    )
    consumer_task = asyncio.create_task(
        consume_detections(bus, camera_id, agg)
    )

    started_at = time.time()
    try:
        await producer_task  # 文件播放完毕
        await asyncio.sleep(0.5)  # 等尾帧
    finally:
        # 视频结束后强制关案
        tail = agg.flush()
        for ev in tail:
            print(f"=== ✅ [CLOSE*] 文件结束强制结案 ===")
            print(f"ID={ev['incident_id']} cam={ev['camera_id']}")
            print(f"  {ev['start_ts']:.3f}s → {ev['end_ts']:.3f}s "
                  f"dur={ev['duration_sec']:.3f}s peak={ev['peak_confidence']:.3f}\n")

        for t in (sampler_task, detector_task, consumer_task):
            t.cancel()
        for t in (sampler_task, detector_task, consumer_task):
            try:
                await t
            except asyncio.CancelledError:
                pass

        print(f"✅ 检测完成，用时 {time.time() - started_at:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
