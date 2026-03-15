import os
import csv
import glob
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple

from events.bus import AsyncBus, Detection, topic_for
from events.frame_discrete import run_frame_source_raw, run_sampler_equal_time_vts
from events.Accident_detect.accident_detector import run_accident_detector_multi
from events.Accident_detect.incident_aggregator import AccidentAggregator


# ---------------------------------------------------------------------
# Source config
# ---------------------------------------------------------------------
BASE_DIR = r"E:\Training\traffic_video"


def find_video_for_cam(cam_id: int) -> str:
    pattern = fr"{BASE_DIR}\*-cam-{cam_id}.mp4"
    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(f"No video file found for cam-{cam_id}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple video files match cam-{cam_id}: {matches}")
    return matches[0]


CAMERA_SOURCES: Dict[str, Dict[str, object]] = {}
for i in range(1, 41):
    cam = f"cam-{i}"
    try:
        src = find_video_for_cam(i)
        CAMERA_SOURCES[cam] = {
            "src": src,
            "enabled": True,
        }
    except FileNotFoundError:
        CAMERA_SOURCES[cam] = {
            "src": "",
            "enabled": False,
        }


def get_source(camera_id: str) -> str:
    meta = CAMERA_SOURCES.get(camera_id)
    if not meta or not meta.get("enabled", True):
        raise KeyError(f"camera_id is not configured or is disabled: {camera_id}")
    src = (meta.get("src") or "").strip()
    if not src:
        raise ValueError(f"camera_id does not provide a valid src: {camera_id}")
    return src


def get_enabled_cameras() -> List[str]:
    cams = []
    for cam, meta in CAMERA_SOURCES.items():
        if meta.get("enabled", True) and str(meta.get("src", "")).strip():
            cams.append(cam)
    return cams


# ---------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------
# None = use all enabled cameras found in BASE_DIR
TEST_CAMERAS = ["cam-1"]
# 例如：
# TEST_CAMERAS = ["cam-1", "cam-2"]
# TEST_CAMERAS = None

TARGET_FPS = 15.0
DETECTOR_BATCH_SIZE = 4
DETECTOR_POLL_MS = 20

OUTPUT_DIR = Path("./full_pipeline_test_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRINT_EVERY_DET_FRAME = False
PRINT_EVENT_PAYLOAD = False

# 如果设为 None，则等待视频自然跑完
MAX_RUN_SECONDS = None

# source 跑完后给下游一点时间清尾
TAIL_DRAIN_SECONDS = 2.0
# ---------------------------------------------------------------------


async def log_accident_stream(
    bus: AsyncBus,
    camera_id: str,
    csv_path: Path,
):
    """
    订阅 accident:<cam>，保存每个 sampled frame 的 detector 输出。
    """
    topic = topic_for("accident", camera_id)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "camera_id",
            "frame_idx",
            "pts_in_video",
            "vts",
            "ts_unix",
            "confidence",
            "happened",
        ])

        async with bus.subscribe(topic, mode="fifo", maxsize=256) as q:
            while True:
                det: Detection = await q.get()

                writer.writerow([
                    det.camera_id,
                    det.frame_idx,
                    f"{det.pts_in_video:.6f}",
                    f"{det.vts:.6f}",
                    f"{det.ts_unix:.6f}",
                    f"{det.confidence:.6f}",
                    int(det.happened),
                ])
                f.flush()

                if PRINT_EVERY_DET_FRAME:
                    print(
                        f"[DET {det.camera_id}] "
                        f"frame={det.frame_idx:04d} "
                        f"pts={det.pts_in_video:.2f} "
                        f"vts={det.vts:.2f} "
                        f"conf={det.confidence:.3f} "
                        f"happened={det.happened}"
                    )
async def log_events_to_csv(bus: AsyncBus, camera_id: str, csv_path: Path):
    """
    订阅 accidents.open:<cam> / accidents.close:<cam>，保存事件到 CSV。
    """
    topic_open = topic_for("accidents.open", camera_id)
    topic_close = topic_for("accidents.close", camera_id)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_type",
            "camera_id",
            "incident_id",
            "session_id",
            "frame_idx",
            "pts_in_video",
            "confidence",
            "start_ts",
            "end_ts",
            "duration_sec",
            "peak_confidence",
            "pos_frames",
            "raw_payload",
        ])

        async def _listen(topic: str):
            async with bus.subscribe(topic, mode="fifo", maxsize=256) as q:
                while True:
                    ev = await q.get()

                    writer.writerow([
                        ev.get("type", ""),
                        ev.get("camera_id", ""),
                        ev.get("incident_id", ""),
                        ev.get("session_id", ""),
                        ev.get("frame_idx", ""),
                        ev.get("pts_in_video", ""),
                        ev.get("confidence", ""),
                        ev.get("start_ts", ""),
                        ev.get("end_ts", ""),
                        ev.get("duration_sec", ""),
                        ev.get("peak_confidence", ""),
                        ev.get("pos_frames", ""),
                        str(ev),
                    ])
                    f.flush()

        await asyncio.gather(_listen(topic_open), _listen(topic_close))

async def log_events(bus: AsyncBus, camera_id: str):
    """
    订阅 accidents.open:<cam> / accidents.close:<cam>，打印聚合事件。
    """
    topic_open = topic_for("accidents.open", camera_id)
    topic_close = topic_for("accidents.close", camera_id)

    async def _listen(topic: str):
        async with bus.subscribe(topic, mode="fifo", maxsize=256) as q:
            while True:
                ev = await q.get()
                if PRINT_EVENT_PAYLOAD:
                    print(f"[EVENT {camera_id}] {topic.split(':')[0]} -> {ev}")
                else:
                    print(f"[EVENT {camera_id}] {topic.split(':')[0]}")

    await asyncio.gather(_listen(topic_open), _listen(topic_close))


async def main():
    bus = AsyncBus()

    if TEST_CAMERAS is None:
        camera_ids = get_enabled_cameras()
    else:
        camera_ids = TEST_CAMERAS

    if not camera_ids:
        raise RuntimeError("No enabled cameras available for test.")

    print("=" * 80)
    print("Full pipeline test starting...")
    print("Selected cameras and sources:")
    for cam in camera_ids:
        print(f"  {cam} -> {get_source(cam)}")
    print("-" * 80)
    print(f"TARGET_FPS          = {TARGET_FPS}")
    print(f"DETECTOR_BATCH_SIZE = {DETECTOR_BATCH_SIZE}")
    print(f"DETECTOR_POLL_MS    = {DETECTOR_POLL_MS}")
    print(f"OUTPUT_DIR          = {OUTPUT_DIR.resolve()}")
    print("=" * 80)

    # -----------------------------------------------------------------
    # 1) Start source tasks
    # -----------------------------------------------------------------
    source_tasks: List[asyncio.Task] = []
    for cam in camera_ids:
        video_path = get_source(cam)
        source_tasks.append(
            asyncio.create_task(
                run_frame_source_raw(bus, cam, video_path)
            )
        )

    # -----------------------------------------------------------------
    # 2) Start sampler tasks
    # -----------------------------------------------------------------
    sampler_tasks: List[asyncio.Task] = []
    for cam in camera_ids:
        sampler_tasks.append(
            asyncio.create_task(
                run_sampler_equal_time_vts(
                    bus,
                    cam,
                    target_fps=TARGET_FPS,
                )
            )
        )

    # -----------------------------------------------------------------
    # 3) Start detector task (shared YOLO)
    # -----------------------------------------------------------------
    detector_task = asyncio.create_task(
        run_accident_detector_multi(
            bus,
            camera_ids=camera_ids,
            batch_size=DETECTOR_BATCH_SIZE,
            poll_ms=DETECTOR_POLL_MS,
        )
    )

    # -----------------------------------------------------------------
    # 4) Start aggregator tasks
    # -----------------------------------------------------------------
    aggregators: List[AccidentAggregator] = []
    aggregator_tasks: List[asyncio.Task] = []
    for cam in camera_ids:
        agg = AccidentAggregator(camera_id=cam, bus=bus)
        aggregators.append(agg)
        aggregator_tasks.append(asyncio.create_task(agg.run()))

    # -----------------------------------------------------------------
    # 5) Start detector result loggers
    # -----------------------------------------------------------------
    det_logger_tasks: List[asyncio.Task] = []
    for cam in camera_ids:
        csv_path = OUTPUT_DIR / f"{cam}_detector_results.csv"
        det_logger_tasks.append(
            asyncio.create_task(log_accident_stream(bus, cam, csv_path))
        )

    # 6) Start event CSV loggers
    event_logger_tasks: List[asyncio.Task] = []
    for cam in camera_ids:
        event_csv_path = OUTPUT_DIR / f"{cam}_event_results.csv"
        event_logger_tasks.append(
            asyncio.create_task(log_events_to_csv(bus, cam, event_csv_path))
        )

    # -----------------------------------------------------------------
    # 7) Wait for source end or timeout
    # -----------------------------------------------------------------
    try:
        if MAX_RUN_SECONDS is None:
            await asyncio.gather(*source_tasks)
        else:
            await asyncio.sleep(MAX_RUN_SECONDS)

    finally:
        # source 结束后给一点时间让 downstream 清尾
        await asyncio.sleep(TAIL_DRAIN_SECONDS)

        # flush aggregators to avoid losing close events
        for agg in aggregators:
            try:
                await agg.flush()
            except Exception as e:
                print(f"[aggregator flush error] {agg.camera_id}: {e}")

        # cancel background tasks
        cancel_tasks = (
            sampler_tasks
            + [detector_task]
            + aggregator_tasks
            + det_logger_tasks
            + event_logger_tasks
        )

        for t in cancel_tasks:
            t.cancel()

        for t in cancel_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[task error] {e}")

        print("\nFull pipeline test finished.")
        print(f"Detector CSV files saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())