import os
import csv
import glob
import asyncio
from pathlib import Path
from typing import Dict, List

from events.bus import AsyncBus, Detection, topic_for

# ====== 按你项目实际模块路径修改 ======
from events.frame_discrete import run_frame_source_raw, run_sampler_equal_time_vts
from events.Accident_detect.accident_detector import run_accident_detector_multi
# =====================================


# ---------------------------------------------------------------------
# Source config
# ---------------------------------------------------------------------
BASE_DIR = r"E:\Training\traffic_video"


def find_video_for_cam(cam_id: int) -> str:
    """
    cam_id = 1 -> find file containing '*-cam-1.mp4'
    """
    pattern = fr"{BASE_DIR}\*-cam-{cam_id}.mp4"
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(f"No video file found for cam-{cam_id}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple video files match cam-{cam_id}: {matches}")
    return matches[0]


CAMERA_SOURCES: Dict[str, Dict[str, object]] = {}
for i in range(1, 2):
    cam = f"cam-{i}"
    try:
        src = find_video_for_cam(i)
        CAMERA_SOURCES[cam] = {
            "src": src,
            "enabled": True,
        }
    except FileNotFoundError:
        # 没找到视频就默认禁用，不直接报错，方便只测部分 camera
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
# 这里手动指定你本次想测哪些 camera；如果为 None，就自动选全部 enabled camera
TEST_CAMERAS = None
# 例如：
# TEST_CAMERAS = ["cam-1", "cam-2"]

TARGET_FPS = 15.0
DETECTOR_BATCH_SIZE = 4
DETECTOR_POLL_MS = 20

OUTPUT_DIR = Path("./detector_test_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRINT_EVERY_FRAME = True

# 如果设为 None，则等视频 source 自然结束
MAX_RUN_SECONDS = None
# ---------------------------------------------------------------------


async def log_accident_stream(
    bus: AsyncBus,
    camera_id: str,
    csv_path: Path,
):
    """
    Subscribe to accident:<cam> and save every sampled-frame detector output.
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

                if PRINT_EVERY_FRAME:
                    print(
                        f"[RESULT {det.camera_id}] "
                        f"frame={det.frame_idx:04d} "
                        f"pts={det.pts_in_video:.2f} "
                        f"vts={det.vts:.2f} "
                        f"conf={det.confidence:.3f} "
                        f"happened={det.happened}"
                    )


async def main():
    bus = AsyncBus()

    if TEST_CAMERAS is None:
        camera_ids = get_enabled_cameras()
    else:
        camera_ids = TEST_CAMERAS

    if not camera_ids:
        raise RuntimeError("No enabled cameras available for test.")

    print("=" * 70)
    print("Enabled camera sources:")
    for cam in camera_ids:
        print(f"  {cam} -> {get_source(cam)}")
    print("=" * 70)

    # 1) source tasks
    source_tasks: List[asyncio.Task] = []
    for cam in camera_ids:
        video_path = get_source(cam)
        source_tasks.append(
            asyncio.create_task(
                run_frame_source_raw(bus, cam, video_path)
            )
        )

    # 2) sampler tasks
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

    # 3) detector task
    detector_task = asyncio.create_task(
        run_accident_detector_multi(
            bus,
            camera_ids=camera_ids,
            batch_size=DETECTOR_BATCH_SIZE,
            poll_ms=DETECTOR_POLL_MS,
        )
    )

    # 4) logger tasks
    logger_tasks: List[asyncio.Task] = []
    for cam in camera_ids:
        csv_path = OUTPUT_DIR / f"{cam}_detector_results.csv"
        logger_tasks.append(
            asyncio.create_task(
                log_accident_stream(bus, cam, csv_path)
            )
        )

    print("Test started.")
    print(f"camera_ids          = {camera_ids}")
    print(f"target_fps          = {TARGET_FPS}")
    print(f"detector_batch_size = {DETECTOR_BATCH_SIZE}")
    print(f"detector_poll_ms    = {DETECTOR_POLL_MS}")
    print(f"output_dir          = {OUTPUT_DIR.resolve()}")
    print("=" * 70)

    try:
        if MAX_RUN_SECONDS is None:
            # wait until all sources finish naturally
            await asyncio.gather(*source_tasks)
        else:
            await asyncio.sleep(MAX_RUN_SECONDS)

    finally:
        # allow pipeline tail to drain
        await asyncio.sleep(2.0)

        # cancel background tasks
        for t in sampler_tasks:
            t.cancel()
        detector_task.cancel()
        for t in logger_tasks:
            t.cancel()

        # await sampler tasks
        for t in sampler_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[sampler task error] {e}")

        # await detector task
        try:
            await detector_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[detector task error] {e}")

        # await logger tasks
        for t in logger_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[logger task error] {e}")

        print("\nTest finished. CSV files written.")


if __name__ == "__main__":
    asyncio.run(main())