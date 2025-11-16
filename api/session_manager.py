# -----------------------------------------------------------------------------
# Copyright (c) 2025
#
# Authors:
#   Liruo Wang
#       School of Electrical Engineering and Computer Science,
#       University of Ottawa
#       lwang032@uottawa.ca
#
# All rights reserved.
# -----------------------------------------------------------------------------

"""
SessionManager: unified scheduler for camera tasks, detectors, and SSE broadcasting.
Handles background database insertion (randomized timestamps within the past week, evidence_text),
and generates embeddings in real time (bge-base-en-v1.5 / 768 dimensions).
"""

from __future__ import annotations

import asyncio, time, json, random
from queue import Queue, Empty
from typing import List, Dict
from datetime import timedelta

from django.http import StreamingHttpResponse, HttpResponse
from django.utils import timezone
from asgiref.sync import sync_to_async

import api.runtime_state as rt
from events.camera_pipeline import SingleFileSession
from events.Accident_detect.accident_detector import run_accident_detector_multi
from events.Weather_detect.weather_detector import run_weather_detector_multi  # ★ 新增天气多路检测器
from events.models import Event, Camera

# ---------- 嵌入模型（768 维） ----------
from sentence_transformers import SentenceTransformer
import numpy as np

_EMBED_MODEL: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        # Downloads weights on first use; stays resident in memory
        _EMBED_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _EMBED_MODEL


def _embed_text(text: str) -> list[float]:
    emb = _get_embedder().encode(text, normalize_embeddings=True)  # np.ndarray (768,)
    return emb.astype(np.float32).tolist()


#tools
def _random_recent_ts():
    now = timezone.now()
    return now - timedelta(seconds=random.randint(0, 7 * 24 * 3600))


def _make_evidence_text(evt: dict) -> str:
    cam = evt.get("camera_id", "unknown")
    conf = float(evt.get("peak_confidence", evt.get("confidence", 0.0)))
    dur = evt.get("duration_sec")
    return (
        f"Accident (closed) on {cam}, peak confidence {conf:.2f}, duration {float(dur):.1f}s."
        if dur is not None
        else f"Accident (open) on {cam}, confidence {conf:.2f}."
    )


# Background queue for database insertion
SAVE_QUEUE: asyncio.Queue = asyncio.Queue(maxsize=512)


@sync_to_async
def _save_event_to_db(evt: dict):
    """Perform a single event insertion: ensure camera exists, generate evidence_text and embedding."""

    cam_id = evt.get("camera_id")
    if not cam_id:
        return
    Camera.objects.get_or_create(camera_id=cam_id)
    ts = _random_recent_ts()
    conf = float(evt.get("peak_confidence", evt.get("confidence", 0.0)))
    text = _make_evidence_text(evt)
    weather = evt.get("weather")
    try:
        vec = _embed_text(text)
    except Exception as e:
        print(f"[EMB] embed failed: {e}")
        vec = None

    # Write to the events table (db_table='events')
    Event.objects.create(
        timestamp=ts,
        camera_id=cam_id,  # Foreign key column 'camera_id' (db_column specified in the model)
        type="accident",
        weather=weather,  # Weather not yet logged: default to 'clear' (can be replaced with event weather later)
        confidence=conf,
        evidence_text=text,
        embedding=vec,  # Write embedding in real time
    )
    print(f"[DB] saved accident event for {cam_id} ({conf:.2f})")


async def _save_worker():
    """Background consumer: fetch events from SAVE_QUEUE and insert them into the database without blocking SSE."""
    while True:
        evt = await SAVE_QUEUE.get()
        try:
            await _save_event_to_db(evt)
        except Exception as e:
            print(f"[save_worker] DB error: {e}")
        finally:
            SAVE_QUEUE.task_done()


# Core Management Class
class SessionManager:
    GLOBAL_SSE_ID = "sse-main"

    @staticmethod
    def register(camera_ids: List[str]) -> Dict:
        results = []
        for cid in camera_ids:
            try:
                from api.camera_map import get_source
                src = get_source(cid)
                rt.INTENDED[cid] = src
                results.append({"camera_id": cid, "src": src, "ok": True})
            except Exception as e:
                results.append({"camera_id": cid, "ok": False, "error": str(e)})

        rt.SSE_PENDING[SessionManager.GLOBAL_SSE_ID] = list(rt.INTENDED.keys())
        print(f"[manager] registered cams={camera_ids}")
        return {
            "session_id": SessionManager.GLOBAL_SSE_ID,
            "results": results,
            "sse_alerts_url": f"/sse/alerts?sse_id={SessionManager.GLOBAL_SSE_ID}",
            "ts": int(time.time()),
        }

    @staticmethod
    def start_all(loop) -> None:
        camera_ids = list(rt.INTENDED.keys())
        if not camera_ids:
            print("[manager] no camera registered.")
            return

        # Start each SingleFileSession (frame source + sampler + aggregator)
        for cid in camera_ids:
            if cid not in rt.SESSIONS:
                sess = SingleFileSession(
                    camera_id=cid,
                    file_path=rt.INTENDED[cid],
                    bus=rt.BUS,
                    session_id=SessionManager.GLOBAL_SSE_ID,
                )
                sess.start(loop=loop)
                rt.SESSIONS[cid] = sess

        active_cams = list(rt.SESSIONS.keys())
        if not active_cams:
            print("[manager] no active cameras.")
            return

        # Start the multi-stream accident detector (YOLO)
        if rt.DETECTOR_TASK is None or rt.DETECTOR_TASK.done():
            async def _run_multi():
                print(f"[manager] accident detector started for {active_cams}")
                await run_accident_detector_multi(
                    rt.BUS,
                    camera_ids=active_cams,
                    batch_size=4,
                    poll_ms=50,
                )

            rt.DETECTOR_TASK = asyncio.run_coroutine_threadsafe(_run_multi(), loop)

        # Start the multi-stream weather detector (CNN)
        # Requires WEATHER_TASK = None in api.runtime_state
        if not hasattr(rt, "WEATHER_TASK"):
            rt.WEATHER_TASK = None

        if rt.WEATHER_TASK is None or rt.WEATHER_TASK.done():
            async def _run_weather():
                print(f"[manager] weather detector started for {active_cams}")
                await run_weather_detector_multi(
                    rt.BUS,
                    camera_ids=active_cams,
                    batch_size=4,
                    poll_ms=100,
                    interval_sec=300,  # detect every 5mins
                )

            rt.WEATHER_TASK = asyncio.run_coroutine_threadsafe(_run_weather(), loop)

    @staticmethod
    def stop_all(loop) -> List[str]:
        stopped = []
        for cid, sess in list(rt.SESSIONS.items()):
            sess.stop(loop=loop)
            stopped.append(cid)
            rt.SESSIONS.pop(cid, None)
            rt.INTENDED.pop(cid, None)

        # stop accident-detector
        if rt.DETECTOR_TASK:
            rt.DETECTOR_TASK.cancel()
            rt.DETECTOR_TASK = None

        # stop climate-detector
        if hasattr(rt, "WEATHER_TASK") and rt.WEATHER_TASK:
            rt.WEATHER_TASK.cancel()
            rt.WEATHER_TASK = None

        print(f"[manager] stopped all cameras")
        return stopped

    @staticmethod
    def stream(loop):
        """SSE broadcasting + database insertion (background queue)"""
        camera_ids = list(rt.INTENDED.keys())
        if not camera_ids:
            return HttpResponse("未注册任何相机源", status=404)

        # Start the background database-insertion worker (run once)
        if not hasattr(rt, "SAVE_WORKER") or rt.SAVE_WORKER is None or rt.SAVE_WORKER.done():
            rt.SAVE_WORKER = asyncio.run_coroutine_threadsafe(_save_worker(), loop)

        q: "Queue[bytes]" = Queue(maxsize=256)

        async def _pipe():
            # Subscribe to accident + weather events
            topics = (
                    [f"accidents.open:{cid}" for cid in camera_ids]
                    + [f"accidents.close:{cid}" for cid in camera_ids]
                    + [f"weather:{cid}" for cid in camera_ids]
            )

            async with rt.BUS.subscribe_many(topics, mode="fifo", maxsize=64) as subq:
                while True:
                    evt = await subq.get()
                    cid = evt.get("camera_id", "unknown")

                    # (1) Weather events → update cache
                    if evt.get("type") == "weather":
                        rt.LAST_WEATHER[cid] = evt.get("weather", "clear")

                    # (2) Accident events → automatically attach weather field before inserting into the database
                    if evt.get("type") == "accident_open":
                        evt["weather"] = rt.LAST_WEATHER.get(cid, "clear")

                        try:
                            SAVE_QUEUE.put_nowait(evt)
                        except asyncio.QueueFull:
                            print("[warn] SAVE_QUEUE full, dropping event")

                    # (3) Unified SSE broadcasting
                    payload = (
                        f"id: {SessionManager.GLOBAL_SSE_ID}-{int(time.time() * 1000)}\n"
                        f"event: {cid}\n"
                        f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
                    )
                    try:
                        q.put_nowait(payload.encode())
                    except Exception:
                        try:
                            _ = q.get_nowait()
                        except Exception:
                            pass
                        q.put_nowait(payload.encode())

        asyncio.run_coroutine_threadsafe(_pipe(), loop)

        def _stream():
            # Initial connection message
            yield f": connected sse_id={SessionManager.GLOBAL_SSE_ID}\n\n".encode()
            last = time.time()
            try:
                while True:
                    try:
                        chunk = q.get(timeout=1.0)
                        yield chunk
                    except Empty:
                        if time.time() - last >= 10:
                            # heartbeat
                            yield b": ping\n\n"
                            last = time.time()
            finally:
                print("[manager] SSE closed")

        resp = StreamingHttpResponse(_stream(), content_type="text/event-stream")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        return resp
