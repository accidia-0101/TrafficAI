# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, asyncio
from queue import Queue, Empty
from django.http import JsonResponse, HttpRequest, StreamingHttpResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

import api.runtime_state as rt
from api.camera_map import get_source
from events.session_single import SingleFileSession
from events.Accident_detect.accident_detector import run_accident_detector_multi


def _json(req: HttpRequest) -> dict:
    try:
        return json.loads(req.body.decode("utf-8")) if req.body else {}
    except Exception:
        return {}


# -------- /api/play --------
@csrf_exempt
@require_POST
def play_view(request: HttpRequest):
    body = _json(request)
    cid = (body.get("camera_id") or "").strip()
    if not cid:
        return JsonResponse({"ok": False, "error": "camera_id 缺失"}, status=400)
    try:
        src = get_source(cid)
        rt.INTENDED[cid] = src
        return JsonResponse({
            "ok": True,
            "camera_id": cid,
            "sse_alerts_url": f"/sse/alerts?camera_id={cid}",
            "ts": int(time.time())
        })
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)


# -------- /sse/alerts --------
@require_GET
def alerts_stream(request: HttpRequest):
    cid = (request.GET.get("camera_id") or "").strip()
    if not cid:
        return HttpResponse("camera_id 缺失", status=400)

    loop = rt.ensure_bg_loop()

    # 启动会话
    if cid not in rt.SESSIONS:
        if cid not in rt.INTENDED:
            return HttpResponse("未登记该 camera_id", status=404)
        sess = SingleFileSession(camera_id=cid, file_path=rt.INTENDED[cid], bus=rt.BUS)
        sess.start(loop=loop)
        rt.SESSIONS[cid] = sess

    # 启动检测器（一次）
    if rt.DETECTOR_TASK is None or getattr(rt.DETECTOR_TASK, "done", lambda: False)():
        async def _run_multi():
            await run_accident_detector_multi(rt.BUS, camera_ids=list(rt.SESSIONS.keys()), batch_size=4, poll_ms=20)
        rt.DETECTOR_TASK = asyncio.run_coroutine_threadsafe(_run_multi(), loop)
        print(f"[detector] started for cams={list(rt.SESSIONS.keys())}")

    # ====== SSE 输出 ======
    q: "Queue[bytes]" = Queue(maxsize=256)

    def _start_pipe(topic: str):
        async def _pipe():
            async with rt.BUS.subscribe(topic, mode="fifo", maxsize=64) as subq:
                while True:
                    evt = await subq.get()
                    payload = {
                        "camera_id": cid,
                        "type": getattr(evt, "type", "accident"),
                        "happened": getattr(evt, "happened", True),
                        "confidence": getattr(evt, "confidence", 0.0),
                        "ts_unix": getattr(evt, "ts_unix", time.time()),
                        "frame_idx": getattr(evt, "frame_idx", None),
                        "pts_in_video": getattr(evt, "pts_in_video", None),
                    }
                    data = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
                    try:
                        q.put_nowait(data)
                    except Exception:
                        try: _ = q.get_nowait()
                        except Exception: pass
                        q.put_nowait(data)
        return asyncio.run_coroutine_threadsafe(_pipe(), loop)

    subs = [
        _start_pipe(f"accidents.open:{cid}"),
        _start_pipe(f"accidents.close:{cid}")
    ]

    def _stream():
        yield b": connected\n\n"
        last = time.time()
        try:
            while True:
                try:
                    chunk = q.get(timeout=1.0)
                    yield chunk
                except Empty:
                    now = time.time()
                    if now - last >= 10:
                        yield b": ping\n\n"
                        last = now
        finally:
            for s in subs:
                s.cancel()

    resp = StreamingHttpResponse(_stream(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp


# -------- /api/stop --------
@csrf_exempt
@require_POST
def stop_view(request: HttpRequest):
    body = _json(request)
    cid = (body.get("camera_id") or "").strip()
    if not cid:
        return JsonResponse({"ok": False, "error": "camera_id 缺失"}, status=400)

    sess = rt.SESSIONS.pop(cid, None)
    if not sess:
        return JsonResponse({"ok": False, "error": "该相机未运行"}, status=404)

    loop = rt.ensure_bg_loop()
    try:
        sess.stop(loop=loop)
        rt.INTENDED.pop(cid, None)
        return JsonResponse({"ok": True, "camera_id": cid, "stopped": True, "ts": int(time.time())})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
