# -*- coding: utf-8 -*-
from __future__ import annotations
from __future__ import annotations

import json
import time

from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .session_manager import SESSION_MANAGER


def _json_body(request: HttpRequest) -> dict:
    try:
        return json.loads(request.body.decode("utf-8")) if request.body else {}
    except Exception:
        return {}

@csrf_exempt
@require_POST
def play_view(request: HttpRequest):
    body = _json_body(request)
    camera_id = (body.get("camera_id") or "").strip()
    idem_key = (request.headers.get("Idempotency-Key") or body.get("idempotency_key") or "").strip()

    if not camera_id:
        return JsonResponse({"ok": False, "error": "invalid_camera_id"}, status=400)

    try:
        sess = SESSION_MANAGER.create_session(camera_id, idem_key=idem_key, allow_parallel=False)
    except FileNotFoundError as e:
        # camera 无映射 或 文件不存在
        msg = str(e)
        if msg == "camera_unmapped":
            return JsonResponse({"ok": False, "error": "camera_unmapped"}, status=404)
        return JsonResponse({"ok": False, "error": "video_not_found"}, status=404)
    except RuntimeError as e:
        # 同一 camera 正在使用
        msg = str(e)
        if msg.startswith("camera_locked:"):
            busy_id = msg.split(":", 1)[1]
            return JsonResponse({"ok": False, "error": "camera_locked", "session_id": busy_id}, status=423)
        return JsonResponse({"ok": False, "error": "internal_error", "detail": msg}, status=500)
    except Exception as e:
        return JsonResponse({"ok": False, "error": "internal_error", "detail": str(e)}, status=500)

    # 组装返回
    meta = sess.source_meta
    data = {
        "ok": True,
        "message": "session ready",
        "session_id": sess.session_id,
        "camera_id": sess.camera_id,
        "video_id": sess.video_id,
        "analysis_state": "ready",
        "source_meta": {
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
            "duration_sec": meta.duration_sec,
            "has_true_pts": meta.has_true_pts,
        },
        "sample_fps": sess.sample_fps,
        "server_ack_ts": sess.server_ack_ts,
        "sse_url": f"/sse/analysis?session_id={sess.session_id}&camera_id={sess.camera_id}"
    }
    return JsonResponse(data, status=200)



@csrf_exempt
@require_POST
def stop_view(request: HttpRequest):
    body = _json_body(request)
    session_id = (body.get("session_id") or "").strip()
    reason = (body.get("reason") or "stopped").strip()  # "stopped" | "eof" | "error"

    if not session_id:
        return JsonResponse({"ok": False, "error": "invalid_session_id"}, status=400)

    ok, released_cam, sess = SESSION_MANAGER.stop_session(session_id, reason=reason)
    if not ok:
        return JsonResponse({"ok": False, "error": "session_not_found"}, status=404)

    data = {
        "ok": True,
        "session_id": session_id,
        "status": sess.status if sess else "stopped",
        "released_camera": released_cam,
        "reason": reason,
        "server_ts": time.time()
    }
    return JsonResponse(data, status=200)
