# from configs.camera_sources import get_source
#
# # ...
# camera_id = (body.get("camera_id") or "").strip()
# try:
#     file_path = get_source(camera_id)
# except Exception as e:
#     return JsonResponse({"ok": False, "error": str(e)}, status=400)
#
# # 后面全部沿用你现有流程：
# # session = SESSION_MANAGER.ensure_single_file_session(camera_id=camera_id, file_path=file_path, ...)
# # session.start_if_needed()
# # return JsonResponse({...})
CAMERA_SOURCES: dict[str, dict] = {
    "cam-1": {
        "src": r"E:\Training\Recording 2025-10-30 172929.mp4",
    },
    "cam-2": {
        "src": r"E:\Training\Recording 2025-11-02 172630.mp4",

    }
}

def get_source(camera_id: str) -> str:
    """查表并做一下合法性校验；不存在或未启用则抛异常"""
    meta = CAMERA_SOURCES.get(camera_id)
    if not meta or not meta.get("enabled", True):
        raise KeyError(f"camera_id 未配置或未启用: {camera_id}")
    src = (meta.get("src") or "").strip()
    if not src:
        raise ValueError(f"camera_id 未提供有效 src: {camera_id}")
    return src
