# -*- coding: utf-8 -*-
"""
Motion Evidence Stage
---------------------
Subscribe:
    motion.score:<camera_id>

Publish:
    motion:<camera_id>

This stage is intentionally lightweight.
It only converts motion score payload into MotionEvidence,
keeping the final bus contract stable.
"""

from __future__ import annotations

import asyncio
from typing import List, Dict, Any

from events.bus import AsyncBus, MotionEvidence, topic_for

_TOPIC_IN_BASE = "motion.score"
_TOPIC_OUT_BASE = "motion"

_SUB_QUEUE_SIZE = 128
_LOG_PER_FRAME = False

# Placeholder threshold only for filling active flag.
# Final fusion / usage in aggregator is intentionally left for later.
_ACTIVE_THR = 0.35


def _to_motion_evidence(payload: Dict[str, Any]) -> MotionEvidence:
    score_motion = float(payload.get("score_motion", 0.0))
    return MotionEvidence(
        type="motion",
        camera_id=str(payload["camera_id"]),
        ts_unix=float(payload["ts_unix"]),
        frame_idx=int(payload["frame_idx"]),
        pts_in_video=float(payload["pts_in_video"]),
        vts=float(payload["vts"]),
        dt_vts=float(payload.get("dt_vts", 0.0)),
        valid=bool(payload.get("valid", False)),
        repeated_sample=bool(payload.get("repeated_sample", False)),
        score_motion=score_motion,
        score_mag=float(payload.get("score_mag", 0.0)),
        score_ori=float(payload.get("score_ori", 0.0)),
        active=bool(score_motion >= _ACTIVE_THR),
    )


class MotionEvidencePublisher:
    def __init__(self, camera_id: str, bus: AsyncBus) -> None:
        self.camera_id = camera_id
        self.bus = bus

    async def run(self) -> None:
        topic_in = topic_for(_TOPIC_IN_BASE, self.camera_id)

        async with self.bus.subscribe(topic_in, mode="fifo", maxsize=_SUB_QUEUE_SIZE) as q:
            while True:
                payload: Dict[str, Any] = await q.get()
                ev = _to_motion_evidence(payload)

                if _LOG_PER_FRAME:
                    print(
                        f"[MOTION-EVIDENCE {self.camera_id}] "
                        f"frame={ev.frame_idx:04d} "
                        f"vts={ev.vts:.2f} "
                        f"valid={ev.valid} "
                        f"repeat={ev.repeated_sample} "
                        f"motion={ev.score_motion:.3f} "
                        f"mag={ev.score_mag:.3f} "
                        f"ori={ev.score_ori:.3f} "
                        f"active={ev.active}"
                    )

                await self.bus.publish_partitioned(_TOPIC_OUT_BASE, self.camera_id, ev)


async def run_motion_evidence_stage(
    bus: AsyncBus,
    *,
    camera_id: str,
) -> None:
    publisher = MotionEvidencePublisher(camera_id=camera_id, bus=bus)
    await publisher.run()


async def run_motion_evidence_stage_multi(
    bus: AsyncBus,
    *,
    camera_ids: List[str],
) -> None:
    tasks = [
        asyncio.create_task(run_motion_evidence_stage(bus, camera_id=cam))
        for cam in camera_ids
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                pass