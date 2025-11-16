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
from __future__ import annotations

import asyncio
from typing import List, Optional

from events.bus import AsyncBus
from events.frame_discrete import run_frame_source_raw, run_sampler_equal_time_vts


class SingleFileSession:
    """
    Single-stream video detection session:
    - frames_raw → equal-time sampling → frames → YOLO → accident → aggregator → open/close
    - Automatically flushes and closes events when the video ends
    """


    def __init__(
        self,
        *,
        camera_id: str,
        file_path: str,
        bus: AsyncBus,
        sampler_fps: float = 15.0,
        session_id: Optional[str] = None,
    ) -> None:
        self.camera_id = camera_id
        self.file_path = file_path
        self.bus = bus
        self.sampler_fps = sampler_fps
        self.session_id = session_id or f"sess-{camera_id}"
        self._tasks: List[asyncio.Task] = []
        self._running = False

    def start(self, *, loop) -> None:
        if self._running:
            return

        async def _start():
            from events.Accident_detect.incident_aggregator import AccidentAggregator
            self._running = True

            aggregator = AccidentAggregator(
                camera_id=self.camera_id,
                bus=self.bus,
                session_id=self.session_id,
            )

            # Start the frame source, sampler, and aggregator
            t_src = asyncio.create_task(run_frame_source_raw(self.bus, self.camera_id, self.file_path))
            t_smp = asyncio.create_task(run_sampler_equal_time_vts(self.bus, self.camera_id))
            t_agg = asyncio.create_task(aggregator.run())
            self._tasks = [t_src, t_smp, t_agg]
            print(f"[session] started {self.camera_id}")

            try:
                # Wait for the video to finish playing
                await t_src
                print(f"[frame_source] {self.camera_id} finished, releasing video")

                # Stop the sampler to prevent further frame dispatch
                t_smp.cancel()
                await asyncio.gather(t_smp, return_exceptions=True)

                # Add a draining window: allow 0.5–1s for detectors and aggregators to process in-flight frames
                await asyncio.sleep(0.8)

                # Then flush to finalize and close any remaining events
                await aggregator.flush()

                # Finally, safely cancel the aggregator (it runs a long loop)
                t_agg.cancel()
                await asyncio.gather(t_agg, return_exceptions=True)

            except asyncio.CancelledError:
                try:
                    await aggregator.flush()
                except Exception:
                    pass
            except Exception as e:
                print(f"[session] error {self.camera_id}: {e}")
            finally:
                self._running = False
                print(f"[session] finished {self.camera_id}")

        asyncio.run_coroutine_threadsafe(_start(), loop)


    # Stop tasks for this camera
    def stop(self, *, loop) -> None:
        if not self._running:
            return

        async def _stop():
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
            self._running = False
            print(f"[session] stopped {self.camera_id}")

        asyncio.run_coroutine_threadsafe(_stop(), loop)
