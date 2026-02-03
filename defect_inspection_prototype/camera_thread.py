
# camera_thread.py
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional, Tuple, Callable, Any

import numpy as np
import cv2

from hik_camera import HikCamera


def resize_keep_aspect_gray(img: np.ndarray, max_side: int) -> np.ndarray:
    """Keep aspect ratio, downscale only. INTER_AREA is best for downscale."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / float(m)
    nw, nh = int(round(w * s)), int(round(h * s))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class FramePack:
    frame_id: int
    ts_wall: float          # time.time()
    ts_perf: float          # time.perf_counter()
    mono_full: np.ndarray   # full-res mono
    mono_small: Optional[np.ndarray]  # derived from mono_full (same frame)


class CameraThread:
    """
    One camera = one thread.
    Thread owns ALL SDK calls; others only read latest / request snapshot.

    Key improvement:
      - request_snapshot_pack() returns (mono_full + mono_small) from the SAME grabbed frame.
      - This prevents "sensor frame != defect frame" mismatch (off-center / blur).
    """

    def __init__(
        self,
        cam_id: int,
        dev_index: Optional[int] = None,
        cam_factory: Optional[Callable[[], Any]] = None,

        target_fps: float | None = None,

        sensor_size: int | None = 640,   # max side for mono_small; None disables small
        copy_on_grab: bool = True,       # safer if SDK reuses internal buffers
        latest_copy: bool = False,       # copy when reading latest getters
        grab_timeout_ms: int = 300,

        init_exposure_us: Optional[float] = None,
        init_gain: Optional[float] = None,

        cam_kwargs: Optional[dict] = None,
    ):
        self.cam_id = cam_id
        self.dev_index = dev_index
        self.target_fps = target_fps
        self.sensor_size = sensor_size
        self.copy_on_grab = copy_on_grab
        self.latest_copy = latest_copy
        self.grab_timeout_ms = grab_timeout_ms

        self.init_exposure_us = init_exposure_us
        self.init_gain = init_gain
        self.cam_kwargs = cam_kwargs or {}

        if cam_factory is not None:
            self.cam_factory = cam_factory
        else:
            if dev_index is None:
                raise ValueError("CameraThread: either dev_index or cam_factory must be provided")
            self.cam_factory = lambda: HikCamera(dev_index=dev_index, **self.cam_kwargs)

        self._lock = threading.Lock()
        self._latest_pack: Optional[FramePack] = None
        self._last_err: Optional[str] = None

        # Snapshot request: send skip_frames (int) and respond with FramePack
        self._snap_req: Queue[int] = Queue(maxsize=1)
        self._snap_resp: Queue[FramePack] = Queue(maxsize=1)

        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

        self._frame_id = 0

    def start(self):
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join(timeout=2)

    # ----------------------------
    # Latest getters (fast)
    # ----------------------------
    def get_latest_pack(self) -> Optional[FramePack]:
        with self._lock:
            pack = self._latest_pack
        if pack is None:
            return None
        if not self.latest_copy:
            return pack

        # Make safe copies for callers (optional)
        mono_full = pack.mono_full.copy()
        mono_small = pack.mono_small.copy() if pack.mono_small is not None else None
        return FramePack(
            frame_id=pack.frame_id,
            ts_wall=pack.ts_wall,
            ts_perf=pack.ts_perf,
            mono_full=mono_full,
            mono_small=mono_small,
        )

    def get_latest_mono_full(self) -> Tuple[Optional[np.ndarray], float]:
        pack = self.get_latest_pack()
        if pack is None:
            return None, 0.0
        return pack.mono_full, pack.ts_wall

    def get_latest_mono_small(self) -> Tuple[Optional[np.ndarray], float]:
        pack = self.get_latest_pack()
        if pack is None:
            return None, 0.0
        return pack.mono_small, pack.ts_wall

    def get_latest_bgr_small(self) -> Tuple[Optional[np.ndarray], float]:
        mono, ts = self.get_latest_mono_small()
        if mono is None:
            return None, ts
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR), ts

    def get_latest_bgr_full(self) -> Tuple[Optional[np.ndarray], float]:
        mono, ts = self.get_latest_mono_full()
        if mono is None:
            return None, ts
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR), ts

    def get_latest_bgr(self):
        bgr, ts = self.get_latest_bgr_small()
        if bgr is not None:
            return bgr, ts
        return self.get_latest_bgr_full()

    def get_last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_err

    # ----------------------------
    # Snapshot pack (the important fix)
    # ----------------------------
    def request_snapshot_pack(self, timeout: float = 0.5, skip_frames: int = 0) -> Optional[FramePack]:
        """
        Returns a FramePack from the SAME grab:
          mono_small is derived from mono_full of that frame.

        skip_frames:
          if you suspect you're snapping too early (motion blur),
          set skip_frames=1 or 2 to wait N new frames after the request is seen.
        """
        # clear old responses
        try:
            while True:
                self._snap_resp.get_nowait()
        except Empty:
            pass

        # signal request (store skip_frames)
        try:
            self._snap_req.put_nowait(int(max(0, skip_frames)))
        except Exception:
            # queue full -> ignore
            pass

        try:
            return self._snap_resp.get(timeout=timeout)
        except Empty:
            return None

    # ----------------------------
    # Thread loop
    # ----------------------------
    def _run(self):
        min_dt = (1.0 / self.target_fps) if (self.target_fps and self.target_fps > 0) else 0.0

        cam = None
        pending_snap_skip: Optional[int] = None

        try:
            cam = self.cam_factory()

            if (self.init_exposure_us is not None) and (self.init_gain is not None):
                try:
                    cam.set_exposure_gain(self.init_exposure_us, self.init_gain)
                except Exception as e:
                    with self._lock:
                        self._last_err = f"set_exposure_gain failed: {e}"

            while not self._stop.is_set():
                loop_t0 = time.time()

                # Receive snapshot request (if any)
                if pending_snap_skip is None:
                    try:
                        pending_snap_skip = self._snap_req.get_nowait()
                    except Empty:
                        pending_snap_skip = None

                # Grab
                try:
                    mono_raw = cam.grab_mono_np(timeout_ms=self.grab_timeout_ms)
                    ts_wall = time.time()
                    ts_perf = time.perf_counter()
                except Exception as e:
                    with self._lock:
                        self._last_err = f"grab_mono_np failed: {e}"
                    time.sleep(0.01)
                    continue

                mono_full = mono_raw.copy() if self.copy_on_grab else mono_raw

                mono_small = None
                if self.sensor_size:
                    try:
                        mono_small = resize_keep_aspect_gray(mono_full, self.sensor_size)
                    except Exception:
                        mono_small = None

                self._frame_id += 1
                pack = FramePack(
                    frame_id=self._frame_id,
                    ts_wall=ts_wall,
                    ts_perf=ts_perf,
                    mono_full=mono_full,
                    mono_small=mono_small,
                )

                with self._lock:
                    self._latest_pack = pack
                    self._last_err = None

                # Fulfill snapshot request only after skip_frames have passed
                if pending_snap_skip is not None:
                    if pending_snap_skip > 0:
                        pending_snap_skip -= 1
                    else:
                        # return a safe copy to caller
                        snap_pack = FramePack(
                            frame_id=pack.frame_id,
                            ts_wall=pack.ts_wall,
                            ts_perf=pack.ts_perf,
                            mono_full=pack.mono_full.copy(),
                            mono_small=(pack.mono_small.copy() if pack.mono_small is not None else None),
                        )
                        try:
                            # clear then put (keep newest)
                            try:
                                while True:
                                    self._snap_resp.get_nowait()
                            except Empty:
                                pass
                            self._snap_resp.put_nowait(snap_pack)
                        except Exception:
                            pass
                        pending_snap_skip = None

                # FPS cap
                if min_dt > 0:
                    dt = time.time() - loop_t0
                    if dt < min_dt:
                        time.sleep(min_dt - dt)

        finally:
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass
