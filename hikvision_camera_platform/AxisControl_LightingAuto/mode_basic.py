# mode_basic.py
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from typing import Callable
import threading


class BasicMode:
    """
    純相機 + YOLO 即時顯示
    - camera_off => black
    - camera_on & model_off => raw
    - camera_on & model_on  => yolo overlay
    """

    def __init__(
        self,
        get_cam: Callable[[], object | None],
        cam_lock: threading.Lock,
        get_yolo: Callable[[], object | None],
        yolo_lock: threading.Lock,
        black_size=(640, 480),
    ) -> None:
        self.get_cam = get_cam
        self.cam_lock = cam_lock
        self.get_yolo = get_yolo
        self.yolo_lock = yolo_lock
        self.black_size = black_size

    def _black_frame(self) -> Image.Image:
        return Image.new("RGB", self.black_size, (0, 0, 0))

    def on_tick(self, camera_on: bool, model_on: bool) -> Image.Image:
        cam = self.get_cam()
        if (not camera_on) or (cam is None):
            return self._black_frame()

        # grab
        with self.cam_lock:
            try:
                img_pil: Image.Image = cam.grab_frame()
            except Exception:
                return self._black_frame()

        yolo = self.get_yolo()
        if (not model_on) or (yolo is None):
            return img_pil

        # infer
        frame_rgb = np.array(img_pil)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        with self.yolo_lock:
            draw_bgr, _info = yolo.infer_frame(frame_bgr)

        draw_rgb = cv2.cvtColor(draw_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(draw_rgb)
