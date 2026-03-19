# mode_plc.py
from __future__ import annotations

import os
import time
import cv2
import numpy as np
from PIL import Image
from typing import Callable
import threading

from pymodbus.client import ModbusTcpClient


class PLCMode:
    """
    PLC/Modbus window + 回寫結果 + FINAL overlay
    - CAPTURE_START ~ CAPTURE_END 為一顆足粒 window
    - window 內推論，res_log 記錄 OK/NG
    - 出現任一 NG => FINAL=NG
    - CAPTURE_END 上升沿：回寫 RESULT_CODE + RESULT_VALID
    - ROBOT_DONE 上升沿：清狀態 + 清 RESULT_VALID/RESULT_CODE
    """

    def __init__(
        self,
        cfg: dict,
        get_cam: Callable[[], object | None],
        cam_lock: threading.Lock,
        get_yolo: Callable[[], object | None],
        yolo_lock: threading.Lock,
        black_size=(640, 480),
    ) -> None:
        self.cfg = cfg
        self.get_cam = get_cam
        self.cam_lock = cam_lock
        self.get_yolo = get_yolo
        self.yolo_lock = yolo_lock
        self.black_size = black_size

        # cycle state
        self.cycle_lock = threading.Lock()
        self.cycle_active = False
        self.res_log: list[str] = []
        self.ng_seen = False
        self._plc_prev = {"capture_start": False, "capture_end": False, "robot_done": False}

        # modbus client cache
        self._modbus_lock = threading.Lock()
        self._modbus_client: ModbusTcpClient | None = None
        self._modbus_cfg: tuple[str, int, float] | None = None
        self._modbus_io_lock = threading.Lock()

        # final overlay state
        self._final_lock = threading.Lock()
        self._final_text: str | None = None
        self._final_until: float = 0.0
        self._final_color_bgr: tuple[int, int, int] = (0, 255, 0)

    # -----------------
    # public hooks
    # -----------------
    def on_enter(self, camera_on: bool, model_on: bool) -> None:
        # 進入 PLC mode：同步一次 PC_READY
        try:
            self.set_pc_ready(camera_on=camera_on, model_on=model_on)
        except Exception:
            pass

    def on_exit(self) -> None:
        # 離開 PLC mode：收乾淨，避免 PLC 端讀到殘留
        try:
            self._clear_final_overlay()
        except Exception:
            pass
        try:
            with self.cycle_lock:
                self.cycle_active = False
                self.res_log = []
                self.ng_seen = False
        except Exception:
            pass

        try:
            # PC_READY 拉 0
            self._hr_write_bool(self.cfg["PC_READY_HR"], False)
        except Exception:
            pass
        try:
            # 清 RESULT
            self._hr_write_bool(self.cfg["RESULT_VALID_HR"], False)
            self._hr_write_int(self.cfg["RESULT_CODE_HR"], 0)
        except Exception:
            pass

    def set_pc_ready(self, camera_on: bool, model_on: bool) -> None:
        ready = bool(camera_on and model_on)
        self._hr_write_bool(self.cfg["PC_READY_HR"], ready)

    def on_tick(self, camera_on: bool, model_on: bool) -> Image.Image:
        # 1) poll PLC edges
        start_rise, end_rise, done_rise = self._poll_plc_edges()

        if start_rise:
            cam = self.get_cam()
            yolo = self.get_yolo()
            if camera_on and model_on and (cam is not None) and (yolo is not None):
                self._cycle_start()

        if end_rise:
            try:
                self._cycle_end_and_send_result()
            except Exception:
                pass

        if done_rise:
            try:
                self._robot_done_cleanup()
            except Exception:
                pass

        # 2) grab frame
        cam = self.get_cam()
        if (not camera_on) or (cam is None):
            return self._black_frame()

        with self.cam_lock:
            try:
                img_pil: Image.Image = cam.grab_frame()
            except Exception:
                return self._black_frame()

        def pil_with_final(img: Image.Image) -> Image.Image:
            frame_rgb = np.array(img)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = self._apply_final_overlay_bgr(frame_bgr)
            frame_rgb2 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb2)

        # 3) infer gate
        yolo = self.get_yolo()
        if (not model_on) or (yolo is None):
            return pil_with_final(img_pil)

        with self.cycle_lock:
            active_now = bool(self.cycle_active)

        if self.cfg.get("INFER_ONLY_WHEN_CYCLE_ACTIVE", True) and (not active_now):
            return pil_with_final(img_pil)

        # 4) infer
        frame_rgb = np.array(img_pil)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        with self.yolo_lock:
            draw_bgr, info = yolo.infer_frame(frame_bgr)

        status = str(info.get("status", "OK"))

        # 5) window log
        if active_now:
            with self.cycle_lock:
                self.res_log.append(status)
                if status == "NG":
                    self.ng_seen = True
                max_len = int(self.cfg.get("MAX_LOG_LEN", 500))
                if len(self.res_log) > max_len:
                    self.res_log[:] = self.res_log[-max_len:]

        draw_bgr = self._apply_final_overlay_bgr(draw_bgr)
        draw_rgb = cv2.cvtColor(draw_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(draw_rgb)

    # -----------------
    # internal helpers
    # -----------------
    def _black_frame(self) -> Image.Image:
        return Image.new("RGB", self.black_size, (0, 0, 0))

    # ===== Modbus cached client =====
    def _get_modbus_client(self) -> ModbusTcpClient:
        host = self.cfg["PLC_HOST"]
        port = int(self.cfg["PLC_PORT"])
        timeout = float(self.cfg["MODBUS_TIMEOUT"])

        with self._modbus_lock:
            if self._modbus_client is None or self._modbus_cfg != (host, port, timeout):
                try:
                    if self._modbus_client is not None:
                        self._modbus_client.close()
                except Exception:
                    pass
                self._modbus_client = ModbusTcpClient(host=host, port=port, timeout=timeout)
                self._modbus_cfg = (host, port, timeout)

            if not self._modbus_client.connect():
                raise ConnectionError(f"Modbus connect failed: {host}:{port}")

            return self._modbus_client

    def _hr_read_int(self, hr_addr: int) -> int:
        with self._modbus_io_lock:
            client = self._get_modbus_client()
            rr = client.read_holding_registers(address=int(hr_addr), count=1)
            if rr.isError():  # type: ignore[attr-defined]
                raise RuntimeError(f"read_holding_registers error: {rr}")
            return int(rr.registers[0])

    def _hr_write_int(self, hr_addr: int, value: int) -> None:
        with self._modbus_io_lock:
            client = self._get_modbus_client()
            rr = client.write_register(address=int(hr_addr), value=int(value))
            if rr.isError():  # type: ignore[attr-defined]
                raise RuntimeError(f"write_register error: {rr}")

    def _hr_read_bool(self, hr_addr: int) -> bool:
        return self._hr_read_int(hr_addr) != 0

    def _hr_write_bool(self, hr_addr: int, value: bool) -> None:
        self._hr_write_int(hr_addr, 1 if value else 0)

    # ===== PLC edge polling =====
    def _poll_plc_edges(self) -> tuple[bool, bool, bool]:
        try:
            cap_start = self._hr_read_bool(self.cfg["CAPTURE_START_HR"])
            cap_end = self._hr_read_bool(self.cfg["CAPTURE_END_HR"])
            robot_done = self._hr_read_bool(self.cfg["ROBOT_DONE_HR"])
        except Exception:
            return (False, False, False)

        start_rise = cap_start and (not self._plc_prev["capture_start"])
        end_rise = cap_end and (not self._plc_prev["capture_end"])
        done_rise = robot_done and (not self._plc_prev["robot_done"])

        self._plc_prev["capture_start"] = cap_start
        self._plc_prev["capture_end"] = cap_end
        self._plc_prev["robot_done"] = robot_done
        return (start_rise, end_rise, done_rise)

    # ===== cycle =====
    def _flush_camera_buffer(self, n: int = 5) -> None:
        cam = self.get_cam()
        if cam is None:
            return
        with self.cam_lock:
            if hasattr(cam, "clear_buffer"):
                try:
                    cam.clear_buffer()
                    return
                except Exception:
                    pass
            for _ in range(max(0, n)):
                try:
                    cam.grab_frame()
                except Exception:
                    break

    def _cycle_start(self) -> None:
        with self.cycle_lock:
            self.cycle_active = True
            self.ng_seen = False
            self.res_log = []

        # 避免上一顆殘留 RESULT_VALID
        try:
            self._hr_write_bool(self.cfg["RESULT_VALID_HR"], False)
            self._hr_write_int(self.cfg["RESULT_CODE_HR"], 0)
        except Exception:
            pass

        self._clear_final_overlay()
        self._flush_camera_buffer(int(self.cfg.get("FLUSH_N", 5)))

    def _append_cycle_log(self, final_code: int, seq: list[str]) -> None:
        os.makedirs(self.cfg["CYCLE_LOG_DIR"], exist_ok=True)
        path = os.path.join(self.cfg["CYCLE_LOG_DIR"], self.cfg["CYCLE_LOG_NAME"])

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        final_str = "OK" if final_code == 1 else "NG" if final_code == 2 else str(final_code)
        seq_str = " ".join(seq)
        line = f"{ts} | FINAL={final_str} | N={len(seq)} | RES=[{seq_str}]\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    def _cycle_end_and_send_result(self) -> None:
        with self.cycle_lock:
            if not self.cycle_active:
                return
            self.cycle_active = False

            CODE_EMPTY = int(self.cfg["RESULT_CODE_EMPTY"])
            CODE_OK = int(self.cfg["RESULT_CODE_OK"])
            CODE_NG = int(self.cfg["RESULT_CODE_NG"])

            if len(self.res_log) == 0:
                final_code = CODE_EMPTY
            else:
                final_code = CODE_NG if self.ng_seen else CODE_OK

            seq_snapshot = list(self.res_log)

            # FINAL overlay 2 秒
            if final_code == CODE_OK:
                self._set_final_overlay("FINAL=OK", (0, 255, 0), duration_sec=2.0)
            elif final_code == CODE_NG:
                self._set_final_overlay("FINAL=NG", (0, 0, 255), duration_sec=2.0)
            else:
                self._set_final_overlay("FINAL=EMPTY", (0, 255, 255), duration_sec=2.0)

        # write file log（不影響主流程）
        try:
            self._append_cycle_log(final_code=final_code, seq=seq_snapshot)
        except Exception:
            pass

        # 先寫 CODE 再拉 VALID
        self._hr_write_int(self.cfg["RESULT_CODE_HR"], int(final_code))
        self._hr_write_bool(self.cfg["RESULT_VALID_HR"], True)

    def _robot_done_cleanup(self) -> None:
        with self.cycle_lock:
            self.cycle_active = False
            self.ng_seen = False
            self.res_log = []

        self._hr_write_bool(self.cfg["RESULT_VALID_HR"], False)
        self._hr_write_int(self.cfg["RESULT_CODE_HR"], 0)

    # ===== FINAL overlay =====
    def _set_final_overlay(self, text: str, color_bgr: tuple[int, int, int], duration_sec: float = 2.0) -> None:
        with self._final_lock:
            self._final_text = text
            self._final_color_bgr = color_bgr
            self._final_until = time.time() + float(duration_sec)

    def _clear_final_overlay(self) -> None:
        with self._final_lock:
            self._final_text = None
            self._final_until = 0.0

    def _apply_final_overlay_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        now = time.time()
        with self._final_lock:
            if (self._final_text is None) or (now > self._final_until):
                self._final_text = None
                self._final_until = 0.0
                return img_bgr
            text = self._final_text
            color = self._final_color_bgr

        h, w = img_bgr.shape[:2]
        org = (int(w * 0.22), int(h * 0.92))

        cv2.putText(img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 5, color, 3, cv2.LINE_AA)
        return img_bgr
