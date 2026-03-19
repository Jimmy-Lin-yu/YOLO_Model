# hik_camera.py (NO PIL, numpy + OpenCV only)

from __future__ import annotations

import os
import ctypes
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np
import cv2
import MvCameraControl_class as MvCC


class HikCamera:
    """
    Hikvision SDK wrapper for pipeline usage (fast & clean)
    - Output: Mono8 numpy (H,W) uint8
    - Optional: BGR numpy (H,W,3) uint8 for YOLO/OpenCV
    - Snapshot default: no disk IO (fastest)
    - Enforce single-thread SDK calls (recommended with your CameraThread design)
    """

    def __init__(self, dev_index: int = 0, debug: bool = False, enforce_single_thread: bool = True):
        self.cam = None
        self.is_grabbing = False
        self.debug = debug
        self.enforce_single_thread = enforce_single_thread

        self._owner_tid: Optional[int] = None  # for enforcing single-thread SDK access
        self._open_device(dev_index)

    # -------------------------
    # Thread guard (no lock, but forbids cross-thread SDK calls)
    # -------------------------
    def _check_thread(self):
        if not self.enforce_single_thread:
            return
        tid = threading.get_ident()
        if self._owner_tid is None:
            self._owner_tid = tid
        elif self._owner_tid != tid:
            raise RuntimeError(
                f"[HikCamera] SDK called from different thread! "
                f"owner_tid={self._owner_tid}, current_tid={tid}. "
                f"Fix: call camera SDK only in CameraThread, or set enforce_single_thread=False."
            )

    # -------------------------
    # Public API
    # -------------------------
    def set_exposure_gain(self, exposure_us: float, gain: float):
        """Set exposure time (us) and gain. Call this BEFORE starting CameraThread (recommended)."""
        self._check_thread()
        if not self.cam:
            raise RuntimeError("Camera not opened")

        self.cam.MV_CC_SetEnumValue("ExposureAuto", MvCC.MV_EXPOSURE_AUTO_MODE_OFF)
        self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))
        self.cam.MV_CC_SetFloatValue("Gain", float(gain))

    def set_fps(self, fps: float, enable: bool = True):
        """
        Control camera-side FPS (GenICam nodes).
        Works only if the camera supports these nodes.
        """
        self._check_thread()
        if not self.cam:
            raise RuntimeError("Camera not opened")

        fps = float(fps)
        if fps <= 0:
            enable = False

        # 1) enable/disable framerate control
        try:
            ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", bool(enable))
            if ret != 0:
                print(f"[HikCamera] Set AcquisitionFrameRateEnable failed 0x{ret:x}", flush=True)
        except Exception as e:
            print(f"[HikCamera] Set AcquisitionFrameRateEnable exception: {repr(e)}", flush=True)

        # 2) set target fps
        if enable:
            try:
                ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", fps)
                if ret != 0:
                    print(f"[HikCamera] Set AcquisitionFrameRate={fps} failed 0x{ret:x}", flush=True)
            except Exception as e:
                print(f"[HikCamera] Set AcquisitionFrameRate exception: {repr(e)}", flush=True)
            

    def grab_mono_np(self, timeout_ms: int = 3000) -> np.ndarray:
        """Return Mono8 numpy: (H,W) uint8"""
        return self._get_frame_mono_np(timeout_ms=timeout_ms)

    def grab_bgr(self, timeout_ms: int = 3000) -> np.ndarray:
        """Return BGR numpy: (H,W,3) uint8 (for YOLO/OpenCV)"""
        mono = self.grab_mono_np(timeout_ms=timeout_ms)
        # Note: converting full-res every frame costs CPU.
        # If you want faster sensor path: resize mono first, then cvtColor on small image.
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    def clear_buffer(self, flush_n: int = 6, timeout_ms: int = 30):
        """
        Clear stale frames (best-effort).
        If SDK has ClearImageBuffer use it; else drain N frames.
        """
        self._check_thread()
        if not self.cam:
            return

        # 1) try SDK clear
        try:
            if hasattr(self.cam, "MV_CC_ClearImageBuffer"):
                ret = self.cam.MV_CC_ClearImageBuffer()
                if ret == 0:
                    return
        except Exception:
            pass

        # 2) fallback: drain frames
        for _ in range(max(0, flush_n)):
            try:
                _ = self._get_frame_mono_np(timeout_ms=timeout_ms)
            except Exception:
                break

    def trigger_snapshot_bgr(
        self,
        clear_buffer: bool = True,
        flush_n: int = 6,
        timeout_ms: int = 3000,
        out_dir: Optional[str] = None,
        prefix: str = "shot",
    ) -> np.ndarray:
        """
        Snapshot (default no disk IO): returns BGR ndarray
        - clear_buffer=True helps avoid getting previous part's leftover frame
        - out_dir != None -> also save jpg (optional)
        """
        self._check_thread()

        if clear_buffer:
            self.clear_buffer(flush_n=flush_n, timeout_ms=30)

        bgr = self.grab_bgr(timeout_ms=timeout_ms)

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(out_dir, f"{prefix}_{ts}.jpg")
            cv2.imwrite(path, bgr)

        return bgr

    def close(self):
        """Close camera and release SDK resources (call after CameraThread stop)."""
        self._check_thread()
        if not self.cam:
            return

        if self.is_grabbing:
            try:
                self.cam.MV_CC_StopGrabbing()
            except Exception:
                pass
        try:
            self.cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            self.cam.MV_CC_DestroyHandle()
        except Exception:
            pass

        self.cam = None
        self.is_grabbing = False

    # -------------------------
    # Device enumeration
    # -------------------------
    @staticmethod
    def list_devices():
        lst = MvCC.MV_CC_DEVICE_INFO_LIST()
        layer_type = MvCC.MV_GIGE_DEVICE | MvCC.MV_USB_DEVICE
        ret = MvCC.MvCamera.MV_CC_EnumDevices(layer_type, lst)
        print(f"[HikCamera] EnumDevices ret = 0x{ret:x}, nDeviceNum = {lst.nDeviceNum}")

        if ret != 0:
            print("  -> EnumDevices failed. Check SDK/permission.")
            return
        if lst.nDeviceNum == 0:
            print("  -> No cameras found.")
            return

        def _decode(arr) -> str:
            b = bytes(bytearray(arr))
            return b.split(b"\0", 1)[0].decode("utf-8", errors="ignore")

        for i in range(lst.nDeviceNum):
            dev_info = ctypes.cast(
                lst.pDeviceInfo[i], ctypes.POINTER(MvCC.MV_CC_DEVICE_INFO)
            ).contents

            if dev_info.nTLayerType == MvCC.MV_GIGE_DEVICE:
                gi = dev_info.SpecialInfo.stGigEInfo
                ip_int = gi.nCurrentIp
                ip = f"{ip_int & 0xff}.{(ip_int>>8)&0xff}.{(ip_int>>16)&0xff}.{(ip_int>>24)&0xff}"
                model = _decode(gi.chModelName)
                sn = _decode(gi.chSerialNumber)
                print(f"  [{i}] GigE  Model={model}  IP={ip}  SN={sn}")
            elif dev_info.nTLayerType == MvCC.MV_USB_DEVICE:
                ui = dev_info.SpecialInfo.stUsb3VInfo
                model = _decode(ui.chModelName)
                sn = _decode(ui.chSerialNumber)
                print(f"  [{i}] USB   Model={model}  SN={sn}")
            else:
                print(f"  [{i}] Unknown layer type: {dev_info.nTLayerType}")

    # -------------------------
    # Internal: open device
    # -------------------------
    def _open_device(self, index: int):
        # thread guard not used here (init stage)
        lst = MvCC.MV_CC_DEVICE_INFO_LIST()
        ret = MvCC.MvCamera.MV_CC_EnumDevices(MvCC.MV_GIGE_DEVICE | MvCC.MV_USB_DEVICE, lst)
        if ret != 0 or lst.nDeviceNum == 0:
            raise RuntimeError("No Hik camera found")

        if index >= lst.nDeviceNum:
            raise RuntimeError(f"Device index {index} out of range, found {lst.nDeviceNum} devices")

        dev_info = ctypes.cast(
            lst.pDeviceInfo[index], ctypes.POINTER(MvCC.MV_CC_DEVICE_INFO)
        ).contents

        self.cam = MvCC.MvCamera()

        ret = self.cam.MV_CC_CreateHandle(dev_info)
        if ret != 0:
            self.cam = None
            raise RuntimeError(f"CreateHandle failed 0x{ret:x}")

        ret = self.cam.MV_CC_OpenDevice(MvCC.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.cam = None
            raise RuntimeError(f"OpenDevice failed 0x{ret:x}")

        # Mono8 output (recommended for bandwidth)
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", MvCC.PixelType_Gvsp_Mono8)
        if ret != 0:
            print(f"[HikCamera] Warning: set PixelFormat failed 0x{ret:x}")

        # Trigger off (continuous grabbing)
        self.cam.MV_CC_SetEnumValue("TriggerMode", MvCC.MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetCommandValue("AcquisitionStart")

        # GigE packet size optimization
        if dev_info.nTLayerType == MvCC.MV_GIGE_DEVICE:
            pkt = self.cam.MV_CC_GetOptimalPacketSize()
            if pkt > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt)

        self.cam.MV_CC_SetImageNodeNum(10)

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"StartGrabbing failed 0x{ret:x}")

        self.is_grabbing = True
        time.sleep(0.1)

    # -------------------------
    # Internal: grab mono numpy
    # -------------------------
    def _get_frame_mono_np(self, timeout_ms: int = 3000) -> np.ndarray:
        self._check_thread()
        if not self.cam:
            raise RuntimeError("Camera not opened")

        if not self.is_grabbing:
            self.cam.MV_CC_StartGrabbing()
            self.is_grabbing = True

        frame = MvCC.MV_FRAME_OUT()
        ret = self.cam.MV_CC_GetImageBuffer(frame, timeout_ms)
        if ret != 0:
            if self.debug:
                print(f"[HikCamera] GetImageBuffer failed ret=0x{ret:x}")
            raise RuntimeError(f"GetImageBuffer failed 0x{ret:x}")

        try:
            buf_type = frame.stFrameInfo.enPixelType
            width = int(frame.stFrameInfo.nWidth)
            height = int(frame.stFrameInfo.nHeight)
            size = int(frame.stFrameInfo.nFrameLen)

            if self.debug:
                print(f"[HikCamera] frame: type=0x{buf_type:x}, w={width}, h={height}, size={size}")

            # Mono8: directly copy into numpy
            if buf_type == MvCC.PixelType_Gvsp_Mono8:
                out = np.empty((height, width), dtype=np.uint8)
                ctypes.memmove(out.ctypes.data, frame.pBufAddr, size)
                return out

            # Non-Mono8: convert to Mono8 via SDK
            dst_size = width * height
            dst = np.empty((dst_size,), dtype=np.uint8)

            convert_param = MvCC.MV_CC_PIXEL_CONVERT_PARAM()
            convert_param.nWidth = width
            convert_param.nHeight = height
            convert_param.pSrcData = frame.pBufAddr
            convert_param.nSrcDataLen = size
            convert_param.enSrcPixelType = buf_type
            convert_param.enDstPixelType = MvCC.PixelType_Gvsp_Mono8
            convert_param.pDstBuffer = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            convert_param.nDstBufferSize = dst.nbytes

            ret = self.cam.MV_CC_ConvertPixelType(convert_param)
            if ret != 0:
                raise RuntimeError(f"ConvertPixelType failed 0x{ret:x}")

            return dst.reshape((height, width))

        finally:
            self.cam.MV_CC_FreeImageBuffer(frame)
