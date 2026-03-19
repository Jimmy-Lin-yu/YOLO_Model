# hik_camera.py
import os
import sys
import ctypes
from pathlib import Path
from ctypes import byref
from datetime import datetime

from PIL import Image
import MvCameraControl_class as MvCC  # 已把 MvImport 路徑加入 PYTHONPATH


PIXEL_TYPE = {
    MvCC.PixelType_Gvsp_Mono8: "L",
    MvCC.PixelType_Gvsp_RGB8_Packed: "RGB",
}


class HikCamera:
    """
    簡易包裝海康 SDK：
    - 初始化就開啟裝置並開始取流
    - set_exposure_gain() 設曝光 / 增益
    - grab_frame() 取得一張 PIL.Image（連續取流用）
    - grab_and_save(path) 拍照並存檔
    - trigger_snapshot(dir) 觸發截圖到指定資料夾
    """

    def __init__(self, dev_index: int = 0):
        self.cam = None
        self.is_grabbing = False
        self._open_device(dev_index)

    # ---------- Public API ----------

    def set_exposure_gain(self, exposure_us: float, gain: float):
        """設定曝光時間（微秒）與增益"""
        if not self.cam:
            raise RuntimeError("Camera not opened")

        self.cam.MV_CC_SetEnumValue("ExposureAuto", MvCC.MV_EXPOSURE_AUTO_MODE_OFF)
        self.cam.MV_CC_SetFloatValue("ExposureTime", exposure_us)
        self.cam.MV_CC_SetFloatValue("Gain", gain)

    def grab_frame(self, timeout_ms: int = 3000) -> Image.Image:
        """
        取得一張影像（PIL.Image），不存檔。
        可在 UI 連續呼叫達到「持續流」效果。
        """
        return self._get_frame(timeout_ms=timeout_ms)

    def grab_and_save(self, path: str = "image.jpg", timeout_ms: int = 3000) -> str:
        """
        拍一張圖並存檔，回傳絕對路徑。
        """
        img = self._get_frame(timeout_ms=timeout_ms)
        img.save(path, "JPEG")
        return str(Path(path).resolve())
    
    def clear_buffer(self, flush_n: int = 8, timeout_ms: int = 50):
        """
        嘗試清掉相機/SDK 內部影像 buffer。
        1) 若 SDK 有 MV_CC_ClearImageBuffer() 就直接清
        2) 否則 fallback：快速 grab N 幀把舊幀讀掉
        """
        if not self.cam:
            return

        # 1) 先試 SDK 的 clear buffer API
        try:
            if hasattr(self.cam, "MV_CC_ClearImageBuffer"):
                ret = self.cam.MV_CC_ClearImageBuffer()
                # ret == 0 通常代表成功
                if ret == 0:
                    return
        except Exception:
            pass

        # 2) fallback：快速讀掉幾幀
        for _ in range(max(0, flush_n)):
            try:
                _ = self._get_frame(timeout_ms=timeout_ms)
            except Exception:
                break

    def trigger_snapshot(self, out_dir: str = "snapshots", prefix: str = "shot") -> str:
        """
        觸發截圖：以 timestamp 命名 JPG 檔，存到 out_dir。
        回傳檔案完整路徑。
        """
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{ts}.jpg"
        path = os.path.join(out_dir, filename)
        return self.grab_and_save(path)

    def close(self):
        """關閉相機與釋放資源"""
        if self.cam:
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


# ----------------列出目前能偵測到的所有海康相機---------------------
    @staticmethod
    def list_devices():
        """列出目前能偵測到的所有海康相機（GigE / USB）。"""
        lst = MvCC.MV_CC_DEVICE_INFO_LIST()
        layer_type = MvCC.MV_GIGE_DEVICE | MvCC.MV_USB_DEVICE
        ret = MvCC.MvCamera.MV_CC_EnumDevices(layer_type, lst)
        print(f"[HikCamera] EnumDevices ret = 0x{ret:x}, nDeviceNum = {lst.nDeviceNum}")

        if ret != 0:
            print("  -> EnumDevices 失敗，請檢查 SDK / 權限。")
            return

        if lst.nDeviceNum == 0:
            print("  -> 沒有偵測到任何相機。")
            return

        import ctypes

        def _decode_c_char_array(arr) -> str:
            """將 ctypes 的 char/ubyte array 轉成 Python 字串。"""
            try:
                b = bytes(arr)
            except TypeError:
                # 有些版本可能不是 bytes-like，就包一層 bytearray
                b = bytes(bytearray(arr))
            # 去掉尾端的 \0 再 decode
            return b.split(b"\0", 1)[0].decode("utf-8", errors="ignore")

        for i in range(lst.nDeviceNum):
            dev_info = ctypes.cast(
                lst.pDeviceInfo[i], ctypes.POINTER(MvCC.MV_CC_DEVICE_INFO)
            ).contents

            if dev_info.nTLayerType == MvCC.MV_GIGE_DEVICE:
                gi = dev_info.SpecialInfo.stGigEInfo
                ip_int = gi.nCurrentIp
                ip = f"{ip_int & 0xff}." \
                     f"{(ip_int >> 8) & 0xff}." \
                     f"{(ip_int >> 16) & 0xff}." \
                     f"{(ip_int >> 24) & 0xff}"
                model = _decode_c_char_array(gi.chModelName)
                sn = _decode_c_char_array(gi.chSerialNumber)
                print(f"  [{i}] GigE  Model={model}  IP={ip}  SN={sn}")

            elif dev_info.nTLayerType == MvCC.MV_USB_DEVICE:
                ui = dev_info.SpecialInfo.stUsb3VInfo
                model = _decode_c_char_array(ui.chModelName)
                sn = _decode_c_char_array(ui.chSerialNumber)
                print(f"  [{i}] USB   Model={model}  SN={sn}")

            else:
                print(f"  [{i}] Unknown layer type: {dev_info.nTLayerType}")

    # ---------- Internal ----------

    def _open_device(self, index: int):
        lst = MvCC.MV_CC_DEVICE_INFO_LIST()
        ret = MvCC.MvCamera.MV_CC_EnumDevices(
            MvCC.MV_GIGE_DEVICE | MvCC.MV_USB_DEVICE, lst
        )
        if ret != 0 or lst.nDeviceNum == 0:
            raise RuntimeError("No Hik camera found")

        if index >= lst.nDeviceNum:
            raise RuntimeError(f"Device index {index} out of range, found {lst.nDeviceNum} devices")

        dev_info = ctypes.cast(
            lst.pDeviceInfo[index], ctypes.POINTER(MvCC.MV_CC_DEVICE_INFO)
        ).contents

        self.cam = MvCC.MvCamera()

        # 1) 建 handle
        ret = self.cam.MV_CC_CreateHandle(dev_info)
        if ret != 0:
            self.cam = None
            raise RuntimeError(f"CreateHandle 失敗 0x{ret:x}")

        # 2) 開裝置
        ret = self.cam.MV_CC_OpenDevice(MvCC.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.cam = None
            raise RuntimeError(f"OpenDevice 失敗 0x{ret:x}")

        # 3) 設定 PixelFormat
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", MvCC.PixelType_Gvsp_Mono8)
        if ret != 0:
            print(f"[HikCamera] 警告：設定 PixelFormat 失敗 0x{ret:x}")

        # 4) 關閉觸發
        self.cam.MV_CC_SetEnumValue("TriggerMode", MvCC.MV_TRIGGER_MODE_OFF)

        # 5) AcquisitionStart
        self.cam.MV_CC_SetCommandValue("AcquisitionStart")

        # 6) GigE 封包優化
        if dev_info.nTLayerType == MvCC.MV_GIGE_DEVICE:
            pkt = self.cam.MV_CC_GetOptimalPacketSize()
            if pkt > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt)

        self.cam.MV_CC_SetImageNodeNum(10)

        # 7) StartGrabbing
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"StartGrabbing 失敗 0x{ret:x}")
        
        # 只拿 Mono8（你已經有設）
        self.cam.MV_CC_SetEnumValue("PixelFormat", MvCC.PixelType_Gvsp_Mono8)

        # 啟用 FrameRate 限制，先降到 3~5 fps
        # self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        # self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 3.0)


        self.is_grabbing = True

        import time
        time.sleep(0.1)



    def _get_frame(self, timeout_ms: int = 3000) -> Image.Image:
        if not self.cam:
            raise RuntimeError("Camera not opened")

        if not self.is_grabbing:
            print("[HikCamera] 尚未開始抓圖，呼叫 StartGrabbing()")
            self.cam.MV_CC_StartGrabbing()
            self.is_grabbing = True

        frame = MvCC.MV_FRAME_OUT()
        ret = self.cam.MV_CC_GetImageBuffer(frame, timeout_ms)
        if ret != 0:
            # 這裡先印出 ret 再丟錯誤
            print(f"[HikCamera] GetImageBuffer 失敗，ret = 0x{ret:x}")
            raise RuntimeError(f"GetImageBuffer 失敗 0x{ret:x}")

        try:
            buf_type = frame.stFrameInfo.enPixelType
            width    = frame.stFrameInfo.nWidth
            height   = frame.stFrameInfo.nHeight
            size     = frame.stFrameInfo.nFrameLen

            # 把 frame 相關資訊印出來
            print(
                f"[HikCamera] frame info: "
                f"pixel_type=0x{buf_type:x}, width={width}, height={height}, size={size}"
            )

            # 如果已經是 Mono8，就直接用
            if buf_type == MvCC.PixelType_Gvsp_Mono8:
                print("[HikCamera] 影像已是 Mono8，不需要轉換")
                buf = (ctypes.c_ubyte * size)()
                ctypes.memmove(buf, frame.pBufAddr, size)
                return Image.frombytes("L", (width, height), buf, "raw")

            # 否則用 SDK 轉成 Mono8
            print("[HikCamera] 影像不是 Mono8，呼叫 ConvertPixelType() 轉換…")
            dst_size = width * height
            dst_buf = (ctypes.c_ubyte * dst_size)()

            convert_param = MvCC.MV_CC_PIXEL_CONVERT_PARAM()
            convert_param.nWidth           = width
            convert_param.nHeight          = height
            convert_param.pSrcData         = frame.pBufAddr
            convert_param.nSrcDataLen      = size
            convert_param.enSrcPixelType   = buf_type
            convert_param.enDstPixelType   = MvCC.PixelType_Gvsp_Mono8
            convert_param.pDstBuffer       = dst_buf
            convert_param.nDstBufferSize   = dst_size

            ret = self.cam.MV_CC_ConvertPixelType(convert_param)
            # ★ 這裡把 ret 印出來
            print(f"[HikCamera] ConvertPixelType ret = 0x{ret:x}")

            if ret != 0:
                raise RuntimeError(f"ConvertPixelType 失敗 0x{ret:x}")

            return Image.frombytes("L", (width, height), dst_buf, "raw")

        finally:
            self.cam.MV_CC_FreeImageBuffer(frame)


