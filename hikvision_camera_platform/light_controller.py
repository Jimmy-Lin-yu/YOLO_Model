

# light_controller.py
from ctypes import CDLL, Structure, POINTER, c_int, c_char, byref
from pathlib import Path

SDK_PATH = "/app/AxisControl_LightingAuto/SW_SDK.dll"
dll = CDLL(SDK_PATH)

class _DevInfo(Structure):
    _fields_ = [
        ("UID", c_int * 6),
        ("address", c_int),
        ("productModel", c_char * 128),
        ("zoneSize", c_int),
        ("colorSize", c_int),
    ]

def _list2c(arr):
    c_arr = (c_int * len(arr))()
    for i, v in enumerate(arr):
        c_arr[i] = v
    return c_arr

class LightController:
    """1Z4C 光源：設定 DIM, W, R, G, B"""

    def __init__(self, com_port: int):
        self.addr = None
        self._open(com_port)

    # ---------- Public API ----------
    def set_dim_rgb(self, dim: int, w: int, r: int, g: int, b: int):
        colors = _list2c([w, r, g, b])
        ok = dll.setZoneAndColors(
            0, 1, dim, colors, self.addr, 10
        )
        if not ok:
            raise RuntimeError("setZoneAndColors failed")

    def close(self):
        # SDK 無 close；可在結束前設 dim=0 熄燈
        try:
            self.set_dim_rgb(0, 0, 0, 0, 0)
        except Exception:
            pass

    # ---------- Internal ----------
    def _open(self, com_no: int):
        dll.init.argtypes  = [c_int, POINTER(c_int)]
        dll.init.restype   = POINTER(_DevInfo)
        num = c_int()
        infos = dll.init(com_no, byref(num))
        if num.value == 0:
            raise RuntimeError("No light source found on COM%d" % com_no)

        self.addr = infos[0].address  # 直接用現成地址
