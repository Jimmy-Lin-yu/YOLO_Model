# realtime_ui.py
import threading
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from hik_camera import HikCamera
from yolo_inference import YOLORealtimeInspector

import base64
import os
import time
# Modbus TCP (Holding Registers)
from pymodbus.client import ModbusTcpClient


# -------------------------
# Global objects
# -------------------------
cam = None
cam_lock = threading.Lock()

yolo_inspector: YOLORealtimeInspector | None = None
yolo_lock = threading.Lock()

BLACK_SIZE = (640, 480)


def _black_frame() -> Image.Image:
    return Image.new("RGB", BLACK_SIZE, (0, 0, 0))


def load_logo_base64(path: str = "evertech_logo.png") -> str:
    path = "/app/AxisControl_LightingAuto/evertech_logo.png"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Logo 檔案不存在：{path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


LOGO_BASE64 = load_logo_base64("evertech_logo.png")


# ============================================================
# Modbus HR helper (cached connection)
# ============================================================
_modbus_lock = threading.Lock()
_modbus_client: ModbusTcpClient | None = None
_modbus_cfg: tuple[str, int, float] | None = None


def _get_modbus_client(host: str, port: int, timeout: float) -> ModbusTcpClient:
    global _modbus_client, _modbus_cfg
    with _modbus_lock:
        if _modbus_client is None or _modbus_cfg != (host, port, timeout):
            try:
                if _modbus_client is not None:
                    _modbus_client.close()
            except Exception:
                pass
            _modbus_client = ModbusTcpClient(host=host, port=port, timeout=timeout)
            _modbus_cfg = (host, port, timeout)

        if not _modbus_client.connect():
            raise ConnectionError(f"Modbus connect failed: {host}:{port}")
        return _modbus_client


# 你原本用 _modbus_lock，這邊用同一把就好；如果你真的想用 _modbus_io_lock，記得先定義它
_modbus_io_lock = threading.Lock()

def _hr_read_int(host: str, port: int, unit_id: int, hr_addr: int, timeout: float) -> int:
    with _modbus_io_lock:
        client = _get_modbus_client(host, port, timeout)

        # ✅ 不要傳 unit/slave（你的版本不吃）
        rr = client.read_holding_registers(address=hr_addr, count=1)

        if rr.isError():  # type: ignore[attr-defined]
            raise RuntimeError(f"read_holding_registers error: {rr}")
        return int(rr.registers[0])


def _hr_write_int(host: str, port: int, unit_id: int, hr_addr: int, value: int, timeout: float) -> None:
    with _modbus_io_lock:
        client = _get_modbus_client(host, port, timeout)

        # ✅ 不要傳 unit/slave（你的版本不吃）
        rr = client.write_register(address=hr_addr, value=int(value))

        if rr.isError():  # type: ignore[attr-defined]
            raise RuntimeError(f"write_register error: {rr}")


def _hr_read_bool(host: str, port: int, unit_id: int, hr_addr: int, timeout: float) -> bool:
    return _hr_read_int(host, port, unit_id, hr_addr, timeout) != 0


def _hr_write_bool(host: str, port: int, unit_id: int, hr_addr: int, value: bool, timeout: float) -> None:
    _hr_write_int(host, port, unit_id, hr_addr, 1 if value else 0, timeout)



# ============================================================
# Cycle state (CAPTURE_START ~ CAPTURE_END)
# ============================================================
cycle_lock = threading.Lock()
cycle_active = False
res_log: list[str] = []
ng_seen = False

# edge detection (previous PLC signals)
_plc_prev = {
    "capture_start": False,
    "capture_end": False,
    "robot_done": False,
}


def _flush_camera_buffer(n: int = 5) -> None:
    global cam
    if cam is None:
        return
    with cam_lock:
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


def _cycle_start(cfg: dict) -> None:
    global cycle_active, res_log, ng_seen
    with cycle_lock:
        cycle_active = True
        ng_seen = False
        res_log = []

    # 避免上一顆殘留的 RESULT_VALID
    try:
        _hr_write_bool(cfg["PLC_HOST"], cfg["PLC_PORT"], cfg["PLC_UNIT_ID"], cfg["RESULT_VALID_HR"], False, cfg["MODBUS_TIMEOUT"])
        _hr_write_int(cfg["PLC_HOST"], cfg["PLC_PORT"], cfg["PLC_UNIT_ID"], cfg["RESULT_CODE_HR"], 0, cfg["MODBUS_TIMEOUT"])
    except Exception:
        pass
    
    _clear_final_overlay() # 開新 cycle 時把舊 FINAL 文字清掉
    _flush_camera_buffer(cfg["FLUSH_N"])

def _append_cycle_log(cfg: dict, final_code: int, seq: list[str]) -> None:
    """
    每顆足粒結束（CAPTURE_END）時，寫一行 log：
    time, final_code, len, sequence
    """
    os.makedirs(cfg["CYCLE_LOG_DIR"], exist_ok=True)
    path = os.path.join(cfg["CYCLE_LOG_DIR"], cfg["CYCLE_LOG_NAME"])

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    final_str = "OK" if final_code == 1 else "NG" if final_code == 2 else str(final_code)
    seq_str = " ".join(seq)

    line = f"{ts} | FINAL={final_str} | N={len(seq)} | RES=[{seq_str}]\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _cycle_end_and_send_result(cfg: dict) -> None:
    """
    CAPTURE_END 上升沿：結束窗 → 統計結果 → 回寫 RESULT_VALID + RESULT_CODE
    """
    global cycle_active, ng_seen, res_log
    with cycle_lock:
        if not cycle_active:
            return
        cycle_active = False

        # ✅ 從 cfg 讀取代碼（方便你在 build_demo() 改）
        CODE_EMPTY = int(cfg["RESULT_CODE_EMPTY"])
        CODE_OK = int(cfg["RESULT_CODE_OK"])
        CODE_NG = int(cfg["RESULT_CODE_NG"])

        if len(res_log) == 0:
            final_code = CODE_EMPTY
        else:
            final_code = CODE_NG if ng_seen else CODE_OK
        
        seq_snapshot = list(res_log)

        # ✅ 設定 FINAL overlay（顯示 2 秒）
        if final_code == CODE_OK:
            _set_final_overlay("FINAL=OK", (0, 255, 0), duration_sec=2.0)
        elif final_code == CODE_NG:
            _set_final_overlay("FINAL=NG", (0, 0, 255), duration_sec=2.0)
        else:
            # 例如空窗：你也可以選擇不顯示，或顯示 FINAL=EMPTY
            _set_final_overlay("FINAL=EMPTY", (0, 255, 255), duration_sec=2.0)

    #（若你有寫檔 log 就照舊）
    try:
        _append_cycle_log(cfg, final_code=final_code, seq=seq_snapshot)
    except Exception:
        pass

    # 先寫 CODE 再拉 VALID
    _hr_write_int(cfg["PLC_HOST"], cfg["PLC_PORT"], cfg["PLC_UNIT_ID"], cfg["RESULT_CODE_HR"], final_code, cfg["MODBUS_TIMEOUT"])
    _hr_write_bool(cfg["PLC_HOST"], cfg["PLC_PORT"], cfg["PLC_UNIT_ID"], cfg["RESULT_VALID_HR"], True, cfg["MODBUS_TIMEOUT"])


def _robot_done_cleanup(cfg: dict) -> None:
    """
    ROBOT_DONE 上升沿：PLC 告知分流完成 → PC 清空 res / 清掉 RESULT_VALID，準備下一顆
    """
    global cycle_active, res_log, ng_seen
    with cycle_lock:
        cycle_active = False
        ng_seen = False
        res_log = []

    # 清 PLC 結果（避免下一顆誤讀）
    _hr_write_bool(cfg["PLC_HOST"], cfg["PLC_PORT"], cfg["PLC_UNIT_ID"], cfg["RESULT_VALID_HR"], False, cfg["MODBUS_TIMEOUT"])
    _hr_write_int(cfg["PLC_HOST"], cfg["PLC_PORT"], cfg["PLC_UNIT_ID"], cfg["RESULT_CODE_HR"], 0, cfg["MODBUS_TIMEOUT"])


def _poll_plc_edges(cfg: dict) -> tuple[bool, bool, bool]:
    """
    讀 PLC HR，回傳 (start_rise, end_rise, done_rise)
    """
    global _plc_prev

    cap_start = cap_end = robot_done = False
    try:
        cap_start = _hr_read_bool(cfg["PLC_HOST"], cfg["PLC_PORT"], cfg["PLC_UNIT_ID"], cfg["CAPTURE_START_HR"], cfg["MODBUS_TIMEOUT"])
        cap_end = _hr_read_bool(cfg["PLC_HOST"], cfg["PLC_PORT"], cfg["PLC_UNIT_ID"], cfg["CAPTURE_END_HR"], cfg["MODBUS_TIMEOUT"])
        robot_done = _hr_read_bool(cfg["PLC_HOST"], cfg["PLC_PORT"], cfg["PLC_UNIT_ID"], cfg["ROBOT_DONE_HR"], cfg["MODBUS_TIMEOUT"])
    except Exception:
        # PLC 連線/讀取短暫失敗時，不讓 UI 掛掉
        return (False, False, False)

    start_rise = cap_start and (not _plc_prev["capture_start"])
    end_rise = cap_end and (not _plc_prev["capture_end"])
    done_rise = robot_done and (not _plc_prev["robot_done"])

    _plc_prev["capture_start"] = cap_start
    _plc_prev["capture_end"] = cap_end
    _plc_prev["robot_done"] = robot_done

    return (start_rise, end_rise, done_rise)


def _set_pc_ready(cfg: dict, camera_on: bool, model_on: bool) -> None:
    """
    PC_READY = camera_on AND model_on
    """
    ready = bool(camera_on and model_on)
    _hr_write_bool(cfg["PLC_HOST"], cfg["PLC_PORT"], cfg["PLC_UNIT_ID"], cfg["PC_READY_HR"], ready, cfg["MODBUS_TIMEOUT"])


# ============================================================
# ✅ FINAL overlay (show FINAL=OK/NG for a short time)
# ============================================================
_final_lock = threading.Lock()
_final_text: str | None = None
_final_until: float = 0.0
_final_color_bgr: tuple[int, int, int] = (0, 255, 0)  # default green


def _set_final_overlay(text: str, color_bgr: tuple[int, int, int], duration_sec: float = 2.0) -> None:
    """設定「最終結果」疊字，顯示 duration_sec 秒後自動消失。"""
    global _final_text, _final_until, _final_color_bgr
    with _final_lock:
        _final_text = text
        _final_color_bgr = color_bgr
        _final_until = time.time() + float(duration_sec)


def _clear_final_overlay() -> None:
    global _final_text, _final_until
    with _final_lock:
        _final_text = None
        _final_until = 0.0


def _apply_final_overlay_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    若 FINAL overlay 還在有效時間內，就在 img_bgr 上疊字後回傳。
    位置：org = (w*0.22, h*0.92)
    """
    global _final_text, _final_until

    now = time.time()
    with _final_lock:
        if (_final_text is None) or (now > _final_until):
            # 超過時間就清掉
            _final_text = None
            _final_until = 0.0
            return img_bgr
        text = _final_text
        color = _final_color_bgr

    h, w = img_bgr.shape[:2]
    org = (int(w * 0.22), int(h * 0.92))

    # 讓字清楚一點（白底描邊 + 彩色字）
    cv2.putText(
        img_bgr,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        5,              # 字體大小
        (255, 255, 255),  # 白色描邊
        6,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_bgr,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        5,
        color,            # OK 綠 / NG 紅
        3,
        cv2.LINE_AA,
    )
    return img_bgr



# ============================================================
# UI control
# ============================================================
def toggle_camera(
    camera_on: bool,
    model_on: bool,
    exposure_us: float,
    gain: float,
    cfg: dict,
) -> Tuple[bool, gr.Button, str]:
    global cam
    msg_html = ""

    if not camera_on:
        try:
            with cam_lock:
                cam = HikCamera(dev_index=0)
                cam.set_exposure_gain(exposure_us, gain)

            camera_on = True
            cam_btn_update = gr.update(value="Camera：Open", variant="primary")

            # 更新 PC_READY
            try:
                _set_pc_ready(cfg, camera_on=camera_on, model_on=model_on)
            except Exception as e:
                msg_html = f'<span class="msg-warning">PC_READY 寫入 PLC 失敗：{e}</span>'

        except Exception as e:
            camera_on = False
            cam = None
            cam_btn_update = gr.update(value="Camera : Close", variant="secondary")
            msg_html = f'<span class="msg-error">相機錯誤：{e}</span>'
    else:
        with cam_lock:
            if cam is not None:
                cam.close()
                cam = None

        camera_on = False
        cam_btn_update = gr.update(value="Camera : Close", variant="secondary")

        # 關相機一定要 PC_READY=0
        try:
            _set_pc_ready(cfg, camera_on=camera_on, model_on=model_on)
        except Exception as e:
            msg_html = f'<span class="msg-warning">PC_READY 寫入 PLC 失敗：{e}</span>'

    return camera_on, cam_btn_update, msg_html


def toggle_model(
    model_on: bool,
    camera_on: bool,
    yolo_conf: float,
    cfg: dict,
) -> Tuple[bool, gr.Button, str]:
    global yolo_inspector
    msg_html = ""

    if not model_on:
        try:
            with yolo_lock:
                if yolo_inspector is None or cfg.get("FORCE_RELOAD_MODEL", False):
                    yolo_inspector = YOLORealtimeInspector.run_with_timer(
                        "載入YOLO模型",
                        YOLORealtimeInspector,
                        cfg["YOLO_WEIGHT_PATH"],
                        conf=yolo_conf,
                        defect_classes=None,
                    )
                else:
                    # 不重載也至少更新 conf（不然你調 YOLO_CONF 不會生效）
                    yolo_inspector.conf = yolo_conf

            model_on = True
            model_btn_update = gr.update(value="Model : Open", variant="primary")

            if not camera_on:
                msg_html = '<span class="msg-warning">未偵測到畫面，請先打開鏡頭</span>'

            # 更新 PC_READY
            try:
                _set_pc_ready(cfg, camera_on=camera_on, model_on=model_on)
            except Exception as e:
                msg_html = f'<span class="msg-warning">PC_READY 寫入 PLC 失敗：{e}</span>'

        except Exception as e:
            model_on = False
            model_btn_update = gr.update(value="Model : Close", variant="secondary")
            msg_html = f'<span class="msg-error">模型載入錯誤：{e}</span>'
    else:
        model_on = False
        model_btn_update = gr.update(value="Model : Close", variant="secondary")
        # 關模型 → PC_READY 依 camera_on 決定（這裡是 AND，所以會變 0）
        try:
            _set_pc_ready(cfg, camera_on=camera_on, model_on=model_on)
        except Exception as e:
            msg_html = f'<span class="msg-warning">PC_READY 寫入 PLC 失敗：{e}</span>'

    return model_on, model_btn_update, msg_html


def stream_frame(camera_on: bool, model_on: bool, cfg: dict) -> Image.Image:
    """
    Timer tick:
    - PLC HR 控制 window：CAPTURE_START ~ CAPTURE_END 是一顆足粒
    - window 內每偵推論，res_log 追加 OK/NG
    - 只要 window 內出現 NG -> final = NG
    - CAPTURE_END 上升沿回寫 RESULT_VALID + RESULT_CODE
    - ROBOT_DONE 上升沿清空 res_log，清 PLC 的 RESULT_VALID/RESULT_CODE
    """
    global cam, yolo_inspector, res_log, ng_seen, cycle_active

    # 1) poll PLC edges
    start_rise, end_rise, done_rise = _poll_plc_edges(cfg)

    if start_rise:
        # 只有在 camera+model ready 才開窗（避免空窗）
        if camera_on and model_on and (cam is not None) and (yolo_inspector is not None):
            _cycle_start(cfg)

    if end_rise:
        try:
            _cycle_end_and_send_result(cfg)
        except Exception:
            pass

    if done_rise:
        try:
            _robot_done_cleanup(cfg)
        except Exception:
            pass

    # 2) grab frame
    if not camera_on or cam is None:
        return _black_frame()

    with cam_lock:
        try:
            img_pil: Image.Image = cam.grab_frame()
        except Exception:
            return _black_frame()

    # 你要把 FINAL 疊字畫上去，用的是 OpenCV 的 cv2.putText，它只能畫在 numpy 的 BGR 圖上，所以每次只要你想「疊字」，就得做這個轉換流程：PIL (RGB) → numpy (RGB) → BGR → 画字 → RGB → PIL
    def _pil_with_final(img_pil: Image.Image) -> Image.Image:
        frame_rgb = np.array(img_pil)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = _apply_final_overlay_bgr(frame_bgr)
        frame_rgb2 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb2)


    # 3) infer
    if not model_on or yolo_inspector is None:
        return _pil_with_final(img_pil)
    with cycle_lock:
        active_now = bool(cycle_active)

    # 建議：只在 window 內推論，避免窗外結果混進 res
    if cfg["INFER_ONLY_WHEN_CYCLE_ACTIVE"] and (not active_now):
        return _pil_with_final(img_pil)
    
    frame_rgb = np.array(img_pil)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    with yolo_lock:
        draw_bgr, info = yolo_inspector.infer_frame(frame_bgr)

    status = str(info.get("status", "OK"))  # "OK" / "NG"

    # 4) window 內記錄 res_log
    if active_now:
        with cycle_lock:
            res_log.append(status)
            if status == "NG":
                ng_seen = True
            if len(res_log) > cfg["MAX_LOG_LEN"]:
                res_log[:] = res_log[-cfg["MAX_LOG_LEN"] :]

    draw_bgr = _apply_final_overlay_bgr(draw_bgr)
    draw_rgb = cv2.cvtColor(draw_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(draw_rgb)


# ============================================================
# UI layout
# ============================================================
CSS = """
:root {
    --app-bg: #f4f6fb;
    --panel-bg: #ffffff;
    --accent: #00a86b;
    --accent-soft: #e3fff1;
    --danger: #ff4d4f;
    --warning: #e6a100;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 8px !important;
    margin: 0 !important;
    box-sizing: border-box;
}

body { background: var(--app-bg); }

.app-wrapper { width: 100%; margin: 0 auto; }

.brand-banner {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 150px;
    margin: 12px auto 4px auto;
    font-family: "Microsoft JhengHei", "Noto Sans TC", system-ui, sans-serif;
    text-align: center;
    transform: translateX(-320px);
}

.brand-banner img { height: 120px; width: auto; }

.brand-banner, .brand-banner .title-main, .brand-banner .title-sub {
    font-family: "Microsoft JhengHei", "微軟正黑體","Noto Sans TC","PingFang TC",system-ui,sans-serif;
}

.brand-banner .title-main, .brand-banner .title-sub {
    font-size: 50px;
    font-weight: 800;
}

.image-card, .control-card {
    background: var(--panel-bg);
    border-radius: 18px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.06);
    padding: 12px;
}

.image-card { min-height: 76vh; }

#main_image img { width: 100%; height: 100%; object-fit: contain; }

#msg_area { min-height: 28px; text-align: center; font-size: 16px; }

.msg-error { color: var(--danger); font-weight: 600; }
.msg-warning { color: var(--warning); font-weight: 600; }

.status-button {
    width: 100%;
    border-radius: 16px !important;
    font-size: 28px !important;
    padding: 20px 8px !important;
    margin-bottom: 18px !important;
    border: 1px solid #bbbbbb !important;
    font-weight: 600 !important;
}

.status-button.secondary { background-color: #f0f0f0 !important; color: #333 !important; }
.status-button.primary { background-color: #99ff99 !important; color: #145c23 !important; }

.main-row { width: 100%; margin: 0; }
"""


def build_demo() -> gr.Blocks:
    # ============================================================
    # ✅ 你要方便改的參數（集中這裡）
    # ============================================================
    EXPOSURE_US = 3000.0
    GAIN = 8.0
    YOLO_CONF = 0.5
    TIMER_SEC = 0.05  #多久讀一次PLC

    # ============================================================
    # ✅ PLC HR 位址（全部 HR，0/1 表示狀態）
    # 注意：pymodbus 多數情況是 0-based address
    # ============================================================
    cfg = {
        "PLC_HOST": "192.168.0.222",
        "PLC_PORT": 502,
        "PLC_UNIT_ID": 1,
        "MODBUS_TIMEOUT": 0.2,

        # model weight
        "YOLO_WEIGHT_PATH": "/app/AxisControl_LightingAuto/best_251230.pt",

        # PLC → PC (HR)
        "CAPTURE_START_HR": 55,  # CAPTURE_START=1
        "CAPTURE_END_HR": 56,    # CAPTURE_END=1
        "ROBOT_DONE_HR": 57,     # ROBOT_DONE=1

        # PC → PLC (HR)
        "PC_READY_HR": 50,       # PC_READY = (camera && model)
        "RESULT_VALID_HR": 51,   # RESULT_VALID=1 表示結果可讀
        "RESULT_CODE_HR": 52,    # RESULT_CODE: 1=OK, 2=NG, 0=clear


        # RESULT_CODE mapping 
        "RESULT_CODE_EMPTY": 3,  # 無結果
        "RESULT_CODE_OK": 1,     # OK
        "RESULT_CODE_NG": 2,     # NG

        # window / logging strategy
        "FLUSH_N": 5,
        "MAX_LOG_LEN": 500,
        "INFER_ONLY_WHEN_CYCLE_ACTIVE": True,

        "CYCLE_LOG_DIR": "logs",
        "CYCLE_LOG_NAME": "cycle.log",

        "FORCE_RELOAD_MODEL": False,
    }

    logo_html = f"""
    <div class="brand-banner">
        <img src="data:image/png;base64,{LOGO_BASE64}" alt="EVERTECH Logo" />
        <div class="brand-text">
            <div class="title-main">ET-Vision (Alpha)</div>
            <div class="title-sub">AOI x AI Defect Inspection Platform </div> 
        </div>
    </div>
    """
    #<div class="title-sub">AOI x AI 即時瑕疵檢測系統</div> 

    with gr.Blocks(title="即時瑕疵檢測系統") as demo:
        gr.HTML(f"<style>{CSS}</style>")

        with gr.Column(elem_classes="app-wrapper"):
            gr.HTML(logo_html)
            msg_html = gr.HTML("", elem_id="msg_area")

            with gr.Row(equal_height=True, elem_classes="main-row"):
                with gr.Column(scale=9, elem_classes="image-card"):
                    image_view = gr.Image(
                        value=_black_frame(),
                        label=None,
                        show_label=False,
                        type="pil",
                        height=800,
                        elem_id="main_image",
                    )

                with gr.Column(scale=2, elem_classes="control-card"):
                    cam_btn = gr.Button(
                        # "相機：關閉中",
                        "Camera:Close",
                        elem_classes="status-button",
                        variant="secondary",
                    )
                    model_btn = gr.Button(
                        # "模型：未啟動",
                        "Model:Close",
                        elem_classes="status-button",
                        variant="secondary",
                    )

            camera_state = gr.State(False)
            model_state = gr.State(False)

            cam_btn.click(
                fn=lambda camera_on, model_on: toggle_camera(
                    camera_on=camera_on,
                    model_on=model_on,
                    exposure_us=EXPOSURE_US,
                    gain=GAIN,
                    cfg=cfg,
                ),
                inputs=[camera_state, model_state],
                outputs=[camera_state, cam_btn, msg_html],
            )

            model_btn.click(
                fn=lambda model_on, camera_on: toggle_model(
                    model_on=model_on,
                    camera_on=camera_on,
                    yolo_conf=YOLO_CONF,
                    cfg=cfg,
                ),
                inputs=[model_state, camera_state],
                outputs=[model_state, model_btn, msg_html],
            )

            timer = gr.Timer(TIMER_SEC, active=True)
            timer.tick(
                fn=lambda camera_on, model_on: stream_frame(camera_on, model_on, cfg),
                inputs=[camera_state, model_state],
                outputs=image_view,
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue().launch(server_name="0.0.0.0", server_port=7777)

