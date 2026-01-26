# ui.py
from __future__ import annotations

import os
import base64
import threading
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from hik_camera import HikCamera
from yolo_inference import YOLORealtimeInspector

from mode_basic import BasicMode
from mode_plc import PLCMode


# =========================
# Mode names (統一管理)
# =========================
MODE_AUTO = "AUTO"       # 原本 PLC
MODE_MANUAL = "MANUAL"   # 原本 BASIC


# =========================
# Shared runtime objects
# =========================
cam = None
cam_lock = threading.Lock()

yolo_inspector: YOLORealtimeInspector | None = None
yolo_lock = threading.Lock()

BLACK_SIZE = (640, 480)

# =========================
# Tunable parameters (集中改)
# =========================
EXPOSURE_US = 3000.0
GAIN = 8.0
YOLO_CONF = 0.2
TIMER_SEC = 0.05  # Timer tick interval (sec)

YOLO_WEIGHT_PATH = "/app/AxisControl_LightingAuto/best_251230.pt"


# =========================
# PLC cfg (給 PLCMode 用)  （名字保留不影響）
# =========================
PLC_CFG = {
    "PLC_HOST": "192.168.0.222",
    "PLC_PORT": 502,
    "PLC_UNIT_ID": 1,
    "MODBUS_TIMEOUT": 0.2,

    # PLC → PC (HR)
    "CAPTURE_START_HR": 55,
    "CAPTURE_END_HR": 56,
    "ROBOT_DONE_HR": 57,

    # PC → PLC (HR)
    "PC_READY_HR": 50,
    "RESULT_VALID_HR": 51,
    "RESULT_CODE_HR": 52,

    # RESULT_CODE mapping
    "RESULT_CODE_EMPTY": 3,
    "RESULT_CODE_OK": 1,
    "RESULT_CODE_NG": 2,

    # window / logging strategy
    "FLUSH_N": 5,
    "MAX_LOG_LEN": 500,
    "INFER_ONLY_WHEN_CYCLE_ACTIVE": True,

    "CYCLE_LOG_DIR": "logs",
    "CYCLE_LOG_NAME": "cycle.log",
}


# =========================
# Utils
# =========================
def _black_frame() -> Image.Image:
    return Image.new("RGB", BLACK_SIZE, (0, 0, 0))


def load_logo_base64(path: str = "evertech_logo.png") -> str:
    path = "/app/AxisControl_LightingAuto/evertech_logo.png"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Logo 檔案不存在：{path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


LOGO_BASE64 = load_logo_base64("evertech_logo.png")


# =========================
# Mode instances
# =========================
basic_mode = BasicMode(
    get_cam=lambda: cam,
    cam_lock=cam_lock,
    get_yolo=lambda: yolo_inspector,
    yolo_lock=yolo_lock,
    black_size=BLACK_SIZE,
)

plc_mode = PLCMode(
    cfg=PLC_CFG,
    get_cam=lambda: cam,
    cam_lock=cam_lock,
    get_yolo=lambda: yolo_inspector,
    yolo_lock=yolo_lock,
    black_size=BLACK_SIZE,
)


# =========================
# UI actions
# =========================
def toggle_camera(camera_on: bool, model_on: bool, mode: str) -> Tuple[bool, gr.Button, str]:
    """
    只管相機開關與 exposure/gain
    Auto 模式才會寫 PC_READY（避免 Manual 調試一直打 PLC）
    """
    global cam
    msg_html = ""

    if not camera_on:
        try:
            with cam_lock:
                cam = HikCamera(dev_index=0)
                cam.set_exposure_gain(EXPOSURE_US, GAIN)

            camera_on = True
            cam_btn_update = gr.update(value="Camera：Open", variant="primary")

        except Exception as e:
            camera_on = False
            cam = None
            cam_btn_update = gr.update(value="Camera：Close", variant="secondary")
            msg_html = f'<span class="msg-error">相機錯誤：{e}</span>'
    else:
        with cam_lock:
            if cam is not None:
                cam.close()
                cam = None

        camera_on = False
        cam_btn_update = gr.update(value="Camera：Close", variant="secondary")

    # 只有 Auto 模式才更新 PC_READY
    if mode == MODE_AUTO:
        try:
            plc_mode.set_pc_ready(camera_on=camera_on, model_on=model_on)
        except Exception as e:
            msg_html = f'<span class="msg-warning">PC_READY 寫入 PLC 失敗：{e}</span>'

    return camera_on, cam_btn_update, msg_html


def toggle_model(model_on: bool, camera_on: bool, mode: str) -> Tuple[bool, gr.Button, str]:
    """
    只管模型開關與載入/更新 conf
    """
    global yolo_inspector
    msg_html = ""

    if not model_on:
        try:
            with yolo_lock:
                if yolo_inspector is None:
                    yolo_inspector = YOLORealtimeInspector.run_with_timer(
                        "載入YOLO模型",
                        YOLORealtimeInspector,
                        YOLO_WEIGHT_PATH,
                        conf=YOLO_CONF,
                        defect_classes=None,
                    )
                else:
                    yolo_inspector.conf = YOLO_CONF

            model_on = True
            model_btn_update = gr.update(value="Model：Open", variant="primary")

            if not camera_on:
                msg_html = '<span class="msg-warning">未偵測到畫面，請先打開鏡頭</span>'

        except Exception as e:
            model_on = False
            model_btn_update = gr.update(value="Model：Close", variant="secondary")
            msg_html = f'<span class="msg-error">模型載入錯誤：{e}</span>'
    else:
        model_on = False
        model_btn_update = gr.update(value="Model：Close", variant="secondary")

    # 只有 Auto 模式才更新 PC_READY
    if mode == MODE_AUTO:
        try:
            plc_mode.set_pc_ready(camera_on=camera_on, model_on=model_on)
        except Exception as e:
            msg_html = f'<span class="msg-warning">PC_READY 寫入 PLC 失敗：{e}</span>'

    return model_on, model_btn_update, msg_html


def toggle_mode(current_mode: str, camera_on: bool, model_on: bool) -> Tuple[str, gr.Button, str]:
    """
    右上角按鈕：Manual <-> Auto
    - Auto：PLC/Modbus window + 回寫
    - Manual：純相機 + YOLO
    """
    if current_mode == MODE_AUTO:
        # Auto -> Manual：把 PLC 狀態收乾淨
        try:
            plc_mode.on_exit()
        except Exception:
            pass
        new_mode = MODE_MANUAL
    else:
        # Manual -> Auto：初始化
        try:
            plc_mode.on_enter(camera_on=camera_on, model_on=model_on)
        except Exception:
            pass
        new_mode = MODE_AUTO

    btn_update = gr.update(
        value=f"Mode : {'Auto' if new_mode == MODE_AUTO else 'Manual'}",
        variant="primary" if new_mode == MODE_AUTO else "secondary",
    )
    msg_html = f'<span class="msg-warning">Mode：{"Auto" if new_mode == MODE_AUTO else "Manual"}</span>'
    return new_mode, btn_update, msg_html


def stream_router(camera_on: bool, model_on: bool, mode: str) -> Image.Image:
    """
    Timer 永遠呼叫這個，內部再分流到兩個 mode class
    """
    if mode == MODE_AUTO:
        return plc_mode.on_tick(camera_on=camera_on, model_on=model_on)
    return basic_mode.on_tick(camera_on=camera_on, model_on=model_on)


# =========================
# UI layout + CSS
# =========================
CSS = """
:root {
    --app-bg: #f4f6fb;
    --panel-bg: #ffffff;
    --accent: #00a86b;
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

/* ✅ 右上角 mode button：外層容器也要 fixed，才不會在底部留一條 */
#mode_btn{
    position: fixed !important;
    top: 14px !important;
    right: 14px !important;
    z-index: 99999 !important;

    width: auto !important;
    min-width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;

    /* ✅ 強制改掉 Gradio theme 的按鈕顏色（primary/secondary 都一起改） */
    --button-primary-background-fill: #1677ff;
    --button-primary-background-fill-hover: #0958d9;
    --button-primary-text-color: #ffffff;

    --button-secondary-background-fill: #1677ff;
    --button-secondary-background-fill-hover: #0958d9;
    --button-secondary-text-color: #ffffff;
}

/* ✅ button 本體 */
#mode_btn button{
    background-color: #1677ff !important;
    background-image: none !important;
    color: #ffffff !important;
    border: 1px solid #1677ff !important;

    border-radius: 12px !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
    line-height: 1.1 !important;

    width: auto !important;
}

/* hover */
#mode_btn button:hover{
    background-color: #0958d9 !important;
    border-color: #0958d9 !important;
}

"""


def build_demo() -> gr.Blocks:
    logo_html = f"""
    <div class="brand-banner">
        <img src="data:image/png;base64,{LOGO_BASE64}" alt="EVERTECH Logo" />
        <div class="brand-text">
            <div class="title-main">ET-Vision (Alpha)</div>
            <div class="title-sub">AOI x AI Defect Inspection Platform</div>
        </div>
    </div>
    """

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
                    cam_btn = gr.Button("Camera：Close", elem_classes="status-button", variant="secondary")
                    model_btn = gr.Button("Model：Close", elem_classes="status-button", variant="secondary")

            # States
            camera_state = gr.State(False)
            model_state = gr.State(False)
            mode_state = gr.State(MODE_AUTO)  # 預設 Auto（原 PLC）

            # ✅ 右上角 mode button
            mode_btn = gr.Button("Mode : Auto", elem_id="mode_btn", variant="secondary")

            # Events
            cam_btn.click(
                fn=toggle_camera,
                inputs=[camera_state, model_state, mode_state],
                outputs=[camera_state, cam_btn, msg_html],
            )

            model_btn.click(
                fn=toggle_model,
                inputs=[model_state, camera_state, mode_state],
                outputs=[model_state, model_btn, msg_html],
            )

            mode_btn.click(
                fn=toggle_mode,
                inputs=[mode_state, camera_state, model_state],
                outputs=[mode_state, mode_btn, msg_html],
            )

            timer = gr.Timer(TIMER_SEC, active=True)
            timer.tick(
                fn=stream_router,
                inputs=[camera_state, model_state, mode_state],
                outputs=image_view,
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue().launch(server_name="0.0.0.0", server_port=7777)
