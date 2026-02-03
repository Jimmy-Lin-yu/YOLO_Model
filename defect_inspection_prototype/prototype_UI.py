# realtime_ui.py
import threading
from typing import Tuple

import gradio as gr
import numpy as np
import base64
import os

from pipeline_main import PipelineRuntime


# -------- paths (沿用你原本) --------
DEFECT_WEIGHT = "/app/best_251230.pt"
SENSOR_WEIGHT = "/app/best_target_capture.pt"


# -------- pipeline runtime (只留一份全域) --------
runtime = PipelineRuntime(
    num_cams=1,          # UI demo 顯示 1 支；你要 5 支就改 5
    target_fps=15,
    preview_cam_id=0,
    persist_dir="out_persist",
    serialize_per_cam=True,   # ✅ 確保：PLC(only) 完成後才進 next sensor
)

BLACK_SIZE = (640, 480)  # (W,H)


def _black_rgb() -> np.ndarray:
    w, h = BLACK_SIZE
    return np.zeros((h, w, 3), dtype=np.uint8)


def load_logo_base64(path: str = "evertech_logo.png") -> str:
    path = "/app/image.png"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Logo 檔案不存在：{path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


LOGO_BASE64 = load_logo_base64("evertech_logo.png")


# ---------- UI actions ----------
def toggle_camera(camera_on: bool) -> Tuple[bool, gr.Button, str]:
    msg_html = ""
    if not camera_on:
        try:
            runtime.start_camera(exposure_us=360.0, gain=8.0)
            camera_on = True
            cam_btn_update = gr.update(value="Camera : Open", variant="primary")
        except Exception as e:
            camera_on = False
            cam_btn_update = gr.update(value="Camera : Close", variant="secondary")
            msg_html = f'<span class="msg-error">相機錯誤：{e}</span>'
    else:
        runtime.stop_camera()
        camera_on = False
        cam_btn_update = gr.update(value="Camera : Close", variant="secondary")

    return camera_on, cam_btn_update, msg_html


def toggle_model(model_on: bool, camera_on: bool) -> Tuple[bool, gr.Button, str]:
    msg_html = ""

    if not model_on:
        if not camera_on:
            msg_html = '<span class="msg-warning">請先打開鏡頭</span>'
            return False, gr.update(value="Model : Close", variant="secondary"), msg_html

        try:
            runtime.start_model(
                sensor_weight_path=SENSOR_WEIGHT,
                defect_weight_path=DEFECT_WEIGHT,
                sensor_conf=0.9,
                sensor_iou=0.45,
                defect_conf=0.3,
                defect_iou=0.45,
                defect_imgsz= 1280,
                # defect_imgsz=None, 
            )
            model_on = True
            model_btn_update = gr.update(value="Model : Open", variant="primary")
        except Exception as e:
            model_on = False
            model_btn_update = gr.update(value="Model : Close", variant="secondary")
            msg_html = f'<span class="msg-error">模型啟動錯誤：{e}</span>'
    else:
        runtime.stop_model()
        model_on = False
        model_btn_update = gr.update(value="Model : Close", variant="secondary")

    return model_on, model_btn_update, msg_html


def get_ui_frame(camera_on: bool, model_on: bool) -> np.ndarray:
    if not camera_on:
        return _black_rgb()
    return runtime.get_ui_frame_rgb()


# ---------- Gradio UI (你的 CSS/版面完整保留) ----------
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

.brand-banner .title-main, .brand-banner .title-sub {
    font-family: "Microsoft JhengHei", "微軟正黑體", "Noto Sans TC", "PingFang TC", system-ui, sans-serif;
    font-size: 50px;
    font-weight: 800;
    letter-spacing: 0.04em;
}

.image-card, .control-card {
    background: var(--panel-bg);
    border-radius: 18px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.06);
    padding: 12px;
}

.image-card { min-height: 76vh; }

#main_image img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

#msg_area {
    min-height: 28px;
    text-align: center;
    font-size: 16px;
}

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

.status-button.secondary {
    background-color: #f0f0f0 !important;
    color: #333333 !important;
}

.status-button.primary {
    background-color: #99ff99 !important;
    color: #145c23 !important;
}

.main-row { width: 100%; margin: 0; }
"""


def build_demo() -> gr.Blocks:
    logo_html = f"""
    <div class="brand-banner">
        <img src="data:image/png;base64,{LOGO_BASE64}" alt="EVERTECH Logo" />
        <div class="brand-text">
            <div class="title-main">ET-Vision (Alpha)</div>
            <div class="title-sub">AOI x AI Defect Inspection Platform </div>
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
                        value=_black_rgb(),
                        label=None,
                        show_label=False,
                        type="numpy",   # ✅ 不用 PIL
                        height=800,
                        elem_id="main_image",
                    )

                with gr.Column(scale=2, elem_classes="control-card"):
                    cam_btn = gr.Button("Camera：Close", elem_classes="status-button", variant="secondary")
                    model_btn = gr.Button("Model：Close", elem_classes="status-button", variant="secondary")

            camera_state = gr.State(False)
            model_state = gr.State(False)

            cam_btn.click(fn=toggle_camera, inputs=[camera_state], outputs=[camera_state, cam_btn, msg_html])
            model_btn.click(fn=toggle_model, inputs=[model_state, camera_state], outputs=[model_state, model_btn, msg_html])

            timer = gr.Timer(0.001, active=True)
            timer.tick(fn=get_ui_frame, inputs=[camera_state, model_state], outputs=image_view)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue().launch(server_name="0.0.0.0", server_port=7787)
