
# # realtime_ui.py
import threading
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from hik_camera import HikCamera
from yolo_inference import YOLORealtimeInspector  # 換成你的檔名

import base64
import os

# --- 全域相機 / 模型物件與 lock ---
cam = None
cam_lock = threading.Lock()

yolo_inspector: YOLORealtimeInspector | None = None
yolo_lock = threading.Lock()

# 黑畫面尺寸
BLACK_SIZE = (640, 480)


def _black_frame() -> Image.Image:
    return Image.new("RGB", BLACK_SIZE, (0, 0, 0))


def load_logo_base64(path: str = "evertech_logo.png") -> str:
    """讀取 logo 檔案並轉成 base64 文字，給 HTML <img> 使用。"""
    path="/app/AxisControl_LightingAuto/evertech_logo.png"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Logo 檔案不存在：{path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

LOGO_BASE64 = load_logo_base64("evertech_logo.png")



# ---------- 控制邏輯 ----------

def toggle_camera(camera_on: bool) -> Tuple[bool, gr.Button, str]:
    """
    切換相機開關：
    - 回傳新的 camera_on 狀態
    - 相機按鈕的外觀更新 (gr.update)
    - 上方訊息 HTML
    """
    global cam
    msg_html = ""

    if not camera_on:
        # 要開啟相機
        try:
            with cam_lock:
                cam = HikCamera(dev_index=0)
                
                # 🔸 一開相機就直接設定曝光 / 增益
                cam.set_exposure_gain(3000.0, 8.0)
                #  ↑ 這兩個數字就是你要的「曝光時間 μs」跟「增益」

            camera_on = True
            cam_btn_update = gr.update(
                value="Camera : Open",
                variant="primary",
            )
        except Exception as e:  # noqa: BLE001
            camera_on = False
            cam = None
            cam_btn_update = gr.update(
                value="Camera:Close",
                variant="secondary",
            )
            msg_html = f'<span class="msg-error">相機錯誤：{e}</span>'
    else:
        # 要關閉相機
        with cam_lock:
            if cam is not None:
                cam.close()
                cam = None
        camera_on = False
        cam_btn_update = gr.update(
            value="Camera : Close",
            variant="secondary",
        )

    return camera_on, cam_btn_update, msg_html


def toggle_model(model_on: bool, camera_on: bool) -> Tuple[bool, gr.Button, str]:
    """
    切換模型開關：
    - 回傳新的 model_on 狀態
    - 模型按鈕的外觀更新 (gr.update)
    - 上方訊息 HTML
    """
    global yolo_inspector
    msg_html = ""

    if not model_on:
        # 要啟動模型：若尚未載入，這邊載入 YOLO 模型
        try:
            with yolo_lock:
                if yolo_inspector is None:
                    yolo_inspector = YOLORealtimeInspector.run_with_timer(
                        "載入YOLO模型",
                        YOLORealtimeInspector,
                        "/app/AxisControl_LightingAuto/best_251203.pt",   # ← 換成你的 best.pt 路徑
                        conf=0.2,
                        defect_classes=None  # 或 [0,1,...] 指定瑕疵 class
                    )
            model_on = True
            model_btn_update = gr.update(
                value="Model : Open",
                variant="primary",
            )
            if not camera_on:
                msg_html = '<span class="msg-warning">未偵測到畫面，請先打開鏡頭</span>'
        except Exception as e:  # noqa: BLE001
            model_on = False
            model_btn_update = gr.update(
                value="Model : Close",
                variant="secondary",
            )
            msg_html = f'<span class="msg-error">模型載入錯誤：{e}</span>'
    else:
        # 關閉模型（物件先保留，下次啟動不用重載）
        model_on = False
        model_btn_update = gr.update(
            value="Model : Close",
            variant="secondary",
        )

    return model_on, model_btn_update, msg_html


def stream_frame(camera_on: bool, model_on: bool) -> Image.Image:
    """
    給 Gradio Timer 連續呼叫：
    - 相機關閉：黑畫面
    - 相機開啟、模型未啟動：原始畫面
    - 相機開啟、模型啟動：YOLO 推論後的畫面（上面會有 Result: OK/NG）
    """
    global cam, yolo_inspector

    if not camera_on or cam is None:
        return _black_frame()

    # 取相機影像（PIL）
    with cam_lock:
        try:
            img_pil: Image.Image = cam.grab_frame()
        except Exception:  # noqa: BLE001
            return _black_frame()

    if not model_on or yolo_inspector is None:
        return img_pil

    # PIL (RGB) → numpy BGR
    frame_rgb = np.array(img_pil)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # YOLO 推論
    with yolo_lock:
        draw_bgr, info = yolo_inspector.infer_frame(frame_bgr)
        # info["status"], info["num_defect"] 要拿來做別的顯示可以再擴充

    # BGR → PIL (RGB)
    draw_rgb = cv2.cvtColor(draw_bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(draw_rgb)
    return out_img


def take_snapshot(camera_on: bool) -> str:
    """目前 UI 沒用到，可以留著備用。"""
    global cam
    if not camera_on or cam is None:
        return "相機未開啟，無法截圖"
    with cam_lock:
        try:
            path = cam.trigger_snapshot(out_dir="snapshots", prefix="shot")
        except Exception as e:  # noqa: BLE001
            return f"截圖失敗：{e}"
    return f"已儲存截圖：{path}"


# def set_default_exposure_gain(camera_on: bool) -> str:
#     """目前 UI 沒用到，可以留著備用。"""
#     global cam
#     if not camera_on or cam is None:
#         return "相機未開啟，無法設定曝光/增益"
#     with cam_lock:
#         try:
#             cam.set_exposure_gain(5000.0, 8.0)
#         except Exception as e:  # noqa: BLE001
#             return f"設定曝光/增益失敗：{e}"
#     return "已設定曝光 = 20000 μs, 增益 = 8.0"


# ---------- Gradio 介面 + 美編 ----------

CSS = """
:root {
    --app-bg: #f4f6fb;
    --panel-bg: #ffffff;
    --accent: #00a86b;
    --accent-soft: #e3fff1;
    --danger: #ff4d4f;
    --warning: #e6a100;
}

/* 讓 Gradio 本身容器吃滿整個螢幕寬，左右不留多的白邊 */
.gradio-container {
    max-width: 100% !important;
    padding: 0 8px !important;   /* 想更貼邊可以改成 0  */
    margin: 0 !important;
    box-sizing: border-box;
}

body {
    background: var(--app-bg);
}

/* 整體 wrapper 也用滿寬 */
.app-wrapper {
    width: 100%;
    margin: 0 auto;
}

/* ✅ 頂端品牌 Banner：Logo + 文字 */
.brand-banner {
    display: flex;
    align-items: center;
    justify-content: center;              /* ⬅ 讓整條 Logo＋文字水平置中 */
    gap: 150px;
    margin: 12px auto 4px auto;          /* ⬅ 左右 margin auto，一樣居中 */
    font-family: "Microsoft JhengHei", "Noto Sans TC", system-ui, sans-serif;
    text-align: center;                  /* ⬅ 文字也一起置中（避免偏左） */
    transform: translateX(-320px);       /* 負值往左，正值往右 */
}

.brand-banner img {
    height: 120px;
    width: auto;
}

.brand-banner,
.brand-banner .title-main,
.brand-banner .title-sub {
    font-family: "Microsoft JhengHei", "微軟正黑體",
                 "Noto Sans TC", "PingFang TC",
                 system-ui, sans-serif;
}

.brand-banner .title-main {
    font-size: 50px;
    font-weight: 800;
    letter-spacing: 0.04em;
}

.brand-banner .title-sub {
    font-size: 50px;
    font-weight: 800;
    letter-spacing: 0.03em;
}

/* 左右兩個 panel 卡片風格 */
.image-card, .control-card {
    background: var(--panel-bg);
    border-radius: 18px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.06);
    padding: 12px;
}

/* 讓影像區高度吃滿視窗大部分 */
.image-card {
    min-height: 76vh;
}

/* 影像真正的 <img> 填滿卡片 */
#main_image img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* 訊息區 */
#msg_area {
    min-height: 28px;
    text-align: center;
    font-size: 16px;
}

.msg-error {
    color: var(--danger);
    font-weight: 600;
}

.msg-warning {
    color: var(--warning);
    font-weight: 600;
}

/* 兩顆大開關按鈕 */
.status-button {
    width: 100%;
    border-radius: 16px !important;
    font-size: 28px !important;
    padding: 20px 8px !important;
    margin-bottom: 18px !important;
    border: 1px solid #bbbbbb !important;
    font-weight: 600 !important;
}

/* OFF 狀態 */
.status-button.secondary {
    background-color: #f0f0f0 !important;
    color: #333333 !important;
}

/* ON 狀態 */
.status-button.primary {
    background-color: #99ff99 !important;
    color: #145c23 !important;
}

/* 主 Row 拉滿寬、左右貼邊 */
.main-row {
    width: 100%;
    margin: 0;
}
"""
def build_demo() -> gr.Blocks:
    # 先組一段 logo + 標題的 HTML，嵌入 base64 圖片
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
            # ✅ 頂端品牌 Banner（沒有任何下載 / 分享按鈕）
            gr.HTML(logo_html)

            # 訊息列
            msg_html = gr.HTML("", elem_id="msg_area")

            # main-row 讓左右兩塊吃滿整個畫面，比例 9:2
            with gr.Row(equal_height=True, elem_classes="main-row"):
                # 左：影像區（大部分寬度）
                with gr.Column(scale=9, elem_classes="image-card"):
                    image_view = gr.Image(
                        value=_black_frame(),
                        label=None,
                        show_label=False,
                        type="pil",
                        height=800,          # 想再高可以繼續調
                        elem_id="main_image"
                    )

                # 右：控制區（窄一點，貼右邊）
                with gr.Column(scale=2, elem_classes="control-card"):
                    cam_btn = gr.Button(
                        "Camera：Close",
                        elem_classes="status-button",
                        variant="secondary",
                    )
                    model_btn = gr.Button(
                        "Model：Close",
                        elem_classes="status-button",
                        variant="secondary",
                    )

            # 狀態
            camera_state = gr.State(False)
            model_state = gr.State(False)

            cam_btn.click(
                fn=toggle_camera,
                inputs=[camera_state],
                outputs=[camera_state, cam_btn, msg_html],
            )

            model_btn.click(
                fn=toggle_model,
                inputs=[model_state, camera_state],
                outputs=[model_state, model_btn, msg_html],
            )

            # Timer 連續更新影像
            timer = gr.Timer(0.05, active=True)
            timer.tick(
                fn=stream_frame,
                inputs=[camera_state, model_state],
                outputs=image_view,
            )

    return demo




if __name__ == "__main__":
    demo = build_demo()
    demo.queue().launch(server_name="0.0.0.0", server_port=7777)