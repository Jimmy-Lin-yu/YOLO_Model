# control_panel.py
import io, requests, gradio as gr
from PIL import Image
from light_controller import LightController      # 仍在本機
# ------------------------------------------------------------------
# 預設參數
DEFAULTS = {
    "api":      "http://192.168.0.100:8080",  # ← 依遠端相機服務實際位址修改
    "exposure": 800.0,  # μs
    "gain":       0.0,  # dB
    "dim":       350,
    "w":        1024,
    "r":           0,
    "g":           0,
    "b":        1024,
}

# ---------- 初始化本地打光機 ----------
try:
    light = LightController(com_port=8)   # ←請改成你的 COM 口
except Exception as e:
    raise SystemExit(f"打光機初始化失敗：{e}")

# ---------- 回呼：套用設定並拍照 ----------
def apply_and_shoot(api_base, exposure, gain, dim, w, r, g, b):
    """
    api_base : 例如 http://192.168.0.100:8080
    其餘參數：UI Slider 取得的數值
    """
    try:
        # 1) 本機打光
        light.set_dim_rgb(dim, w, r, g, b)

        # 2) REST API → 相機參數
        endpoint_param = f"{api_base.rstrip('/')}/set_cam_param"
        req = {"exposure": exposure, "gain": gain}
        requests.post(endpoint_param, json=req, timeout=3).raise_for_status()

        # 3) REST API → 拍照
        endpoint_snap = f"{api_base.rstrip('/')}/snapshot"
        resp = requests.post(endpoint_snap, timeout=10)
        resp.raise_for_status()

        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return img, "✅ 完成拍照並更新參數"
    except Exception as err:
        return None, f"❌ 失敗：{err}"

# ---------- Gradio UI ----------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🖥️ 工業相機 (遠端) + 打光機 (本地) 控制台")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🌐 相機 API 設定")
            api_box = gr.Textbox(value=DEFAULTS["api"], label="Camera API Base URL")

            gr.Markdown("### 📷 相機參數")
            exp  = gr.Slider(50, 200_000, value=DEFAULTS["exposure"], step=10,
                             label="Exposure (μs)")
            gain = gr.Slider(0, 24, value=DEFAULTS["gain"], step=0.1,
                             label="Gain (dB)")

            gr.Markdown("### 💡 打光參數")
            dim = gr.Slider(0, 1024, value=DEFAULTS["dim"], label="DIM")
            w   = gr.Slider(0, 1024, value=DEFAULTS["w"], label="White")
            r   = gr.Slider(0, 1024, value=DEFAULTS["r"], label="Red")
            g   = gr.Slider(0, 1024, value=DEFAULTS["g"], label="Green")
            b   = gr.Slider(0, 1024, value=DEFAULTS["b"], label="Blue")

            run_btn = gr.Button("🚀 套用並拍照", variant="primary")

        with gr.Column():
            img_out = gr.Image(label="最新影像", type="pil")
            msg_box = gr.Textbox(label="訊息", interactive=False)

    run_btn.click(
        fn=apply_and_shoot,
        inputs=[api_box, exp, gain, dim, w, r, g, b],
        outputs=[img_out, msg_box],
    )

    gr.Markdown(
        "#### 使用說明\n"
        "- **Camera API Base URL** 請填遠端相機服務位址，例如 `http://192.168.0.100:8080`\n"
        "- 相機端需先啟動 `cam_service.py`，並確保 `/set_cam_param`、`/snapshot` 端點可用\n"
        "- 如需重新偵測硬體，請重啟本程式"
    )

# ---------- 啟動 ----------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
