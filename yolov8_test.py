from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import time

# ---------- 計時＋記錄 ----------
def run_inference_with_timer(image_filename, func, *args, **kwargs):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "inference.log")

    start_time = datetime.now()
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    start = time.time()

    result = func(*args, **kwargs)

    end = time.time()
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    duration = round(end - start, 3)

    log_message = (
        f"[{image_filename}]\n"
        f"開始時間: {start_time_str}\n"
        f"結束時間: {end_time_str}\n"
        f"耗時: {duration} 秒\n"
        f"{'-'*40}\n"
    )

    print(f"🕒 推論 {image_filename} 耗時：{duration} 秒")

    with open(log_path, "a") as f:
        f.write(log_message)

    return result

# ---------- 主流程 ----------
def process_images_with_yolo(model_path, image_folder, output_folder):
    """
    使用 YOLO 模型處理資料夾內所有圖片，並保存裁切圖像（以原圖座標為準）

    :param model_path: YOLO 模型權重路徑
    :param image_folder: 圖片來源資料夾
    :param output_folder: 處理後圖片輸出資料夾
    """
    model = YOLO(model_path)
    print(f"模型目前使用的設備：{model.device}")
    os.makedirs(output_folder, exist_ok=True)

    for image_filename in os.listdir(image_folder):
        if not image_filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(image_folder, image_filename)
        image = cv2.imread(image_path)
        h_img, w_img = image.shape[:2]

        # 推論 + 計時
        results = run_inference_with_timer(image_filename, model, image_path)

        for result in results:
            obb = result.obb

            if obb.xyxy.shape[0] == 0:
                print(f"⚠️ {image_filename} 沒有偵測結果，跳過處理")
                with open("logs/inference.log", "a") as f:
                    f.write(f"[{image_filename}] ❌ 沒有偵測結果，已跳過\n{'-'*40}\n")
                continue

            # 取第一個框示範（如要全部框請加迴圈）
            x1, y1, x2, y2 = obb.xyxy.cpu().numpy()[0].astype(int)

            # 邊界防呆
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                print(f"⚠️ {image_filename} 裁切框過小，跳過")
                continue

            # 原圖裁切（畫質保留）
            cropped_image = image[y1:y2, x1:x2]

            output_name = os.path.splitext(image_filename)[0] + "_crop.png"
            output_path = os.path.join(output_folder, output_name)
            cv2.imwrite(output_path, cropped_image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            print(f"✅ 已保存處理後圖片到: {output_path}")

    print("📦 所有圖片處理完成！")

# ✅ 執行入口
if __name__ == "__main__": 
    model_path = "/app/obb_runs/yolov8n-obb2/weights/best.pt"
    image_folder = "/app/dataset/YOLO_seg/Train/images"
    output_folder = "/app/inference_image"

    process_images_with_yolo(model_path, image_folder, output_folder)
