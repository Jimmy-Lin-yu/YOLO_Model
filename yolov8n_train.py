from ultralytics import YOLO
import time
from datetime import datetime

import time
import os
from datetime import datetime

def run_with_timer(description, func, *args, log_file_path=None, **kwargs):
    """
    執行 func 並紀錄花費時間，同時記錄 log 到檔案

    :param description: 描述這段操作，例如 '訓練 YOLO 模型'
    :param func: 要執行的函式
    :param args: 傳給 func 的參數
    :param kwargs: 傳給 func 的關鍵字參數
    :param log_file_path: 自訂 log 檔路徑（可選）
    :return: func 的回傳值
    """
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"🟢 [{description}] 開始時間：{start_time_str}")
    start = time.time()

    result = func(*args, **kwargs)

    end = time.time()
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    duration = end - start
    minutes, seconds = divmod(duration, 60)
    time_used = f"{int(minutes)} 分 {int(seconds)} 秒"

    print(f"✅ [{description}] 結束時間：{end_time_str}")
    print(f"⏱️ 總共花費時間：{time_used}\n")

    # 儲存 log
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    if not log_file_path:
        log_file_path = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

    with open(log_file_path, 'a') as log_file:
        log_file.write(f"[{description}]\n")
        log_file.write(f"開始時間：{start_time_str}\n")
        log_file.write(f"結束時間：{end_time_str}\n")
        log_file.write(f"總共花費時間：{time_used}\n")
        log_file.write("-" * 40 + "\n")

    return result


def main():
    # 載入模型設定（建立新的模型）
    model = YOLO("yolov8n-obb.yaml")  # 使用 OBB 模型結構
    print(f"模型目前使用的設備：{model.device}")

    # 訓練模型，並記錄花費時間
    results = run_with_timer(
        "訓練 YOLOv8 OBB 模型",
        model.train,
        data="/workspace/dataset/YOLO_seg/dataset.yaml",  # 資料集的 .yaml 設定路徑
        epochs=100,         # 訓練總次數
        imgsz=640,         # 輸入圖像大小
        batch=16,          # 批次大小（可依 GPU 調整）
        project="obb_runs",  # 存放訓練結果的資料夾
        name="yolov8n-obb",  # 任務名稱
        verbose=True         # 顯示詳細輸出
    )

    # optional: 顯示訓練結果圖
    results.plot()

if __name__ == "__main__":
    main()
