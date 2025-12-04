import os
import time
import glob
import yaml
import torch
import random
import subprocess
from datetime import datetime
from ultralytics import YOLO

# os.environ["TORCH_FORCE_DISABLE_INFERENCE_MODE"] = "1"   # 避免 Ultralytics 自動啟用 inference_mode

# ---------- 公用：計時器 ----------
def run_with_timer(description, func, *args, log_file_path=None, **kwargs):
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"🟢 [{description}] 開始時間：{start_time_str}")
    t0 = time.time()
    result = func(*args, **kwargs)
    t1 = time.time()
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    minutes, seconds = divmod(int(t1 - t0), 60)
    time_used = f"{minutes} 分 {seconds} 秒"
    print(f"✅ [{description}] 結束時間：{end_time_str}")
    print(f"⏱️ 總共花費時間：{time_used}\n")

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    if not log_file_path:
        log_file_path = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    with open(log_file_path, 'a', encoding="utf-8") as f:
        f.write(f"[{description}]\n開始時間：{start_time_str}\n結束時間：{end_time_str}\n總共花費時間：{time_used}\n")
        f.write("-" * 40 + "\n")
    return result

# ---------- 可重現性 ----------
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 已設定 random seed = {seed}（cudnn deterministic=True）")

# ---------- 依賴：先確保 pycocotools 存在 ----------
def ensure_pycocotools():
    try:
        import pycocotools  # noqa: F401
        print("✅ 已安裝 pycocotools")
    except Exception:
        print("📦 安裝 pycocotools 中…")
        subprocess.run(
            ["python3", "-m", "pip", "install", "-q", "pycocotools>=2.0.6"],
            check=True
        )
        print("✅ pycocotools 安裝完成")

# ---------- 資料/設定健檢 ----------
def assert_no_background_class(dataset_yaml_path: str):
    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names")
    if isinstance(names, dict):
        name_list = [names[k] for k in sorted(names.keys())]
    else:
        name_list = list(names)
    lowered = [str(n).strip().lower() for n in name_list]
    forbidden = {"background", "bg", "ok"}
    hit = [n for n in lowered if n in forbidden]
    if hit:
        raise ValueError(
            f"❌ dataset.yaml 的 names 內含不應存在的類別：{hit}\n"
            f"請刪除 background/ok，僅保留缺陷類（例如：['NG']）。\n"
            f"OK 影像應該沒有 label 檔。"
        )
    print(f"✅ 已檢查 dataset.yaml：names = {name_list}（未包含 background/ok）")

def quick_dataset_sanity(dataset_yaml_path: str):
    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.get("path", "")
    train_rel = cfg.get("train")
    val_rel = cfg.get("val") or cfg.get("validation") or cfg.get("val_dir")

    def _count_pair(img_dir_rel):
        img_dir = os.path.join(base, img_dir_rel)
        lbl_dir = img_dir.replace(os.sep + "images", os.sep + "labels")
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        n_img = len(img_paths)
        n_with_lbl = 0
        n_without_lbl = 0
        for p in img_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            lp = os.path.join(lbl_dir, stem + ".txt")
            if os.path.exists(lp) and os.path.getsize(lp) > 0:
                n_with_lbl += 1
            else:
                n_without_lbl += 1
        return n_img, n_with_lbl, n_without_lbl

    for split in [("train", train_rel), ("val", val_rel)]:
        if split[1]:
            n_img, n_with, n_without = _count_pair(split[1])
            print(f"📦 {split[0]}：總影像 {n_img}｜有標註(NG) {n_with} ｜無標註(OK) {n_without}")
    print("💡 無標註影像即為 OK（影像等級為 OK）。")

# ---------- 主流程 ----------
def main():
    # 裝置與種子
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {'cuda:0' if device==0 else 'cpu'}")
    set_seed(42)

    # 可調參數
    dataset_yaml = "/app/dataset/YOLO_seg/dataset.yaml"
    epochs = 250
    imgsz = 1024
    conf = 0.03
    iou = 0.45

    # 依賴 & 健檢
    ensure_pycocotools()
    assert os.path.isfile(dataset_yaml), f"找不到 dataset.yaml：{dataset_yaml}"
    assert_no_background_class(dataset_yaml)
    quick_dataset_sanity(dataset_yaml)

    # 模型（用預訓練權重）
    model = YOLO("yolo11s-seg.pt")

    # 訓練
    run_with_timer(
        "訓練 yolo11s-seg.pt 模型",
        model.train,
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=-1,
        device=0,
        project="seg_runs",
        name="y11sseg_V2_251203",
        # mosaic=1.0, mixup=0.0, close_mosaic=10,
        # translate=0.2, scale=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        verbose=True
    )

if __name__ == "__main__":
    main()
