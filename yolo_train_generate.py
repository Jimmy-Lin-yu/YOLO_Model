import os
import time
import glob
import yaml
import torch
import random
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime
from ultralytics import YOLO

# os.environ["TORCH_FORCE_DISABLE_INFERENCE_MODE"] = "1"

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


# ---------- Model YAML 讀寫 ----------
def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )


def print_layers(model_cfg: dict):
    print("\n===== backbone =====")
    for i, layer in enumerate(model_cfg.get("backbone", [])):
        print(f"[B{i}] {layer}")

    print("\n===== head =====")
    for i, layer in enumerate(model_cfg.get("head", [])):
        print(f"[H{i}] {layer}")
    print()


def update_model_yaml(
    template_yaml_path: str,
    output_yaml_path: str,
    nc: int = None,
    scales: dict = None,
    backbone_updates: dict = None,
    head_updates: dict = None
):
    """
    backbone_updates / head_updates 格式：
    {
        layer_index: {
            "from": ...,
            "repeats": ...,
            "module": ...,
            "args": [...]
        }
    }

    例如：
    backbone_updates = {
        2: {"repeats": 3},
        7: {"args": [768, 3, 2]}
    }
    """

    cfg = load_yaml(template_yaml_path)
    new_cfg = deepcopy(cfg)

    if nc is not None:
        new_cfg["nc"] = int(nc)

    if scales is not None:
        new_cfg["scales"] = scales

    def _apply_updates(section_name: str, updates: dict):
        if not updates:
            return

        section = new_cfg.get(section_name, [])
        for idx, patch in updates.items():
            if idx < 0 or idx >= len(section):
                raise IndexError(f"{section_name}[{idx}] 超出範圍，該 section 只有 {len(section)} 層")

            old_layer = section[idx]
            # YOLO layer format: [from, repeats, module, args]
            new_layer = deepcopy(old_layer)

            if "from" in patch:
                new_layer[0] = patch["from"]
            if "repeats" in patch:
                new_layer[1] = patch["repeats"]
            if "module" in patch:
                new_layer[2] = patch["module"]
            if "args" in patch:
                new_layer[3] = patch["args"]

            section[idx] = new_layer
            print(f"🛠 已更新 {section_name}[{idx}]")
            print(f"   舊: {old_layer}")
            print(f"   新: {new_layer}")

        new_cfg[section_name] = section

    _apply_updates("backbone", backbone_updates)
    _apply_updates("head", head_updates)

    save_yaml(new_cfg, output_yaml_path)
    print(f"✅ 已輸出修改後的 model yaml：{output_yaml_path}")
    return output_yaml_path


# ---------- 主流程 ----------
def main():
    # 裝置與種子
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {'cuda:0' if device == 0 else 'cpu'}")
    set_seed(42)

    # ========= 你主要改這裡 =========
    dataset_yaml = "/app/dataset/YOLO_seg/dataset.yaml"

    # 原始官方 / 你自己的 model yaml 模板
    template_model_yaml = "/app/models/yolo11-seg-template.yaml"

    # 輸出修改後的 yaml
    modified_model_yaml = "/app/models/generated/yolo11-custom-seg.yaml"

    # 訓練參數
    epochs = 250
    imgsz = 1024
    conf = 0.03
    iou = 0.45

    # 你的資料集類別數
    custom_nc = 1

    # 你要不要真的改架構
    # backbone / head 的 index 是該 section 內的索引，不是全域索引
    backbone_updates = {
        # 範例1：把 backbone 第2層 repeats 從 2 改成 3
        # 2: {"repeats": 3},

        # 範例2：把 backbone 第7層 Conv 輸出通道改成 768
        # 原本可能是 [1024, 3, 2]
        # 這種改法會影響後面接續層，請自己確認通道相容
        # 7: {"args": [768, 3, 2]},
    }

    head_updates = {
        # 範例：把 head 第2層 C3k2 repeats 改成 3
        # 2: {"repeats": 3},

        # 如果你真的要改最後 segmentation head，也能這樣改
        # 但這最容易壞，先不要亂動
        # 10: {"args": [custom_nc, 32, 256, 512, True]},
    }

    # 是否使用預訓練權重
    use_pretrained_pt = False
    pretrained_pt_path = "yolo11n-seg.pt"
    # ========= 你主要改這裡 =========

    # 依賴 & 健檢
    ensure_pycocotools()
    assert os.path.isfile(dataset_yaml), f"找不到 dataset.yaml：{dataset_yaml}"
    assert os.path.isfile(template_model_yaml), f"找不到 model yaml：{template_model_yaml}"

    assert_no_background_class(dataset_yaml)
    quick_dataset_sanity(dataset_yaml)

    # 先列印原始 layer 給你看
    raw_cfg = load_yaml(template_model_yaml)
    print("📄 原始 model yaml layer：")
    print_layers(raw_cfg)

    # 產生修改後的 model yaml
    update_model_yaml(
        template_yaml_path=template_model_yaml,
        output_yaml_path=modified_model_yaml,
        nc=custom_nc,
        backbone_updates=backbone_updates,
        head_updates=head_updates
    )

    # 再列印修改後 layer
    new_cfg = load_yaml(modified_model_yaml)
    print("📄 修改後 model yaml layer：")
    print_layers(new_cfg)

    # 建立模型
    if use_pretrained_pt:
        print(f"📦 嘗試載入預訓練權重：{pretrained_pt_path}")
        model = YOLO(pretrained_pt_path)
        # 注意：如果你要真正套用「改過的結構」，單純 YOLO(pt) 不會吃你新 yaml 結構。
        # 大改架構時建議直接用 YOLO(modified_model_yaml)
        print("⚠️ 你目前是用 .pt 載入，若已改動架構，建議改成 YOLO(modified_model_yaml)")
    else:
        print(f"🧠 以自訂 YAML 建立模型：{modified_model_yaml}")
        model = YOLO(modified_model_yaml)

    # 訓練
    run_with_timer(
        "訓練自訂 segmentation 模型",
        model.train,
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=-1,
        device=device,
        project="seg_runs",
        name="y11_custom_seg",
        verbose=True
    )

    # CUDA / cuDNN 狀態
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    print("cuDNN enabled:", torch.backends.cudnn.enabled)
    print("cuDNN benchmark:", torch.backends.cudnn.benchmark)
    print("cuDNN deterministic:", torch.backends.cudnn.deterministic)


if __name__ == "__main__":
    main()