import os
import glob
import gc
import time
import json
import csv as _csv
from datetime import datetime

import cv2
import yaml
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 允許的影像副檔名
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ---------------------------
# 共用：簡易計時 + 日誌
# ---------------------------
def run_with_timer(tag, func, *args, **kwargs):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "inference.log")

    t0 = time.time()
    start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🟢 [{tag}] 開始：{start}")

    result = func(*args, **kwargs)

    t1 = time.time()
    end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dur = round(t1 - t0, 3)
    print(f"✅ [{tag}] 結束：{end}｜耗時 {dur} 秒\n")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{tag}]\n開始：{start}\n結束：{end}\n耗時：{dur} 秒\n{'-'*40}\n")
    return result


# ---------------------------
# 工具：由 dataset.yaml 取得 val 影像/標註路徑
# ---------------------------
def get_val_dirs_from_yaml(data_yaml):
    cfg = yaml.safe_load(open(data_yaml, 'r', encoding='utf-8'))
    base = cfg.get('path', '')
    img_dir = os.path.join(base, cfg.get('val') or cfg.get('validation') or cfg.get('val_dir'))
    # img_dir = os.path.join(base, cfg.get('train') or cfg.get('train') or cfg.get('train_dir'))
    # 若要改用 train，就把上面那行改掉
    assert img_dir and os.path.isdir(img_dir), f"val images 不存在：{img_dir}"
    lbl_dir = img_dir.replace(os.sep + 'images', os.sep + 'labels')
    assert os.path.isdir(lbl_dir), f"val labels 不存在：{lbl_dir}"
    return img_dir, lbl_dir


# ---------------------------
# def1：把「模型預測」畫成四角點（每個框四個點）
# ---------------------------
def draw_pred_four_points(image_bgr, result, color=(0, 255, 0), r=6, thick=-1, with_score=True):
    canvas = image_bgr.copy()

    xyxy = None
    conf = None
    cls = None
    if getattr(result, "boxes", None) is not None:
        b = result.boxes
        if hasattr(b, "xyxy") and b.xyxy is not None:
            xyxy = b.xyxy.detach().cpu().numpy().astype(int)
        elif getattr(b, "data", None) is not None:
            xyxy = b.data[:, :4].detach().cpu().numpy().astype(int)
        if getattr(b, "conf", None) is not None:
            conf = b.conf.detach().cpu().numpy()
        if getattr(b, "cls", None) is not None:
            cls = b.cls.detach().cpu().numpy()

    # 若 box 為空但有 masks，就用外接框
    if (xyxy is None or len(xyxy) == 0) and getattr(result, "masks", None) is not None:
        m = result.masks
        if getattr(m, "xy", None) is not None and len(m.xy):
            xyxy = []
            for poly in m.xy:
                xs = poly[:, 0]
                ys = poly[:, 1]
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                xyxy.append([x1, y1, x2, y2])
            xyxy = np.array(xyxy, dtype=int)

    if xyxy is None or len(xyxy) == 0:
        return canvas

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        for (x, y) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            cv2.circle(canvas, (x, y), r, color, thick)
        if with_score and conf is not None:
            label = f"{(int(cls[i]) if cls is not None else 0)}:{conf[i]:.2f}"
            cv2.putText(canvas, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return canvas


# ---------------------------
# 讀取 per-instance attrs：labels/../attrs_instances/<stem>.json
# （若找不到，退回 labels/../attrs/<stem>.json）
# ---------------------------
def _load_instance_buckets(lbl_dir, image_path):
    """
    回傳一個 list，順序要跟 YOLO txt 每行對應：
        ["0.2", "0.15", "0.1", ...]
    支援幾種格式：
      1) ["0.2", "0.15", ...]
      2) [{"bucket": "0.2"}, {"bucket": "0.15"}, ...]
      3) [{"attr": ">0.2", "bbox": [...], "label": "NG"}, ...]  ← 你現在的格式
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    base_dir = os.path.abspath(os.path.join(lbl_dir, os.pardir))

    # 優先用 attrs_instances，其次 attrs（相容舊資料）
    for sub in ("attrs_instances", "attrs"):
        jf = os.path.join(base_dir, sub, stem + ".json")
        if os.path.isfile(jf):
            try:
                data = json.load(open(jf, "r", encoding="utf-8"))
            except Exception:
                return []
            buckets = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        # 直接是一個 bucket 字串
                        buckets.append(item)
                    elif isinstance(item, dict):
                        # 舊版：用 "bucket"
                        if "bucket" in item:
                            buckets.append(str(item["bucket"]))
                        # 你現在這版：用 "attr"
                        elif "attr" in item:
                            buckets.append(str(item["attr"]))
            return buckets
    return []


# ---------------------------
# 由 GT label 在圖上畫四角點 + attrs_instances
# ---------------------------
def draw_four_points_from_gt_with_attrs(image_bgr, image_path, lbl_dir,
                                        point_color=(255, 0, 0),
                                        attr_color=(255, 0, 0),
                                        r=6, thick=-1):
    """
    中間圖：
      - 畫出 GT 四角點
      - 每一顆瑕疵顯示對應的 attrs_instances（例如 NG | >0.2）
    """
    canvas = image_bgr.copy()
    stem = os.path.splitext(os.path.basename(image_path))[0]
    lbl_path = os.path.join(lbl_dir, stem + ".txt")
    if (not os.path.exists(lbl_path)) or os.path.getsize(lbl_path) == 0:
        return canvas

    H, W = canvas.shape[:2]

    # 讀取 per-instance attrs（順序需與 label 每行一致）
    inst_attrs = _load_instance_buckets(lbl_dir, image_path)

    with open(lbl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) <= 1:
                continue
            vals = list(map(float, parts[1:]))

            if len(vals) == 4:
                cx, cy, w, h = vals
                x1 = (cx - w / 2) * W
                y1 = (cy - h / 2) * H
                x2 = (cx + w / 2) * W
                y2 = (cy + h / 2) * H
            else:
                xs = np.array(vals[0::2]) * W
                ys = np.array(vals[1::2]) * H
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # 畫四角點
            for (x, y) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                cv2.circle(canvas, (x, y), r, point_color, thick)

            # 取對應的 attr
            attr = inst_attrs[idx] if idx < len(inst_attrs) else ""
            tag = "NG" if not attr else f"NG | {attr}"

            # 標在框上方，略往上移一點
            cv2.putText(canvas, tag,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        attr_color, 2, cv2.LINE_AA)

    return canvas

# ---------------------------
# 在圖上寫上檔名（紅字＋白色外框）
# ---------------------------
def put_text_with_outline(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale=1.5, color=(0, 0, 255), thickness=3,
                          outline_color=(255, 255, 255), outline_thickness=7):
    cv2.putText(img, text, org, font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


# ---------------------------
# def2：輸出「原圖 / GT / Pred」三聯圖
# ---------------------------
def compose_triptych_gt_vs_pred(image_bgr, result, image_path, data_yaml,
                                pred_box_color=(0, 255, 0), pred_txt_color=(0, 255, 0),
                                show_filename=True):
    H, W = image_bgr.shape[:2]
    _, lbl_dir = get_val_dirs_from_yaml(data_yaml)

    # 中間：GT + attrs_instances
    mid = draw_four_points_from_gt_with_attrs(
        image_bgr, image_path, lbl_dir,
        point_color=(255, 0, 0),
        attr_color=(255, 0, 0)
    )

    # 右邊：Pred box + 四角點（信心值只畫一次）
    right = image_bgr.copy()
    if getattr(result, "boxes", None) is not None and result.boxes is not None:
        b = result.boxes
        if hasattr(b, "xyxy") and b.xyxy is not None:
            xyxy = b.xyxy.detach().cpu().numpy().astype(int)
        else:
            xyxy = b.data[:, :4].detach().cpu().numpy().astype(int)
        confs = (b.conf.detach().cpu().numpy() if getattr(b, "conf", None) is not None else None)
        clss = (b.cls.detach().cpu().numpy() if getattr(b, "cls", None) is not None else None)

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            cv2.rectangle(right, (x1, y1), (x2, y2), pred_box_color, 2)

            if confs is not None:
                if clss is None or int(clss[i]) == 0:
                    label = f"NG {confs[i]:.2f}"
                else:
                    label = f"{int(clss[i])} {confs[i]:.2f}"

                # 只畫這一層文字，不讓 draw_pred_four_points 再畫一次
                cv2.putText(right, label,
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            pred_txt_color, 2, cv2.LINE_AA)

    # 這裡關掉 score，讓它只畫四角點，不再畫文字，避免重疊
    right = draw_pred_four_points(right, result, color=(0, 255, 0), with_score=False)

    pad = 10
    canvas = np.ones((H, W * 3 + pad * 2, 3), dtype=np.uint8) * 255
    canvas[:, :W] = image_bgr
    canvas[:, W + pad:W * 2 + pad] = mid
    canvas[:, W * 2 + pad * 2:] = right
    cv2.line(canvas, (W + pad // 2, 0), (W + pad // 2, H), (180, 180, 180), 2)
    cv2.line(canvas, (W * 2 + pad + pad // 2, 0), (W * 2 + pad + pad // 2, H), (180, 180, 180), 2)

    if show_filename:
        fname = os.path.basename(image_path)
        put_text_with_outline(canvas, fname, org=(16, 46))

    return canvas


# ---------------------------
# def3：釋放顯存
# ---------------------------
def free_cuda(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


# ---------------------------
# def4：影像等級評估（NG/OK）
# ---------------------------
def eval_image_level(model, data_yaml, conf=0.25, iou=0.5, imgsz=960, device=0, max_det=300):
    img_dir, lbl_dir = get_val_dirs_from_yaml(data_yaml)
    paths = [p for p in sorted(glob.glob(os.path.join(img_dir, '*.*')))
             if os.path.splitext(p)[1].lower() in IMG_EXTS]
    assert paths, f'找不到影像於 {img_dir}'

    def gt_is_ng(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(lbl_dir, stem + '.txt')
        return os.path.exists(lp) and os.path.getsize(lp) > 0

    TP = TN = FP = FN = 0
    use_half = torch.cuda.is_available()

    for p in paths:
        r = model.predict(
            source=p,
            conf=conf, iou=iou, imgsz=imgsz,
            device=device, batch=1, stream=False, verbose=False,
            save=False, retina_masks=False, half=use_half, amp=True, max_det=max_det
        )[0]

        has_det = False
        if getattr(r, "boxes", None) is not None and hasattr(r.boxes, "shape"):
            has_det = (r.boxes.shape[0] > 0)
        if (not has_det) and getattr(r, "masks", None) is not None and getattr(r.masks, "data", None) is not None:
            has_det = (r.masks.data.shape[0] > 0)

        gt = gt_is_ng(p)
        if gt and has_det:
            TP += 1
        elif (not gt) and (not has_det):
            TN += 1
        elif (not gt) and has_det:
            FP += 1
        else:
            FN += 1

        free_cuda(r)

    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    fpr = FP / (FP + TN) if (FP + TN) else 0.0

    print(f'Images={len(paths)} | TP={TP} TN={TN} FP={FP} FN={FN}')
    print(f'Precision(img)={prec:.3f}  Recall(img)={rec:.3f}  F1(img)={f1:.3f}  FPR(img)={fpr:.3f}')
    return dict(images=len(paths), TP=TP, TN=TN, FP=FP, FN=FN,
                precision=prec, recall=rec, f1=f1, fpr=fpr)


# ---------------------------
# 取得並鎖定裝置
# ---------------------------
def choose_device():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if torch.cuda.is_available():
        return 0  # Ultralytics 使用整數 index 表示 GPU
    return 'cpu'


# ---------------------------
# Image-level 混淆矩陣畫圖
# ---------------------------
def _plot_confusion_matrix(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=128)
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def confusion_matrix_image_level(model, data_yaml, out_path,
                                 conf=0.25, iou=0.5, imgsz=1024, device=0, max_det=300):
    img_dir, lbl_dir = get_val_dirs_from_yaml(data_yaml)
    paths = [p for p in sorted(glob.glob(os.path.join(img_dir, '*.*')))
             if os.path.splitext(p)[1].lower() in IMG_EXTS]
    assert paths, f'找不到影像於 {img_dir}'

    def gt_is_ng(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(lbl_dir, stem + '.txt')
        return os.path.exists(lp) and os.path.getsize(lp) > 0

    use_half = torch.cuda.is_available()
    TP = TN = FP = FN = 0
    pred_instance_total = 0
    gt_instance_total = 0

    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(lbl_dir, stem + '.txt')
        if os.path.exists(lp) and os.path.getsize(lp) > 0:
            with open(lp, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        gt_instance_total += 1

        r = model.predict(source=p, conf=conf, iou=iou, imgsz=imgsz, device=device,
                          batch=1, stream=False, verbose=False, save=False,
                          retina_masks=False, half=use_half, amp=True, max_det=max_det)[0]

        has_det = False
        n_inst = 0
        if getattr(r, "boxes", None) is not None and hasattr(r.boxes, "shape"):
            n_inst = int(r.boxes.shape[0])
            has_det = (n_inst > 0)
        if (not has_det) and getattr(r, "masks", None) is not None and getattr(r.masks, "data", None) is not None:
            n_inst = int(r.masks.data.shape[0])
            has_det = (n_inst > 0)

        pred_instance_total += n_inst

        gt = gt_is_ng(p)
        if gt and has_det:
            TP += 1
        elif (not gt) and (not has_det):
            TN += 1
        elif (not gt) and has_det:
            FP += 1
        else:
            FN += 1

        free_cuda(r)

    cm = np.array([[TP, 0],
                   [FN, 0]], dtype=np.int64)
    cm[0, 1] = FP
    cm[1, 1] = TN

    print(f"[Image-level] Images={len(paths)}  TP={TP} TN={TN} FP={FP} FN={FN}")
    print(f"  Predicted defect instances (sum of boxes/masks): {pred_instance_total}")
    print(f"  GT defect instances (from labels): {gt_instance_total}")

    _plot_confusion_matrix(cm, labels=["NG", "background"],
                           title="Confusion Matrix (Image-level)",
                           out_path=out_path)

    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                pred_instances=pred_instance_total,
                gt_instances=gt_instance_total)


# ---------------------------
# YOLO txt → bbox（xyxy）
# ---------------------------
def _yolo_label_to_bboxes(lbl_path, H, W):
    boxes = []
    if (not os.path.exists(lbl_path)) or os.path.getsize(lbl_path) == 0:
        return boxes
    with open(lbl_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 1:
                continue
            vals = list(map(float, parts[1:]))
            if len(vals) == 4:
                cx, cy, w, h = vals
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)
            else:
                xs = (np.array(vals[0::2]) * W)
                ys = (np.array(vals[1::2]) * H)
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    return boxes


# ---------------------------
# Ultralytics result → bbox 清單（xyxy）
# ---------------------------
def _pred_to_bboxes(result):
    boxes = []
    if getattr(result, "boxes", None) is not None and getattr(result.boxes, "xyxy", None) is not None:
        xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(int)
        for x1, y1, x2, y2 in xyxy:
            if x2 > x1 and y2 > y1:
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
        return boxes

    if getattr(result, "masks", None) is not None and getattr(result.masks, "xy", None) is not None:
        for poly in result.masks.xy:
            xs = poly[:, 0]
            ys = poly[:, 1]
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    return boxes


# ---------------------------
# IoU（xyxy）
# ---------------------------
def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------
# 2x2 混淆矩陣畫圖（實例等級）
# ---------------------------
def _plot_cm(cm2x2, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm2x2, cmap='Blues')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    for i in range(cm2x2.shape[0]):
        for j in range(cm2x2.shape[1]):
            ax.text(j, i, f"{int(cm2x2[i, j])}",
                    ha='center', va='center',
                    color='white' if cm2x2[i, j] > cm2x2.max() * 0.5 else 'black')
    fig.colorbar(im, ax=ax)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ==========================================================
# 實例等級混淆矩陣
# ==========================================================
def confusion_matrix_instance_level(model, data_yaml, out_path,
                                    iou_thr=0.5, conf=0.25, iou=0.5, imgsz=1024,
                                    device=0, max_det=300):
    img_dir, lbl_dir = get_val_dirs_from_yaml(data_yaml)
    paths = [p for p in sorted(glob.glob(os.path.join(img_dir, '*.*')))
             if os.path.splitext(p)[1].lower() in IMG_EXTS]
    assert paths, f'找不到影像於 {img_dir}'

    use_half = torch.cuda.is_available()
    TP = FP = FN = 0
    total_gt = 0
    total_pred = 0

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(lbl_dir, stem + '.txt')

        gt_boxes = _yolo_label_to_bboxes(lp, H, W)
        r = model.predict(source=p, conf=conf, iou=iou, imgsz=imgsz, device=device,
                          batch=1, stream=False, verbose=False, save=False,
                          retina_masks=False, half=use_half, amp=True, max_det=max_det)[0]
        pred_boxes = _pred_to_bboxes(r)

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        matches = []
        for gi, g in enumerate(gt_boxes):
            for pi, pr in enumerate(pred_boxes):
                iou_val = _iou_xyxy(g, pr)
                if iou_val >= iou_thr:
                    matches.append((iou_val, gi, pi))
        matches.sort(reverse=True)
        used_g, used_p = set(), set()
        for iou_val, gi, pi in matches:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            TP += 1
        FN += (len(gt_boxes) - len(used_g))
        FP += (len(pred_boxes) - len(used_p))

        try:
            del r
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    cm = np.array([[TP, FP],
                   [FN, 0]], dtype=np.int64)
    _plot_cm(cm, labels=['NG', 'background'],
             title='Confusion Matrix (Instance-level)', out_path=out_path)

    print(f"[Instance-level IoU≥{iou_thr}] GT instances={total_gt} | Pred instances={total_pred}")
    print(f"TP={TP} FP={FP} FN={FN} | Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")
    print(f"Saved confusion matrix: {out_path}")

    return dict(
        gt_instances=total_gt,
        pred_instances=total_pred,
        TP=TP, FP=FP, FN=FN,
        precision=precision, recall=recall, f1=f1
    )


# ---------------------------
# YOLO txt → bbox + bucket（從 attrs_instances 讀 bucket）
# ---------------------------
def _boxes_from_label(lbl_dir, image_path, H, W):
    """
    回傳 [(x1, y1, x2, y2, bucket), ...]
    bucket 只從 attrs_instances / attrs 的 json 取得，不再由 bbox 尺寸推算。
    """
    boxes = []
    stem = os.path.splitext(os.path.basename(image_path))[0]
    lbl_path = os.path.join(lbl_dir, stem + ".txt")
    if (not os.path.exists(lbl_path)) or os.path.getsize(lbl_path) == 0:
        return boxes

    inst_buckets = _load_instance_buckets(lbl_dir, image_path)

    with open(lbl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            ps = line.strip().split()
            if len(ps) <= 1:
                continue
            vs = list(map(float, ps[1:]))

            if len(vs) == 4:
                cx, cy, w, h = vs
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)
            else:
                xs = np.array(vs[0::2]) * W
                ys = np.array(vs[1::2]) * H
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)

            bucket = inst_buckets[idx] if idx < len(inst_buckets) else None
            boxes.append((x1, y1, x2, y2, bucket))
    return boxes


# ---------------------------
# 產生三聯圖 + 同步寫 image-level log（你要的格式）
# ---------------------------
def export_triptychs(model, data_yaml, out_dir,
                     log_txt_path=None,
                     conf=0.25, iou=0.5, imgsz=960, device=0):
    """
    每張圖：
      1. 存三聯圖
      2. 若 log_txt_path 不為 None，寫一行 image-level log
    """
    os.makedirs(out_dir, exist_ok=True)
    img_dir, lbl_dir = get_val_dirs_from_yaml(data_yaml)
    paths = [p for p in sorted(glob.glob(os.path.join(img_dir, '*.*')))
             if os.path.splitext(p)[1].lower() in IMG_EXTS]
    assert paths, f'找不到影像於 {img_dir}'

    use_half = torch.cuda.is_available()
    buckets = ['>0.2', '0.2', '0.15', '0.1']
    log_lines = []

    for p in paths:
        img = cv2.imread(p)
        assert img is not None, f"讀不到影像：{p}"
        H, W = img.shape[:2]

        stem = os.path.splitext(os.path.basename(p))[0]

        # 1) 讀 GT（位置 + bucket from attrs_instances）
        gts = _boxes_from_label(lbl_dir, p, H, W)
        gt_cnt = {b: 0 for b in buckets}
        for *_, b in gts:
            if b in gt_cnt:
                gt_cnt[b] += 1
        # ★ 關鍵修改：有沒有標註 = 看 gts 長度，而不是 bucket 的數量
        gt_total = len(gts)

        # 2) 推論
        res = model.predict(
            source=p,
            conf=conf, iou=iou, imgsz=imgsz,
            device=device, batch=1, stream=False, verbose=False,
            save=False, retina_masks=False, half=use_half, amp=True
        )[0]

        has_det = False
        if getattr(res, "boxes", None) is not None and hasattr(res.boxes, "shape"):
            has_det = (res.boxes.shape[0] > 0)
        if (not has_det) and getattr(res, "masks", None) is not None and getattr(res.masks, "data", None) is not None:
            has_det = (res.masks.data.shape[0] > 0)

        # 3) 三聯圖
        trip = compose_triptych_gt_vs_pred(img, res, p, data_yaml)
        outp = os.path.join(out_dir, stem + "_triptych.png")
        cv2.imwrite(outp, trip)
        print(f"🖼️ 已輸出：{outp}")

        # 4) image-level log
        if log_txt_path is not None:
            if gt_total == 0:
                # 沒有標註 → OK 圖
                if not has_det:
                    line = f"{stem}.jpg  OK標註  推論結果: OK圖"
                else:
                    line = f"{stem}.jpg  OK標註  推論結果: 誤判-多檢-(有偵測到瑕疵)"
            else:
                gt_part = " ".join([f"{b}: {gt_cnt[b]}顆" for b in buckets])
                if not has_det:
                    # 有標註但完全沒偵測 → 全部遺漏
                    miss_part = "、".join(
                        [f"{b}: {gt_cnt[b]}顆" for b in buckets if gt_cnt[b] > 0]
                    ) or "無"
                    line = (
                        f"{stem}.jpg  NG標註: {gt_part}  |  "
                        f"推論結果:  無偵測到瑕疵   "
                        f"實際: 錯誤判斷-遺漏-:   {miss_part}"
                    )
                else:
                    # 有標註且至少偵測到一顆 → image-level 先視為正確
                    line = (
                        f"{stem}.jpg  NG標註: {gt_part}  |  "
                        f"推論結果:  有偵測到瑕疵   "
                        f"實際: 正確判斷-有偵測-"
                    )
            log_lines.append(line)

        free_cuda(res, trip, img)

    if log_txt_path is not None and log_lines:
        os.makedirs(os.path.dirname(log_txt_path), exist_ok=True)
        with open(log_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"已輸出三聯圖影像等級 log：{log_txt_path}")


# ---------------------------
# 逐圖 instance 匹配 log（IoU 版本，bucket 由 attrs_instances 來）
# ---------------------------
def export_instance_match_log(model, data_yaml, out_text_path, out_csv_path,
                              iou_thr=0.5, conf=0.25, iou=0.5, imgsz=1024, device=0, max_det=300):
    img_dir, lbl_dir = get_val_dirs_from_yaml(data_yaml)
    paths = [p for p in sorted(glob.glob(os.path.join(img_dir, '*.*')))
             if os.path.splitext(p)[1].lower() in IMG_EXTS]
    os.makedirs(os.path.dirname(out_text_path), exist_ok=True)

    use_half = torch.cuda.is_available()
    txt_lines = []
    csv_rows = []
    buckets = ['>0.2', '0.2', '0.15', '0.1']

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        stem = os.path.splitext(os.path.basename(p))[0]

        # GT: 位置 + bucket(from attrs_instances)
        gts = _boxes_from_label(lbl_dir, p, H, W)  # (x1,y1,x2,y2,bucket)

        r = model.predict(source=p, conf=conf, iou=iou, imgsz=imgsz, device=device,
                          batch=1, stream=False, verbose=False, save=False,
                          retina_masks=False, half=use_half, amp=True, max_det=max_det)[0]
        preds_xyxy = _pred_to_bboxes(r)
        # preds 不需要 bucket，因為 FP 只統計總數
        preds = [(x1, y1, x2, y2) for (x1, y1, x2, y2) in preds_xyxy]

        gt_cnt = {b: 0 for b in buckets}
        for *_, b in gts:
            if b in gt_cnt:
                gt_cnt[b] += 1

        # IoU 匹配
        matches = []
        for gi, g in enumerate(gts):
            for pi, pr in enumerate(preds):
                iouv = _iou_xyxy(g[:4], pr[:4])
                if iouv >= iou_thr:
                    matches.append((iouv, gi, pi))
        matches.sort(reverse=True)

        used_g, used_p = set(), set()
        TPb = {b: 0 for b in buckets}
        for iouv, gi, pi in matches:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            gb = gts[gi][4]
            if gb in TPb:
                TPb[gb] += 1

        FNb = {b: 0 for b in buckets}
        for gi, g in enumerate(gts):
            if gi not in used_g and g[4] in FNb:
                FNb[g[4]] += 1

        # FP：不用 bucket，只要總數
        FP_total = 0
        for pi, _ in enumerate(preds):
            if pi not in used_p:
                FP_total += 1

        if len(gts) == 0:
            line = f"{stem}.jpg  OK標註  推論結果: OK圖"
        else:
            gt_part = " ".join([f"{k}: {gt_cnt[k]}顆" for k in buckets])
            tp_part = "、".join([f"{k}: {TPb[k]}顆" for k in buckets if TPb[k] > 0]) or "無"
            fn_part = "、".join([f"{k}: {FNb[k]}顆" for k in buckets if FNb[k] > 0]) or "無"
            line = (f"{stem}.jpg  NG標註: {gt_part}  |  "
                    f"推論結果: 正確判斷: {tp_part}；錯誤判斷(遺漏): {fn_part}；誤判: {FP_total}顆")
        txt_lines.append(line)

        csv_row = {
            "filename": stem + ".jpg",
            "gt_>0.2": gt_cnt['>0.2'], "gt_0.2": gt_cnt['0.2'], "gt_0.15": gt_cnt['0.15'], "gt_0.1": gt_cnt['0.1'],
            "tp_>0.2": TPb['>0.2'], "tp_0.2": TPb['0.2'], "tp_0.15": TPb['0.15'], "tp_0.1": TPb['0.1'],
            "fn_>0.2": FNb['>0.2'], "fn_0.2": FNb['0.2'], "fn_0.15": FNb['0.15'], "fn_0.1": FNb['0.1'],
            "fp_total": FP_total
        }
        csv_rows.append(csv_row)

        free_cuda(r)

    with open(out_text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    keys = list(csv_rows[0].keys()) if csv_rows else []
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)

    print(f"已輸出逐圖文字 log：{out_text_path}")
    print(f"已輸出逐圖 CSV：{out_csv_path}")
    return out_csv_path


def plot_summary_from_log(csv_path, out_png_path):
    buckets = ['>0.2', '0.2', '0.15', '0.1']
    TPb = {b: 0 for b in buckets}
    FNb = {b: 0 for b in buckets}
    FP_total = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        r = _csv.DictReader(f)
        for row in r:
            for b in buckets:
                TPb[b] += int(row[f"tp_{b}"])
                FNb[b] += int(row[f"fn_{b}"])
                FN_total = sum(FNb[b] for b in buckets)
            FP_total += int(row["fp_total"])

    x = np.arange(len(buckets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6), dpi=130)
    ax.bar(x - width / 2, [TPb[b] for b in buckets], width,
           label='Correct (TP)')
    ax.bar(x + width / 2, [FNb[b] for b in buckets], width,
           label='Missed (FN)')

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("Counts")
    ax.set_title("")  # 用 suptitle 取代
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    # 上方兩行標題
    fig.suptitle(
        f"Instance Matching Summary (Preview)\nTotal False Negatives: {FN_total}",
        fontsize=18
    )

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.90])  # 留空給 suptitle
    plt.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"已輸出統計圖：{out_png_path}")


# ---------------------------
# main 流程
# ---------------------------
def main():
    # 你自己的路徑
    model_path = "/app/seg_runs/y11sseg_V3_251203/weights/best.pt"
    data_yaml = "/app/dataset/YOLO_seg/dataset.yaml"
    out_dir = "/app/inference_image"
    os.makedirs(out_dir, exist_ok=True)

    # 選擇裝置
    device = choose_device()

    # 載入模型
    model = YOLO(model_path)
    if device == 0:
        model.to("cuda:0")
        print("✅ 使用 GPU: cuda:0")
    else:
        print("⚠️ 未偵測到 CUDA，改用 CPU")

    # 1) 三聯圖 + image-level log
    triptych_log_txt = os.path.join(out_dir, "image_level_from_triptych.log")
    run_with_timer(
        "輸出三聯圖 + image-level log",
        export_triptychs,
        model, data_yaml, out_dir,
        triptych_log_txt,
        conf=0.25, iou=0.5, imgsz=1024, device=device,
    )

    # 2) Image-level NG/OK 整體指標
    run_with_timer(
        "影像等級 NG/OK 指標",
        eval_image_level,
        model, data_yaml,
        conf=0.25, iou=0.5, imgsz=1024, device=device,
    )

    # 3) Image-level 混淆矩陣圖
    img_cm_path = os.path.join(out_dir, "confmat_image_level.png")
    run_with_timer(
        "Image-level 混淆矩陣",
        confusion_matrix_image_level,
        model, data_yaml, img_cm_path,
        conf=0.25, iou=0.5, imgsz=1024, device=device,
    )

    # 4) Instance-level 混淆矩陣圖
    inst_cm_path = os.path.join(out_dir, "confmat_instance_level.png")
    run_with_timer(
        "Instance-level 混淆矩陣",
        confusion_matrix_instance_level,
        model, data_yaml, inst_cm_path,
        iou_thr=0.5, conf=0.25, iou=0.5, imgsz=1024, device=device,
    )

    # 5) 逐圖 instance match log + CSV
    inst_txt = os.path.join(out_dir, "instance_match.log")
    inst_csv = os.path.join(out_dir, "instance_match.csv")
    run_with_timer(
        "輸出逐圖 instance log + CSV",
        export_instance_match_log,
        model, data_yaml,
        inst_txt, inst_csv,
        iou_thr=0.5, conf=0.25, iou=0.5, imgsz=1024, device=device,
    )

    # 6) 根據 CSV 畫總結圖
    summary_png = os.path.join(out_dir, "instance_match_summary.png")
    run_with_timer(
        "繪製 instance 統計圖",
        plot_summary_from_log,
        inst_csv, summary_png,
    )

    # 7) 收尾
    free_cuda(model)


if __name__ == "__main__":
    main()
