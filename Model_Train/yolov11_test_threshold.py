#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import csv
import json
from itertools import product
from datetime import datetime

import yaml
import torch
from PIL import Image
from ultralytics import YOLO

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ==========================================================
# Dataset utils
# ==========================================================
def _resolve_path(base: str, p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base, p))


def get_val_dirs_from_yaml(data_yaml: str):
    cfg = yaml.safe_load(open(data_yaml, "r", encoding="utf-8"))
    base = cfg.get("path", "") or ""

    val_key = cfg.get("val") or cfg.get("validation") or cfg.get("val_dir")
    assert val_key, f"dataset.yaml 找不到 val/validation/val_dir：{data_yaml}"

    img_dir = _resolve_path(base, val_key)
    assert os.path.isdir(img_dir), f"val images 不存在：{img_dir}"

    if (os.sep + "images") in img_dir:
        lbl_dir = img_dir.replace(os.sep + "images", os.sep + "labels")
    else:
        lbl_dir = os.path.join(os.path.dirname(img_dir), "labels")

    assert os.path.isdir(lbl_dir), f"val labels 不存在：{lbl_dir}"
    return img_dir, lbl_dir


def list_images(img_dir: str):
    paths = [p for p in sorted(glob.glob(os.path.join(img_dir, "*.*")))
             if os.path.splitext(p)[1].lower() in IMG_EXTS]
    assert paths, f"找不到影像於 {img_dir}"
    return paths


def choose_device():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return 0 if torch.cuda.is_available() else "cpu"


def free_cuda(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ==========================================================
# Param grid (你要的實驗矩陣)
# ==========================================================
def build_param_grid(conf_list, imgsz_list, iou_list):
    grid = []
    for conf, imgsz, iou in product(conf_list, imgsz_list, iou_list):
        grid.append({"conf": float(conf), "imgsz": int(imgsz), "iou": float(iou)})
    return grid


# ==========================================================
# Shared: bbox helpers (不畫圖，只做計算)
# ==========================================================
def _read_hw(image_path: str):
    # 只取 H,W 做 label 解碼，不用 cv2
    with Image.open(image_path) as im:
        w, h = im.size
    return h, w


def _gt_label_path(lbl_dir: str, image_path: str):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(lbl_dir, stem + ".txt")


def _gt_is_ng(lbl_dir: str, image_path: str) -> bool:
    lp = _gt_label_path(lbl_dir, image_path)
    return os.path.exists(lp) and os.path.getsize(lp) > 0


def _yolo_label_to_bboxes(lbl_path, H, W):
    """
    支援：
      - bbox: class cx cy w h
      - polygon: class x1 y1 x2 y2 ...（normalized）
    回傳 list[[x1,y1,x2,y2], ...] (int)
    """
    boxes = []
    if (not os.path.exists(lbl_path)) or os.path.getsize(lbl_path) == 0:
        return boxes

    with open(lbl_path, "r", encoding="utf-8") as f:
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
                xs = vals[0::2]
                ys = vals[1::2]
                x1 = int(min(xs) * W)
                x2 = int(max(xs) * W)
                y1 = int(min(ys) * H)
                y2 = int(max(ys) * H)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

    return boxes


def _pred_to_bboxes(result):
    """
    從 Ultralytics result 取出 xyxy boxes
    - 有 boxes 就用 boxes
    - 沒 boxes 但有 masks，就用 mask polygon 外接框
    """
    boxes = []

    if getattr(result, "boxes", None) is not None and getattr(result.boxes, "xyxy", None) is not None:
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        for x1, y1, x2, y2 in xyxy:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
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


def _has_any_detection(result):
    if getattr(result, "boxes", None) is not None and hasattr(result.boxes, "shape"):
        if int(result.boxes.shape[0]) > 0:
            return True
    if getattr(result, "masks", None) is not None and getattr(result.masks, "data", None) is not None:
        if int(result.masks.data.shape[0]) > 0:
            return True
    return False


# ==========================================================
# Metrics
# ==========================================================
def eval_image_level(model, paths, lbl_dir, conf, imgsz, iou, device, max_det=300):
    TP = TN = FP = FN = 0
    use_half = torch.cuda.is_available()

    for p in paths:
        gt_ng = _gt_is_ng(lbl_dir, p)

        r = model.predict(
            source=p,
            conf=conf,
            iou=iou,       # 推論 NMS IoU threshold（你的 grid 參數之一）
            imgsz=imgsz,
            device=device,
            batch=1,
            stream=False,
            verbose=False,
            save=False,
            retina_masks=False,
            half=use_half,
            amp=True,
            max_det=max_det,
        )[0]

        has_det = _has_any_detection(r)

        if gt_ng and has_det:
            TP += 1
        elif (not gt_ng) and (not has_det):
            TN += 1
        elif (not gt_ng) and has_det:
            FP += 1
        else:
            FN += 1

        free_cuda(r)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "img_TP": TP, "img_TN": TN, "img_FP": FP, "img_FN": FN,
        "img_precision": precision,
        "img_recall": recall,
        "img_f1": f1,
    }


def eval_instance_level(model, paths, lbl_dir, conf, imgsz, iou, iou_match, device, max_det=300):
    """
    Instance-level：
      - GT: label txt 轉成 bbox（polygon 也會轉外接框）
      - Pred: result boxes/masks 轉 bbox
      - 用 iou_match 做 greedy matching 算 TP/FP/FN
    """
    TP = FP = FN = 0
    total_gt = 0
    total_pred = 0
    use_half = torch.cuda.is_available()

    for p in paths:
        H, W = _read_hw(p)
        lp = _gt_label_path(lbl_dir, p)
        gt_boxes = _yolo_label_to_bboxes(lp, H, W)

        r = model.predict(
            source=p,
            conf=conf,
            iou=iou,      # 推論 NMS IoU threshold（你的 grid 參數之一）
            imgsz=imgsz,
            device=device,
            batch=1,
            stream=False,
            verbose=False,
            save=False,
            retina_masks=False,
            half=use_half,
            amp=True,
            max_det=max_det,
        )[0]

        pred_boxes = _pred_to_bboxes(r)

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        # matching candidates
        matches = []
        for gi, g in enumerate(gt_boxes):
            for pi, pr in enumerate(pred_boxes):
                v = _iou_xyxy(g, pr)
                if v >= iou_match:
                    matches.append((v, gi, pi))

        matches.sort(reverse=True)

        used_g, used_p = set(), set()
        for v, gi, pi in matches:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            TP += 1

        FN += (len(gt_boxes) - len(used_g))
        FP += (len(pred_boxes) - len(used_p))

        free_cuda(r)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "inst_gt": total_gt,
        "inst_pred": total_pred,
        "inst_TP": TP, "inst_FP": FP, "inst_FN": FN,
        "inst_precision": precision,
        "inst_recall": recall,
        "inst_f1": f1,
    }


# ==========================================================
# Grid runner
# ==========================================================
def pick_best(rows, mode="instance"):
    # mode="instance"：先以 instance-level F1 再看 image-level F1
    if mode == "instance":
        return max(rows, key=lambda d: (d["inst_f1"], d["inst_recall"], d["inst_precision"],
                                        d["img_f1"], d["img_recall"], d["img_precision"]))
    # mode="image"
    return max(rows, key=lambda d: (d["img_f1"], d["img_recall"], d["img_precision"],
                                    d["inst_f1"], d["inst_recall"], d["inst_precision"]))


def run_grid_search(model, paths, lbl_dir, grid, device, iou_match=0.5, max_det=300):
    rows = []
    t0 = time.time()

    for idx, hp in enumerate(grid, 1):
        conf, imgsz, iou = hp["conf"], hp["imgsz"], hp["iou"]

        img_m = eval_image_level(model, paths, lbl_dir, conf, imgsz, iou, device, max_det=max_det)
        inst_m = eval_instance_level(model, paths, lbl_dir, conf, imgsz, iou, iou_match, device, max_det=max_det)

        row = {
            **hp,
            "iou_match": float(iou_match),
            **img_m,
            **inst_m,
        }
        rows.append(row)

        print(
            f"[{idx:02d}/{len(grid)}] conf={conf:.2f} imgsz={imgsz} iou={iou:.2f} | "
            f"IMG(P={row['img_precision']:.3f} R={row['img_recall']:.3f}) "
            f"INST(P={row['inst_precision']:.3f} R={row['inst_recall']:.3f})"
        )

    best_inst = pick_best(rows, mode="instance")
    best_img = pick_best(rows, mode="image")

    dur = round(time.time() - t0, 2)
    print("\n" + "=" * 80)
    print(f"Done. combos={len(grid)} time={dur}s  (iou_match={iou_match})")

    print(
        f"[BEST by INSTANCE] conf={best_inst['conf']:.2f} imgsz={best_inst['imgsz']} iou={best_inst['iou']:.2f} | "
        f"INST(P={best_inst['inst_precision']:.3f} R={best_inst['inst_recall']:.3f} F1={best_inst['inst_f1']:.3f}) | "
        f"IMG(P={best_inst['img_precision']:.3f} R={best_inst['img_recall']:.3f} F1={best_inst['img_f1']:.3f})"
    )

    print(
        f"[BEST by IMAGE]    conf={best_img['conf']:.2f} imgsz={best_img['imgsz']} iou={best_img['iou']:.2f} | "
        f"IMG(P={best_img['img_precision']:.3f} R={best_img['img_recall']:.3f} F1={best_img['img_f1']:.3f}) | "
        f"INST(P={best_img['inst_precision']:.3f} R={best_img['inst_recall']:.3f} F1={best_img['inst_f1']:.3f})"
    )
    print("=" * 80 + "\n")

    return rows, best_inst, best_img


def save_results(rows, best_inst, best_img, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(out_dir, f"grid_results_{ts}.csv")
    best_path = os.path.join(out_dir, f"best_{ts}.json")

    keys = [
        "conf", "imgsz", "iou", "iou_match",
        "img_TP", "img_TN", "img_FP", "img_FN", "img_precision", "img_recall", "img_f1",
        "inst_gt", "inst_pred", "inst_TP", "inst_FP", "inst_FN", "inst_precision", "inst_recall", "inst_f1",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    payload = {
        "best_by_instance": best_inst,
        "best_by_image": best_img,
    }
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved:\n- {csv_path}\n- {best_path}")


# ==========================================================
# main
# ==========================================================
def main():
    model_path = "/app/seg_runs/y11sseg_V2_251203/weights/best.pt"
    data_yaml = "/app/dataset/YOLO_seg/dataset.yaml"
    out_dir = "/app/inference_gridsearch"

    # 你的 3 個推論超參數 grid
    CONF_LIST = [0.10, 0.15, 0.20, 0.25]
    IMGSZ_LIST = [512, 1024, 1536]
    IOU_LIST = [0.50, 0.25, 0.15]

    # Instance-level matching IoU（固定常數，不算推論超參數）
    IOU_MATCH = 0.50

    MAX_DET = 300

    device = choose_device()
    model = YOLO(model_path)
    if device == 0:
        model.to("cuda:0")
        print("✅ 使用 GPU: cuda:0")
    else:
        print("⚠️ 未偵測到 CUDA，改用 CPU")

    img_dir, lbl_dir = get_val_dirs_from_yaml(data_yaml)
    paths = list_images(img_dir)

    grid = build_param_grid(CONF_LIST, IMGSZ_LIST, IOU_LIST)

    rows, best_inst, best_img = run_grid_search(
        model=model,
        paths=paths,
        lbl_dir=lbl_dir,
        grid=grid,
        device=device,
        iou_match=IOU_MATCH,
        max_det=MAX_DET,
    )

    save_results(rows, best_inst, best_img, out_dir)
    free_cuda(model)


if __name__ == "__main__":
    main()
