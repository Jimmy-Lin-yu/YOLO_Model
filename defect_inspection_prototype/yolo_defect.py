# yolo_defect.py  (no PIL, return serializable detections)
from __future__ import annotations

from time import time
from typing import Dict, Any, Tuple, List, Optional

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time

class YOLODefectInspector:
    """
    瑕疵檢測（純推論輸出，給 pipeline 三分流用）：
    - status: OK/NG
    - num_defect: 瑕疵數量（可用 defect_classes 過濾）
    - detections: 可序列化的 bbox/cls/conf/area/category
    - size_info: 尺寸分級統計（可序列化）
    """

    def __init__(
        self,
        weight_path: str,
        device: str | None = None,
        conf: float = 0.5,
        iou: float = 0.45,
        imgsz: int | None = 640,
        defect_classes: list[int] | None = None,
        fuse: bool = True,
        print_infer_time: bool = True,
    ):
        self.weight_path = weight_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.defect_classes = defect_classes
        self.print_infer_time = bool(print_infer_time)

        self.model = YOLO(self.weight_path)
        self.model.to(self.device)

        if fuse:
            try:
                self.model.fuse()
            except Exception:
                pass

        # 你原本的像素面積門檻（依你的標定來）
        self._t1 = 8.9 * 8.5
        self._t2 = 15.1 * 15.9
        self._t3 = 24.1 * 24

    # ---------------------------
    # 尺寸分類（依 bbox 面積）
    # ---------------------------
    def _size_category(self, area: float) -> str:
        if area <= self._t1:
            return "0.1mm"
        if area <= self._t2:
            return "0.2mm"
        if area <= self._t3:
            return "0.4mm"
        return ">0.5mm"

    # ---------------------------
    # 把 result 轉成可序列化 detections + size_info
    # ---------------------------
    def _extract_detections_and_sizes(self, result) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        回傳：
        detections: [
          {"bbox":[x1,y1,x2,y2], "cls":int, "conf":float, "w":float, "h":float, "area":float, "category":str},
          ...
        ]
        size_info: {"boxes": [...同 detections 的 bbox/size...], "counts": {...}}
        """
        counts = {"0.1mm": 0, "0.2mm": 0, "0.4mm": 0, ">0.5mm": 0}
        detections: List[Dict[str, Any]] = []

        if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
            return detections, {"boxes": [], "counts": counts}

        b = result.boxes
        xyxy = b.xyxy.detach().cpu().numpy()
        conf = b.conf.detach().cpu().numpy() if getattr(b, "conf", None) is not None else None
        cls = b.cls.detach().cpu().numpy().astype(int) if getattr(b, "cls", None) is not None else None

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            w = float(x2 - x1)
            h = float(y2 - y1)
            area = w * h
            cat = self._size_category(area)
            counts[cat] += 1

            det = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "cls": int(cls[i]) if cls is not None else 0,
                "conf": float(conf[i]) if conf is not None else 0.0,
                "w": w,
                "h": h,
                "area": float(area),
                "category": cat,
            }
            detections.append(det)

        size_info = {"boxes": detections, "counts": counts}
        return detections, size_info

    # ---------------------------
    # 瑕疵數量（可用 defect_classes 過濾）
    # ---------------------------
    def _count_defects_from_dets(self, detections: List[Dict[str, Any]]) -> int:
        if not detections:
            return 0
        if self.defect_classes is None:
            return len(detections)
        allow = set(int(x) for x in self.defect_classes)
        return sum(1 for d in detections if int(d["cls"]) in allow)

    # ---------------------------
    # ✅ 對單張做瑕疵推論（預設不畫圖）
    # ---------------------------
    @torch.inference_mode()
    def infer(self, frame_bgr: np.ndarray, draw: bool = True):
        kwargs = dict(
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )
        # ✅ imgsz=None -> 不傳；有值才傳
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz

        # ✅ 計時（GPU 需要 synchronize 才準）
        use_cuda = str(self.device).startswith("cuda") and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        results = self.model.predict(frame_bgr, **kwargs)

        if use_cuda:
            torch.cuda.synchronize()
        infer_time_s = time.perf_counter() - t0

        result = results[0]

        detections, size_info = self._extract_detections_and_sizes(result)
        num_defect = self._count_defects_from_dets(detections)

        if num_defect == 0:
            status = "OK"
        else:
            status = "NG"

        info: Dict[str, Any] = {
            "status": status,
            "num_defect": int(num_defect),
            "detections": detections,   # ✅ 可序列化（UI/Persist 用）
            "size_info": size_info,     # ✅ 可序列化（Persist 用）
            "infer_time_s": float(infer_time_s),
            "infer_time_ms": float(infer_time_s * 1000.0),
        }

        if getattr(self, "print_infer_time", False):
            print(f"[YOLODefect] infer_time={infer_time_s:.4f}s ({infer_time_s*1000:.1f}ms)")


        if not draw:
            return frame_bgr, info

        # --- draw=True: OpenCV 畫（不使用 PIL） ---
        out = frame_bgr.copy()
        color = (0, 255, 0) if status == "OK" else (0, 0, 255)

        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                f'{d["category"]}',
                (x1 + 6, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            out,
            f"{status} defects={num_defect}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            color,
            3,
            cv2.LINE_AA,
        )

        return out, info
