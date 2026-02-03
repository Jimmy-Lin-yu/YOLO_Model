from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List, Callable

import time
import torch
import numpy as np
from ultralytics import YOLO


class YOLOSensor:
    """
    只做「有沒有物體」判斷（trigger 用）
    - 回傳 has_obj + 可序列化 info
    - 若 has_obj=False：外層應直接丟掉該 frame（continue）
    - 若 has_obj=True：print「偵測到足粒」，並可透過 trigger_cb 觸發相機/流程
    """

    def __init__(
        self,
        weight_path: str,
        device: str | None = None,
        conf: float = 0.9,
        iou: float = 0.45,
        imgsz: int | None = 320,
        classes: list[int] | None = None,
        fuse: bool = True,
        max_det: int = 1,
        min_best_conf: float | None = None,
        print_infer_time: bool = True,   # ✅ 每次 has_object 都印推論時間

        require_centered: bool = True,
        center_tol_ratio_xy: Tuple[float, float] = (0.15, 0.15),
        center_tol_px_xy: Tuple[int | None, int | None] = (None, None),

        # ✅ Trigger 設定
        trigger_cb: Optional[Callable[[Dict[str, Any]], None]] = None,  # 由外部傳入：真正的 trigger 動作
        trigger_on_rising_edge: bool = True,                            # True: 只有「無->有」才 trigger
        trigger_min_interval_s: float = 0.2,                            # 避免抖動/跳針
        print_trigger: bool = True,                                     # 印「偵測到足粒」
    ):
        self.weight_path = weight_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes
        self.max_det = max_det
        self.min_best_conf = min_best_conf
        self.print_infer_time = bool(print_infer_time)

        self.require_centered = require_centered
        self.center_tol_ratio_xy = center_tol_ratio_xy
        self.center_tol_px_xy = center_tol_px_xy
        self.enable_crop = False  # TODO: 未來可改成參數

        self.trigger_cb = trigger_cb
        self.trigger_on_rising_edge = trigger_on_rising_edge
        self.trigger_min_interval_s = float(trigger_min_interval_s)
        self.print_trigger = bool(print_trigger)

        self._prev_has_obj = False
        self._last_trigger_t = 0.0

        self.model = YOLO(self.weight_path)
        self.model.to(self.device)

        if fuse:
            try:
                self.model.fuse()
            except Exception:
                pass

    @staticmethod
    def is_bbox_center_near_frame_center(
        bbox_xyxy: Optional[List[float]],
        frame_shape_hw: Tuple[int, int],
        tol_ratio_xy: Tuple[float, float] = (0.15, 0.15),
        tol_px_xy: Tuple[int | None, int | None] = (None, None),
        # ✅ 新增：不切邊模式
        require_full_in_frame: bool = False,
        # ✅ 新增：邊界安全距離（避免 bbox 太靠邊也算切邊風險）
        edge_margin_ratio_xy: Tuple[float, float] = (0.0, 0.0),
        edge_margin_px_xy: Tuple[int | None, int | None] = (None, None),
    ) -> Tuple[bool, Dict[str, Any]]:
        h, w = int(frame_shape_hw[0]), int(frame_shape_hw[1])

        fx = w / 2.0
        fy = h / 2.0

        if bbox_xyxy is None:
            return False, {
                "ok": False,
                "center_ok": False,
                "edge_ok": False,
                "reason": "no_bbox",
                "bbox_center": None,
                "frame_center": [fx, fy],
            }

        x1, y1, x2, y2 = map(float, bbox_xyxy)

        # bbox 基本合法性
        if x2 <= x1 or y2 <= y1:
            return False, {
                "ok": False,
                "center_ok": False,
                "edge_ok": False,
                "reason": "invalid_bbox",
                "bbox_xyxy": [x1, y1, x2, y2],
                "frame_center": [fx, fy],
            }

        # --- 中心判斷 ---
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        tol_x = float(tol_px_xy[0]) if tol_px_xy[0] is not None else float(w * tol_ratio_xy[0])
        tol_y = float(tol_px_xy[1]) if tol_px_xy[1] is not None else float(h * tol_ratio_xy[1])

        dx = cx - fx
        dy = cy - fy
        center_ok = (abs(dx) <= tol_x) and (abs(dy) <= tol_y)

        # --- 不切邊判斷（bbox 必須完全在畫面內，且可選 margin）---
        margin_x = float(edge_margin_px_xy[0]) if edge_margin_px_xy[0] is not None else float(w * edge_margin_ratio_xy[0])
        margin_y = float(edge_margin_px_xy[1]) if edge_margin_px_xy[1] is not None else float(h * edge_margin_ratio_xy[1])

        edge_ok = (x1 >= margin_x) and (y1 >= margin_y) and (x2 <= (w - margin_x)) and (y2 <= (h - margin_y))

        # ✅ 最終判斷：若 require_full_in_frame=True -> 必須 center_ok 且 edge_ok
        ok = bool(center_ok and (edge_ok if require_full_in_frame else True))

        reason = "ok"
        if not ok:
            if require_full_in_frame and not edge_ok:
                reason = "touch_edge_or_outside"
            elif not center_ok:
                reason = "center_not_ok"
            else:
                reason = "not_ok"

        return ok, {
            "ok": ok,
            "center_ok": bool(center_ok),
            "edge_ok": bool(edge_ok),
            "reason": reason,
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "bbox_center": [float(cx), float(cy)],
            "frame_center": [float(fx), float(fy)],
            "dx": float(dx),
            "dy": float(dy),
            "tol_x": float(tol_x),
            "tol_y": float(tol_y),
            "margin_x": float(margin_x),
            "margin_y": float(margin_y),
        }


    @torch.inference_mode()
    def has_object(self, frame_bgr: np.ndarray, min_det: int = 1) -> Tuple[bool, Dict[str, Any]]:
        # ✅ 計時（GPU 需 synchronize 才準）
        use_cuda = str(self.device).startswith("cuda") and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
            
        
        results = self.model.predict(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            max_det=self.max_det,
            verbose=False,
            device=self.device,
        )

        if use_cuda:
            torch.cuda.synchronize()
        infer_time_s = time.perf_counter() - t0
        r = results[0]

        num_det = 0
        best_conf = 0.0
        best_bbox = None
        best_cls = None

        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            b = r.boxes
            num_det = int(len(b))

            confs = b.conf.detach().cpu().numpy()
            idx = int(np.argmax(confs))
            best_conf = float(confs[idx])

            xyxy = b.xyxy.detach().cpu().numpy()[idx]
            best_bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]

            if getattr(b, "cls", None) is not None:
                best_cls = int(b.cls.detach().cpu().numpy()[idx])

        has_obj = (num_det >= min_det)

        # conf gate
        if has_obj and self.min_best_conf is not None and best_conf < float(self.min_best_conf):
            has_obj = False

        # center gate
        center_ok, center_dbg = self.is_bbox_center_near_frame_center(
            best_bbox,
            frame_shape_hw=(frame_bgr.shape[0], frame_bgr.shape[1]),
            tol_ratio_xy=self.center_tol_ratio_xy,
            tol_px_xy=self.center_tol_px_xy,
            require_full_in_frame=(not self.enable_crop),   # ✅ 不切邊 => 要求 bbox 完全在畫面內 + 正中心
            edge_margin_ratio_xy=(0.0, 0.0),                # ✅ 不想留 margin 就 0
            edge_margin_px_xy=(5, 5),                       # ✅ 或改成 (5,5) 更保守
        )
        if self.require_centered:
            has_obj = bool(has_obj and center_ok)

        info: Dict[str, Any] = {
            "has_obj": bool(has_obj),
            "drop_frame": bool(not has_obj),  # ✅ 外層看到 True 就 continue（丟掉此 frame）
            "num_det": int(num_det),
            "best_conf": float(best_conf),
            "best_bbox": best_bbox,
            "best_cls": best_cls,

            # ✅ 推論耗時
            "infer_time_s": float(infer_time_s),
            "infer_time_ms": float(infer_time_s * 1000.0),

            "center_ok": bool(center_dbg.get("center_ok", False)),
            "bbox_center": center_dbg.get("bbox_center", None),
            "frame_center": center_dbg.get("frame_center", None),
            "dx": center_dbg.get("dx", None),
            "dy": center_dbg.get("dy", None),
            "tol_x": center_dbg.get("tol_x", None),
            "tol_y": center_dbg.get("tol_y", None),
        }

        # ✅ 有偵測到才 trigger
        now = time.time()
        should_trigger = bool(has_obj)

        if self.print_infer_time:
            print(f"[YOLOSensor] infer_time={infer_time_s:.4f}s ({infer_time_s*1000:.1f}ms)")

        if self.trigger_on_rising_edge:
            # 只有「上一張沒物體、這張有物體」才 trigger
            should_trigger = bool(should_trigger and (not self._prev_has_obj))

        if should_trigger and (now - self._last_trigger_t) < self.trigger_min_interval_s:
            should_trigger = False

        if should_trigger:
            if self.print_trigger:
                print(f"[YOLOSensor] 偵測到足粒 (conf={best_conf:.3f}, cls={best_cls}, center_ok={info['center_ok']})")
            if self.trigger_cb is not None:
                # 由外部注入真正的 trigger 動作（觸發相機/通知 PLC/請求 full-res snapshot）
                self.trigger_cb(info)

            self._last_trigger_t = now

        self._prev_has_obj = bool(has_obj)
        return bool(has_obj), info
