import os
import time
from datetime import datetime

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

class YOLORealtimeInspector:
    """
    簡單 YOLO 推論封裝：
    - 給一張 BGR frame
    - 回傳畫好結果的 frame + 基本資訊 (OK / NG、瑕疵數)
    """

    def __init__(self,
                 weight_path: str,
                 device: str | None = None,
                 conf: float = 0.5,
                 iou: float = 0.45,
                 defect_classes: list[int] | None = None,
                 font_path: str = r"/app/fonts/NotoSansCJK-Regular.ttc"):
        """
        :param weight_path: 訓練好的 best.pt 路徑
        :param device: 'cuda' 或 'cpu'，不給就自動偵測
        :param conf: YOLO 信心閾值
        :param iou: YOLO NMS IoU 閾值
        :param defect_classes: 要算成「瑕疵」的 class id 列表，None = 全部框都算瑕疵
        """
        self.weight_path = weight_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.iou = iou
        self.defect_classes = defect_classes

        # 載入模型
        self.model = YOLO(self.weight_path)
        self.model.to(self.device)

        # 載入中文字型
        self.font = ImageFont.truetype(font_path, size=60)

    # ---------------------------
    # 共用：簡易計時 + 日誌（放在 class 裡）
    # ---------------------------
    @staticmethod
    def run_with_timer(tag, func, *args, **kwargs):
        """
        用法：
        inspector = YOLORealtimeInspector.run_with_timer(
            "載入模型", YOLORealtimeInspector, "best.pt"
        )
        """
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
    # 工具：把「模型預測」畫成四角點（放在 class 裡）
    # ---------------------------
    @staticmethod
    def draw_pred_four_points(image_bgr, result, color=(0, 255, 0),
                              r=6, thick=-1, with_score=True,
                              custom_labels=None):
        """
        custom_labels: list[str]，若提供，則用來取代原本的 cls:conf 顯示
        """
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

            if with_score:
                label_text = None

                # 優先使用自訂尺寸 label
                if custom_labels is not None and i < len(custom_labels):
                    if custom_labels[i] is not None:
                        label_text = str(custom_labels[i])

                # 沒給自訂的話就 fallback 回原本的 conf 顯示
                if label_text is None and conf is not None:
                    label_text = f"{(int(cls[i]) if cls is not None else 0)}:{conf[i]:.2f}"

                if label_text is not None:
                                        
                    font_scale = 0.9      # 字變大
                    thickness = 3         # 線條變粗

                    # 往右一點、稍微再往上一點
                    label_x = x1 + 8
                    label_y = max(0, y1 - 10)

                    cv2.putText(
                        canvas,
                        label_text,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )
                    
        return canvas


    # ---------------------------
    # 內部：計算每顆瑕疵 bbox 像素尺寸並分級
    # ---------------------------
    def _analyze_defect_sizes(self, result):
        """
        回傳:
        {
            "boxes": [
                {
                    "bbox": [x1, y1, x2, y2],
                    "w": w_px,
                    "h": h_px,
                    "area": area_px,
                    "category": "0.1mm" / "0.15mm" / "0.2mm" / ">0.2mm"
                },
                ...
            ],
            "counts": {... 各尺寸數量 ...}
        }
        """
        if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
            return {
                "boxes": [],
                "counts": {
                    "0.1mm": 0,
                    "0.2mm": 0,
                    "0.4mm": 0,
                    ">0.5mm": 0,
                },
            }

        xyxy = result.boxes.xyxy.detach().cpu().numpy()

        # 面積門檻
        t1 = 8.9 * 8.5          # 0.1mm 上限
        t2 = 15.1 * 15.9       # 0.15mm 上限
        t3 = 24.1 * 24       # 0.2mm 上限

        box_infos = []
        counts = {"0.1mm": 0, "0.2mm": 0, "0.4mm": 0, ">0.5mm": 0}

        for (x1, y1, x2, y2) in xyxy:
            w = float(x2 - x1)
            h = float(y2 - y1)
            area = w * h

            if area <= t1:
                cat = "0.1mm"
            elif area <= t2:
                cat = "0.2mm"
            elif area <= t3:
                cat = "0.4mm"
            else:
                cat = ">0.5mm"

            counts[cat] += 1
            box_infos.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "w": w,
                    "h": h,
                    "area": area,
                    "category": cat,
                }
            )

        return {"boxes": box_infos, "counts": counts}


    # ---------------------------
    # 內部：計算瑕疵數量
    # ---------------------------
    def _count_defects(self, result) -> int:
        if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
            return 0

        cls = result.boxes.cls.detach().cpu().numpy().astype(int)

        if self.defect_classes is None:
            return int(len(cls))

        mask = np.isin(cls, np.array(self.defect_classes, dtype=int))
        return int(mask.sum())

    # ---------------------------
    # 對單張 frame 做推論並畫結果
    # ---------------------------
    def infer_frame(self, frame_bgr):
        """
        :param frame_bgr: OpenCV 取得的 BGR 影像
        :return:
            draw_img: 已畫好結果的 BGR 影像
            info: dict, 包含 status('OK'/'NG')、num_defect、raw_result
        """
        results = self.model(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device
        )
        result = results[0]

        # 計算瑕疵數量
        num_defect = self._count_defects(result)

        if num_defect == 0:
            status = "OK"
            text = "结果: OK"
            color = (0, 255, 0)
        else:
            status = "NG"
            text = f"结果: NG 瑕疵{num_defect}顆"
            color = (0, 0, 255)

        # ★ 先計算每顆瑕疵的尺寸區間
        size_info = self._analyze_defect_sizes(result)
        # 取出每個 bbox 的 category，例如 "0.1mm" ...
        size_labels = [b["category"] for b in size_info["boxes"]]




        # 畫四角點
        draw_img = self.draw_pred_four_points(
            frame_bgr,
            result,
            color=color,
            with_score=True,
            custom_labels=size_labels,
        )

        h, w = draw_img.shape[:2]
        org = (int(w * 0.22), int(h * 0.92))  # 位置可自己調

        # --- 用 PIL 畫中文 ---
        # OpenCV BGR → PIL RGB
        rgb_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        draw = ImageDraw.Draw(pil_img)

        # PIL 用的是 RGB，所以顏色要反過來
        r, g, b = color[2], color[1], color[0]

        draw.text(org, text, font=self.font, fill=(r, g, b))

        # 再轉回 OpenCV BGR
        draw_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        info = {
            "status": status,
            "num_defect": num_defect,
            "raw_result": result,
            "size_info": size_info,
        }
        return draw_img, info
