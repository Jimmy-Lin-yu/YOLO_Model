# aoi_flat_processor.py
import os
import cv2
import numpy as np
from typing import Iterable, Tuple, List, Optional

class FlatFieldProcessor:
    """
    批次或單張的平場校正（flat-field）處理器。
    - 以高斯低頻背景除法去除大面積亮度不均/陰影。
    - 預設將結果輸出為 8-bit 灰階（以 128 為中性亮度）。

    參數
    ----
    flat_sigma : int | float
        高斯模糊的 sigma。建議 30~60：越大越能去大陰影。
    exts : Iterable[str]
        支援的影像副檔名。
    scale : float
        cv2.divide 的縮放，預設 128 讓結果落在可視灰階範圍。
    """
    def __init__(
        self,
        flat_sigma: float = 8,
        exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        scale: float = 128.0,
    ):
        self.flat_sigma = float(flat_sigma)
        self.exts = tuple(exts)
        self.scale = float(scale)

    # ---------- 公用 API ----------
    def process_image_array(self, img_bgr: np.ndarray) -> np.ndarray:
        """輸入 BGR 或灰階影像陣列，回傳平場後的 8-bit 灰階。"""
        gray = self._to_gray(img_bgr)
        return self._flatfield(gray)

    def process_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        處理單張檔案；若未給 output_path，會在同資料夾輸出 *_flat.png。
        回傳實際輸出路徑。
        """
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {input_path}")

        flat = self.process_image_array(img)
        if output_path is None:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(os.path.dirname(input_path), f"{base}_flat.png")

        self._ensure_dir_for_file(output_path)
        cv2.imwrite(output_path, flat)
        return output_path

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = False,
    ) -> Tuple[int, int, List[str]]:
        """
        批次處理資料夾。回傳 (ok_count, fail_count, fail_list)。
        若 recursive=True，會遞迴處理子資料夾並保持相對路徑。
        """
        paths = self._list_images(input_dir, recursive=recursive)
        if not paths:
            raise FileNotFoundError(f"No images found in: {input_dir}")

        ok, fail = 0, 0
        fail_list: List[str] = []
        for ipath in paths:
            rel = os.path.relpath(ipath, input_dir)
            base = os.path.splitext(rel)[0] + "_flat.png"
            opath = os.path.join(output_dir, base)

            try:
                img = cv2.imread(ipath, cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(ipath)
                flat = self.process_image_array(img)
                self._ensure_dir_for_file(opath)
                cv2.imwrite(opath, flat)
                ok += 1
                print(f"[OK] {ipath} -> {opath}")
            except Exception as e:
                print(f"[FAIL] {ipath}: {e}")
                fail += 1
                fail_list.append(ipath)

        print(f"\nDone. OK={ok}, FAIL={fail}. Output -> {output_dir}")
        return ok, fail, fail_list

    # ---------- 內部工具 ----------
    @staticmethod
    def _ensure_dir_for_file(p: str) -> None:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    def _list_images(self, folder: str, recursive: bool = False) -> List[str]:
        if not recursive:
            return sorted(
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(self.exts)
            )
        paths: List[str] = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(self.exts):
                    paths.append(os.path.join(root, f))
        return sorted(paths)

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    def _flatfield(self, gray: np.ndarray) -> np.ndarray:
        bg = cv2.GaussianBlur(gray, (0, 0), self.flat_sigma, self.flat_sigma)
        bg = np.maximum(bg, 1)  # 避免除以 0
        flat = cv2.divide(gray, bg, scale=self.scale)
        return flat.astype(np.uint8)


# ---------- 範例用法（直接改路徑執行） ----------
if __name__ == "__main__":
    # 1) 批次處理資料夾
    INPUT_DIR  = "/app/NG_10"
    OUTPUT_DIR = "/app/out_flat_only"
    proc = FlatFieldProcessor(flat_sigma=8)
    proc.process_folder(INPUT_DIR, OUTPUT_DIR, recursive=False)

    # 2) 或處理單張
    proc.process_file("/app/Tt_NG_4.jpg", "/app/out_flat_only/Tt_NG_4_flat.png")

    # 3) 或直接拿陣列用
    import cv2
    img = cv2.imread("/app/Tt_NG_4.jpg", cv2.IMREAD_COLOR)
    flat = proc.process_image_array(img)

# # aoi_flat_processor_residual.py
# import os, cv2, numpy as np
# from typing import Iterable, Tuple, List, Optional

# class FlatFieldProcessor:
#     """
#     批次或單張的平場校正處理器（加入「殘差不吃點」分支）
#     - 保留原本：Gaussian 平場 → 輸出 *_flat.png
#     - 新增：Residual 暗點偵測支路（可開關），輸出熱圖/遮罩/疊色，不會把小黑點一起磨掉
#     """

#     def __init__(
#         self,
#         flat_sigma: float = 45,                           # 平場的高斯 sigma（建議 30~60）
#         exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
#         scale: float = 128.0,

#         # === 殘差支路（可關） ===
#         use_residual_branch: bool = True,                # 打開：會另外輸出殘差熱圖/遮罩/疊色
#         base_sigma: float = 7.0,                         # 做超平滑基底的 Gaussian σ（要明顯大於瑕疵尺寸）
#         residual_k: float = 3.5,                         # z-score 門檻（3~4 常用）
#         min_area: int = 25,                              # 最小連通面積
#         edge_tau: float = 0.35,                          # 邊緣抑制；越小抑制越強(0.25~0.5)
#         heat_cap: float = 5.0,                           # 熱圖視覺上限：z/heat_cap -> 0~1
#     ):
#         self.flat_sigma = float(flat_sigma)
#         self.exts = tuple(exts)
#         self.scale = float(scale)

#         # residual 分支
#         self.use_residual_branch = bool(use_residual_branch)
#         self.base_sigma = float(base_sigma)
#         self.residual_k = float(residual_k)
#         self.min_area = int(min_area)
#         self.edge_tau = float(edge_tau)
#         self.heat_cap = float(heat_cap)

#     # ---------- 公用 API（相容舊版） ----------
#     def process_image_array(self, img_bgr: np.ndarray) -> np.ndarray:
#         """輸入 BGR/灰階，回傳『平場後』8-bit 灰階。"""
#         gray = self._to_gray(img_bgr)
#         return self._flatfield_gaussian(gray)

#     def process_file(self, input_path: str, output_path: Optional[str] = None) -> str:
#         """
#         只做平場並存檔（相容舊版）。
#         若你想同時輸出 residual 熱圖/遮罩/疊色，請改用 process_and_save_file()。
#         """
#         img = cv2.imread(input_path, cv2.IMREAD_COLOR)
#         if img is None:
#             raise FileNotFoundError(f"Cannot read image: {input_path}")

#         flat = self.process_image_array(img)
#         if output_path is None:
#             base = os.path.splitext(os.path.basename(input_path))[0]
#             output_path = os.path.join(os.path.dirname(input_path), f"{base}_flat.png")
#         self._ensure_dir_for_file(output_path)
#         cv2.imwrite(output_path, flat)
#         return output_path

#     def process_folder(self, input_dir: str, output_dir: str, recursive: bool = False) -> Tuple[int, int, List[str]]:
#         """
#         批次：平場 +（若開啟）殘差支路輸出。
#         會輸出：*_flat.png，若開 residual：再輸出 *_residual_heat.png / *_residual_mask.png / *_residual_overlay.png
#         """
#         paths = self._list_images(input_dir, recursive=recursive)
#         if not paths:
#             raise FileNotFoundError(f"No images found in: {input_dir}")

#         ok, fail = 0, 0
#         fails: List[str] = []
#         for ipath in paths:
#             try:
#                 self.process_and_save_file(ipath, output_dir)
#                 ok += 1
#             except Exception as e:
#                 print(f"[FAIL] {ipath}: {e}")
#                 fails.append(ipath); fail += 1
#         print(f"\nDone. OK={ok}, FAIL={fail}. Output -> {output_dir}")
#         return ok, fail, fails

#     # ---------- 新增：單張處理並把所有結果存到 output_dir ----------
#     def process_and_save_file(self, input_path: str, output_dir: str) -> None:
#         img = cv2.imread(input_path, cv2.IMREAD_COLOR)
#         if img is None:
#             raise FileNotFoundError(input_path)
#         gray = self._to_gray(img)

#         # 1) 平場（做為訓練主輸入）
#         flat = self._flatfield_gaussian(gray)

#         base = os.path.splitext(os.path.basename(input_path))[0]
#         op_flat = os.path.join(output_dir, f"{base}_flat.png")
#         self._ensure_dir_for_file(op_flat)
#         cv2.imwrite(op_flat, flat)

#         # 2) 殘差分支（可關閉）
#         if self.use_residual_branch:
#             mask, heat = self._dark_spot_residual(flat, s_base=self.base_sigma,
#                                                   k=self.residual_k, min_area=self.min_area,
#                                                   edge_tau=self.edge_tau, heat_cap=self.heat_cap)

#             op_heat = os.path.join(output_dir, f"{base}_residual_heat.png")
#             op_mask = os.path.join(output_dir, f"{base}_residual_mask.png")
#             op_ovl  = os.path.join(output_dir, f"{base}_residual_overlay.png")

#             cv2.imwrite(op_heat, heat)
#             cv2.imwrite(op_mask, mask)
#             cv2.imwrite(op_ovl, self._overlay_heat(flat, heat, alpha=0.65))

#         print(f"[OK] {input_path} -> {op_flat}")

#     # ---------- 內部：Gaussian 平場（修正 dtype，避免吃點） ----------
#     def _flatfield_gaussian(self, gray: np.ndarray) -> np.ndarray:
#         g  = gray.astype(np.float32)
#         bg = cv2.GaussianBlur(g, (0, 0), self.flat_sigma, self.flat_sigma)
#         bg = np.maximum(bg, 1e-6).astype(np.float32)           # 避免除 0
#         out32 = cv2.divide(g, bg, scale=self.scale, dtype=cv2.CV_32F)
#         return np.clip(out32, 0, 255).astype(np.uint8)

#     # ---------- 內部：暗點殘差偵測（不吃掉黑點） ----------
#     def _dark_spot_residual(self, gray_u8: np.ndarray,
#                              s_base: float = 7.0, k: float = 3.5, min_area: int = 25,
#                              edge_tau: float = 0.35, heat_cap: float = 5.0):
#         """
#         step1: 做超平滑基底 base（只拿來減光照/紋理，不直接當輸入）
#         step2: 殘差 res = base - gray（只留變暗）→ robust z-score（MAD）
#         step3: 邊緣抑制權重 w（Sobel）乘在 z 上：邊緣更不容易誤報
#         step4: z_w >= k → mask，過濾小面積。回傳 (mask, heatmap)
#         """
#         g = gray_u8.astype(np.float32)
#         base = cv2.GaussianBlur(g, (0, 0), s_base, s_base)     # 也可改 bilateral

#         res = (base - g)                                       # 暗殘差（亮點會是負的、被剪掉）
#         res[res < 0] = 0

#         # Robust noise estimation: MAD
#         med = np.median(res)
#         mad = np.median(np.abs(res - med)) + 1e-6
#         sigma = 1.4826 * mad
#         z = (res - med) / (sigma + 1e-6)

#         # 邊緣抑制（避免把結構邊緣當缺陷）
#         w = self._edge_weight(gray_u8, tau=edge_tau)           # 0~1，邊緣越小
#         z_w = z * w

#         # 二值化與小面積過濾
#         bw = (z_w >= k).astype(np.uint8) * 255
#         bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
#         n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
#         keep = np.zeros_like(bw)
#         for i in range(1, n):
#             if stats[i, cv2.CC_STAT_AREA] >= min_area:
#                 keep[lab == i] = 255

#         # 視覺熱圖（方便人眼檢視，不影響演算法）
#         heat = np.clip(z / heat_cap, 0, 1)
#         heat_u8 = (heat * 255).astype(np.uint8)
#         return keep, cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

#     # ---------- 小工具 ----------
#     @staticmethod
#     def _edge_weight(gray: np.ndarray, tau: float = 0.35) -> np.ndarray:
#         gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
#         gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
#         g  = np.sqrt(gx*gx + gy*gy)
#         g  = (g - g.min()) / (g.max() - g.min() + 1e-6)
#         w  = np.clip(1.0 - g / max(tau, 1e-3), 0.0, 1.0)
#         return cv2.GaussianBlur(w, (5,5), 0)

#     @staticmethod
#     def _overlay_heat(gray_u8: np.ndarray, heat_bgr: np.ndarray, alpha: float = 0.65) -> np.ndarray:
#         base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
#         return (base*(1-alpha) + heat_bgr.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)

#     @staticmethod
#     def _ensure_dir_for_file(p: str) -> None:
#         d = os.path.dirname(p)
#         if d: os.makedirs(d, exist_ok=True)

#     def _list_images(self, folder: str, recursive: bool = False) -> List[str]:
#         if not recursive:
#             return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(self.exts))
#         paths: List[str] = []
#         for root, _, files in os.walk(folder):
#             for f in files:
#                 if f.lower().endswith(self.exts):
#                     paths.append(os.path.join(root, f))
#         return sorted(paths)

#     @staticmethod
#     def _to_gray(img: np.ndarray) -> np.ndarray:
#         return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


# # ---------- 範例用法 ----------
# if __name__ == "__main__":
#     INPUT_DIR  = "/app/Test_NG_Back_14"
#     OUTPUT_DIR = "/app/out_flat_only"

#     # 推薦初始參數（依你的金屬件畫面）
#     proc = FlatFieldProcessor(
#         flat_sigma=10,            # 平場去陰影
#         use_residual_branch=True, # 開啟殘差分支，不吃掉黑點
#         base_sigma=5,           # 殘差基底尺度，要比瑕疵大（約 2~3x）
#         residual_k=10,           # z-score 門檻：越小抓得越多
#         min_area=25,              # 最小面積
#         edge_tau=0.35,            # 邊緣抑制
#     )

#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     proc.process_folder(INPUT_DIR, OUTPUT_DIR, recursive=False)





# flat_field_processor_v2.py
# import os, cv2, numpy as np
# from typing import Iterable, List, Tuple, Optional, Dict

# class FlatFieldProcessor:
#     """
#     平場處理器（v2：可選前處理模組）
#     - 保留原本的 Gaussian 平場 API（process_image_array/process_file/process_folder）
#     - 新增四個可切換模組：
#         1) 模板式平場（median flat）      -> use_template_flat
#         2) 高光 inpaint/白名單            -> use_glare_inpaint
#         3) local z-normalization          -> use_local_znorm
#         4) 多尺度 LoG/White-Tophat 通道   -> use_log_tophat
#     - 新增 process_pipeline(...) 可一次跑完整條（依 True/False 決定）
#     """

#     def __init__(
#         self,
#         flat_sigma: float = 45.0,
#         exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
#         scale: float = 128.0,

#         # 4 模組的開關 -----------------------------------------
#         use_template_flat: bool = False,
#         use_glare_inpaint: bool = False,
#         use_local_znorm: bool = False,
#         use_log_tophat: bool = False,

#         # (1) 模板式平場參數 -----------------------------------
#         flat_template: Optional[np.ndarray] = None,   # 你也可後續用 set_flat_template(...) 設定
#         eps_flat: float = 1e-6,

#         # (2) 高光 inpaint 參數 --------------------------------
#         glare_mask: Optional[np.ndarray] = None,  # 可直接給固定白名單 (0/255)
#         glare_p_high: float = 99.7,               # 自動偵測高亮分位數
#         glare_tophat_ks: int = 17,                # 補抓小亮突起
#         glare_open_ks: int = 3,
#         glare_dilate_px: int = 2,
#         inpaint_radius: int = 3,
#         blend_sigma: int = 7,
#         blend_alpha: float = 0.85,

#         # (3) local z-norm 參數 --------------------------------
#         znorm_win: int = 11,   # 建議 9~13
#         znorm_clip: float = 3.0,  # clip 到 ±3σ

#         # (4) LoG/Tophat 參數 ----------------------------------
#         log_sigmas: Tuple[float, ...] = (2.0, 3.0, 4.0, 5.0),
#         tophat_ks: int = 17,   # 結構元直徑，接近亮點直徑
#     ):
#         # 原本參數
#         self.flat_sigma = float(flat_sigma)
#         self.exts = tuple(exts)
#         self.scale = float(scale)

#         # 開關
#         self.use_template_flat = bool(use_template_flat)
#         self.use_glare_inpaint = bool(use_glare_inpaint)
#         self.use_local_znorm   = bool(use_local_znorm)
#         self.use_log_tophat    = bool(use_log_tophat)

#         # (1) 模板式平場
#         self.flat_template = flat_template  # 灰階 8-bit 或 float
#         self.eps_flat = float(eps_flat)

#         # (2) 高光 inpaint
#         self._glare_mask_fixed = glare_mask  # 若不為 None，直接使用
#         self.glare_p_high = float(glare_p_high)
#         self.glare_tophat_ks = int(glare_tophat_ks)
#         self.glare_open_ks = int(glare_open_ks)
#         self.glare_dilate_px = int(glare_dilate_px)
#         self.inpaint_radius = int(inpaint_radius)
#         self.blend_sigma = int(blend_sigma)
#         self.blend_alpha = float(blend_alpha)

#         # (3) z-norm
#         self.znorm_win = int(znorm_win)
#         self.znorm_clip = float(znorm_clip)

#         # (4) LoG/Tophat
#         self.log_sigmas = tuple(float(s) for s in log_sigmas)
#         self.tophat_ks = int(tophat_ks)

#     # ---------------- 公用 API（舊） ----------------
#     def process_image_array(self, img_bgr: np.ndarray) -> np.ndarray:
#         """輸入 BGR/灰階，回傳『Gaussian 平場後』灰階（相容舊版）。"""
#         gray = self._to_gray(img_bgr)
#         return self._flatfield_gaussian(gray)

#     def process_file(self, input_path: str, output_path: Optional[str] = None) -> str:
#         img = cv2.imread(input_path, cv2.IMREAD_COLOR)
#         if img is None:
#             raise FileNotFoundError(f"Cannot read image: {input_path}")
#         flat = self.process_image_array(img)
#         if output_path is None:
#             base = os.path.splitext(os.path.basename(input_path))[0]
#             output_path = os.path.join(os.path.dirname(input_path), f"{base}_flat.png")
#         self._ensure_dir_for_file(output_path)
#         cv2.imwrite(output_path, flat)
#         return output_path

#     def process_folder(self, input_dir: str, output_dir: str, recursive: bool = False) -> Tuple[int, int, List[str]]:
#         paths = self._list_images(input_dir, recursive=recursive)
#         if not paths:
#             raise FileNotFoundError(f"No images found in: {input_dir}")
#         ok, fail = 0, 0; fail_list: List[str] = []
#         for ipath in paths:
#             try:
#                 img = cv2.imread(ipath, cv2.IMREAD_COLOR)
#                 if img is None: raise FileNotFoundError(ipath)
#                 out = self.process_image_array(img)
#                 rel = os.path.relpath(ipath, input_dir); base = os.path.splitext(rel)[0] + "_flat.png"
#                 opath = os.path.join(output_dir, base); self._ensure_dir_for_file(opath)
#                 cv2.imwrite(opath, out); ok += 1
#                 print(f"[OK] {ipath} -> {opath}")
#             except Exception as e:
#                 print(f"[FAIL] {ipath}: {e}"); fail += 1; fail_list.append(ipath)
#         print(f"\nDone. OK={ok}, FAIL={fail}. Output -> {output_dir}")
#         return ok, fail, fail_list

#     # ---------------- 新：可切換完整流程 ----------------
#     def process_pipeline(self, img_bgr: np.ndarray, output_mode: str = "flat") -> Dict[str, np.ndarray]:
#         """
#         依 True/False 執行：
#           對位(外部做) -> [模板平場 or Gaussian平場] -> [高光inpaint] -> [local z-norm] -> [LoG/Tophat]
#         output_mode:
#           - "flat" : 回傳 {'flat': 灰階}（預設）
#           - "dict" : 回傳 {'flat','deglare','znorm','log_dark','tophat_white','fused'}
#           - "stack3": 回傳 {'stack3': 3ch影像}（平場/LoG/Tophat 組成，供可微調模型）
#         """
#         gray0 = self._to_gray(img_bgr)

#         # (1) 平場：模板優先，否則 Gaussian
#         if self.use_template_flat and self.flat_template is not None:
#             flat = self._flatfield_template(gray0, self.flat_template)
#         else:
#             flat = self._flatfield_gaussian(gray0)

#         # (2) 高光 inpaint
#         deglare = flat.copy()
#         if self.use_glare_inpaint:
#             gmask = self._glare_mask_fixed if self._glare_mask_fixed is not None else self._auto_glare_mask(flat)
#             deglare = self._inpaint_blend_gray(flat, gmask, r=self.inpaint_radius, sigma=self.blend_sigma, alpha=self.blend_alpha)

#         # (3) local z-norm（輸出 8-bit）
#         znorm = self._local_znorm_u8(deglare, win=self.znorm_win, clip=self.znorm_clip) if self.use_local_znorm else deglare

#         # (4) LoG / Tophat 分數圖（基於 znorm 後效果最好）
#         log_dark_u8, tophat_u8, fused_u8 = None, None, None
#         if self.use_log_tophat:
#             log_dark_u8 = self._dark_log_u8(znorm, self.log_sigmas)
#             tophat_u8   = self._white_tophat_u8(znorm, self.tophat_ks)
#             fused_u8    = np.maximum(log_dark_u8, tophat_u8)

#         # 輸出組合
#         if output_mode == "stack3":
#             # 用 3ch 組合一張輸入（適合可微調的模型）
#             if self.use_log_tophat and (log_dark_u8 is not None) and (tophat_u8 is not None):
#                 stack3 = cv2.merge([znorm, log_dark_u8, tophat_u8])
#             else:
#                 # 沒開 LoG/Tophat 時，複製成 3ch
#                 stack3 = cv2.merge([znorm, znorm, znorm])
#             return {"stack3": stack3}

#         if output_mode == "dict":
#             out = {"flat": flat, "deglare": deglare, "znorm": znorm}
#             if self.use_log_tophat:
#                 out.update({"log_dark": log_dark_u8, "tophat_white": tophat_u8, "fused": fused_u8})
#             return out

#         # default: "flat"（傳回最終灰階，若開 z-norm 就是 znorm，否則 deglare/flat）
#         return {"flat": znorm}

#     # ----------------- 模組工具（1）模板平場 -----------------
#     def set_flat_template(self, tpl_gray: np.ndarray) -> None:
#         """直接設定模板（灰階）。"""
#         self.flat_template = self._to_gray(tpl_gray)

#     def build_flat_template(self, image_paths: List[str], max_n: int = 30) -> np.ndarray:
#         """
#         由多張 OK 灰階生成「中位數模板」。假設已對位；若未對位請先外部處理。
#         """
#         paths = image_paths[:max_n]
#         imgs = []
#         for p in paths:
#             im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
#             if im is not None: imgs.append(im.astype(np.float32))
#         if not imgs: raise RuntimeError("No images to build flat template.")
#         stack = np.stack(imgs, axis=0)
#         tpl = np.median(stack, axis=0).astype(np.float32)
#         tpl_u8 = np.clip(tpl, 0, 255).astype(np.uint8)
#         self.flat_template = tpl_u8
#         return tpl_u8

#     def _flatfield_template(self, gray: np.ndarray, tpl: np.ndarray) -> np.ndarray:
#         g = gray.astype(np.float32); f = self._to_gray(tpl).astype(np.float32)
#         f = f / (np.mean(f) + self.eps_flat)
#         out = g / (f + self.eps_flat)
#         out = (self._norm01(out) * 255).astype(np.uint8)
#         return out

#     def _flatfield_gaussian(self, gray: np.ndarray) -> np.ndarray:
#         # 兩邊都用 float32，避免 dtype 衝突
#         g  = gray.astype(np.float32)
#         bg = cv2.GaussianBlur(g, (0, 0), self.flat_sigma, self.flat_sigma)
#         # 避免除以 0：用一個很小的常數保底
#         bg = np.maximum(bg, 1e-6).astype(np.float32)
#         # 指定輸出型別為 float32，再轉回 8-bit
#         out32 = cv2.divide(g, bg, scale=self.scale, dtype=cv2.CV_32F)
#         return np.clip(out32, 0, 255).astype(np.uint8)

#     # ----------------- 模組工具（2）高光 inpaint -----------------
#     def set_glare_mask(self, mask_u8: np.ndarray) -> None:
#         """設定固定白名單（0/255）。"""
#         self._glare_mask_fixed = (mask_u8 if mask_u8.ndim == 2 else cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY))

#     def _auto_glare_mask(self, gray: np.ndarray) -> np.ndarray:
#         p = float(np.percentile(gray.reshape(-1).astype(np.float32), self.glare_p_high))
#         m1 = (gray >= max(1, int(p))).astype(np.uint8) * 255
#         # white-tophat 補抓小亮點
#         se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.glare_tophat_ks, self.glare_tophat_ks))
#         opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)
#         wth = cv2.subtract(gray, opened)
#         q2 = float(np.percentile(wth.reshape(-1).astype(np.float32), 99.0))
#         m2 = (wth >= max(1, int(q2))).astype(np.uint8) * 255
#         m = cv2.bitwise_or(m1, m2)
#         if self.glare_open_ks > 1:
#             se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.glare_open_ks, self.glare_open_ks))
#             m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se2, iterations=1)
#         if self.glare_dilate_px > 0:
#             se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.glare_dilate_px+1, 2*self.glare_dilate_px+1))
#             m = cv2.dilate(m, se3, 1)
#         return m

#     def _inpaint_blend_gray(self, gray: np.ndarray, mask: np.ndarray, r=3, sigma=7, alpha=0.85) -> np.ndarray:
#         if mask is None or np.max(mask) == 0:
#             return gray.copy()
#         # inpaint 需要 3ch/1ch 8bit；對灰階直接 inpaint
#         inp = cv2.inpaint(gray, mask, r, cv2.INPAINT_TELEA)
#         w = cv2.GaussianBlur(mask, (sigma|1, sigma|1), 0).astype(np.float32) / 255.0
#         out = (gray.astype(np.float32) * (1 - alpha*w) + inp.astype(np.float32) * (alpha*w)).clip(0,255)
#         return out.astype(np.uint8)

#     # ----------------- 模組工具（3）local z-norm -----------------
#     def _local_znorm_u8(self, gray: np.ndarray, win: int = 11, clip: float = 3.0) -> np.ndarray:
#         f = gray.astype(np.float32)
#         mean = cv2.blur(f, (win, win))
#         mean2 = cv2.blur(f*f, (win, win))
#         std = np.sqrt(np.maximum(mean2 - mean*mean, 1e-6))
#         z = (f - mean) / std
#         z = np.clip(z, -clip, clip)
#         z01 = (z + clip) / (2.0*clip)
#         return (z01 * 255).astype(np.uint8)

#     # ----------------- 模組工具（4）LoG / White-Tophat ---------
#     def _dark_log_u8(self, gray: np.ndarray, sigmas: Tuple[float, ...]) -> np.ndarray:
#         g = gray.astype(np.float32)
#         resp = np.zeros_like(g, np.float32)
#         for s in sigmas:
#             lap = cv2.Laplacian(cv2.GaussianBlur(g, (0,0), s), cv2.CV_32F, ksize=3)
#             resp = np.maximum(resp, - (s*s) * lap)   # 暗點 -> 正
#         return (self._norm01(resp) * 255).astype(np.uint8)

#     def _white_tophat_u8(self, gray: np.ndarray, k: int) -> np.ndarray:
#         se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
#         opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)
#         wth = cv2.subtract(gray, opened)  # 亮凸
#         return (self._norm01(wth) * 255).astype(np.uint8)

#     # ----------------- 共用小工具 -----------------
#     @staticmethod
#     def _ensure_dir_for_file(p: str) -> None:
#         d = os.path.dirname(p);  os.makedirs(d, exist_ok=True) if d else None

#     def _list_images(self, folder: str, recursive: bool = False) -> List[str]:
#         if not recursive:
#             return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(self.exts))
#         paths: List[str] = []
#         for root, _, files in os.walk(folder):
#             for f in files:
#                 if f.lower().endswith(self.exts):
#                     paths.append(os.path.join(root, f))
#         return sorted(paths)

#     @staticmethod
#     def _to_gray(img: np.ndarray) -> np.ndarray:
#         return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

#     @staticmethod
#     def _norm01(x: np.ndarray) -> np.ndarray:
#         x = x.astype(np.float32); m, M = np.min(x), np.max(x)
#         if M - m < 1e-6: return np.zeros_like(x, np.float32)
#         return (x - m) / (M - m + 1e-6)


# if __name__ == "__main__":
#     # ========= 路徑與模式 =========
#     RUN_MODE = "folder"   # "folder" 或 "file"

#     # folder 模式
#     INPUT_DIR  = "/app/Test_NG_Back_14"
#     OUTPUT_DIR = "/app/out_flat_only"

#     # file  模式
#     INPUT_IMAGE  = "/app/Tt_NG_4.jpg"
#     OUTPUT_IMAGE = "/app/out_flat_only/Tt_NG_4_flat.png"

#     # ========= 前處理開關 =========
#     USE_TEMPLATE_FLAT = False  # True: 使用模板式平場；False: 用 Gaussian 平場
#     USE_GLARE_INPAINT = False   # True: 高光 inpaint/白名單遮蔽
#     USE_LOCAL_ZNORM   = False   # True: 啟用 local z-normalization
#     USE_LOG_TOPHAT    = False   # True: 產生 LoG/Tophat 分數（可供後融合）

#     # （可選）模板與白名單
#     TEMPLATE_PATH   = None                    # 例如 "/app/ok_template.png"
#     GLARE_MASK_PATH = None                    # 例如 "/app/whitelist_mask.png"（0/255）

#     # ========= 參數（可調）=========
#     FLAT_SIGMA = 30                       # Gaussian 平場 sigma（備用）
#     # 其餘參數用 class 預設即可；需要再調就傳進去

#     # ========= 建立處理器 =========
#     proc = FlatFieldProcessor(
#         flat_sigma=FLAT_SIGMA,
#         use_template_flat=USE_TEMPLATE_FLAT,
#         use_glare_inpaint=USE_GLARE_INPAINT,
#         use_local_znorm=USE_LOCAL_ZNORM,
#         use_log_tophat=USE_LOG_TOPHAT,
#     )

#     # 載入模板（若開啟）
#     if USE_TEMPLATE_FLAT and TEMPLATE_PATH and os.path.exists(TEMPLATE_PATH):
#         tpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
#         if tpl is None:
#             raise FileNotFoundError(TEMPLATE_PATH)
#         proc.set_flat_template(tpl)

#     # 載入白名單（若開啟）
#     if USE_GLARE_INPAINT and GLARE_MASK_PATH and os.path.exists(GLARE_MASK_PATH):
#         m = cv2.imread(GLARE_MASK_PATH, cv2.IMREAD_GRAYSCALE)
#         if m is None:
#             raise FileNotFoundError(GLARE_MASK_PATH)
#         proc.set_glare_mask(m)

#     # ========= 執行 =========
#     if RUN_MODE == "folder":
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
#         paths = proc._list_images(INPUT_DIR, recursive=False)
#         if not paths:
#             raise FileNotFoundError(f"No images found in: {INPUT_DIR}")

#         ok, fail = 0, 0
#         for ipath in paths:
#             img = cv2.imread(ipath, cv2.IMREAD_COLOR)
#             if img is None:
#                 print(f"[WARN] cannot read: {ipath}")
#                 fail += 1
#                 continue

#             # 只輸出「最終灰階」（視開關而定是 flat/deglare/znorm）
#             out_dict = proc.process_pipeline(img, output_mode="flat")
#             out_gray = out_dict["flat"]

#             rel = os.path.relpath(ipath, INPUT_DIR)
#             base = os.path.splitext(rel)[0] + "_flat.png"
#             opath = os.path.join(OUTPUT_DIR, base)
#             os.makedirs(os.path.dirname(opath), exist_ok=True)
#             cv2.imwrite(opath, out_gray)
#             ok += 1
#             print(f"[OK] {ipath} -> {opath}")

#         print(f"\nDone. OK={ok}, FAIL={fail}. Output -> {OUTPUT_DIR}")

#     elif RUN_MODE == "file":
#         img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
#         if img is None:
#             raise FileNotFoundError(INPUT_IMAGE)
#         out_gray = proc.process_pipeline(img, output_mode="flat")["flat"]
#         os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
#         cv2.imwrite(OUTPUT_IMAGE, out_gray)
#         print("[OK] saved:", OUTPUT_IMAGE)

#     else:
#         raise ValueError("RUN_MODE must be 'folder' or 'file'")
