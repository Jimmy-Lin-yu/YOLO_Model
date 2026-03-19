import os
import cv2
import numpy as np
import zipfile
import shutil
from datetime import datetime
import yaml
from cvat_sdk import make_client
import json

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class CVATDownloader:
    def __init__(self, host, username, password, project_id,
                 write_empty_label=True):
        """
        write_empty_label:
            True  -> 為 OK 圖片建立 0-byte 空白標註檔（之後統計更穩定）
            False -> OK 圖片不建立 labels 檔
        """
        self.host = host
        self.username = username
        self.password = password
        self.project_id = project_id
        self.write_empty_label = write_empty_label  #OK（無 shape）→ 空白標註

        # 初始化 CVAT client
        self.client = make_client(host=self.host)
        self.client.login((self.username, self.password))
        self.project = self.client.projects.retrieve(self.project_id)

        # 取得標籤：CVAT label_id -> label_name
        labels = {item.id: item.name for item in self.project.get_labels()}
        # class_map：label_id -> 連續 class index（依 id 排序）
        sorted_ids = sorted(labels.keys())
        self.class_map = {lid: idx for idx, lid in enumerate(sorted_ids)}
        self.labels = labels

    # ---------- 全量清空（安全刪除） ----------
    def clean_dataset_root(self, dataset_name: str):
        root = os.path.abspath(os.path.join("dataset", dataset_name))
        if not root.startswith(os.path.abspath("dataset") + os.sep):
            raise RuntimeError(f"Refusing to remove non-dataset path: {root}")
        if os.path.isdir(root):
            shutil.rmtree(root)
            print(f"🧹 已清空舊資料夾：{root}")

    def prepare_folders(self, dataset_name):
        base_dir = os.path.join("dataset", dataset_name)
        folders = [
            os.path.join(base_dir, "Train", "images"),
            os.path.join(base_dir, "Train", "labels"),
            os.path.join(base_dir, "Train", "attrs_instances"),
            os.path.join(base_dir, "Test",  "images"),
            os.path.join(base_dir, "Test",  "labels"),
            os.path.join(base_dir, "Test",  "attrs_instances"),
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            print(f"📁 資料夾已建立：{folder}")

    def download_data(self, dataset_name):
        tasks = self.project.get_tasks()
        print(f"總共有 {len(tasks)} 個任務")

        for task in tasks:
            subset = getattr(task, "subset", None)
            if subset not in ("Train", "Test"):
                print(f"未定義資料屬性，跳過 task.id: {task.id}")
                continue

            # 逐幀原始檔名對照表
            frame_name_map = self._build_frame_name_map(task)

            # 取 shapes 並按幀分組
            anns = task.get_annotations().get("shapes", [])
            shapes_by_frame = {}
            for shape in anns:
                shapes_by_frame.setdefault(shape.frame, []).append(shape)

            # 任務總幀數（為了包含沒有標註的 OK）
            total_frames = self._get_total_frames(task)
            print(f"任務 {task.id}：總幀數 {total_frames}，含標註幀數 {len(shapes_by_frame)}")

            for frame_idx in range(total_frames):
                shapes = shapes_by_frame.get(frame_idx, [])  # 可能無標註（OK）
                filename = self._get_frame_filename(task, frame_idx, frame_name_map)
                self._save_frame_and_label(task, frame_idx, shapes, subset, dataset_name, filename)

    # ---------- 取得每幀的原始檔名 ----------
    def _build_frame_name_map(self, task):
        name_map = {}
        try:
            infos = task.get_frames_info()
            for i, info in enumerate(infos):
                fname = getattr(info, "name", None)
                if fname is None and isinstance(info, dict):
                    fname = info.get("name")
                if fname:
                    name_map[i] = os.path.basename(fname)
        except Exception:
            pass

        if not name_map:
            try:
                meta = task.get_meta()
                frames = meta.get("frames") if isinstance(meta, dict) else None
                if isinstance(frames, list):
                    for i, fr in enumerate(frames):
                        if isinstance(fr, dict) and "name" in fr:
                            name_map[i] = os.path.basename(fr["name"])
            except Exception:
                pass

        return name_map

    def _get_frame_filename(self, task, frame_idx, frame_name_map):
        fname = frame_name_map.get(frame_idx)
        if not fname:
            fname = f"{task.id}_{frame_idx}.jpg"  # fallback
        fname = fname.replace("\\", "/").split("/")[-1]
        return fname

    def _get_total_frames(self, task):
        try:
            meta = task.get_meta()
            if isinstance(meta, dict) and "size" in meta:
                return int(meta["size"])
        except Exception:
            pass
        try:
            info = task.get_frames_info()
            return len(info)
        except Exception:
            pass
        print("⚠️ 無法取得完整幀數，將僅遍歷有標註的幀（可能漏 OK）。")
        return max([s.frame for s in task.get_annotations().get("shapes", [type('x', (object,), {'frame': 0})()])]) + 1

    # ---------------- 主要：儲存影像 / labels / attrs / attrs_instances ----------------
    def _save_frame_and_label(self, task, frame_idx, shapes, subset, dataset_name, filename):
        # 讀影格
        try:
            frame = self._decode_frame(task.get_frame(frame_idx))
        except Exception as e:
            print(f"獲取任務 {task.id} 的幀 {frame_idx} 時出錯: {e}")
            return

        h, w = frame.shape[:2]
        base = os.path.join("dataset", dataset_name, subset)
        img_out_dir = os.path.join(base, "images")
        lbl_out_dir = os.path.join(base, "labels")
        inst_dir = os.path.join(base, "attrs_instances")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        os.makedirs(inst_dir, exist_ok=True)

        img_path = os.path.join(img_out_dir, filename)
        stem, _ext = os.path.splitext(filename)
        lbl_path = os.path.join(lbl_out_dir, stem + ".txt")
        inst_json_path = os.path.join(inst_dir, stem + ".json")

        # 防同名（跨 task）
        if any(os.path.exists(p) for p in (img_path, lbl_path, inst_json_path)):
            stem = f"{stem}_t{task.id}"
            img_path = os.path.join(img_out_dir, stem + _ext)
            lbl_path = os.path.join(lbl_out_dir, stem + ".txt")
            inst_json_path = os.path.join(inst_dir, stem + ".json")

        # 寫圖片
        if not cv2.imwrite(img_path, frame):
            img_path = os.path.join(img_out_dir, stem + ".jpg")
            cv2.imwrite(img_path, frame)

        label_id_to_name = {lid: name for lid, name in self.labels.items()}

        # -------- OK（無 shape）→ 空白標註 + attrs=[] + instances=[] --------
        if len(shapes) == 0:
            if self.write_empty_label:
                open(lbl_path, 'w').close()
            with open(inst_json_path, "w", encoding="utf-8") as jf:
                json.dump([], jf, ensure_ascii=False, indent=2)
            print(f"已儲存 OK：{img_path} | labels 空 | attrs=[] | instances=[]")
            return

        # -------- NG → YOLO seg labels + attrs(list) + attrs_instances(list of dict) --------
        # 1) YOLO segmentation txt
        with open(lbl_path, 'w', encoding='utf-8') as f:
            for shape in shapes:
                lid = getattr(shape, 'label_id', None)
                if lid is None and isinstance(shape, dict):
                    lid = shape.get('label_id')
                cls = self.class_map[lid]
                pts = getattr(shape, 'points', None)
                if pts is None and isinstance(shape, dict):
                    pts = shape.get('points', [])
                norm = [round(coord / (w if idx % 2 == 0 else h), 6)
                        for idx, coord in enumerate(pts)]
                f.write(f"{cls} " + " ".join(map(str, norm)) + "\n")


        # 2) 逐顆 instances（你要的新 JSON，沒有 iou）
        instances = self._collect_instance_records(
            filename=filename, shapes=shapes, W=w, H=h, label_id_to_name=label_id_to_name
        )
        with open(inst_json_path, "w", encoding="utf-8") as jf:
            json.dump(instances, jf, ensure_ascii=False, indent=2)

        print(f"已儲存 NG：{img_path}\n  labels: {lbl_path}\n  inst  : {inst_json_path} -> {len(instances)} 筆")

    def _decode_frame(self, frame_bytes):
        return cv2.imdecode(np.frombuffer(frame_bytes.read(), np.uint8), cv2.IMREAD_COLOR)

    def create_dataset_yaml(self, dataset_name):
        base_path = os.path.abspath(os.path.join("dataset", dataset_name))
        yaml_dict = {
            'path': base_path,
            'train': 'Train/images',
            'val':   'Test/images',
            'nc':    len(self.class_map),
            'names': [self.labels[lid] for lid in sorted(self.class_map.keys())]
        }
        yaml_path = os.path.join("dataset", dataset_name, "dataset.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, allow_unicode=True)
        print(f"已建立 dataset.yaml：{yaml_path}")

    def compress_dataset(self, dataset_name):
        folder = os.path.join("dataset", dataset_name)
        date_str = datetime.now().strftime("%Y%m%d")
        zip_name = f"{dataset_name}_{date_str}.zip"
        zip_fp = os.path.join("dataset", zip_name)
        with zipfile.ZipFile(zip_fp, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for r, d, files in os.walk(folder):
                for file in files:
                    fp = os.path.join(r, file)
                    arc = os.path.relpath(fp, start=folder)
                    zipf.write(fp, arc)
        print(f"壓縮完成：{zip_fp}")

    # ---------- 統計（沿用） ----------
    @staticmethod
    def _is_image(p: str) -> bool:
        return os.path.splitext(p)[1].lower() in IMG_EXTS

    @staticmethod
    def _count_instances_in_label(label_file: str) -> int:
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                cnt = 0
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    parts = s.split()
                    if len(parts) >= 2:
                        cnt += 1
                return cnt
        except FileNotFoundError:
            return 0

    def summarize_dataset(self, dataset_name: str):
        base_dir = os.path.join("dataset", dataset_name)

        def scan_split(split: str):
            img_dir = os.path.join(base_dir, split, "images")
            lbl_dir = os.path.join(base_dir, split, "labels")
            if not os.path.isdir(img_dir):
                return {"images": 0, "ok_images": 0, "ng_images": 0, "ng_instances": 0}

            all_imgs = [p for p in sorted(os.listdir(img_dir)) if self._is_image(p)]
            n_images = len(all_imgs)

            n_ok_images = 0
            n_ng_images = 0
            n_ng_instances_total = 0

            for img_name in all_imgs:
                stem, _ = os.path.splitext(img_name)
                lbl_path = os.path.join(lbl_dir, stem + ".txt")
                n_inst = self._count_instances_in_label(lbl_path)
                if n_inst > 0:
                    n_ng_images += 1
                    n_ng_instances_total += n_inst
                else:
                    n_ok_images += 1

            return {
                "images": n_images,
                "ok_images": n_ok_images,
                "ng_images": n_ng_images,
                "ng_instances": n_ng_instances_total,
            }

        tr = scan_split("Train")
        te = scan_split("Test")

        print("\n================  資料統計  ================")
        print(f"[Train]  影像數：{tr['images']} | OK 張數：{tr['ok_images']} | NG 張數：{tr['ng_images']} | NG 標註實例總數：{tr['ng_instances']}")
        print(f"[Test ]  影像數：{te['images']} | OK 張數：{te['ok_images']} | NG 張數：{te['ng_images']} | NG 標註實例總數：{te['ng_instances']}")
        print("============================================\n")

        return {"Train": tr, "Test": te}

    def summarize_size_attributes(
        self,
        target_label_name: str = "NG",
        size_keys = (">0.2", "0.2", "0.15", "0.1")
    ):
        label_id_to_name = {lid: name for lid, name in self.labels.items()}
        subsets = ["Train", "Test"]

        def _new_counter():
            return {k: 0 for k in size_keys} | {"others": 0}

        stats = {s: _new_counter() for s in subsets}
        stats["Total"] = _new_counter()

        def _norm_val(v: str) -> str:
            s = str(v).strip().lower().replace(" ", "")
            if s.endswith("x"):
                s = s[:-1]
            return s

        tasks = self.project.get_tasks()
        for task in tasks:
            subset = getattr(task, "subset", None)
            if subset not in subsets:
                continue
            anns = task.get_annotations().get("shapes", [])
            for sh in anns:
                lid = getattr(sh, "label_id", None)
                if lid is None and isinstance(sh, dict):
                    lid = sh.get("label_id")
                if lid is None or label_id_to_name.get(lid) != target_label_name:
                    continue
                attrs = getattr(sh, "attributes", None)
                if attrs is None and isinstance(sh, dict):
                    attrs = sh.get("attributes", [])
                hit_key = None
                if isinstance(attrs, list):
                    for a in attrs:
                        val = None
                        if hasattr(a, "value"):
                            val = a.value
                        elif isinstance(a, dict):
                            val = a.get("value")
                        if val is None:
                            continue
                        nv = _norm_val(val)
                        for k in size_keys:
                            if _norm_val(k) == nv:
                                hit_key = k
                                break
                        if hit_key:
                            break
                if hit_key is None:
                    stats[subset]["others"] += 1
                    stats["Total"]["others"] += 1
                else:
                    stats[subset][hit_key] += 1
                    stats["Total"][hit_key] += 1

        print("\n====== Size Attribute Summary (NG only) ======")
        for split in ["Train", "Test", "Total"]:
            row = " | ".join([f"{k}: {stats[split][k]}" for k in (*size_keys, "others")])
            print(f"{split:>5}: {row}")
        print("=============================================\n")

        return stats

    def plot_dataset_and_size_summary(
        self,
        dataset_name: str,
        ds_stats: dict,
        size_stats: dict,
        out_path: str = None
    ):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        size_keys = [">0.2", "0.2", "0.15", "0.1"]
        if out_path is None:
            out_path = os.path.join("dataset", dataset_name, "summary.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 左表格
        tbl_rows = []
        for split in ["Train", "Test"]:
            s = ds_stats.get(split, {})
            tbl_rows.append([split, s.get("images", 0), s.get("ok_images", 0),
                             s.get("ng_images", 0), s.get("ng_instances", 0)])

        # 右圖
        train_counts = [size_stats.get("Train", {}).get(k, 0) for k in size_keys]
        test_counts  = [size_stats.get("Test",  {}).get(k, 0) for k in size_keys]

        plt.figure(figsize=(14, 6), dpi=180)
        gs = GridSpec(1, 2, width_ratios=[1.1, 1.9])

        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        table_data = [["Split", "Images", "OK Images", "NG Images", "NG Instances"], *tbl_rows]
        tbl = ax1.table(cellText=table_data, loc='center', cellLoc='center',
                        colColours=['#E6EEF8']*5)
        tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.2, 1.6)
        ax1.set_title("Dataset Summary", fontsize=14, pad=10)

        ax2 = plt.subplot(gs[1])
        x = np.arange(len(size_keys)); w = 0.36
        b1 = ax2.bar(x - w/2, train_counts, width=w, label='Train')
        b2 = ax2.bar(x + w/2, test_counts,  width=w, label='Test')
        ax2.set_xticks(x); ax2.set_xticklabels(size_keys, fontsize=11)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("NG size distribution by split", fontsize=14)
        ax2.legend(frameon=False)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax2.annotate(f"{int(h)}", xy=(bar.get_x()+bar.get_width()/2, h),
                             xytext=(0, 4), textcoords="offset points",
                             ha='center', va='bottom', fontsize=10)

        plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()
        print(f"🖼️ 已輸出統整圖：{out_path}")
        return out_path

    # ----------------- 工具：屬性 / bbox / instances -----------------
    _SIZE_KEYS = (">0.2", "0.2", "0.15", "0.1")
    _TARGET_LABEL_NAME = "NG"

    @staticmethod
    def _norm_attr(v: str) -> str:
        s = str(v).strip().lower().replace(" ", "")
        if s.endswith("x"):
            s = s[:-1]
        return s

    @staticmethod
    def _categorize_size_bucket_by_box(x_ratio: float):
        if x_ratio > 0.2:   return ">0.2"
        if x_ratio >= 0.2:  return "0.2"
        if x_ratio >= 0.15: return "0.15"
        if x_ratio >= 0.1:  return "0.1"
        return None

    @staticmethod
    def _bbox_from_points(pts, W, H):
        xs = np.array(pts[0::2]); ys = np.array(pts[1::2])
        xs = np.clip(xs, 0, W - 1); ys = np.clip(ys, 0, H - 1)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        return x1, y1, x2, y2

    def _collect_size_attrs_from_shapes(self, shapes, W, H, label_id_to_name):
        buckets = []
        keys_norm = {self._norm_attr(k): k for k in self._SIZE_KEYS}
        for sh in shapes:
            lid = getattr(sh, "label_id", None)
            if lid is None and isinstance(sh, dict):
                lid = sh.get("label_id")
            if lid is None or label_id_to_name.get(lid) != self._TARGET_LABEL_NAME:
                continue
            attrs = getattr(sh, "attributes", None)
            if attrs is None and isinstance(sh, dict):
                attrs = sh.get("attributes", [])
            hit = None
            if isinstance(attrs, list):
                for a in attrs:
                    val = None
                    if hasattr(a, "value"): val = a.value
                    elif isinstance(a, dict): val = a.get("value")
                    if val is None: continue
                    nv = self._norm_attr(val)
                    if nv in keys_norm:
                        hit = keys_norm[nv]; break
            if hit is None:
                pts = getattr(sh, "points", None)
                if pts is None and isinstance(sh, dict):
                    pts = sh.get("points", [])
                if pts:
                    x1, y1, x2, y2 = self._bbox_from_points(pts, W, H)
                    ratio = max((x2 - x1) / W, (y2 - y1) / H)
                    hit = self._categorize_size_bucket_by_box(ratio)
            if hit is not None and hit not in buckets:
                buckets.append(hit)
        return buckets

    def _collect_instance_records(self, filename: str, shapes, W: int, H: int, label_id_to_name: dict):
        """
        逐顆輸出：
        {
          "filename": <檔名>,
          "bbox": [x1,y1,x2,y2],   # 像素
          "attr": "0.2" / ">0.2" / "0.15" / "0.1" / "",  # 若無屬性→空字串
          "label": "<label_name>"
        }
        """
        records = []
        keys_norm = {self._norm_attr(k): k for k in self._SIZE_KEYS}

        for sh in shapes:
            lid = getattr(sh, "label_id", None)
            if lid is None and isinstance(sh, dict):
                lid = sh.get("label_id")
            label_name = label_id_to_name.get(lid, str(lid))

            pts = getattr(sh, "points", None)
            if pts is None and isinstance(sh, dict):
                pts = sh.get("points", [])
            if not pts:
                continue
            x1, y1, x2, y2 = self._bbox_from_points(pts, W, H)

            # 找屬性
            attr_val = ""
            attrs = getattr(sh, "attributes", None)
            if attrs is None and isinstance(sh, dict):
                attrs = sh.get("attributes", [])
            if isinstance(attrs, list):
                for a in attrs:
                    val = None
                    if hasattr(a, "value"): val = a.value
                    elif isinstance(a, dict): val = a.get("value")
                    if val is None: continue
                    nv = self._norm_attr(val)
                    if nv in keys_norm:
                        attr_val = keys_norm[nv]
                        break

            # 若完全沒屬性可選，保留空字串；（可改：fallback 以 bbox 比例估）
            records.append({
                "filename": filename,
                "bbox": [x1, y1, x2, y2],
                "attr": attr_val,
                "label": label_name
            })
        return records


if __name__ == '__main__':
    host = "http://192.168.0.5:8080/"
    username = "Jimmy"
    password = "301123350jIMMY"
    project_id = 8
    dataset_name = "YOLO_seg"

    dl = CVATDownloader(host, username, password, project_id,
                        write_empty_label=True)

    dl.clean_dataset_root(dataset_name)
    dl.prepare_folders(dataset_name)

    dl.download_data(dataset_name)
    dl.create_dataset_yaml(dataset_name)

    # 統計 + 圖
    stats = dl.summarize_dataset(dataset_name)
    size_stats = dl.summarize_size_attributes(
        target_label_name="NG",
        size_keys=( ">0.2", "0.2", "0.15", "0.1" )
    )
    dl.plot_dataset_and_size_summary(dataset_name, stats, size_stats)

    dl.compress_dataset(dataset_name)
