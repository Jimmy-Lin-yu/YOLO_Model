import os
import cv2
import numpy as np
from cvat_sdk import make_client
import zipfile
from datetime import datetime
import yaml

class CVATDownloader:
    def __init__(self, host, username, password, project_id):
        self.host = host
        self.username = username
        self.password = password
        self.project_id = project_id
        # 初始化 CVAT client
        self.client = make_client(host=self.host)
        self.client.login((self.username, self.password))
        self.project = self.client.projects.retrieve(self.project_id)
        # 取得標籤對照：CVAT label_id -> label_name
        labels = {item.id: item.name for item in self.project.get_labels()}
        # 建立 class_map：從 label_id 到連續的 class index
        sorted_ids = sorted(labels.keys())
        self.class_map = {lid: idx for idx, lid in enumerate(sorted_ids)}
        self.labels = labels

    def prepare_folders(self, dataset_name):
        base_dir = os.path.join("dataset", dataset_name)
        folders = [
            os.path.join(base_dir, "Train", "images"),
            os.path.join(base_dir, "Train", "labels"),
            os.path.join(base_dir, "Test",  "images"),
            os.path.join(base_dir, "Test",  "labels"),
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            print(f"資料夾已建立：{folder}")

    def download_data(self, dataset_name):
        tasks = self.project.get_tasks()
        print(f"總共有 {len(tasks)} 個任務")

        for task in tasks:
            subset = task.subset
            if subset not in ("Train", "Test"):
                print(f"未定義資料屬性，跳過 task.id: {task.id}")
                continue

            ann = task.get_annotations()["shapes"]
            # 按影格分組
            frames = {}
            for shape in ann:
                # 不過濾任何形狀類型，全部下載
                frames.setdefault(shape.frame, []).append(shape)

            print(f"任務 {task.id} 共 {sum(len(v) for v in frames.values())} 個標註形狀")

            for frame_idx, shapes in frames.items():
                self._save_frame_and_label(task, frame_idx, shapes, subset, dataset_name)

    def _save_frame_and_label(self, task, frame_idx, shapes, subset, dataset_name):
        # 讀取影格
        try:
            frame = self._decode_frame(task.get_frame(frame_idx))
        except Exception as e:
            print(f"獲取任務 {task.id} 的幀 {frame_idx} 時出現錯誤: {e}")
            return

        h, w = frame.shape[:2]
        base = os.path.join("dataset", dataset_name, subset)
        img_path = os.path.join(base, "images", f"{task.id}_{frame_idx}.jpg")
        lbl_path = os.path.join(base, "labels", f"{task.id}_{frame_idx}.txt")

        # 儲存圖片
        cv2.imwrite(img_path, frame)
        # 寫入標註 (一行 per shape)
        with open(lbl_path, 'w') as f:
            for shape in shapes:
                lid = shape['label_id']
                cls = self.class_map[lid]
                pts = shape['points']  # 動態點列表
                # normalize 所有點座標
                norm = [round(coord / (w if idx % 2 == 0 else h), 6)
                        for idx, coord in enumerate(pts)]
                line = f"{cls} " + " ".join(map(str, norm)) + "\n"
                f.write(line)
        print(f"已儲存：{img_path} & {lbl_path}")

    def _decode_frame(self, frame_bytes):
        return cv2.imdecode(
            np.frombuffer(frame_bytes.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

    def create_dataset_yaml(self, dataset_name):
        # 建立 dataset.yaml，使用 Ultralytics YOLO segmentation 格式
        base_path = os.path.abspath(os.path.join("dataset", dataset_name))
        yaml_dict = {
            'path': base_path,
            'train': 'Train/images',
            'val':   'Test/images',
            'nc':    len(self.class_map),
            'names': [self.labels[lid] for lid in sorted(self.class_map.keys())]
        }
        yaml_path = os.path.join("dataset", dataset_name, "dataset.yaml")
        with open(yaml_path, 'w') as f:
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

if __name__ == '__main__':
    host = "http://192.168.0.173:8080/"
    username = "Jimmy"
    password = "301123350jIMMY"
    project_id = 1
    dataset_name = "YOLO_seg"

    dl = CVATDownloader(host, username, password, project_id)
    dl.prepare_folders(dataset_name)
    dl.download_data(dataset_name)
    dl.create_dataset_yaml(dataset_name)
    dl.compress_dataset(dataset_name)