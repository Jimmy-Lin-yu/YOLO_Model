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
        self.client = make_client(host=self.host)
        self.client.login((self.username, self.password))
        self.project = self.client.projects.retrieve(self.project_id)
        self.labels = {item.id: item.name for item in self.project.get_labels()}

    def download_data(self, dataset_name):
        base_dir = os.path.join("dataset", dataset_name)
        os.makedirs(base_dir, exist_ok=True)

        tasks = self.project.get_tasks()
        print(f"總共有 {len(tasks)} 個任務")
        for task in tasks:
            subset = task.subset
            if subset not in ("Train", "Test"):
                print(f"未定義資料屬性，跳過 task.id: {task.id}")
                continue

            annotations = task.get_annotations()["shapes"]
            print(f"任務 {task.id} 有 {len(annotations)} 個標注")

            frames_dict = {}
            for shape in annotations:
                frames_dict.setdefault(shape.frame, []).append(shape)

            for frame_idx, shapes in frames_dict.items():
                self._save_frame_and_label(task, frame_idx, shapes, subset, dataset_name)

    def prepare_folders(self, dataset_name):
        base_dir = os.path.join("dataset", dataset_name)
        folders = [
            os.path.join(base_dir, "Train", "images"),
            os.path.join(base_dir, "Train", "labels"),  # ✅ 改為 labels
            os.path.join(base_dir, "Test", "images"),
            os.path.join(base_dir, "Test", "labels"),   # ✅ 改為 labels
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            print(f"資料夾已建立：{folder}")

    def _save_frame_and_label(self, task, frame_idx, shapes, subset, dataset_name):
        try:
            frame = self._decode_frame(task.get_frame(frame_idx))
        except Exception as e:
            print(f"獲取任務 {task.id} 的幀 {frame_idx} 時出現錯誤: {str(e)}")
            return

        image_height, image_width = frame.shape[:2]
        base_subset = os.path.join("dataset", dataset_name, subset)
        image_path = os.path.join(base_subset, "images", f"{task.id}_{frame_idx}.jpg")
        label_path = os.path.join(base_subset, "labels", f"{task.id}_{frame_idx}.txt")  # ✅ 改為 labels

        print(f"保存圖像到: {image_path}")
        cv2.imwrite(image_path, frame)

        print(f"保存標注到: {label_path}")
        with open(label_path, 'w') as f:
            for shape in shapes:
                label_id = 0
                #label_id = shape["label_id"]
                points = shape['points']
                normalized_points = [
                    round(points[i] / (image_width if i % 2 == 0 else image_height), 5)
                    for i in range(8)
                ]
                line = f"0 {' '.join(map(str, normalized_points))}\n"
                f.write(line)

    def _decode_frame(self, frame_bytes):
        return cv2.imdecode(np.frombuffer(frame_bytes.read(), np.uint8), cv2.IMREAD_COLOR)

    def compress_dataset(self, dataset_name):
        dataset_folder = os.path.join("dataset", dataset_name)
        date_str = datetime.now().strftime("%Y%m%d")
        zip_filename = f"{dataset_name}_{date_str}.zip"
        zip_filepath = os.path.join("dataset", zip_filename)
        print(f"開始壓縮 {dataset_folder} 到 {zip_filepath}")
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dataset_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=dataset_folder)
                    zipf.write(file_path, arcname)
        print(f"壓縮完成：{zip_filepath}")

    def create_dataset_yaml(self, dataset_name):
        yaml_path = os.path.join("dataset", dataset_name, "dataset.yaml")
        yaml_dict = {
            "train": f"/workspace/dataset/{dataset_name}/Train/images",
            "val": f"/workspace/dataset/{dataset_name}/Test/images",
            "nc": len(self.labels),
            "names": list(self.labels.values())
        }

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, allow_unicode=True)

        print(f"已建立 dataset.yaml：{yaml_path}")


if __name__ == '__main__':
    host = "http://192.168.1.8:8080/"
    username = "admin"    
    password = "301123350jIMMY"
    project_id = 1
    dataset_name = "YOLO_seg"

    downloader = CVATDownloader(host, username, password, project_id)
    downloader.prepare_folders(dataset_name)
    downloader.download_data(dataset_name)
    downloader.create_dataset_yaml(dataset_name)
    downloader.compress_dataset(dataset_name)
