# pipeline_main.py
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

from yolo_defect import YOLODefectInspector
from yolo_sensor import YOLOSensor
from camera_thread import CameraThread


# -------------------------
# Data (with precise timing)
# -------------------------
@dataclass
class TriggerEvent:
    cam_id: int
    cycle_id: int
    t_first_hit: float   # perf_counter: first time sensor sees object in this cycle
    t_trigger: float     # perf_counter: when we enqueue trigger
    mono_full: np.ndarray


@dataclass
class PLCPayload:
    cam_id: int
    cycle_id: int
    status: str
    num_defect: int

    # perf timing
    t_first_hit: float
    t_trigger: float
    t_plc_ready: float   # perf_counter: after snapshot+infer done (right before enqueue PLC)

    # breakdown (ms)
    q_wait_ms: float         # trigger->defect_worker_get
    snap_ms: float
    infer_ms: float
    trigger_to_snap_ms: float
    trigger_to_infer_ms: float


@dataclass
class PersistPayload:
    cam_id: int
    ts_wall: float
    status: str
    num_defect: int
    snap_bgr: np.ndarray
    log: Dict[str, Any]


@dataclass
class UIPayload:
    cam_id: int
    ts_wall: float
    status: str
    num_defect: int
    snap_bgr: np.ndarray
    boxes: List[List[int]]
    labels: List[str]


class CycleLogger:
    """
    Print FULL CT when next object first appears:
    prev(sensor_first_hit) -> next(sensor_first_hit)
    Also show PLC->next_sensor waiting time.
    """
    def __init__(self, num_cams: int):
        self._lock = threading.Lock()
        self._last_done: List[Optional[Dict[str, Any]]] = [None] * num_cams

    def note_plc_sent(self, cam_id: int, rec: Dict[str, Any]):
        with self._lock:
            self._last_done[cam_id] = rec

    def maybe_print_full_ct_on_next_first_hit(self, cam_id: int, t_next_first_hit: float):
        with self._lock:
            prev = self._last_done[cam_id]

        if not prev:
            return

        if prev.get("t_first_hit", 0.0) <= 0.0 or prev.get("t_plc_sent", 0.0) <= 0.0:
            return

        full_ct_ms = (t_next_first_hit - prev["t_first_hit"]) * 1000.0
        wait_after_plc_ms = (t_next_first_hit - prev["t_plc_sent"]) * 1000.0

        print(
            f"[CT_FULL] cam={cam_id} prev_cycle={prev['cycle_id']} "
            f"sensor->next_sensor={full_ct_ms:.1f}ms "
            f"PLC->next_sensor={wait_after_plc_ms:.1f}ms "
            f"(prev sensor->PLC={prev.get('sensor_to_plc_ms', 0.0):.1f}ms status={prev.get('status','?')} defects={prev.get('num_defect','?')})"
        )


# -------------------------
# Workers
# -------------------------
class SensorWorker:
    """
    sensor(偵測到足粒) -> trigger event
    - strict serialize_per_cam: if busy, do NOT run sensor (so chain becomes PLC->next sensor)
    """
    def __init__(
        self,
        sensor: YOLOSensor,
        cams: List[CameraThread],
        trigger_q: Queue,
        camera_on_evt: threading.Event,
        model_on_evt: threading.Event,

        busy_flags: List[threading.Event],
        serialize_per_cam: bool,
        cycle_logger: CycleLogger,

        per_cam_interval_s: float = 0.03,
        trigger_after_n: int = 2,
        cooldown_s: float = 0.1,
        min_det: int = 1,
    ):
        self.sensor = sensor
        self.cams = cams
        self.trigger_q = trigger_q

        self.camera_on_evt = camera_on_evt
        self.model_on_evt = model_on_evt

        self.busy_flags = busy_flags
        self.serialize_per_cam = serialize_per_cam
        self.cycle_logger = cycle_logger

        self.per_cam_interval_s = per_cam_interval_s
        self.trigger_after_n = max(1, trigger_after_n)
        self.cooldown_s = cooldown_s
        self.min_det = min_det

        n = len(cams)
        self._last_check = [0.0] * n
        self._hit_streak = [0] * n
        self._in_object = [False] * n
        self._last_trigger_wall = [0.0] * n

        # for first-hit timing (per object)
        self._has_first_hit = [False] * n
        self._t_first_hit = [0.0] * n

        # cycle id increases ONLY when we actually trigger
        self._cycle_id = [0] * n

        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join(timeout=2)

    def _run(self):
        rr = 0
        n = len(self.cams)

        while not self._stop.is_set():
            if not self.camera_on_evt.is_set() or not self.model_on_evt.is_set():
                time.sleep(0.02)
                continue

            cam_id = rr % n
            rr += 1

            # strict: PLC done then next sensor (so full CT matches your chain)
            if self.serialize_per_cam and self.busy_flags[cam_id].is_set():
                time.sleep(0.001)
                continue

            now_wall = time.time()
            if now_wall - self._last_check[cam_id] < self.per_cam_interval_s:
                time.sleep(0.001)
                continue
            self._last_check[cam_id] = now_wall

            if now_wall - self._last_trigger_wall[cam_id] < self.cooldown_s:
                continue

            # 影像取得(從camerathread)
            frame_bgr, _ = self.cams[cam_id].get_latest_bgr_small()
            if frame_bgr is None:
                frame_bgr, _ = self.cams[cam_id].get_latest_bgr()
                if frame_bgr is None:
                    continue

            t_now = time.perf_counter()
            has_obj, _ = self.sensor.has_object(frame_bgr, min_det=self.min_det)

            if not has_obj:
                self._hit_streak[cam_id] = 0
                self._in_object[cam_id] = False
                self._has_first_hit[cam_id] = False
                continue

            # first hit for this object
            if not self._has_first_hit[cam_id]:
                self._t_first_hit[cam_id] = t_now
                self._has_first_hit[cam_id] = True

                # when next object first appears, we can print FULL CT of previous cycle
                self.cycle_logger.maybe_print_full_ct_on_next_first_hit(cam_id, t_now)

            self._hit_streak[cam_id] += 1

            if self._hit_streak[cam_id] < self.trigger_after_n:
                continue

            if self._in_object[cam_id]:
                continue

            # TRIGGER
            self._cycle_id[cam_id] += 1
            cycle_id = self._cycle_id[cam_id]

            t_trigger = time.perf_counter()
            evt = TriggerEvent(
                cam_id=cam_id,
                cycle_id=cycle_id,
                t_first_hit=self._t_first_hit[cam_id],
                t_trigger=t_trigger,
                mono_full=self.cams[cam_id].get_latest_pack().mono_full,
            )

            try:
                self.trigger_q.put(evt, timeout=0.05)
                if self.serialize_per_cam:
                    self.busy_flags[cam_id].set()
            except Exception:
                # enqueue failed -> do NOT block this cam
                pass

            self._in_object[cam_id] = True
            self._hit_streak[cam_id] = 0
            self._last_trigger_wall[cam_id] = time.time()
            # keep first_hit info until next object reset


class DefectWorker:
    """
    trigger -> snapshot(full) -> defect infer -> PLC / persist / UI
    If fail, clear busy to avoid deadlock.
    """
    def __init__(
        self,
        defect: YOLODefectInspector,
        cams: List[CameraThread],
        trigger_q: Queue,
        plc_q: Queue,
        persist_q: Queue,
        ui_q: Queue,
        camera_on_evt: threading.Event,
        model_on_evt: threading.Event,

        busy_flags: List[threading.Event],
        serialize_per_cam: bool,

        snapshot_timeout: float = 0.5,
    ):
        self.defect = defect
        self.cams = cams
        self.trigger_q = trigger_q
        self.plc_q = plc_q
        self.persist_q = persist_q
        self.ui_q = ui_q
        self.camera_on_evt = camera_on_evt
        self.model_on_evt = model_on_evt

        self.busy_flags = busy_flags
        self.serialize_per_cam = serialize_per_cam

        self.snapshot_timeout = snapshot_timeout

        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join(timeout=2)

    def _clear_busy(self, cam_id: int):
        if self.serialize_per_cam:
            self.busy_flags[cam_id].clear()

    def _run(self):
        while not self._stop.is_set():
            if not self.camera_on_evt.is_set() or not self.model_on_evt.is_set():
                time.sleep(0.02)
                continue

            try:
                evt: TriggerEvent = self.trigger_q.get(timeout=0.2)
            except Empty:
                continue

            t_evt_get = time.perf_counter()
            q_wait_ms = (t_evt_get - evt.t_trigger) * 1000.0

            camt = self.cams[evt.cam_id]

            # snapshot timing
            t_snap0 = time.perf_counter()
            pack = camt.request_snapshot_pack(timeout=self.snapshot_timeout)
            if pack is None:
                continue  # 或 return / handle timeout

            t_snap1 = time.perf_counter()

            # 影像取得(從sensorthread)
            # 強制 sensor 與 defect 一定同一張
            snap_bgr = cv2.cvtColor(evt.mono_full, cv2.COLOR_GRAY2BGR) 

            # infer timing
            t_inf0 = time.perf_counter()
            _, info = self.defect.infer(snap_bgr, draw=False)
            t_inf1 = time.perf_counter()

            status = str(info["status"])
            num_defect = int(info["num_defect"])

            snap_ms = (t_snap1 - t_snap0) * 1000.0
            infer_ms = (t_inf1 - t_inf0) * 1000.0
            trigger_to_snap_ms = (t_snap1 - evt.t_trigger) * 1000.0
            trigger_to_infer_ms = (t_inf1 - evt.t_trigger) * 1000.0

            t_plc_ready = time.perf_counter()

            # ---- PLC payload (if fail -> clear busy)
            p_plc = PLCPayload(
                cam_id=evt.cam_id,
                cycle_id=evt.cycle_id,
                status=status,
                num_defect=num_defect,
                t_first_hit=evt.t_first_hit,
                t_trigger=evt.t_trigger,
                t_plc_ready=t_plc_ready,
                q_wait_ms=q_wait_ms,
                snap_ms=snap_ms,
                infer_ms=infer_ms,
                trigger_to_snap_ms=trigger_to_snap_ms,
                trigger_to_infer_ms=trigger_to_infer_ms,
            )

            try:
                self.plc_q.put(p_plc, timeout=0.05)
            except Exception:
                self._clear_busy(evt.cam_id)
                continue

            # ---- persist (does NOT block PLC chain)
            try:
                ts_wall = time.time()
                size_info = info.get("size_info", {})
                boxes: List[List[int]] = []
                labels: List[str] = []
                for b in size_info.get("boxes", []):
                    x1, y1, x2, y2 = b["bbox"]
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    labels.append(str(b.get("category", "")))

                log = {
                    "cam_id": evt.cam_id,
                    "cycle_id": evt.cycle_id,
                    "ts_wall": ts_wall,
                    "status": status,
                    "num_defect": num_defect,
                    "q_wait_ms": q_wait_ms,
                    "snap_ms": snap_ms,
                    "infer_ms": infer_ms,
                    "sensor_to_trigger_ms": (evt.t_trigger - evt.t_first_hit) * 1000.0,
                }

                self.persist_q.put_nowait(PersistPayload(evt.cam_id, ts_wall, status, num_defect, snap_bgr, log))
                self.ui_q.put_nowait(UIPayload(evt.cam_id, ts_wall, status, num_defect, snap_bgr, boxes, labels))
            except Exception:
                pass


class PLCWorker:
    """
    Prints CT:
    - [CT] : sensor(first_hit) -> trigger -> snapshot -> infer -> PLC(sent)
    - [CT_FULL] printed in SensorWorker when next object first appears.
    Also clears busy flag (so next sensor can run).
    """
    def __init__(
        self,
        plc_q: Queue,
        model_on_evt: threading.Event,
        num_cams: int,
        busy_flags: List[threading.Event],
        serialize_per_cam: bool,
        cycle_logger: CycleLogger,
    ):
        self.plc_q = plc_q
        self.model_on_evt = model_on_evt
        self.num_cams = num_cams

        self.busy_flags = busy_flags
        self.serialize_per_cam = serialize_per_cam
        self.cycle_logger = cycle_logger

        self._last_plc_sent = [None] * num_cams  # perf_counter

        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            try:
                p: PLCPayload = self.plc_q.get(timeout=0.2)
            except Empty:
                continue

            # even if model is turned off, clear busy to avoid deadlock
            t_plc_sent = time.perf_counter()

            sensor_to_trigger_ms = (p.t_trigger - p.t_first_hit) * 1000.0
            trigger_to_plc_ms = (t_plc_sent - p.t_trigger) * 1000.0
            sensor_to_plc_ms = (t_plc_sent - p.t_first_hit) * 1000.0

            last_plc = self._last_plc_sent[p.cam_id]
            plc_to_plc_ms = None if last_plc is None else (t_plc_sent - last_plc) * 1000.0
            self._last_plc_sent[p.cam_id] = t_plc_sent

            msg = (
                f"[CT] cam={p.cam_id} cycle={p.cycle_id} status={p.status} defects={p.num_defect} "
                f"sensor->trigger={sensor_to_trigger_ms:.1f}ms "
                f"qwait={p.q_wait_ms:.1f}ms "
                f"trigger->snapshot={p.trigger_to_snap_ms:.1f}ms "
                f"snapshot={p.snap_ms:.1f}ms "
                f"infer={p.infer_ms:.1f}ms "
                f"trigger->PLC={trigger_to_plc_ms:.1f}ms "
                f"sensor->PLC={sensor_to_plc_ms:.1f}ms "
            )
            if plc_to_plc_ms is not None:
                msg += f"PLC->PLC={plc_to_plc_ms:.1f}ms "

            print(msg)

            # store for FULL CT printing when next sensor first_hit happens
            self.cycle_logger.note_plc_sent(
                p.cam_id,
                {
                    "cycle_id": p.cycle_id,
                    "t_first_hit": p.t_first_hit,
                    "t_plc_sent": t_plc_sent,
                    "sensor_to_plc_ms": sensor_to_plc_ms,
                    "status": p.status,
                    "num_defect": p.num_defect,
                },
            )

            # release next cycle
            if self.serialize_per_cam:
                self.busy_flags[p.cam_id].clear()


class PersistWorker:
    def __init__(self, persist_q: Queue, out_dir: str = "out_persist", save_ext: str = "png"):
        self.persist_q = persist_q
        self.out_dir = out_dir
        self.save_ext = save_ext.lower().lstrip(".")
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._count = 0

    def start(self):
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join(timeout=2)

    def _run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        import json

        while not self._stop.is_set():
            try:
                p: PersistPayload = self.persist_q.get(timeout=0.2)
            except Empty:
                continue

            self._count += 1
            count = self._count
            ts_ms = int(p.ts_wall * 1000)

            stem = f"cam{p.cam_id}_{ts_ms}_{p.status}_cnt{count:06d}"
            img_path = os.path.join(self.out_dir, f"{stem}_raw.{self.save_ext}")
            log_path = os.path.join(self.out_dir, f"{stem}.json")
            p.log["count"] = count

            if self.save_ext == "png":
                cv2.imwrite(img_path, p.snap_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            elif self.save_ext in ("jpg", "jpeg"):
                cv2.imwrite(img_path, p.snap_bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])
            else:
                cv2.imwrite(img_path, p.snap_bgr)

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(p.log, f, ensure_ascii=False, indent=2)

            print(f"[PERSIST] {img_path}")


class UiOverlayWorker:
    def __init__(
        self,
        cams: List[CameraThread],
        ui_q: Queue,
        camera_on_evt: threading.Event,
        model_on_evt: threading.Event,
        preview_cam_id: int = 0,
        display_max_side: int = 960,
        idle_fps: int = 10,
    ):
        self.cams = cams
        self.ui_q = ui_q
        self.camera_on_evt = camera_on_evt
        self.model_on_evt = model_on_evt
        self.preview_cam_id = preview_cam_id
        self.display_max_side = display_max_side
        self.idle_dt = 1.0 / float(max(1, idle_fps))

        self._lock = threading.Lock()
        self._latest_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join(timeout=2)

    def get_latest_rgb(self) -> np.ndarray:
        with self._lock:
            bgr = self._latest_bgr.copy()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _set_latest(self, bgr: np.ndarray):
        bgr2 = self._resize_keep_aspect(bgr, self.display_max_side)
        with self._lock:
            self._latest_bgr = bgr2

    @staticmethod
    def _resize_keep_aspect(bgr: np.ndarray, max_side: int) -> np.ndarray:
        h, w = bgr.shape[:2]
        m = max(h, w)
        if m <= max_side:
            return bgr
        s = max_side / float(m)
        nw, nh = int(round(w * s)), int(round(h * s))
        return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    def _draw_overlay(self, p: UIPayload) -> np.ndarray:
        bgr = p.snap_bgr.copy()
        color = (0, 255, 0) if p.status == "OK" else (0, 0, 255)

        for i, box in enumerate(p.boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 3)
            if i < len(p.labels) and p.labels[i]:
                cv2.putText(
                    bgr, p.labels[i], (x1 + 8, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
                )

        h, w = bgr.shape[:2]
        text = f"Result: {p.status}  defects={p.num_defect}"
        org = (int(w * 0.05), int(h * 0.92))
        cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 6, cv2.LINE_AA)
        return bgr

    def _run(self):
        while not self._stop.is_set():
            if not self.camera_on_evt.is_set():
                self._set_latest(np.zeros((480, 640, 3), dtype=np.uint8))
                time.sleep(0.05)
                continue

            if not self.model_on_evt.is_set():
                cam_id = min(max(0, self.preview_cam_id), len(self.cams) - 1)
                bgr, _ = self.cams[cam_id].get_latest_bgr_small()
                if bgr is None:
                    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
                self._set_latest(bgr)
                time.sleep(self.idle_dt)
                continue

            try:
                p: UIPayload = self.ui_q.get(timeout=0.2)
            except Empty:
                continue

            bgr = self._draw_overlay(p)
            self._set_latest(bgr)


# -------------------------
# Runtime
# -------------------------
class PipelineRuntime:
    def __init__(
        self,
        num_cams: int = 1,
        target_fps: float = 10,
        preview_cam_id: int = 0,
        persist_dir: str = "out_persist",
        serialize_per_cam: bool = True,   # ✅ 핵심：PLC done 才允許下一顆 sensor
    ):
        self.num_cams = num_cams
        self.target_fps = target_fps
        self.preview_cam_id = preview_cam_id
        self.persist_dir = persist_dir
        self.serialize_per_cam = serialize_per_cam

        self.camera_on_evt = threading.Event()
        self.model_on_evt = threading.Event()

        self.cam_threads: List[CameraThread] = []
        self.sensor: Optional[YOLOSensor] = None
        self.defect: Optional[YOLODefectInspector] = None

        self.trigger_q: Queue = Queue(maxsize=32)
        self.plc_q: Queue = Queue(maxsize=64)
        self.persist_q: Queue = Queue(maxsize=64)
        self.ui_q: Queue = Queue(maxsize=16)

        self.busy_flags: List[threading.Event] = [threading.Event() for _ in range(num_cams)]
        self.cycle_logger = CycleLogger(num_cams=num_cams)

        self.sensor_w: Optional[SensorWorker] = None
        self.defect_w: Optional[DefectWorker] = None
        self.plc_w: Optional[PLCWorker] = None
        self.persist_w: Optional[PersistWorker] = None
        self.ui_w: Optional[UiOverlayWorker] = None

        self._lock = threading.Lock()

    # -------------------------
    # Camera
    # -------------------------
    def start_camera(self, exposure_us: float = 3000.0, gain: float = 8.0, sensor_max_side: int = 640):
        with self._lock:
            if self.camera_on_evt.is_set():
                return

            self.cam_threads = []
            self._drain_queue(self.trigger_q)
            self._drain_queue(self.plc_q)
            self._drain_queue(self.persist_q)
            self._drain_queue(self.ui_q)
            for e in self.busy_flags:
                e.clear()

            for i in range(self.num_cams):
                ct = CameraThread(
                    cam_id=i,
                    dev_index=i,
                    target_fps=self.target_fps,
                    sensor_size=sensor_max_side,
                    copy_on_grab=True,
                    latest_copy=False,
                    init_exposure_us=exposure_us,
                    init_gain=gain,
                    cam_kwargs={"debug": False},
                )
                ct.start()
                self.cam_threads.append(ct)

            # UI preview worker
            self.ui_w = UiOverlayWorker(
                cams=self.cam_threads,
                ui_q=self.ui_q,
                camera_on_evt=self.camera_on_evt,
                model_on_evt=self.model_on_evt,
                preview_cam_id=self.preview_cam_id,
                display_max_side=960,
                idle_fps=10,
            )
            self.ui_w.start()

            self.camera_on_evt.set()

    def stop_camera(self):
        with self._lock:
            self.stop_model()

            if self.ui_w:
                try:
                    self.ui_w.stop()
                except Exception:
                    pass
                self.ui_w = None

            for ct in self.cam_threads:
                try:
                    ct.stop()
                except Exception:
                    pass

            self.cam_threads = []
            self.camera_on_evt.clear()

            self._drain_queue(self.trigger_q)
            self._drain_queue(self.plc_q)
            self._drain_queue(self.persist_q)
            self._drain_queue(self.ui_q)
            for e in self.busy_flags:
                e.clear()

    # -------------------------
    # Model
    # -------------------------
    def start_model(
        self,
        sensor_weight_path: str,
        defect_weight_path: str,
        sensor_conf: float = 0.9,
        sensor_iou: float = 0.45,
        defect_conf: float = 0.3,
        defect_iou: float = 0.45,
        defect_imgsz: Optional[int] = 1280,
    ):
        with self._lock:
            if not self.camera_on_evt.is_set():
                raise RuntimeError("Camera is not open")
            if self.model_on_evt.is_set():
                return

            if self.sensor is None:
                self.sensor = YOLOSensor(
                    weight_path=sensor_weight_path,
                    conf=sensor_conf,
                    iou=sensor_iou,
                    imgsz=320,
                    classes=None,
                    fuse=True,
                )

            if self.defect is None:
                self.defect = YOLODefectInspector(
                    weight_path=defect_weight_path,
                    conf=defect_conf,
                    iou=defect_iou,
                    imgsz=defect_imgsz,
                    defect_classes=None,
                    fuse=True,
                )

            self._drain_queue(self.trigger_q)
            self._drain_queue(self.plc_q)
            self._drain_queue(self.persist_q)
            self._drain_queue(self.ui_q)
            for e in self.busy_flags:
                e.clear()

            self.sensor_w = SensorWorker(
                sensor=self.sensor,
                cams=self.cam_threads,
                trigger_q=self.trigger_q,
                camera_on_evt=self.camera_on_evt,
                model_on_evt=self.model_on_evt,
                busy_flags=self.busy_flags,
                serialize_per_cam=self.serialize_per_cam,
                cycle_logger=self.cycle_logger,
                per_cam_interval_s=0.03,
                trigger_after_n=2,
                cooldown_s=0.1,
                min_det=1,
            )

            self.defect_w = DefectWorker(
                defect=self.defect,
                cams=self.cam_threads,
                trigger_q=self.trigger_q,
                plc_q=self.plc_q,
                persist_q=self.persist_q,
                ui_q=self.ui_q,
                camera_on_evt=self.camera_on_evt,
                model_on_evt=self.model_on_evt,
                busy_flags=self.busy_flags,
                serialize_per_cam=self.serialize_per_cam,
                snapshot_timeout=0.5,
            )

            self.plc_w = PLCWorker(
                plc_q=self.plc_q,
                model_on_evt=self.model_on_evt,
                num_cams=self.num_cams,
                busy_flags=self.busy_flags,
                serialize_per_cam=self.serialize_per_cam,
                cycle_logger=self.cycle_logger,
            )

            self.persist_w = PersistWorker(self.persist_q, out_dir=self.persist_dir, save_ext="png")

            self.model_on_evt.set()
            self.sensor_w.start()
            self.defect_w.start()
            self.plc_w.start()
            self.persist_w.start()

    def stop_model(self):
        with self._lock:
            if not self.model_on_evt.is_set():
                return

            self.model_on_evt.clear()

            for w in [self.sensor_w, self.defect_w, self.plc_w, self.persist_w]:
                if w:
                    try:
                        w.stop()
                    except Exception:
                        pass

            self.sensor_w = None
            self.defect_w = None
            self.plc_w = None
            self.persist_w = None

            self._drain_queue(self.trigger_q)
            self._drain_queue(self.plc_q)
            self._drain_queue(self.persist_q)
            self._drain_queue(self.ui_q)
            for e in self.busy_flags:
                e.clear()

    # -------------------------
    # UI
    # -------------------------
    def get_ui_frame_rgb(self) -> np.ndarray:
        if self.ui_w is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self.ui_w.get_latest_rgb()

    @staticmethod
    def _drain_queue(q: Queue):
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass
