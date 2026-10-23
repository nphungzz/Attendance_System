# ============================================================
#  core/pipeline.py  –  Realtime Attendance Pipeline
#  3 thread: main (display) / detect / recog
# ============================================================

import cv2
import os
import time
import numpy as np
import threading
from core.detector   import FaceDetector
from core.stability  import StabilityChecker
from core.quality    import QualityChecker
from core.recognizer import FaceRecognizer
from core.attendance import AttendanceLogger
from config import DETECT_RESIZE_WIDTH, BASE_DIR

COLOR_KNOWN    = (0, 255,   0)
COLOR_UNKNOWN  = (0,   0, 255)
COLOR_CHECKING = (0, 255, 255)

DEBUG_FRAME_DIR = r"C:\Users\admin\Documents\attedance system\Data\frame_crop"
DEBUG_CROP_DIR  = r"C:\Users\admin\Documents\attedance system\Data\crop_face"
os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
os.makedirs(DEBUG_CROP_DIR,  exist_ok=True)


class AttendancePipeline:

    def __init__(self):
        print("[Pipeline] Initializing modules…")
        self.detector   = FaceDetector()
        self.stability  = StabilityChecker()
        self.quality    = QualityChecker()
        self.recognizer = FaceRecognizer()
        self.attendance = AttendanceLogger()

        self._lock             = threading.Lock()
        self._detect_frame     = None
        self._detecting        = False
        self._recog_frame      = None
        self._recognizing      = False
        self._last_result      = None
        self._last_box         = None
        self._last_quality_fail = None   # lý do fail quality check gần nhất
        self._save_counter     = 0

        threading.Thread(target=self._detect_worker, daemon=True).start()
        threading.Thread(target=self._recog_worker,  daemon=True).start()
        print("[Pipeline] Ready  (3-thread mode)")

    # ════════════════════════════════════════════════════════
    #  Thread 2: OpenCV SSD Detect
    # ════════════════════════════════════════════════════════

    def _detect_worker(self):
        while True:
            with self._lock:
                frame = self._detect_frame
                self._detect_frame = None

            if frame is None:
                time.sleep(0.003)
                continue

            h, w = frame.shape[:2]
            scale = DETECT_RESIZE_WIDTH / w
            small = cv2.resize(frame, (DETECT_RESIZE_WIDTH, int(h * scale)))

            face_info = self.detector.largest_face(small)

            with self._lock:
                self._detecting = False

                if face_info is None:
                    self._last_box = None
                    self.stability.reset()
                    continue

                sx, sy, sw, sh = face_info["box"]
                box = (int(sx/scale), int(sy/scale),
                       int(sw/scale), int(sh/scale))

                stable = self.stability.update(box)
                self._last_box = (box, face_info["confidence"], stable)

                if stable and not self._recognizing:
                    face_crop = self.detector.crop_face(frame, box)
                    if face_crop.size == 0:
                        continue

                    # ── Quality Check (3 bước) ───────────────
                    quality = self.quality.check(face_crop, box)

                    if not quality["passed"]:
                        # Lưu lý do fail để hiển thị lên màn hình
                        self._last_quality_fail = quality["fail_reason"]
                        continue

                    # Quality passed → xóa thông báo fail
                    self._last_quality_fail = None

                    # ── Lưu debug frames ─────────────────────
                    self._save_counter += 1
                    idx = self._save_counter

                    frame_debug = frame.copy()
                    x, y, bw, bh = box
                    cv2.rectangle(frame_debug, (x, y), (x+bw, y+bh), COLOR_KNOWN, 2)
                    cv2.imwrite(os.path.join(DEBUG_FRAME_DIR, f"frame_{idx:04d}.jpg"), frame_debug)
                    cv2.imwrite(os.path.join(DEBUG_CROP_DIR,  f"crop_{idx:04d}.jpg"), face_crop)
                    print(f"[Debug] #{idx} | blur:{quality['blur_score']} "
                          f"bright:{quality['brightness']} "
                          f"size:{quality['face_w']}x{quality['face_h']}")

                    self._recog_frame = frame
                    self._recognizing = True

    # ════════════════════════════════════════════════════════
    #  Thread 3: InsightFace Recognition
    # ════════════════════════════════════════════════════════

    def _recog_worker(self):
        while True:
            with self._lock:
                frame = self._recog_frame
                self._recog_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            result = self.recognizer.recognize(frame)

            if result["known"]:
                result["newly_marked"] = self.attendance.mark(result["name"])
            else:
                result["newly_marked"] = False

            with self._lock:
                self._last_result  = result
                self._recognizing  = False

    # ════════════════════════════════════════════════════════
    #  Thread 1 (main): đọc frame + hiển thị
    # ════════════════════════════════════════════════════════

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()

        with self._lock:
            if not self._detecting:
                self._detect_frame = frame
                self._detecting    = True

        with self._lock:
            box_info     = self._last_box
            result       = self._last_result
            quality_fail = self._last_quality_fail

        if box_info is None:
            self._draw_status(display, "No face detected", COLOR_UNKNOWN)
            return display

        box, conf, stable = box_info
        x, y, bw, bh = box
        color = COLOR_KNOWN if stable else COLOR_CHECKING
        cv2.rectangle(display, (x, y), (x+bw, y+bh), color, 2)
        cv2.putText(display, f"conf:{conf:.2f}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if not stable:
            self._draw_status(display, "Stabilizing…", COLOR_CHECKING)
        elif quality_fail is not None:
            # Hiển thị lý do fail quality: Too small / Too dark / Blurry
            self._draw_status(display, f"Quality: {quality_fail}", COLOR_UNKNOWN)
        elif result is None:
            self._draw_status(display, "Identifying…", COLOR_CHECKING)
        else:
            self._draw_result(display, result, box)

        return display

    # ── Drawing helpers ──────────────────────────────────────

    def _draw_result(self, frame, result, box):
        name   = result["name"]
        sim    = result["similarity"]
        known  = result["known"]
        marked = result.get("newly_marked", False)

        color = COLOR_KNOWN if known else COLOR_UNKNOWN
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} ({sim:.2f})", (x, y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "Present" if known else "Unknown", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if marked:
            self._draw_status(frame, f"Marked: {name}", COLOR_KNOWN)

    def _draw_status(self, frame, text, color):
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)