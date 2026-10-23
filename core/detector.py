# ============================================================
#  core/detector.py  –  Face Detection (OpenCV DNN SSD)
# ============================================================

import cv2
import numpy as np
from config import (
    PROTOTXT_PATH, CAFFEMODEL_PATH,
    DETECTION_CONFIDENCE, FACE_PADDING
)


class FaceDetector:
    """
    Phát hiện khuôn mặt bằng OpenCV DNN SSD (ResNet-10).
    """

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

    # ── Public ──────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Trả về danh sách các mặt được phát hiện.

        Mỗi phần tử là dict:
            {
                "box"        : (x, y, w, h),   # pixel trong frame gốc
                "confidence" : float
            }
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < DETECTION_CONFIDENCE:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Đảm bảo không vượt biên ảnh
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            results.append({
                "box"        : (x1, y1, x2 - x1, y2 - y1),
                "confidence" : conf
            })

        return results

    def largest_face(self, frame: np.ndarray) -> dict | None:
        """
        Trả về mặt lớn nhất trong frame (diện tích w*h lớn nhất).
        Trả về None nếu không có mặt nào.
        """
        faces = self.detect(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: f["box"][2] * f["box"][3])

    def crop_face(self, frame: np.ndarray, box: tuple, padding: int = FACE_PADDING) -> np.ndarray:
        """
        Crop khuôn mặt từ frame theo bounding box, có padding.

        Args:
            frame   : ảnh gốc BGR
            box     : (x, y, w, h)
            padding : pixel mở rộng ra 4 phía

        Returns:
            Ảnh BGR đã crop.
        """
        h, w = frame.shape[:2]
        x, y, bw, bh = box

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)

        return frame[y1:y2, x1:x2]
