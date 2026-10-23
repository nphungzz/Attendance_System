# ============================================================
#  core/quality.py  –  Quality Check
#
#  Các bước kiểm tra (theo thứ tự, fail sớm thì dừng):
#    1. Face Size     – mặt đủ lớn trong frame không?
#    2. Brightness    – ảnh không quá tối / quá sáng?
#    3. Blur          – ảnh đủ nét không? (Variance of Laplacian)
# ============================================================

import cv2
import numpy as np
from config import (
    BLUR_THRESHOLD,
    MIN_FACE_SIZE,
    BRIGHTNESS_MIN,
    BRIGHTNESS_MAX,
)


class QualityChecker:
    """
    Kiểm tra chất lượng ảnh khuôn mặt trước khi đưa vào ArcFace.

    Lưu ý:
    - KHÔNG cân bằng sáng (CLAHE) vì InsightFace buffalo_l
      đã tự normalize pixel bên trong model.
    - Chỉ LỌCBỎ ảnh quá kém, không xử lý lại ảnh.
    """

    def __init__(self):
        self.blur_threshold  = BLUR_THRESHOLD
        self.min_face_size   = MIN_FACE_SIZE
        self.brightness_min  = BRIGHTNESS_MIN
        self.brightness_max  = BRIGHTNESS_MAX

    # ── Public ──────────────────────────────────────────────

    def check(self, face_img: np.ndarray, box: tuple = None) -> dict:
        """
        Chạy toàn bộ quality check theo thứ tự ưu tiên.

        Args:
            face_img : ảnh BGR đã crop vùng mặt
            box      : (x, y, w, h) bounding box gốc (để check face size)
                       Nếu None → bỏ qua face size check

        Returns:
            {
                "passed"          : bool,
                "fail_reason"     : str | None,
                "blur_score"      : float,
                "brightness"      : float,
                "face_w"          : int,
                "face_h"          : int,
            }
        """
        result = {
            "passed"      : False,
            "fail_reason" : None,
            "blur_score"  : 0.0,
            "brightness"  : 0.0,
            "face_w"      : 0,
            "face_h"      : 0,
        }

        # ── 1. Face Size Check ───────────────────────────────
        if box is not None:
            _, _, bw, bh = box
            result["face_w"] = bw
            result["face_h"] = bh
            if bw < self.min_face_size or bh < self.min_face_size:
                result["fail_reason"] = f"Too small ({bw}x{bh})"
                return result
        else:
            h, w = face_img.shape[:2]
            result["face_w"] = w
            result["face_h"] = h

        # ── 2. Brightness Check ──────────────────────────────
        gray       = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        result["brightness"] = round(brightness, 1)

        if brightness < self.brightness_min:
            result["fail_reason"] = f"Too dark ({brightness:.0f})"
            return result
        if brightness > self.brightness_max:
            result["fail_reason"] = f"Too bright ({brightness:.0f})"
            return result

        # ── 3. Blur Check (Variance of Laplacian) ────────────
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        result["blur_score"] = round(blur_score, 2)

        if blur_score < self.blur_threshold:
            result["fail_reason"] = f"Blurry ({blur_score:.1f})"
            return result

        # ── All passed ───────────────────────────────────────
        result["passed"] = True
        return result