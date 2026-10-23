# ============================================================
#  core/stability.py  –  Stability Check (không dùng AI)
# ============================================================

from config import STABILITY_THRESHOLD, STABILITY_MIN_COUNT


class StabilityChecker:
    """
    Đảm bảo khuôn mặt đứng yên đủ lâu trước khi embedding.

    Logic:
        - So sánh bounding box frame hiện tại với frame trước.
        - Nếu |Δx|, |Δy|, |Δw|, |Δh| đều < threshold → stable_count++
        - Khi stable_count >= min_count → is_stable() trả về True
        - Nếu box thay đổi nhiều → reset về 0
    """

    def __init__(self,
                 threshold: int = STABILITY_THRESHOLD,
                 min_count: int = STABILITY_MIN_COUNT):
        self.threshold   = threshold
        self.min_count   = min_count
        self._prev_box   = None
        self._count      = 0

    # ── Public ──────────────────────────────────────────────

    def update(self, box: tuple | None) -> bool:
        """
        Cập nhật với bounding box mới.

        Args:
            box : (x, y, w, h) hoặc None nếu không detect được mặt.

        Returns:
            True nếu đang ổn định đủ lâu.
        """
        if box is None:
            self.reset()
            return False

        if self._prev_box is None:
            self._prev_box = box
            self._count = 1
            return False

        dx = abs(box[0] - self._prev_box[0])
        dy = abs(box[1] - self._prev_box[1])
        dw = abs(box[2] - self._prev_box[2])
        dh = abs(box[3] - self._prev_box[3])

        if dx < self.threshold and dy < self.threshold \
                and dw < self.threshold and dh < self.threshold:
            self._count += 1
        else:
            self._count = 1     # reset, không về 0 để bắt frame mới ngay

        self._prev_box = box
        return self._count >= self.min_count

    def is_stable(self) -> bool:
        return self._count >= self.min_count

    def reset(self):
        self._prev_box = None
        self._count    = 0
