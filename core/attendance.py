# ============================================================
#  core/attendance.py  –  Attendance Logger (CSV)
# ============================================================

import os
import csv
from datetime import datetime
from config import ATTENDANCE_CSV


class AttendanceLogger:
    """
    Ghi nhận điểm danh vào CSV.

    - Mỗi sinh viên chỉ được điểm danh 1 lần / session.
    - Nếu file chưa tồn tại sẽ tự tạo header.
    """

    CSV_HEADER = ["StudentID", "Time", "Status"]

    def __init__(self):
        self._marked: set[str] = set()   # tập SV đã điểm danh trong session
        self._ensure_csv()

    # ── Public ──────────────────────────────────────────────

    def mark(self, student_id: str) -> bool:
        """
        Điểm danh một sinh viên.

        Returns:
            True  → điểm danh thành công (lần đầu)
            False → đã điểm danh rồi, bỏ qua
        """
        if student_id in self._marked:
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_row(student_id, timestamp, "Present")
        self._marked.add(student_id)

        print(f"[Attendance] ✓ {student_id} – {timestamp}")
        return True

    def is_marked(self, student_id: str) -> bool:
        return student_id in self._marked

    def get_marked(self) -> set[str]:
        return set(self._marked)

    def reset_session(self):
        """Xóa bộ nhớ session (dùng khi bắt đầu buổi điểm danh mới)."""
        self._marked.clear()

    # ── Private ─────────────────────────────────────────────

    def _ensure_csv(self):
        """Tạo file CSV với header nếu chưa tồn tại."""
        os.makedirs(os.path.dirname(ATTENDANCE_CSV), exist_ok=True)
        if not os.path.exists(ATTENDANCE_CSV):
            with open(ATTENDANCE_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADER)

    def _write_row(self, student_id: str, timestamp: str, status: str):
        with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([student_id, timestamp, status])
