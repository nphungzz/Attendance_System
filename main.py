# ============================================================
#  main.py  –  Realtime Attendance
#  Q=Quit  R=Reload DB  C=Show Checked-in
# ============================================================


import cv2
import time
from core.pipeline import AttendancePipeline
from config import CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT


def main():
    print("=" * 55)
    print("  REALTIME ATTENDANCE SYSTEM")
    print("  Q=Quit  R=Reload DB  C=Show Checked-in")
    print("=" * 55)

    pipeline = AttendancePipeline()

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {CAMERA_SOURCE}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # FPS tracking
    fps        = 0.0
    fps_timer  = time.time()
    fps_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        result_frame = pipeline.process_frame(frame)

        # ── FPS counter ──────────────────────────────────────
        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps       = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # ── Số SV đã điểm danh ───────────────────────────────
        marked_count = len(pipeline.attendance.get_marked())
        cv2.putText(result_frame, f"Checked-in: {marked_count}",
                    (FRAME_WIDTH - 210, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Attendance System", result_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("r"), ord("R")):
            pipeline.recognizer.reload_database()
            print("[Info] Database reloaded.")
        elif key in (ord("c"), ord("C")):
            marked = pipeline.attendance.get_marked()
            print(f"\n[Checked-in] {len(marked)} students:")
            for sid in sorted(marked):
                print(f"  - {sid}")

    cap.release()
    cv2.destroyAllWindows()

    marked = pipeline.attendance.get_marked()
    print(f"\n[Done] {len(marked)} student(s) marked present.")
    for sid in sorted(marked):
        print(f"  ✓ {sid}")


if __name__ == "__main__":
    main()