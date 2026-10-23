# ============================================================
#  config.py  –  Cấu hình toàn bộ hệ thống điểm danh
# ============================================================

import os

# ── Đường dẫn gốc project ───────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath("BASE_DIR"))

# ── Dataset & Database ──────────────────────────────────────
DATASET_DIR      = "DATASET_DIR"
EMBEDDINGS_PATH  = "EMBEDDINGS_PATH"
ATTENDANCE_CSV   = "ATTENDANCE_CSV_path"


# ── Models OpenCV SSD ────────────────────────────────────────
PROTOTXT_PATH    = "PROTOTXT_PATH"
CAFFEMODEL_PATH  = "CAFFEMODEL_PATH"

# ── Face Detection ───────────────────────────────────────────
DETECTION_CONFIDENCE = 0.5      # ngưỡng tin cậy khi detect
FACE_PADDING         = 20       # pixel padding khi crop mặt
 
# ── InsightFace / ArcFace ────────────────────────────────────
INSIGHTFACE_MODEL    = "buffalo_l"
INSIGHTFACE_CTX_ID   = 0        # 0 = GPU (cuda:0), -1 = CPU
 
# ── Stability Check ──────────────────────────────────────────
STABILITY_THRESHOLD  = 15       # pixel, độ lệch tối đa giữa 2 frame
STABILITY_MIN_COUNT  = 5        # số frame liên tiếp phải ổn định
 
# ── Quality Check ────────────────────────────────────────────
# 1. Face Size: mặt phải đủ lớn (tránh embed mặt quá nhỏ → sai)
MIN_FACE_SIZE        = 80       # pixel, cạnh nhỏ nhất của bounding box
 
# 2. Brightness: lọc ảnh quá tối hoặc quá sáng
#    mean pixel value grayscale (0–255)
#    Không dùng CLAHE vì InsightFace buffalo_l tự normalize bên trong
BRIGHTNESS_MIN       = 40       # < 40  → quá tối
BRIGHTNESS_MAX       = 220      # > 220 → quá sáng / ngược sáng
 
# 3. Blur: Variance of Laplacian
BLUR_THRESHOLD       = 120.0    # < 120 → ảnh mờ → bỏ qua
 
# ── Recognition ──────────────────────────────────────────────
COSINE_THRESHOLD     = 0.45     # similarity >= threshold → nhận diện được
 
# ── Performance ──────────────────────────────────────────────
RECOGNITION_EVERY_N_FRAMES = 5
DETECT_RESIZE_WIDTH        = 320
 

# ── Camera / Stream ──────────────────────────────────────────
CAMERA_SOURCE        = "CAMERA_SOURCE"
FRAME_WIDTH          = 1280
FRAME_HEIGHT         = 720